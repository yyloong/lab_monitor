from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# 连续多少次采样 CPU=0 才判定为停滞（避免误报）
_CPU_STALL_THRESHOLD = 3

# 分布式训练的启动器关键词，用于识别父进程
_LAUNCHER_KEYWORDS = {"torchrun", "deepspeed", "accelerate", "mpirun", "mpiexec"}


class ProcessStatus(Enum):
    RUNNING = "running"
    MISSING = "missing"   # 进程消失
    ZOMBIE = "zombie"     # 僵尸进程
    STALLED = "stalled"   # CPU 长时间为 0


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cmdline: str
    status: str
    cpu_percent: float
    memory_mb: float
    is_launcher: bool = False
    children_pids: list[int] = field(default_factory=list)


@dataclass
class ProcessGroupStatus:
    exp_name: str
    status: ProcessStatus
    processes: list[ProcessInfo] = field(default_factory=list)
    alert_reason: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.status == ProcessStatus.RUNNING

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_name": self.exp_name,
            "status": self.status.value,
            "process_count": len(self.processes),
            "alert_reason": self.alert_reason,
            "processes": [
                {
                    "pid": p.pid,
                    "name": p.name,
                    "cmdline": p.cmdline[:120],
                    "status": p.status,
                    "cpu_percent": p.cpu_percent,
                    "memory_mb": p.memory_mb,
                }
                for p in self.processes
            ],
        }


class ProcessMonitor:
    """监控与实验关键词匹配的进程组，支持分布式训练场景。"""

    def __init__(self) -> None:
        # exp_name -> {pid: stall_count}
        self._stall_counters: dict[str, dict[int, int]] = {}
        # exp_name -> last known pids (用于检测进程消失)
        self._known_pids: dict[str, set[int]] = {}

    def check(
        self,
        exp_name: str,
        keywords: list[str],
        cwd: str = "",
        extra_keywords: list[str] | None = None,
        pids: list[int] | None = None,
        gpu_utilizations: list[int] | None = None,
    ) -> ProcessGroupStatus:
        """返回与该实验匹配的进程组状态。

        优先级：pids > 关键词扫描
        - pids 不为空：直接监控指定 PID 及其子进程，跳过关键词扫描
        - pids 为空：通过 keywords / cwd / extra_keywords 扫描匹配
        - gpu_utilizations：该实验相关 GPU 的利用率列表，用于 STALLED 判断
        """
        if pids:
            matched = self._collect_from_pids(pids)
        else:
            matched = self._find_matching_processes(keywords, cwd, extra_keywords or [])

        if not matched:
            if self._known_pids.get(exp_name):
                self._known_pids[exp_name] = set()
                return ProcessGroupStatus(
                    exp_name=exp_name,
                    status=ProcessStatus.MISSING,
                    alert_reason="训练进程已消失（可能异常退出或已完成）",
                )
            if pids:
                return ProcessGroupStatus(
                    exp_name=exp_name,
                    status=ProcessStatus.MISSING,
                    alert_reason=f"指定 PID {pids} 均不存在",
                )
            return ProcessGroupStatus(
                exp_name=exp_name,
                status=ProcessStatus.MISSING,
                alert_reason="未找到匹配的训练进程",
            )

        # 更新已知 PID
        current_pids = {p.pid for p in matched}
        self._known_pids[exp_name] = current_pids

        # 检查僵尸进程
        zombie_procs = [p for p in matched if p.status == "zombie"]
        if zombie_procs:
            return ProcessGroupStatus(
                exp_name=exp_name,
                status=ProcessStatus.ZOMBIE,
                processes=matched,
                alert_reason=f"存在 {len(zombie_procs)} 个僵尸进程: "
                f"{[p.pid for p in zombie_procs]}",
            )

        # 检测 CPU 停滞
        stall_status = self._check_stall(exp_name, matched, gpu_utilizations or [])
        if stall_status:
            return stall_status

        return ProcessGroupStatus(
            exp_name=exp_name,
            status=ProcessStatus.RUNNING,
            processes=matched,
        )

    def reset_known_pids(self, exp_name: str) -> None:
        """重置已知 PID，下次检查不会误报进程消失。"""
        self._known_pids.pop(exp_name, None)
        self._stall_counters.pop(exp_name, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_from_pids(self, pids: list[int]) -> list[ProcessInfo]:
        """直接从指定 PID 收集进程信息，并递归纳入其子进程。"""
        all_pids: set[int] = set()
        launcher_pids: set[int] = set()

        for pid in pids:
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline() or []).lower()
                all_pids.add(pid)
                if any(lk in cmdline for lk in _LAUNCHER_KEYWORDS):
                    launcher_pids.add(pid)
                for child in proc.children(recursive=True):
                    all_pids.add(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        result: list[ProcessInfo] = []
        for pid in all_pids:
            info = self._get_proc_info(pid, launcher_pids)
            if info is not None:
                result.append(info)
        return result

    def _find_matching_processes(
        self,
        keywords: list[str],
        cwd: str = "",
        extra_keywords: list[str] | None = None,
    ) -> list[ProcessInfo]:
        """扫描所有进程，找出命令行包含任意关键词的进程（含分布式 rank 子进程）。

        cwd 不为空时要求进程工作目录包含该字符串。
        extra_keywords 不为空时要求命令行同时包含所有这些关键词（AND 逻辑）。
        """
        matched_pids: set[int] = set()
        launcher_pids: set[int] = set()

        for proc in psutil.process_iter(["pid", "name", "cmdline", "status", "ppid", "cwd"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                name = proc.info["name"] or ""
                combined = f"{name} {cmdline}".lower()

                # OR 匹配：命令行含任意一个关键词
                if not any(kw.lower() in combined for kw in keywords):
                    continue

                # AND 匹配：命令行必须同时含所有 extra_keywords
                if extra_keywords and not all(ek.lower() in combined for ek in extra_keywords):
                    continue

                # cwd 过滤
                if cwd:
                    try:
                        proc_cwd = proc.info.get("cwd") or proc.cwd()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        proc_cwd = ""
                    if cwd.lower() not in (proc_cwd or "").lower():
                        continue

                matched_pids.add(proc.pid)
                if any(lk in combined for lk in _LAUNCHER_KEYWORDS):
                    launcher_pids.add(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 对于启动器进程，将其所有子进程也纳入监控
        for launcher_pid in list(launcher_pids):
            try:
                launcher_proc = psutil.Process(launcher_pid)
                for child in launcher_proc.children(recursive=True):
                    matched_pids.add(child.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        result: list[ProcessInfo] = []
        for pid in matched_pids:
            info = self._get_proc_info(pid, launcher_pids)
            if info is not None:
                result.append(info)

        return result

    @staticmethod
    def _get_proc_info(pid: int, launcher_pids: set[int]) -> ProcessInfo | None:
        try:
            proc = psutil.Process(pid)
            with proc.oneshot():
                cmdline = " ".join(proc.cmdline() or [])
                mem = proc.memory_info().rss / (1024 * 1024)
                cpu = proc.cpu_percent(interval=None)
                children = [c.pid for c in proc.children()]
            return ProcessInfo(
                pid=pid,
                name=proc.name(),
                cmdline=cmdline,
                status=proc.status(),
                cpu_percent=cpu,
                memory_mb=round(mem, 1),
                is_launcher=(pid in launcher_pids),
                children_pids=children,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    @staticmethod
    def _has_active_connections(pid: int) -> bool:
        """检查进程是否有活跃的 TCP/UDP 网络连接（ESTABLISHED 或 SYN_SENT）。"""
        try:
            proc = psutil.Process(pid)
            for conn in proc.net_connections(kind="inet"):
                if conn.status in ("ESTABLISHED", "SYN_SENT", "SYN_RECV"):
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass
        return False

    def _check_stall(
        self, exp_name: str, procs: list[ProcessInfo], gpu_utilizations: list[int] = [],
    ) -> ProcessGroupStatus | None:
        """检测进程组停滞：进程 CPU=0 且 GPU 利用率全为 0 且无活跃网络连接，三条件同时满足才告警。"""
        counters = self._stall_counters.setdefault(exp_name, {})
        worker_procs = [p for p in procs if not p.is_launcher]

        # GPU 是否全部空闲
        gpu_all_idle = (not gpu_utilizations) or all(u == 0 for u in gpu_utilizations)

        stalled_pids: list[int] = []
        for p in worker_procs:
            if p.cpu_percent == 0.0:
                # GPU 还在跑（如 forward/backward），说明进程在等 GPU 结果，不算卡死
                if not gpu_all_idle:
                    counters.pop(p.pid, None)
                    continue
                # 有活跃网络连接，说明在等待外部 API，不算卡死
                if self._has_active_connections(p.pid):
                    counters.pop(p.pid, None)
                    continue
                counters[p.pid] = counters.get(p.pid, 0) + 1
                if counters[p.pid] >= _CPU_STALL_THRESHOLD:
                    stalled_pids.append(p.pid)
            else:
                counters.pop(p.pid, None)

        # 清理已退出进程的计数
        alive_pids = {p.pid for p in procs}
        for pid in list(counters.keys()):
            if pid not in alive_pids:
                del counters[pid]

        if stalled_pids:
            return ProcessGroupStatus(
                exp_name=exp_name,
                status=ProcessStatus.STALLED,
                processes=procs,
                alert_reason=(
                    f"进程 CPU 长时间为 0，疑似卡死 (PIDs: {stalled_pids}，"
                    f"连续 {_CPU_STALL_THRESHOLD} 次检测)"
                ),
            )
        return None
