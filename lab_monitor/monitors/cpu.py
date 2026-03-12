from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class CPUStat:
    cpu_percent_total: float        # 全局 CPU 使用率 %
    cpu_percent_per_core: list[float]
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    load_avg_1m: float              # 1 分钟负载均值（Linux）
    load_avg_5m: float
    process_cpu_percent: float      # 属于该实验的进程 CPU 占用之和
    process_memory_mb: float        # 属于该实验的进程内存之和
    alert: bool = False
    alert_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent_total": self.cpu_percent_total,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "memory_percent": self.memory_percent,
            "load_avg_1m": self.load_avg_1m,
            "load_avg_5m": self.load_avg_5m,
            "process_cpu_percent": self.process_cpu_percent,
            "process_memory_mb": self.process_memory_mb,
            "alert": self.alert,
            "alert_reasons": self.alert_reasons,
        }


@dataclass
class CPUAlertThresholds:
    memory_percent: float = 90.0    # 系统内存占用超过此值告警
    cpu_percent: float = 95.0       # 全局 CPU 超过此值告警


class CPUMonitor:
    """采集系统 CPU / 内存状态，以及实验进程的资源占用。"""

    def collect(
        self,
        process_pids: set[int],
        thresholds: CPUAlertThresholds,
    ) -> CPUStat:
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()

        try:
            load_avg = psutil.getloadavg()
            load_1m, load_5m = load_avg[0], load_avg[1]
        except AttributeError:
            load_1m = load_5m = 0.0

        proc_cpu = 0.0
        proc_mem_mb = 0.0
        for pid in process_pids:
            try:
                p = psutil.Process(pid)
                proc_cpu += p.cpu_percent(interval=None)
                proc_mem_mb += p.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        alert_reasons: list[str] = []
        if mem.percent >= thresholds.memory_percent:
            alert_reasons.append(
                f"系统内存占用 {mem.percent:.1f}% >= {thresholds.memory_percent}%"
            )
        if cpu_total >= thresholds.cpu_percent:
            alert_reasons.append(
                f"CPU 使用率 {cpu_total:.1f}% >= {thresholds.cpu_percent}%"
            )

        return CPUStat(
            cpu_percent_total=cpu_total,
            cpu_percent_per_core=cpu_per_core if isinstance(cpu_per_core, list) else [],
            memory_used_gb=round((mem.total - mem.available) / (1024 ** 3), 2),
            memory_total_gb=round(mem.total / (1024 ** 3), 2),
            memory_percent=mem.percent,
            load_avg_1m=round(load_1m, 2),
            load_avg_5m=round(load_5m, 2),
            process_cpu_percent=round(proc_cpu, 1),
            process_memory_mb=round(proc_mem_mb, 1),
            alert=bool(alert_reasons),
            alert_reasons=alert_reasons,
        )
