from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from lab_monitor.analyzer.llm import create_analyzer
from lab_monitor.monitors.cpu import CPUMonitor
from lab_monitor.monitors.gpu import GPUMonitor
from lab_monitor.monitors.process import ProcessMonitor, ProcessStatus
from lab_monitor.monitors.wandb_monitor import WandbMonitor
from lab_monitor.notifier.feishu import FeiShuNotifier

if TYPE_CHECKING:
    from lab_monitor.config import AppConfig, ExperimentConfig

logger = logging.getLogger(__name__)


class MonitorScheduler:
    """统一调度所有监控任务，每个实验独立轮询，故障隔离。"""

    def __init__(self, config: "AppConfig") -> None:
        self._cfg = config
        self._notifier = FeiShuNotifier(
            webhook_url=config.feishu.webhook_url,
            cooldown_minutes=config.schedule.alert_cooldown_minutes,
            app_id=config.feishu.app_id,
            app_secret=config.feishu.app_secret,
        )
        self._gpu_monitor = GPUMonitor()
        self._cpu_monitor = CPUMonitor()
        self._process_monitor = ProcessMonitor()
        self._wandb_monitor = WandbMonitor(
            entity=config.wandb.entity,
        )
        self._analyzer = create_analyzer(
            provider=config.llm.provider,
            api_key=config.llm.api_key,
            model=config.llm.model,
            base_url=config.llm.base_url,
            vision_enabled=config.llm.vision_enabled,
            max_input_chars=config.llm.max_input_chars,
            max_tokens=config.llm.max_tokens,
        )
        # APScheduler：线程池执行，实验间互不阻塞
        executors = {"default": ThreadPoolExecutor(max_workers=len(config.experiments) * 3 + 2)}
        self._scheduler = BackgroundScheduler(executors=executors, timezone="Asia/Shanghai")
        # exp_name -> 上次发送 GPU 定期报告的时间戳
        self._last_gpu_report: dict[str, float] = {}

    def start(self) -> None:
        """注册所有任务并启动调度器。"""
        sched = self._cfg.schedule
        for exp in self._cfg.experiments:
            safe_name = exp.name.replace(" ", "_")

            self._scheduler.add_job(
                self._run_process_check,
                "interval",
                seconds=sched.process_interval_seconds,
                args=[exp],
                id=f"process_{safe_name}",
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.add_job(
                self._run_gpu_check,
                "interval",
                seconds=sched.gpu_interval_seconds,
                args=[exp],
                id=f"gpu_{safe_name}",
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.add_job(
                self._run_cpu_check,
                "interval",
                seconds=sched.gpu_interval_seconds,
                args=[exp],
                id=f"cpu_{safe_name}",
                max_instances=1,
                coalesce=True,
            )
            if exp.wandb_project:
                self._scheduler.add_job(
                    self._run_wandb_check,
                    "interval",
                    seconds=sched.wandb_interval_seconds,
                    args=[exp],
                    id=f"wandb_{safe_name}",
                    max_instances=1,
                    coalesce=True,
                )

        self._scheduler.start()
        logger.info(
            "Scheduler started. Monitoring %d experiments.", len(self._cfg.experiments)
        )

        # 启动后立即执行一次全量检查，推送接入通知和首次分析
        for exp in self._cfg.experiments:
            self._scheduler.add_job(
                self._run_startup_check,
                args=[exp],
                id=f"startup_{exp.name.replace(' ', '_')}",
            )

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    # ------------------------------------------------------------------
    # Job handlers
    # ------------------------------------------------------------------

    def _run_process_check(self, exp: "ExperimentConfig") -> None:
        """进程健康检查任务。"""
        try:
            # 获取当前 GPU 利用率，用于 STALLED 判断（CPU+GPU+网络三条件）
            gpu_utilizations: list[int] = []
            try:
                gpu_ids = list(exp.gpu_ids)
                if not gpu_ids:
                    proc_status = self._process_monitor.check(
                        exp.name, exp.process_keywords,
                        cwd=exp.process_cwd, extra_keywords=exp.process_extra_keywords,
                        pids=exp.process_pids,
                    )
                    gpu_ids = self._gpu_monitor.resolve_gpu_ids({p.pid for p in proc_status.processes})
                if gpu_ids:
                    stats = self._gpu_monitor.collect(gpu_ids, exp.gpu_alert_thresholds)
                    gpu_utilizations = [s.utilization for s in stats if s is not None]
            except Exception:  # noqa: BLE001
                pass

            status = self._process_monitor.check(
                exp.name, exp.process_keywords,
                cwd=exp.process_cwd, extra_keywords=exp.process_extra_keywords,
                pids=exp.process_pids, gpu_utilizations=gpu_utilizations,
            )
            if not status.is_healthy:
                logger.warning(
                    "[%s] Process alert: %s", exp.name, status.alert_reason
                )
                details = self._format_process_details(status)
                self._notifier.send_process_alert(
                    exp_name=exp.name,
                    reason=status.alert_reason,
                    details=details,
                    dedup_key=f"process_{exp.name}_{status.status.value}",
                )
            else:
                logger.debug("[%s] Processes healthy (%d procs)", exp.name, len(status.processes))
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Process check failed: %s", exp.name, exc)

    def _run_gpu_check(self, exp: "ExperimentConfig") -> None:
        """GPU 状态采集任务。gpu_ids 为空时自动从进程占用推断。"""
        try:
            gpu_ids = list(exp.gpu_ids)
            if not gpu_ids:
                # 先获取该实验的进程 PID，再反查 GPU
                proc_status = self._process_monitor.check(exp.name, exp.process_keywords, cwd=exp.process_cwd, extra_keywords=exp.process_extra_keywords, pids=exp.process_pids)
                pids = {p.pid for p in proc_status.processes}
                gpu_ids = self._gpu_monitor.resolve_gpu_ids(pids)
                if gpu_ids:
                    logger.info("[%s] Auto-detected GPUs from processes: %s", exp.name, gpu_ids)
                else:
                    logger.debug("[%s] No GPUs detected for processes (pids=%s)", exp.name, pids)
                    return

            stats = self._gpu_monitor.collect(gpu_ids, exp.gpu_alert_thresholds)
            if not stats:
                return

            alert_stats = [s for s in stats if s.alert]
            if alert_stats:
                reasons = "; ".join(
                    f"GPU{s.gpu_id}({s.name}): {', '.join(s.alert_reasons)}"
                    for s in alert_stats
                )
                logger.warning("[%s] GPU alert: %s", exp.name, reasons)
                self._notifier.send_alert(
                    title=f"GPU 异常 — {exp.name}",
                    content=reasons,
                    dedup_key=f"gpu_{exp.name}",
                )

            # 定期 GPU 状态报告
            report_interval = self._cfg.schedule.gpu_report_interval_minutes
            if report_interval > 0:
                now = time.monotonic()
                last = self._last_gpu_report.get(exp.name, 0.0)
                if now - last >= report_interval * 60:
                    self._last_gpu_report[exp.name] = now
                    self._notifier.send_gpu_report(
                        exp_name=exp.name,
                        gpu_stats=[s.to_dict() for s in stats],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] GPU check failed: %s", exp.name, exc)

    def _run_cpu_check(self, exp: "ExperimentConfig") -> None:
        """CPU / 内存状态采集任务。"""
        try:
            proc_status = self._process_monitor.check(exp.name, exp.process_keywords, cwd=exp.process_cwd, extra_keywords=exp.process_extra_keywords, pids=exp.process_pids)
            pids = {p.pid for p in proc_status.processes}
            stat = self._cpu_monitor.collect(pids, exp.cpu_alert_thresholds)

            if stat.alert:
                reasons = "；".join(stat.alert_reasons)
                logger.warning("[%s] CPU alert: %s", exp.name, reasons)
                self._notifier.send_alert(
                    title=f"CPU/内存 异常 — {exp.name}",
                    content=reasons,
                    dedup_key=f"cpu_{exp.name}",
                )
            else:
                logger.debug(
                    "[%s] CPU %.1f%%  内存 %.1f%%  进程 CPU %.1f%%",
                    exp.name, stat.cpu_percent_total, stat.memory_percent, stat.process_cpu_percent,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] CPU check failed: %s", exp.name, exc)

    def _run_wandb_check(self, exp: "ExperimentConfig") -> None:
        """wandb 轮询任务：检测新 step 并触发 LLM 分析。"""
        try:
            metric_keys = exp.wandb_metric_keys or None
            snapshot = self._wandb_monitor.get_snapshot(
                project=exp.wandb_project,
                run_id=exp.wandb_run_id,
                run_name=exp.wandb_run_name,
                entity=self._cfg.wandb.entity,
                ignore_missing=exp.wandb_ignore_missing,
                metric_keys=metric_keys,
            )
            if snapshot is None:
                logger.debug("[%s] No wandb snapshot available.", exp.name)
                return

            # run 状态异常告警
            if snapshot.state in ("crashed", "failed"):
                self._notifier.send_alert(
                    title=f"WandB Run 异常 — {exp.name}",
                    content=f"Run `{snapshot.run_name}` 状态: **{snapshot.state}**",
                    dedup_key=f"wandb_state_{exp.name}_{snapshot.run_id}",
                )

            # NaN/Inf 告警
            if snapshot.has_nan_inf:
                self._notifier.send_alert(
                    title=f"指标 NaN/Inf — {exp.name}",
                    content=f"Run `{snapshot.run_name}` step={snapshot.latest_step}\n"
                    f"异常指标: {snapshot.nan_inf_keys}",
                    dedup_key=f"nan_inf_{exp.name}_{snapshot.latest_step}",
                )

            # 新 step 触发分析
            step_check = self._cfg.schedule.wandb_step_check
            if step_check and self._wandb_monitor.has_new_step(exp.name, snapshot):
                logger.info(
                    "[%s] New step detected (%d), triggering LLM analysis.",
                    exp.name,
                    snapshot.latest_step,
                )
                self._trigger_llm_analysis(exp, snapshot)
            else:
                logger.debug("[%s] No new wandb steps.", exp.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Wandb check failed: %s", exp.name, exc)

    def _trigger_llm_analysis(
        self, exp: "ExperimentConfig", snapshot
    ) -> None:
        """拉取历史数据、绘图、调用 LLM 分析并推送。"""
        try:
            metric_keys = exp.wandb_metric_keys or None
            df = self._wandb_monitor.get_history_df(
                project=exp.wandb_project,
                run_id=exp.wandb_run_id,
                run_name=exp.wandb_run_name,
                entity=self._cfg.wandb.entity,
            )
            chart_buf = self._wandb_monitor.plot_metrics(
                df,
                metric_keys=metric_keys,
                title=f"{exp.name} — step {snapshot.latest_step}",
            )
            analysis = self._analyzer.analyze(snapshot, chart_buf)
            analysis_text = analysis.text
            summary = analysis.summary_line()
            if summary:
                analysis_text += f"\n\n---\n{summary}"
            self._notifier.send_wandb_analysis(
                exp_name=exp.name,
                step=snapshot.latest_step,
                analysis_text=analysis_text,
                chart_buf=chart_buf,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] LLM analysis failed: %s", exp.name, exc)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_process_details(status) -> str:
        if not status.processes:
            return "（未找到相关进程）"
        lines = []
        for p in status.processes[:10]:  # 最多展示 10 条
            lines.append(f"- PID {p.pid} `{p.name}` cpu={p.cpu_percent}% mem={p.memory_mb}MB")
        if len(status.processes) > 10:
            lines.append(f"... 共 {len(status.processes)} 个进程")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Startup check
    # ------------------------------------------------------------------

    def _run_startup_check(self, exp: "ExperimentConfig") -> None:
        """启动时立即执行一次：发送接入成功通知 + 首次全量分析。"""
        try:
            lines: list[str] = [f"**实验**: {exp.name}", ""]

            # 进程状态
            process_status = self._process_monitor.check(exp.name, exp.process_keywords, cwd=exp.process_cwd, extra_keywords=exp.process_extra_keywords, pids=exp.process_pids)
            proc_icon = "🟢" if process_status.is_healthy else "🔴"
            proc_count = len(process_status.processes)
            lines.append(
                f"{proc_icon} **进程**: "
                + (f"运行中，共 {proc_count} 个进程" if process_status.is_healthy
                   else f"异常 — {process_status.alert_reason}")
            )

            # GPU 状态
            gpu_ids = list(exp.gpu_ids)
            if not gpu_ids:
                pids = {p.pid for p in process_status.processes}
                gpu_ids = self._gpu_monitor.resolve_gpu_ids(pids)
            if gpu_ids:
                stats = self._gpu_monitor.collect(gpu_ids, exp.gpu_alert_thresholds)
                if stats:
                    for s in stats:
                        icon = "🔴" if s.alert else "🟢"
                        lines.append(
                            f"{icon} **GPU {s.gpu_id}**: "
                            f"利用率 {s.utilization}%  "
                            f"显存 {s.memory_used_mb}/{s.memory_total_mb}MB ({s.memory_percent:.1f}%)  "
                            f"温度 {s.temperature}°C"
                        )
                else:
                    lines.append("⚪ **GPU**: 无法采集（pynvml 不可用）")

            # CPU / 内存状态
            pids = {p.pid for p in process_status.processes}
            cpu_stat = self._cpu_monitor.collect(pids, exp.cpu_alert_thresholds)
            cpu_icon = "🔴" if cpu_stat.alert else "🟢"
            lines.append(
                f"{cpu_icon} **CPU/内存**: "
                f"CPU {cpu_stat.cpu_percent_total:.1f}%  "
                f"内存 {cpu_stat.memory_used_gb}/{cpu_stat.memory_total_gb}GB ({cpu_stat.memory_percent:.1f}%)  "
                f"负载 {cpu_stat.load_avg_1m}/{cpu_stat.load_avg_5m}  "
                f"进程占用 CPU {cpu_stat.process_cpu_percent}% / 内存 {cpu_stat.process_memory_mb}MB"
            )

            # wandb 状态
            wandb_summary = ""
            if exp.wandb_project:
                metric_keys = exp.wandb_metric_keys or None
                snapshot = self._wandb_monitor.get_snapshot(
                    project=exp.wandb_project,
                    run_id=exp.wandb_run_id,
                    run_name=exp.wandb_run_name,
                    entity=self._cfg.wandb.entity,
                    ignore_missing=exp.wandb_ignore_missing,
                    metric_keys=metric_keys,
                )
                if snapshot:
                    # 预置 last_step，避免启动后的第一次定时检查重复触发分析
                    self._wandb_monitor.has_new_step(exp.name, snapshot)
                    state_icon = "🟢" if snapshot.state == "running" else "🟡"
                    lines.append(
                        f"{state_icon} **WandB**: run `{snapshot.run_name}`  "
                        f"状态 {snapshot.state}  当前 step={snapshot.latest_step}"
                    )
                    wandb_summary = f"当前 step={snapshot.latest_step}，状态={snapshot.state}"
                    if snapshot.has_nan_inf:
                        lines.append(f"  ⚠️ 存在 NaN/Inf: {snapshot.nan_inf_keys}")
                else:
                    lines.append("⚪ **WandB**: 未找到活跃 run")

            self._notifier.send_info(
                title=f"Lab Monitor 已接入 — {exp.name}",
                content="\n".join(lines),
            )
            logger.info("[%s] Startup notification sent.", exp.name)

            # 若 wandb 有数据，立即触发一次 LLM 分析
            if exp.wandb_project and wandb_summary:
                snapshot = self._wandb_monitor.get_snapshot(
                    project=exp.wandb_project,
                    run_id=exp.wandb_run_id,
                    run_name=exp.wandb_run_name,
                    entity=self._cfg.wandb.entity,
                    metric_keys=exp.wandb_metric_keys or None,
                )
                if snapshot and snapshot.latest_step > 0:
                    logger.info("[%s] Running initial LLM analysis (step=%d).", exp.name, snapshot.latest_step)
                    self._trigger_llm_analysis(exp, snapshot)

        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Startup check failed: %s", exp.name, exc)
