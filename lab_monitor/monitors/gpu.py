from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lab_monitor.config import ExperimentConfig, GPUAlertThresholds

logger = logging.getLogger(__name__)

try:
    import pynvml

    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False
    logger.warning("pynvml not installed; GPU monitoring disabled.")


@dataclass
class GPUStat:
    gpu_id: int
    name: str
    utilization: int          # %
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    temperature: int          # °C
    power_draw_w: float
    alert: bool = False
    alert_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.alert_reasons is None:
            self.alert_reasons = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "utilization": self.utilization,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_percent": self.memory_percent,
            "temperature": self.temperature,
            "power_draw_w": self.power_draw_w,
            "alert": self.alert,
            "alert_reasons": self.alert_reasons,
        }


class GPUMonitor:
    """采集指定 GPU 的运行状态，检测超阈值异常。"""

    def __init__(self) -> None:
        self._initialized = False
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._initialized = True
            except pynvml.NVMLError as exc:
                logger.error("NVML init failed: %s", exc)

    def resolve_gpu_ids(self, pids: set[int]) -> list[int]:
        """根据进程 PID 集合，自动找出这些进程占用的 GPU 物理索引列表。"""
        if not self._initialized or not pids:
            return []
        found: list[int] = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except pynvml.NVMLError:
                    procs = []
                gpu_pids = {p.pid for p in procs}
                if gpu_pids & pids:
                    found.append(i)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to resolve GPU IDs from pids: %s", exc)
        return sorted(found)

    def collect(
        self,
        gpu_ids: list[int],
        thresholds: "GPUAlertThresholds",
    ) -> list[GPUStat]:
        """采集指定 GPU 列表的状态并标注告警。"""
        if not self._initialized:
            return []

        stats: list[GPUStat] = []
        for gpu_id in gpu_ids:
            stat = self._collect_one(gpu_id, thresholds)
            if stat is not None:
                stats.append(stat)
        return stats

    def _collect_one(
        self,
        gpu_id: int,
        thresholds: "GPUAlertThresholds",
    ) -> GPUStat | None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            except pynvml.NVMLError:
                power = 0.0

            mem_total_mb = mem.total // (1024 * 1024)
            mem_used_mb = mem.used // (1024 * 1024)
            mem_percent = mem.used / mem.total * 100 if mem.total > 0 else 0.0

            alert_reasons: list[str] = []
            if mem_percent >= thresholds.memory_percent:
                alert_reasons.append(
                    f"显存占用 {mem_percent:.1f}% >= {thresholds.memory_percent}%"
                )
            if temp >= thresholds.temperature:
                alert_reasons.append(
                    f"温度 {temp}°C >= {thresholds.temperature}°C"
                )

            return GPUStat(
                gpu_id=gpu_id,
                name=name,
                utilization=util.gpu,
                memory_used_mb=mem_used_mb,
                memory_total_mb=mem_total_mb,
                memory_percent=mem_percent,
                temperature=temp,
                power_draw_w=power,
                alert=bool(alert_reasons),
                alert_reasons=alert_reasons,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to collect GPU %d stats: %s", gpu_id, exc)
            return None

    def __del__(self) -> None:
        if self._initialized and _NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
