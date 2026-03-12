from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeishuConfig:
    webhook_url: str
    app_id: str = ""        # 可选：飞书自建应用 app_id，用于上传图片展示图表
    app_secret: str = ""    # 可选：飞书自建应用 app_secret


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = ""
    vision_enabled: bool = True
    max_input_chars: int = 4000   # 输入文本超过此长度时截断，0 表示不限制
    max_tokens: int = 600         # 模型最大输出 token 数


@dataclass
class WandbConfig:
    entity: str = ""


@dataclass
class GPUAlertThresholds:
    memory_percent: float = 95.0
    temperature: float = 85.0
    utilization: float = 5.0


@dataclass
class CPUAlertThresholds:
    memory_percent: float = 90.0
    cpu_percent: float = 95.0


@dataclass
class ExperimentConfig:
    name: str
    wandb_project: str = ""
    wandb_run_id: str = ""
    wandb_run_name: str = ""   # 按 run name 模糊匹配（包含即可），比 run_id 更直观；run_id 优先
    wandb_ignore_missing: bool = False
    wandb_metric_keys: list[str] = field(default_factory=list)  # 指定关注的核心指标；留空则自动选取
    process_pids: list[int] = field(default_factory=list)       # 直接指定 PID，优先于关键词扫描
    process_keywords: list[str] = field(default_factory=list)
    process_cwd: str = ""
    process_extra_keywords: list[str] = field(default_factory=list)
    gpu_ids: list[int] = field(default_factory=list)
    gpu_alert_thresholds: GPUAlertThresholds = field(default_factory=GPUAlertThresholds)
    cpu_alert_thresholds: CPUAlertThresholds = field(default_factory=CPUAlertThresholds)


@dataclass
class ScheduleConfig:
    gpu_interval_seconds: int = 60
    process_interval_seconds: int = 30
    wandb_interval_seconds: int = 30
    wandb_step_check: bool = True
    alert_cooldown_minutes: int = 10
    gpu_report_interval_minutes: int = 30


@dataclass
class AppConfig:
    feishu: FeishuConfig
    llm: LLMConfig
    wandb: WandbConfig
    experiments: list[ExperimentConfig]
    schedule: ScheduleConfig


def _from_dict(cls, data: dict[str, Any]):
    """Simple dataclass factory that handles nested dataclasses."""
    if data is None:
        data = {}
    hints = cls.__dataclass_fields__
    kwargs: dict[str, Any] = {}
    for name, f in hints.items():
        val = data.get(name)
        ftype = f.type
        # resolve string annotations
        if isinstance(ftype, str):
            ftype = eval(ftype)  # noqa: S307
        origin = getattr(ftype, "__origin__", None)
        if origin is list:
            inner = ftype.__args__[0]
            if hasattr(inner, "__dataclass_fields__"):
                val = [_from_dict(inner, item) for item in (val or [])]
            else:
                val = val if val is not None else []
        elif hasattr(ftype, "__dataclass_fields__") and val is not None:
            val = _from_dict(ftype, val)
        elif hasattr(ftype, "__dataclass_fields__") and val is None:
            val = _from_dict(ftype, {})
        if val is not None:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    feishu = _from_dict(FeishuConfig, raw.get("feishu", {}))
    llm = _from_dict(LLMConfig, raw.get("llm", {}))
    wandb_cfg = _from_dict(WandbConfig, raw.get("wandb", {}))
    schedule = _from_dict(ScheduleConfig, raw.get("schedule", {}))

    experiments = []
    for exp_raw in raw.get("experiments", []):
        thresholds_raw = exp_raw.pop("gpu_alert_thresholds", {})
        cpu_thresholds_raw = exp_raw.pop("cpu_alert_thresholds", {})
        exp = _from_dict(ExperimentConfig, exp_raw)
        exp.gpu_alert_thresholds = _from_dict(GPUAlertThresholds, thresholds_raw)
        exp.cpu_alert_thresholds = _from_dict(CPUAlertThresholds, cpu_thresholds_raw)
        experiments.append(exp)

    # Allow overriding secrets via environment variables
    if not llm.api_key:
        llm.api_key = os.getenv("LLM_API_KEY", "")
    if not feishu.webhook_url or "YOUR_WEBHOOK" in feishu.webhook_url:
        env_url = os.getenv("FEISHU_WEBHOOK_URL", "")
        if env_url:
            feishu.webhook_url = env_url

    return AppConfig(
        feishu=feishu,
        llm=llm,
        wandb=wandb_cfg,
        experiments=experiments,
        schedule=schedule,
    )
