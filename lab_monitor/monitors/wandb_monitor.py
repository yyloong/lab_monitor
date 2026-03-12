from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # 无头模式，不需要 display

logger = logging.getLogger(__name__)

try:
    import wandb
    from wandb import Api as WandbApi

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed; wandb monitoring disabled.")


@dataclass
class RunSnapshot:
    """一次 wandb run 状态快照。"""

    run_id: str
    run_name: str
    state: str                        # running / finished / crashed / failed
    latest_step: int
    metrics: dict[str, float] = field(default_factory=dict)
    has_nan_inf: bool = False
    nan_inf_keys: list[str] = field(default_factory=list)


class WandbMonitor:
    """定期轮询 wandb run 状态，检测新 step，绘制指标曲线。"""

    def __init__(self, entity: str = "") -> None:
        self._api: WandbApi | None = None
        if not _WANDB_AVAILABLE:
            return
        try:
            self._api = WandbApi()
            logger.info("WandB API initialized successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"WandB API 初始化失败: {exc}\n"
                "请先在命令行执行 `wandb login` 完成认证"
            ) from exc
        # exp_name -> last known step
        self._last_steps: dict[str, int] = {}
        # exp_name -> last step that triggered LLM analysis
        self._last_analysis_steps: dict[str, int] = {}

    def get_snapshot(
        self, project: str, run_id: str = "", entity: str = "",
        ignore_missing: bool = False, run_name: str = "",
        metric_keys: list[str] | None = None,
    ) -> RunSnapshot | None:
        """获取指定 run 的最新快照；run_id/run_name 均留空则自动选取最新 running run。

        run_id：精确匹配 wandb 内部 ID。
        run_name：按 run name 包含匹配，比 run_id 更直观；run_id 优先。
        metric_keys：只保留指定的指标列；留空则保留全部。
        ignore_missing=False（默认）：找不到 run 时记录 ERROR 日志，便于调试。
        ignore_missing=True：找不到 run 时静默返回 None（实验尚未启动时使用）。
        """
        if self._api is None:
            return None
        run = self._resolve_run(project, run_id, entity, run_name=run_name)
        if run is None:
            if ignore_missing:
                logger.debug("No run found in project '%s' (ignore_missing=True)", project)
                return None
            raise RuntimeError(
                f"在项目 '{project}' 中找不到 run (run_id={run_id or '<latest>'}, run_name={run_name or '<any>'})。\n"
                "如果实验尚未启动，请在 config.yaml 中设置 wandb_ignore_missing: true"
            )

        try:
            history = run.history(samples=500, pandas=True)
            latest_step = int(history["_step"].max()) if not history.empty else 0

            # 对每列取最近一个有效（非NaN）值，避免不同指标记录频率不同导致缺失
            metrics: dict[str, float] = {}
            nan_inf_keys: list[str] = []
            if not history.empty:
                numeric_cols = [
                    c for c in history.columns
                    if not c.startswith("_") and pd.api.types.is_numeric_dtype(history[c])
                ]
                # 若指定了核心指标，则只保留指定列（取交集）
                if metric_keys:
                    numeric_cols = [c for c in numeric_cols if c in metric_keys]
                for col in numeric_cols:
                    series = history[col].dropna()
                    if series.empty:
                        continue
                    fv = float(series.iloc[-1])
                    metrics[col] = fv
                    if math.isnan(fv) or math.isinf(fv):
                        nan_inf_keys.append(col)

            # 用 run.summary 补充 history 采样中未出现的指标（如 critic 与 actor 记录频率不同）
            try:
                summary = dict(run.summary)
                for k, v in summary.items():
                    if k.startswith("_") or k in metrics:
                        continue
                    if metric_keys and k not in metric_keys:
                        continue
                    try:
                        fv = float(v)
                        if math.isnan(fv) or math.isinf(fv):
                            continue  # summary 中的 nan/inf 无参考意义，跳过
                        metrics[k] = fv
                    except (TypeError, ValueError):
                        pass
            except Exception:  # noqa: BLE001
                pass  # summary 读取失败不影响主流程

            return RunSnapshot(
                run_id=run.id,
                run_name=run.name,
                state=run.state,
                latest_step=latest_step,
                metrics=metrics,
                has_nan_inf=bool(nan_inf_keys),
                nan_inf_keys=nan_inf_keys,
            )
        except Exception as exc:
            raise RuntimeError(f"获取 wandb run 数据失败: {exc}") from exc

    def has_new_step(self, exp_name: str, snapshot: RunSnapshot, step_interval: int = 1) -> bool:
        """判断此次快照是否需要触发分析。

        step_interval=1（默认）：每检测到新 step 即触发。
        step_interval=N（N>1）：仅当距上次触发分析的 step 差值 >= N 时才触发。
        """
        last = self._last_steps.get(exp_name, -1)
        if snapshot.latest_step <= last:
            return False
        # 有新 step，更新已知最新 step
        self._last_steps[exp_name] = snapshot.latest_step

        last_analysis = self._last_analysis_steps.get(exp_name, -1)
        if step_interval <= 1 or (snapshot.latest_step - last_analysis) >= step_interval:
            self._last_analysis_steps[exp_name] = snapshot.latest_step
            return True
        return False

    def get_history_df(
        self, project: str, run_id: str = "", entity: str = "", samples: int = 1000,
        run_name: str = "",
    ) -> pd.DataFrame | None:
        """获取完整历史数据（DataFrame），供绘图使用。失败时抛出异常。"""
        if self._api is None:
            return None
        run = self._resolve_run(project, run_id, entity, run_name=run_name)
        if run is None:
            return None
        try:
            return run.history(samples=samples, pandas=True)
        except Exception as exc:
            raise RuntimeError(f"获取 wandb history 失败: {exc}") from exc

    def plot_metrics(
        self,
        df: pd.DataFrame,
        metric_keys: list[str] | None = None,
        title: str = "Training Metrics",
    ) -> io.BytesIO | None:
        """用 matplotlib 绘制指标折线图，返回 PNG BytesIO。"""
        if df is None or df.empty:
            return None

        # 自动选取要绘制的数值列
        numeric_cols = [
            c for c in df.columns if not c.startswith("_") and pd.api.types.is_numeric_dtype(df[c])
        ]
        if metric_keys:
            cols = [c for c in metric_keys if c in numeric_cols]
        else:
            # 优先选训练核心指标，最多展示 8 条
            priority_kw = ("loss", "acc", "f1", "lr", "reward", "score", "kl", "advantage", "entropy")
            priority = [c for c in numeric_cols if any(k in c.lower() for k in priority_kw)]
            rest = [c for c in numeric_cols if c not in priority]
            cols = (priority + rest)[:8]

        if not cols:
            return None

        step_col = "_step" if "_step" in df.columns else df.index
        n = len(cols)
        ncols = min(2, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(title, fontsize=14)

        for i, col in enumerate(cols):
            ax = axes[i // ncols][i % ncols]
            series = df[col].dropna()
            if isinstance(step_col, str):
                xs = df.loc[series.index, step_col]
            else:
                xs = series.index
            ax.plot(xs, series.values, linewidth=1.5)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_run(self, project: str, run_id: str, entity: str = "", run_name: str = "") -> Any:
        """根据 run_id / run_name 或自动选最新 running run。

        优先级：run_id（精确）> run_name（包含匹配）> 自动最新。
        """
        if run_id:
            path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
            return self._api.run(path)

        project_path = f"{entity}/{project}" if entity else project

        if run_name:
            # 先在 running 状态中按 name 匹配
            for state_filter in ({"state": "running"}, {}):
                runs = list(self._api.runs(
                    path=project_path,
                    filters=state_filter,
                    order="-created_at",
                    per_page=50,
                ))
                matched = [r for r in runs if run_name in r.name]
                if matched:
                    if len(matched) > 1:
                        logger.warning(
                            "run_name '%s' 在项目 '%s' 中匹配到 %d 个 run，使用最新: %s",
                            run_name, project, len(matched), matched[0].name,
                        )
                    logger.info(
                        "run_name '%s' 匹配到 run: %s (id=%s, state=%s)",
                        run_name, matched[0].name, matched[0].id, matched[0].state,
                    )
                    return matched[0]
            logger.warning("run_name '%s' 在项目 '%s' 中未匹配到任何 run", run_name, project)
            return None

        # 查找该 project 下最新的 running run
        filters = {"state": "running"}
        runs = list(self._api.runs(
            path=project_path,
            filters=filters,
            order="-created_at",
            per_page=1,
        ))
        if runs:
            return runs[0]

        # 若无 running，则取最近一次 run（可能 finished）
        runs = list(self._api.runs(
            path=project_path,
            order="-created_at",
            per_page=1,
        ))
        return runs[0] if runs else None
