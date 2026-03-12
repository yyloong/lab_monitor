from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lab_monitor.monitors.wandb_monitor import RunSnapshot

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """你是一个深度学习实验分析助手。
用户会提供训练曲线数据（JSON 格式的最新指标）以及可选的折线图图片。
请简洁地分析以下几个方面（不超过 300 字）：
1. 当前训练进展是否正常
2. loss/acc 趋势判断（收敛/发散/震荡）
3. 是否有过拟合或欠拟合迹象
4. 是否存在异常值（NaN/Inf/突变）
5. 简要建议（如有）
回复使用中文，直接给出结论，不要重复输入数据。"""


@dataclass
class AnalysisResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False

    def summary_line(self) -> str:
        parts = []
        if self.input_tokens or self.output_tokens:
            parts.append(f"token 用量: 输入 {self.input_tokens} / 输出 {self.output_tokens}")
        if self.truncated:
            parts.append("（输入已截断）")
        return "  \n".join(parts)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """截断超长文本，返回 (截断后文本, 是否发生截断)。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n... [已截断，原始长度 {len(text)} 字符]", True


class LLMAnalyzer(ABC):
    """大模型分析器抽象基类。"""

    @abstractmethod
    def analyze(
        self,
        snapshot: "RunSnapshot",
        chart_buf: io.BytesIO | None = None,
    ) -> AnalysisResult:
        """分析 wandb 快照，返回结构化分析结果（含 token 用量）。"""


class OpenAIAnalyzer(LLMAnalyzer):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "",
        vision_enabled: bool = True,
        max_input_chars: int = 4000,
        max_tokens: int = 600,
    ) -> None:
        from openai import OpenAI

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._vision_enabled = vision_enabled
        self._max_input_chars = max_input_chars
        self._max_tokens = max_tokens

    def analyze(
        self,
        snapshot: "RunSnapshot",
        chart_buf: io.BytesIO | None = None,
    ) -> AnalysisResult:
        user_text, truncated = self._build_user_text(snapshot)
        messages = self._build_messages(
            user_text, chart_buf if self._vision_enabled else None
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=0.3,
            )
            usage = resp.usage
            return AnalysisResult(
                text=resp.choices[0].message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI API error: %s", exc)
            return AnalysisResult(text=f"[LLM 分析失败: {exc}]", truncated=truncated)

    def _build_user_text(self, snapshot: "RunSnapshot") -> tuple[str, bool]:
        import json

        metrics_text = json.dumps(snapshot.metrics, ensure_ascii=False, indent=2)
        raw = (
            f"Run: {snapshot.run_name} (step={snapshot.latest_step}, state={snapshot.state})\n"
            f"最新指标:\n{metrics_text}"
        )
        if snapshot.has_nan_inf:
            raw += f"\n⚠️ 以下指标出现 NaN/Inf: {snapshot.nan_inf_keys}"
        return _truncate(raw, self._max_input_chars)

    @staticmethod
    def _build_messages(user_text: str, chart_buf: io.BytesIO | None) -> list[dict]:
        if chart_buf is not None:
            chart_buf.seek(0)
            img_b64 = base64.b64encode(chart_buf.read()).decode()
            return [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                },
            ]
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]


class AnthropicAnalyzer(LLMAnalyzer):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_input_chars: int = 4000,
        max_tokens: int = 600,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_input_chars = max_input_chars
        self._max_tokens = max_tokens

    def analyze(
        self,
        snapshot: "RunSnapshot",
        chart_buf: io.BytesIO | None = None,
    ) -> AnalysisResult:
        import json

        metrics_text = json.dumps(snapshot.metrics, ensure_ascii=False, indent=2)
        raw = (
            f"Run: {snapshot.run_name} (step={snapshot.latest_step}, state={snapshot.state})\n"
            f"最新指标:\n{metrics_text}"
        )
        if snapshot.has_nan_inf:
            raw += f"\n⚠️ 以下指标出现 NaN/Inf: {snapshot.nan_inf_keys}"
        user_text, truncated = _truncate(raw, self._max_input_chars)

        content: list[dict] = []
        if chart_buf is not None:
            chart_buf.seek(0)
            img_b64 = base64.b64encode(chart_buf.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
            })
        content.append({"type": "text", "text": user_text})

        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            usage = resp.usage
            return AnalysisResult(
                text=resp.content[0].text if resp.content else "",
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Anthropic API error: %s", exc)
            return AnalysisResult(text=f"[LLM 分析失败: {exc}]", truncated=truncated)


class ZhipuAnalyzer(LLMAnalyzer):
    """智谱 GLM 分析器（不支持图片输入，仅文本）。"""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4",
        max_input_chars: int = 4000,
        max_tokens: int = 600,
    ) -> None:
        from zhipuai import ZhipuAI

        self._client = ZhipuAI(api_key=api_key)
        self._model = model
        self._max_input_chars = max_input_chars
        self._max_tokens = max_tokens

    def analyze(
        self,
        snapshot: "RunSnapshot",
        chart_buf: io.BytesIO | None = None,
    ) -> AnalysisResult:
        import json

        metrics_text = json.dumps(snapshot.metrics, ensure_ascii=False, indent=2)
        raw = (
            f"Run: {snapshot.run_name} (step={snapshot.latest_step}, state={snapshot.state})\n"
            f"最新指标:\n{metrics_text}"
        )
        if snapshot.has_nan_inf:
            raw += f"\n⚠️ 以下指标出现 NaN/Inf: {snapshot.nan_inf_keys}"
        if chart_buf is not None:
            raw += "\n（图表已生成但当前模型不支持图片输入）"
        user_text, truncated = _truncate(raw, self._max_input_chars)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=self._max_tokens,
                temperature=0.3,
            )
            usage = getattr(resp, "usage", None)
            return AnalysisResult(
                text=resp.choices[0].message.content or "",
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
                truncated=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ZhipuAI API error: %s", exc)
            return AnalysisResult(text=f"[LLM 分析失败: {exc}]", truncated=truncated)


def create_analyzer(
    provider: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
    vision_enabled: bool = True,
    max_input_chars: int = 4000,
    max_tokens: int = 600,
) -> LLMAnalyzer:
    """工厂函数，根据 provider 创建对应的分析器实例。"""
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIAnalyzer(
            api_key=api_key, model=model or "gpt-4o", base_url=base_url,
            vision_enabled=vision_enabled,
            max_input_chars=max_input_chars, max_tokens=max_tokens,
        )
    if provider in ("anthropic", "claude"):
        return AnthropicAnalyzer(
            api_key=api_key, model=model or "claude-3-5-sonnet-20241022",
            max_input_chars=max_input_chars, max_tokens=max_tokens,
        )
    if provider in ("zhipu", "zhipuai", "glm"):
        return ZhipuAnalyzer(
            api_key=api_key, model=model or "glm-4",
            max_input_chars=max_input_chars, max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose from: openai, anthropic, zhipu")
