from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

_FEISHU_TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
_FEISHU_IMAGE_URL = "https://open.feishu.cn/open-apis/im/v1/images"

# 自定义日志级别：NOTIFY 用于标记"消息已成功推送"，颜色显著区别于普通 INFO
NOTIFY = 25
logging.addLevelName(NOTIFY, "NOTIFY")


class FeiShuNotifier:
    """飞书推送。

    仅配置 webhook_url：纯文字/卡片模式。
    额外配置 app_id + app_secret：可上传图片，在卡片中直接展示图表。
    """

    def __init__(
        self,
        webhook_url: str,
        cooldown_minutes: int = 10,
        app_id: str = "",
        app_secret: str = "",
    ) -> None:
        self._url = webhook_url
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._last_sent: dict[str, datetime] = {}
        self._app_id = app_id
        self._app_secret = app_secret
        # tenant_access_token 缓存
        self._token: str = ""
        self._token_expires: datetime = datetime.min

        if app_id and app_secret:
            logger.info("FeiShu image upload enabled (app_id configured).")
        else:
            logger.info("FeiShu image upload disabled (no app_id/app_secret, using text fallback).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, title: str, content: str, dedup_key: str = "") -> bool:
        """发送纯文本告警消息（带限流）。"""
        key = dedup_key or title
        if self._is_throttled(key):
            logger.debug("Alert throttled: %s", key)
            return False
        payload = self._build_text(f"[告警] {title}\n{content}")
        ok = self._post(payload)
        if ok:
            self._update_throttle(key)
            logger.log(NOTIFY, "🚨 告警已推送: %s", title)
        return ok

    def send_info(self, title: str, content: str) -> bool:
        """发送普通信息卡片（不限流）。"""
        payload = self._build_card(title, content, color="blue")
        ok = self._post(payload)
        if ok:
            logger.log(NOTIFY, "📢 通知已推送: %s", title)
        return ok

    def send_gpu_report(self, exp_name: str, gpu_stats: list[dict[str, Any]]) -> bool:
        """发送 GPU 状态卡片报告。"""
        lines = [f"**实验**: {exp_name}", ""]
        for g in gpu_stats:
            status_icon = "🔴" if g.get("alert") else "🟢"
            lines.append(
                f"{status_icon} GPU {g['gpu_id']} ({g['name']})  "
                f"利用率 {g['utilization']}%  "
                f"显存 {g['memory_used_mb']}MB/{g['memory_total_mb']}MB ({g['memory_percent']:.1f}%)  "
                f"温度 {g['temperature']}°C  "
                f"功耗 {g['power_draw_w']:.0f}W"
            )
        payload = self._build_card("GPU 状态报告", "\n".join(lines), color="green")
        ok = self._post(payload)
        if ok:
            logger.log(NOTIFY, "📊 GPU 报告已推送: %s", exp_name)
        return ok

    def send_process_alert(
        self,
        exp_name: str,
        reason: str,
        details: str,
        dedup_key: str = "",
    ) -> bool:
        """发送进程异常告警（带限流）。"""
        key = dedup_key or f"process_{exp_name}"
        if self._is_throttled(key):
            logger.debug("Process alert throttled: %s", key)
            return False
        content = f"**实验**: {exp_name}\n**原因**: {reason}\n\n{details}"
        payload = self._build_card("进程异常告警", content, color="red")
        ok = self._post(payload)
        if ok:
            self._update_throttle(key)
            logger.log(NOTIFY, "🚨 进程告警已推送: %s — %s", exp_name, reason)
        return ok

    def send_wandb_analysis(
        self,
        exp_name: str,
        step: int,
        analysis_text: str,
        chart_buf: io.BytesIO | None = None,
    ) -> bool:
        """发送 wandb 分析报告。有 app 凭据时上传图片展示，否则降级纯文字卡片。"""
        title = f"WandB 分析报告 — {exp_name} (step {step})"
        image_key: str | None = None

        if chart_buf is not None and self._app_id and self._app_secret:
            image_key = self._upload_image(chart_buf)
            if not image_key:
                logger.warning("Image upload failed, falling back to text-only card.")

        if image_key:
            payload = self._build_card_with_image_key(title, analysis_text, image_key)
        else:
            note = ""
            if chart_buf is not None and not self._app_id:
                note = "\n\n> 图表已生成，配置 app_id/app_secret 后可在此处展示"
            payload = self._build_card(title, analysis_text + note, color="wathet")

        ok = self._post(payload)
        if ok:
            logger.log(NOTIFY, "🤖 WandB 分析已推送: %s step=%d", exp_name, step)
        return ok

    # ------------------------------------------------------------------
    # Image upload
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """获取 tenant_access_token，自动续期。"""
        if self._token and datetime.now() < self._token_expires:
            return self._token
        try:
            resp = requests.post(
                _FEISHU_TOKEN_URL,
                json={"app_id": self._app_id, "app_secret": self._app_secret},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.error("Failed to get FeiShu token: %s", data)
                return ""
            self._token = data["tenant_access_token"]
            # token 有效期 2 小时，提前 5 分钟刷新
            self._token_expires = datetime.now() + timedelta(seconds=data.get("expire", 7200) - 300)
            return self._token
        except Exception as exc:  # noqa: BLE001
            logger.error("FeiShu token request failed: %s", exc)
            return ""

    def _upload_image(self, buf: io.BytesIO) -> str | None:
        """上传图片到飞书，返回 image_key；失败返回 None。"""
        token = self._get_token()
        if not token:
            return None
        try:
            buf.seek(0)
            resp = requests.post(
                _FEISHU_IMAGE_URL,
                headers={"Authorization": f"Bearer {token}"},
                files={
                    "image_type": (None, "message"),
                    "image": ("chart.png", buf, "image/png"),
                },
                timeout=30,
            )
            if not resp.ok:
                logger.error(
                    "FeiShu image upload HTTP %s: %s", resp.status_code, resp.text
                )
                return None
            data = resp.json()
            if data.get("code") != 0:
                logger.error("FeiShu image upload API error: %s", data)
                return None
            return data["data"]["image_key"]
        except Exception as exc:  # noqa: BLE001
            logger.error("FeiShu image upload failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text(text: str) -> dict:
        return {"msg_type": "text", "content": {"text": text}}

    @staticmethod
    def _build_card(title: str, content: str, color: str = "blue") -> dict:
        return {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color,
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": content}},
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        ],
                    },
                ],
            },
        }

    @staticmethod
    def _build_card_with_image_key(title: str, content: str, image_key: str) -> dict:
        """构建带真实图片的飞书卡片（需要 image_key）。"""
        return {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "wathet",
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": content}},
                    {"tag": "img", "img_key": image_key, "alt": {"tag": "plain_text", "content": "训练曲线图"}},
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            }
                        ],
                    },
                ],
            },
        }

    # ------------------------------------------------------------------
    # HTTP & throttle helpers
    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> bool:
        try:
            resp = requests.post(
                self._url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("code", 0) != 0:
                logger.warning("FeiShu API error: %s", result)
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send FeiShu message: %s", exc)
            return False

    def _is_throttled(self, key: str) -> bool:
        last = self._last_sent.get(key)
        if last is None:
            return False
        return datetime.now() - last < self._cooldown

    def _update_throttle(self, key: str) -> None:
        self._last_sent[key] = datetime.now()
