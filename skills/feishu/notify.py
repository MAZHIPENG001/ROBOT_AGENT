"""Non-blocking, best-effort Feishu status notifications for DeepSeek VLM inference.

Uses the existing Feishu Tool implementation and the latest authorized chat
recorded by AgentRuntime. Never routes notifications through the LLM.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import lark_oapi as lark

from .tool import feishu_tools

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAT_FILE = PROJECT_ROOT / "state" / "last_feishu_chat.json"
_NOTIFICATIONS: queue.Queue[str] = queue.Queue(maxsize=100)
_START_LOCK = threading.Lock()
_WORKER_STARTED = False
_SEND_TOOL: Any = None


def _chat_id() -> str | None:
    """Only notify the chat previously authorized through the Feishu listener."""
    try:
        chat_id = json.loads(CHAT_FILE.read_text(encoding="utf-8")).get("chat_id")
        if isinstance(chat_id, str) and chat_id.startswith("oc_"):
            return chat_id
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def _send_tool():
    global _SEND_TOOL
    if _SEND_TOOL is None:
        app_id = os.environ.get("FEISHU_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET", "")
        if not app_id or not app_secret:
            raise RuntimeError("缺少 FEISHU_APP_ID / FEISHU_APP_SECRET")
        client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
        # Reuse the existing messaging tool and its authorization check.
        _SEND_TOOL = feishu_tools(client, lambda candidate: candidate == _chat_id())[0]
    return _SEND_TOOL


def _deliver_forever() -> None:
    while True:
        text = _NOTIFICATIONS.get()
        try:
            chat_id = _chat_id()
            if chat_id is None:
                logger.warning("跳过飞书推理通知：尚未记录飞书会话")
                continue
            result = str(_send_tool().invoke({"chat_id": chat_id, "text": text}))
            if "发送成功" not in result:
                logger.warning("飞书推理通知未送达：%s", result)
        except Exception:
            logger.exception("飞书推理通知发送异常（不影响推理）")
        finally:
            _NOTIFICATIONS.task_done()


def notify_feishu(text: str) -> None:
    """Queue a notification without delaying model inference or robot steps."""
    global _WORKER_STARTED
    if os.getenv("ROBOT_VLM_FEISHU_NOTIFY", "true").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return
    if _chat_id() is None:
        logger.warning("未找到飞书会话：先给机器人发一条消息以登记目标")
        return
    try:
        _NOTIFICATIONS.put_nowait(text[:3900])
    except queue.Full:
        logger.warning("飞书通知队列已满，丢弃通知")
        return
    with _START_LOCK:
        if not _WORKER_STARTED:
            threading.Thread(
                target=_deliver_forever,
                name="feishu-inference-notifier",
                daemon=True,
            ).start()
            _WORKER_STARTED = True


def notify_vlm_inference(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate predict(): send start, success, or failure notification."""
    @functools.wraps(func)
    def wrapper(self, observation, *args, **kwargs):
        model = str(getattr(self, "model", "deepseek_vlm"))
        step = getattr(observation, "step_index", "?")
        started = time.monotonic()
        notify_feishu(
            f"🚀 DeepSeek VLM 开始推理\n模型：{model}\n仿真步数：{step}"
        )
        try:
            result = func(self, observation, *args, **kwargs)
        except Exception as exc:
            elapsed = time.monotonic() - started
            # Avoid sending request URLs, credentials, or raw images in an alert.
            notify_feishu(
                f"❌ DeepSeek VLM 推理失败\n模型：{model}\n仿真步数：{step}\n"
                f"耗时：{elapsed:.2f} 秒\n异常类型：{type(exc).__name__}\n"
                "请查看 robot_control_status 或本地日志获取详细错误。"
            )
            raise
        else:
            elapsed = time.monotonic() - started

            actions = getattr(result, "actions", None)
            shape = getattr(actions, "shape", "未知")

            first_action = (
                [round(float(v), 5) for v in actions[0]]
                if actions is not None and len(actions) > 0
                else []
            )

            notify_feishu(
                f"✅ DeepSeek VLM 推理完成\n"
                f"模型：{model}\n"
                f"仿真步数：{step}\n"
                f"耗时：{elapsed:.2f} 秒\n"
                f"返回动作形状：{shape}\n"
                f"第一个动作：{first_action}"
            )

            return result

    return wrapper
