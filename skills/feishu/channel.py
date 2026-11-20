# skills/feishu/channel.py

import json
import logging
import os
import re
import threading

from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from .tool import feishu_tools

logger = logging.getLogger(__name__)


class FeishuSkill:
    def __init__(self):
        self.app_id = os.getenv("FEISHU_APP_ID")
        self.app_secret = os.getenv("FEISHU_APP_SECRET")

        if not self.app_id or not self.app_secret:
            raise RuntimeError("请配置飞书 APP_ID 和 APP_SECRET")

        self.client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .build()
        )

        self._lock = threading.Lock()

        # 消息去重
        self._seen_message_ids = set()
        self._seen_message_order = deque()

        # 记录已收到消息的会话，主动发送只允许这些会话。
        # 个人模式不要求配置 Open ID / Chat ID。
        self._known_chat_ids = set()

        # 单用户顺序处理飞书消息，确保会话指令与确认的先后顺序
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._ws_client = None

    def _is_chat_allowed(self, chat_id: str) -> bool:
        with self._lock:
            return chat_id in self._known_chat_ids

    def tools(self):
        """获取飞书主动发送工具。"""
        return feishu_tools(
            self.client,
            self._is_chat_allowed,
        )

    def _reply(self, message_id: str, text: str):
        """回复收到的飞书消息。"""

        request = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("text")
                .content(
                    json.dumps(
                        {"text": text},
                        ensure_ascii=False,
                    )
                )
                .build()
            )
            .build()
        )

        try:
            response = self.client.im.v1.message.reply(request)

            if not response.success():
                logger.error(
                    "飞书回复失败: %s %s",
                    response.code,
                    response.msg,
                )
        except Exception:
            logger.exception("飞书回复异常")
    
    def remember_chat(self, chat_id: str) -> None:
        """登记可以发送消息的飞书会话。"""

        if chat_id:
            with self._lock:
                self._known_chat_ids.add(chat_id)
    
    def start(self, on_text: Callable[[str, str], str]):
        """
        启动飞书长连接。

        on_text(message, chat_id) -> Agent 回复
        """

        def process(message_id, content, chat_id):
            try:
                answer = on_text(content, chat_id)
                if not isinstance(answer, str) or not answer.strip():
                    answer = "Agent 没有返回有效的文本回复。"
            except Exception:
                logger.exception("Agent 处理失败")
                answer = "处理失败，请检查 Agent 日志。"

            self._reply(message_id, answer)

        def on_message(event):
            if not event.event or not event.event.message:
                return

            message = event.event.message

            if message.message_type != "text":
                return

            sender = getattr(event.event, "sender", None)

            # 忽略机器人自己的消息
            if getattr(sender, "sender_type", None) == "app":
                return

            chat_id = getattr(message, "chat_id", "")
            message_id = getattr(message, "message_id", "")

            if not chat_id or not message_id:
                return

            try:
                content = json.loads(
                    message.content
                ).get("text", "")
            except (ValueError, TypeError, AttributeError):
                return

            if not isinstance(content, str):
                return

            content = re.sub(
                r"<at\b[^>]*>.*?</at>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()

            if not content:
                return

            # 消息去重 + 记录已授权的会话
            with self._lock:
                if message_id in self._seen_message_ids:
                    return

                if len(self._seen_message_order) >= 1000:
                    old_id = self._seen_message_order.popleft()
                    self._seen_message_ids.discard(old_id)

                self._seen_message_ids.add(message_id)
                self._seen_message_order.append(message_id)
                self._known_chat_ids.add(chat_id)

            logger.info("收到飞书消息: chat_id=%s", chat_id)

            # Agent 推理放到线程池执行
            self._executor.submit(
                process,
                message_id,
                content,
                chat_id,
            )

        dispatcher = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(on_message)
            .build()
        )

        self._ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=dispatcher,
            log_level=lark.LogLevel.INFO,
        )

        logger.info("启动飞书长连接")
        self._ws_client.start()