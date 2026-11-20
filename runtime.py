"""单用户 Agent 运行时：飞书消息、JSON 会话、快捷命令、定时回调。

所有以反斜杠开头的快捷命令在这里处理，不经过 DeepSeek。
定时任务和普通对话使用同一个 Agent、同一组注册工具。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import ToolMessage

from agent import ask_agent, build_agent
from skills.registry import SkillRegistry

ROOT = Path(__file__).resolve().parent
CHAT_DIR = ROOT / "cache" / "chat"
ID_PATTERN = r"[A-Za-z0-9_-]+"
logger = logging.getLogger(__name__)

# 仅用于判断后台任务是否已经发送过消息，绝非 Agent 工具白名单。
FEISHU_SEND_TOOLS = {
    "feishu_send_message", "feishu_send_file", "feishu_send_image"
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ChatHistoryManager:
    def __init__(self, directory: Path | str = CHAT_DIR):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active_path = self.directory / "_active.json"
        self._lock = threading.RLock()

    def _path(self, chat_id: str) -> Path:
        if not re.fullmatch(ID_PATTERN, chat_id) or chat_id.startswith("_"):
            raise ValueError("非法会话 ID")
        return self.directory / f"{chat_id}.json"

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"文件内容格式错误：{path.name}")
        return data

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        # 先写临时文件，再原子替换，防止异常中断产生半截 JSON。
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _active_id(self) -> str | None:
        if not self.active_path.exists():
            return None
        chat_id = self._read_json(self.active_path).get("active_id")
        if not chat_id:
            return None
        try:
            path = self._path(chat_id)
        except (ValueError, TypeError):
            return None
        return chat_id if path.is_file() else None

    def _set_active_id(self, chat_id: str | None) -> None:
        self._write_json(self.active_path, {"active_id": chat_id})

    def _load(self, chat_id: str) -> dict:
        path = self._path(chat_id)
        if not path.is_file():
            raise FileNotFoundError(f"找不到会话：{chat_id}")
        record = self._read_json(path)
        if record.get("id") != chat_id or not isinstance(record.get("turns"), list):
            raise ValueError(f"会话文件格式错误：{path.name}")
        return record

    def new_chat(self, title: str = "新对话") -> dict:
        with self._lock:
            chat_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            record = {
                "id": chat_id,
                "title": title.strip()[:80] or "新对话",
                "created_at": _now(),
                "updated_at": _now(),
                "turns": [],
            }
            self._write_json(self._path(chat_id), record)
            self._set_active_id(chat_id)
            return record

    def active_chat(self, create: bool = False) -> dict | None:
        with self._lock:
            active_id = self._active_id()
            if active_id:
                return self._load(active_id)
            return self.new_chat() if create else None

    def continue_chat(self, chat_id: str) -> dict:
        with self._lock:
            record = self._load(chat_id)
            self._set_active_id(chat_id)
            return record

    def end_chat(self) -> None:
        with self._lock:
            self._set_active_id(None)

    def append_turn(self, chat_id: str, user: str, assistant: str) -> None:
        with self._lock:
            record = self._load(chat_id)
            record["turns"].append({
                "user": user,
                "assistant": assistant,
                "created_at": _now(),
            })
            if len(record["turns"]) == 1 and record["title"] == "新对话":
                record["title"] = user.strip().replace("\n", " ")[:60] or "新对话"
            record["updated_at"] = _now()
            self._write_json(self._path(chat_id), record)

    def edit_turn(self, chat_id: str, index: int, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError("只能修改 user 或 assistant 消息")
        if not content.strip():
            raise ValueError("修改后的内容不能为空")
        with self._lock:
            record = self._load(chat_id)
            if not 1 <= index <= len(record["turns"]):
                raise IndexError(f"回合编号超出范围：当前共 {len(record['turns'])} 回合")
            record["turns"][index - 1][role] = content.strip()
            record["updated_at"] = _now()
            self._write_json(self._path(chat_id), record)

    def delete_chat(self, chat_id: str) -> None:
        with self._lock:
            path = self._path(chat_id)
            if not path.is_file():
                raise FileNotFoundError(f"找不到会话：{chat_id}")
            if self._active_id() == chat_id:
                self._set_active_id(None)
            path.unlink()

    def list_chats(self, limit: int = 15) -> str:
        with self._lock:
            chats = []
            for path in self.directory.glob("*.json"):
                if path.name == self.active_path.name:
                    continue
                try:
                    record = self._load(path.stem)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                chats.append(record)
            chats.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
            active = self._active_id()
            if not chats:
                return "暂无历史对话。发送 \\new 开始新对话。"
            rows = ["历史对话（最近更新排前）："]
            for item in chats[:limit]:
                tag = " [当前]" if item["id"] == active else ""
                rows.append(
                    f"{item['id']}{tag} | {item['title']} | {len(item['turns'])} 回合"
                )
            rows.append("使用 \\continue_{ID} 恢复其中一段对话。")
            return "\n".join(rows)

    def show_chat(self, chat_id: str, max_chars: int = 3200) -> str:
        with self._lock:
            record = self._load(chat_id)
            chunks = [f"会话 {chat_id} | {record['title']} | 共 {len(record['turns'])} 回合"]
            for i, turn in enumerate(record["turns"], start=1):
                chunks.append(f"第 {i} 回合\n你：{turn['user']}\nAgent：{turn['assistant']}")
            result = "\n\n".join(chunks)
            if len(result) > max_chars:
                result = result[:max_chars] + "\n……（已截断，可直接查看对应 JSON）"
            return result

class AgentRuntime:
    """统一入口：一个 Agent + 一个 SkillRegistry + 一个聊天历史管理器。"""

    def __init__(self) -> None:
        self.registry = SkillRegistry()
        self.tools = self.registry.load_tools()
        self.agent = build_agent(self.tools)
        self.feishu = self.registry.get("feishu")
        self.scheduler = self.registry.get("scheduler")
        self.chats = ChatHistoryManager(CHAT_DIR)

        self._tools_by_name = {tool.name: tool for tool in self.tools}
        self.target_file = ROOT / "state" / "last_feishu_chat.json"
        self._message_lock = threading.RLock()
        self._target_lock = threading.Lock()
        self._agent_lock = threading.RLock()

        # 恢复飞书真实会话 ID，保证重启后定时任务仍能发送通知。
        last_chat = self._load_target()
        if last_chat:
            self.feishu.remember_chat(last_chat)

    def _load_target(self) -> str | None:
        with self._target_lock:
            try:
                data = json.loads(self.target_file.read_text(encoding="utf-8"))
                chat_id = data.get("chat_id", "")
                if isinstance(chat_id, str) and chat_id.startswith("oc_"):
                    return chat_id
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        return None

    def _save_target(self, chat_id: str) -> None:
        if not isinstance(chat_id, str) or not chat_id.startswith("oc_"):
            raise ValueError("无效飞书会话 ID")
        with self._target_lock:
            self.target_file.parent.mkdir(parents=True, exist_ok=True)
            temp = self.target_file.with_name(
                f".{self.target_file.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temp.open("w", encoding="utf-8") as f:
                    json.dump({"chat_id": chat_id}, f, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp, self.target_file)
            finally:
                temp.unlink(missing_ok=True)

    def _call_tool(self, name: str, args: dict) -> str:
        """快捷命令直接调用已注册的工具，不使用大模型。"""
        selected = self._tools_by_name.get(name)
        if selected is None:
            return f"执行失败：未注册工具 {name}"
        try:
            return str(selected.invoke(args))
        except Exception:
            logger.exception("快捷命令执行失败：%s", name)
            return f"执行失败：{name} 出现异常，请查看日志。"

    @staticmethod
    def _help() -> str:
        return (
            "会话管理：\n"
            "\\help：查看所有快捷命令\n"
            "\\new [标题]：结束当前对话，创建新对话\n"
            "\\end：结束当前对话（保留历史）\n"
            "\\current：查看当前对话\n"
            "\\list：列出历史对话\n"
            "\\continue_{ID}：恢复对话\n"
            "\\show_{ID}：查看对话内容\n"
            "\\edit_{ID}_{N}_user 新内容：修改第 N 回合的用户消息\n"
            "\\edit_{ID}_{N}_assistant 新内容：修改第 N 回合的 Agent 回复\n"
            "\\delete_{ID}：删除指定对话的 JSON 文件\n\n"
            "定时任务：\n"
            "\\tasks：查看所有等待执行的任务\n"
            "\\cancel_{task_ID}：取消指定任务"
        )

    def handle_command(self, message: str) -> str | None:
        """唯一的反斜杠命令入口；普通消息返回 None。"""
        cmd = message.strip()
        if not cmd.startswith("\\"):
            return None

        try:
            if cmd == r"\help":
                return self._help()

            if cmd == r"\tasks":
                return self._call_tool("schedule_list", {})

            match = re.fullmatch(
                r"\\cancel_(?:\{(task_[A-Za-z0-9_-]+)\}|(task_[A-Za-z0-9_-]+))",
                cmd,
            )
            if match:
                return self._call_tool(
                    "schedule_cancel",
                    {"job_id": match.group(1) or match.group(2)},
                )
            if cmd.startswith(r"\cancel"):
                return r"用法：\cancel_{task_123456789abc}；先使用 \tasks 查看 ID。"

            match = re.fullmatch(r"\\new(?:\s+([\s\S]+))?", cmd)
            if match:
                record = self.chats.new_chat(match.group(1) or "新对话")
                return f"已创建新对话：{record['id']}。原对话历史已保存。"

            if cmd == r"\end":
                self.chats.end_chat()
                return "当前对话已结束，历史已保存。下一条普通消息会创建新对话。"

            if cmd == r"\current":
                record = self.chats.active_chat(create=False)
                if record is None:
                    return "当前没有正在进行的对话。"
                return (
                    f"当前对话：{record['id']} | {record['title']} | "
                    f"{len(record['turns'])} 回合"
                )

            if cmd == r"\list":
                return self.chats.list_chats()

            match = re.fullmatch(
                rf"\\(continue|show|delete)_(?:\{{({ID_PATTERN})\}}|({ID_PATTERN}))",
                cmd,
            )
            if match:
                action = match.group(1)
                conversation_id = match.group(2) or match.group(3)
                if action == "continue":
                    record = self.chats.continue_chat(conversation_id)
                    return (
                        f"已恢复对话 {conversation_id}，"
                        f"共 {len(record['turns'])} 回合。"
                    )
                if action == "show":
                    return self.chats.show_chat(conversation_id)
                self.chats.delete_chat(conversation_id)
                return f"已删除对话 {conversation_id}。此操作无法通过快捷命令撤销。"

            match = re.fullmatch(
                rf"\\edit_(?:\{{({ID_PATTERN})\}}|({ID_PATTERN}))_"
                rf"(?:\{{(\d+)\}}|(\d+))_(user|assistant)\s+([\s\S]+)",
                cmd,
            )
            if match:
                conversation_id = match.group(1) or match.group(2)
                turn_index = int(match.group(3) or match.group(4))
                role, content = match.group(5), match.group(6)
                self.chats.edit_turn(conversation_id, turn_index, role, content)
                return (
                    f"已修改对话 {conversation_id} "
                    f"第 {turn_index} 回合的 {role} 消息。"
                )

            return "未知快捷命令，发送 \\help 查看所有命令。"
        except (OSError, ValueError, IndexError, TypeError, KeyError) as exc:
            return f"快捷命令执行失败：{exc}"

    def on_message(self, message: str, chat_id: str) -> str:
        """飞书消息入口：快捷命令直接执行；普通消息才调用 Agent。"""
        with self._message_lock:
            self.feishu.remember_chat(chat_id)
            self._save_target(chat_id)
            reply = self.handle_command(message)
            if reply is not None:
                return reply

            record = self.chats.active_chat(create=True)
            with self._agent_lock:
                answer = ask_agent(
                    self.agent,
                    message,
                    chat_id=chat_id,
                    history=record["turns"],
                )
            self.chats.append_turn(record["id"], message, answer)
            return answer

    @staticmethod
    def _already_sent(messages: list) -> bool:
        """防止定时执行过程中 Agent 主动发送后，又收到重复通知。"""
        by_id = {
            call["id"]: call["name"]
            for msg in messages
            for call in (getattr(msg, "tool_calls", None) or [])
            if call.get("id") and call.get("name")
        }
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            if by_id.get(msg.tool_call_id) not in FEISHU_SEND_TOOLS:
                continue
            value = str(msg.content)
            if "发送成功" in value or "API 已确认消息发送成功" in value:
                return True
        return False

    def on_scheduled_task(self, job_id: str, instruction: str) -> None:
        """定时执行：与普通对话共用同一个 Agent，所有 Tools 均可调用。"""
        chat_id = self._load_target()
        now = datetime.now(self.scheduler.timezone).strftime("%Y-%m-%d %H:%M:%S")
        system_context = (
            "你现在正在执行已到时间的定时任务，而不是创建该定时任务。"
            "按任务要求自行选择系统已注册的工具。"
            f"当前时间：{now}；时区：{self.scheduler.timezone}。"
            "如果只需要通知文本，输出最终答复即可，由程序负责通知。"
            "如果需要发送照片或文件，可自行调用飞书发送工具。"
        )
        if chat_id:
            system_context += f"本次可用的飞书目标 chat_id={chat_id}。"
        else:
            system_context += "没有已记录的飞书目标，不要尝试发送飞书消息。"

        try:
            with self._agent_lock:
                result = self.agent.invoke({
                    "messages": [
                        {"role": "system", "content": system_context},
                        {"role": "user", "content": instruction},
                    ]
                })
            answer = result["messages"][-1].content
            if not isinstance(answer, str):
                answer = str(answer)
            already_sent = self._already_sent(result["messages"])
            logger.info("定时任务 %s 执行完成：%s", job_id, answer)
        except Exception:
            logger.exception("定时任务 %s 执行失败", job_id)
            answer, already_sent = "定时任务执行失败，请查看服务日志。", False

        if already_sent:
            return
        if not chat_id:
            logger.warning("定时任务 %s 执行但没有飞书会话可通知", job_id)
            return

        # 只负责送达结果；SchedulerSkill 本身不依赖飞书。
        result = self._call_tool("feishu_send_message", {
            "chat_id": chat_id,
            "text": f"⏰ 定时任务 {job_id}\n{answer}"[:3900],
        })
        logger.info("定时任务 %s 通知结果：%s", job_id, result)

    def run(self) -> None:
        """启动定时器与飞书监听。"""
        self.scheduler.start(self.on_scheduled_task)
        try:
            logger.info("AgentRuntime 已启动")
            self.feishu.start(self.on_message)
        finally:
            self.scheduler.stop()