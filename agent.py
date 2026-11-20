"""Agent 核心：模型和工具调用；不负责聊天记录的持久化。"""

import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system.md"


def build_agent(tools: list):
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
    )


def ask_agent(agent, text: str, chat_id: str | None = None, history: list[dict] | None = None) -> str:
    """把历史轮次作为输入，返回本轮答案；仅打印本轮新增的工具日志。"""
    messages = []

    # 每次调用时使用实时飞书会话 ID；不把 ID 固化到历史中。
    if chat_id:
        messages.append({
            "role": "system",
            "content": (
                f"当前飞书会话 chat_id={chat_id}。"
                "该 ID 只用于用户明确请求的额外飞书发送操作，"
                "普通聊天由监听器自动回复。"
            ),
        })

    # 只将最近 20 回合发送给模型，但 JSON 文件存储全部历史。
    for turn in (history or [])[-20:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": text})
    print(f"\n[USER] {text}")

    result = agent.invoke({"messages": messages})

    for message in result["messages"][len(messages):]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[AGENT] 调用工具: {call['name']}, 参数: {call['args']}")
        if isinstance(message, ToolMessage):
            print(f"[TOOL RESULT] {message.content}")

    answer = result["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    print(f"[AGENT] 最终回复: {answer}")
    return answer