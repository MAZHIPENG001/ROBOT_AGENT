"""Agent 核心：模型和工具调用；不负责聊天记录的持久化。"""

import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system.md"
SKILLS_DIR = BASE_DIR / "skills"


def load_skill_instructions() -> str:
    """自动读取每个已安装 Skill 的 SKILL.md，组合成模型使用说明。

    与 skills/registry.py 的发现规则一致：
    只考虑 skills/ 下有 __init__.py、且目录名不以 _ 开头的子目录。
    """
    sections = []

    if not SKILLS_DIR.is_dir():
        raise FileNotFoundError(f"技能目录不存在：{SKILLS_DIR}")

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if (
            not skill_dir.is_dir()
            or skill_dir.name.startswith("_")
            or not (skill_dir / "__init__.py").is_file()
        ):
            continue

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            print(f"[SKILL] 未找到文档：{skill_md}")
            continue

        content = skill_md.read_text(encoding="utf-8").strip()
        if not content:
            print(f"[SKILL] 文档为空：{skill_md}")
            continue

        sections.append(f"## {skill_dir.name}\n\n{content}")
        print(f"[SKILL] 已读取文档：{skill_md.relative_to(BASE_DIR)}")

    return "\n\n".join(sections)


def build_agent(tools: list):
    # 通用规则保留在 prompts/system.md；每个技能的说明来自其 SKILL.md。
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    skill_instructions = load_skill_instructions()
    if skill_instructions:
        system_prompt += (
            "\n\n# 已加载的 Skill 文档\n\n"
            "以下内容来自本地各技能的 SKILL.md。"
            "根据用户需求选择已注册的工具，遵循技能的适用范围和注意事项。"
            "工具是否可执行、参数与最终结果以实际注册的工具为准。\n\n"
            f"{skill_instructions}"
        )

    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


def ask_agent(
    agent,
    text: str,
    chat_id: str | None = None,
    history: list[dict] | None = None,
) -> str:
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