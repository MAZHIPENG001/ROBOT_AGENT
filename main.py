"""机器人 Agent 的精简入口。"""

from pathlib import Path

from dotenv import load_dotenv
from runtime import AgentRuntime


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    AgentRuntime().run()


if __name__ == "__main__":
    main()