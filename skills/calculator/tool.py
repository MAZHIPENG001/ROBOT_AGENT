from langchain_core.tools import tool


class CalculatorSkill:
    """计算技能：支持 0～100 以内整数的加减法。"""

    def tools(self) -> list:
        """向 LangChain Agent 注册计算工具。"""

        @tool
        def calculator(a: int, b: int, op: str) -> str:
            """计算两个 0～100 以内整数的加减法。op 只能为 + 或 -。"""

            if not (0 <= a <= 100 and 0 <= b <= 100):
                return "超出技能计算范围，仅支持 0～100 的整数"

            if op == "+":
                return str(a + b)

            if op == "-":
                return str(a - b)

            return "该技能仅支持加法和减法"

        return [calculator]