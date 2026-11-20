
"""自动扫描、初始化并注册所有 Skills。"""

import importlib
import inspect
from pathlib import Path


class SkillRegistry:
    def __init__(self):
        self.skills = {}
        self._discover_skills()

    def _discover_skills(self):
        """扫描 skills/ 目录，自动发现 Skill 类。"""

        skills_dir = Path(__file__).resolve().parent

        for path in sorted(skills_dir.iterdir()):
            if not path.is_dir():
                continue

            if path.name.startswith("_"):
                continue

            if not (path / "__init__.py").exists():
                continue

            # 动态导入 skills.calculator 等包
            module = importlib.import_module(
                f"{__package__}.{path.name}"
            )

            # 根据 __all__ 寻找技能类
            classes = [
                getattr(module, name)
                for name in getattr(module, "__all__", [])
                if inspect.isclass(getattr(module, name, None))
                and name.endswith("Skill")
            ]

            if len(classes) != 1:
                raise RuntimeError(
                    f"{path.name} 应导出且仅导出一个 XxxSkill 类"
                )

            skill_class = classes[0]
            skill = skill_class()

            if not callable(getattr(skill, "tools", None)):
                raise TypeError(
                    f"{skill_class.__name__} 缺少 tools() 方法"
                )

            self.skills[path.name] = skill

            print(f"[SKILL] 已加载: {skill_class.__name__}")

    def load_tools(self) -> list:
        """收集所有 Skill 提供的 LangChain Tools。"""

        tools = []

        for skill in self.skills.values():
            tools.extend(skill.tools())

        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ValueError("发现重复的 Tool 名称")

        return tools

    def get(self, name: str):
        """根据技能名称获取实例。"""
        return self.skills[name]
