"""仅向 SkillRegistry 暴露一个 Skill，核心控制协议可独立导入测试。"""
__all__ = ["RobotControlSkill"]


def __getattr__(name: str):
    if name == "RobotControlSkill":
        from .tool import RobotControlSkill
        return RobotControlSkill
    raise AttributeError(name)
