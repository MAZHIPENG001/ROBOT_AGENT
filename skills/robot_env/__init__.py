"""现有 SkillRegistry 只扫描 skills/ 下一级包，此文件导出二级 LIBERO Skill。"""
from .libero import LiberoSkill

__all__ = ["LiberoSkill"]
