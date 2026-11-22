"""所有驱动共用的 I/O 校验；动作空间名称代表物理含义，不能只比较维度。"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RobotConfig
from .types import ActionChunk, Observation


def validate_observation(observation: Observation, config: RobotConfig) -> None:
    missing = config.observation.images.keys() - observation.images.keys()
    if missing:
        raise ValueError(f"机器人缺少配置的相机：{sorted(missing)}")
    for name in config.observation.images:
        image = np.asarray(observation.images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not image.size:
            raise ValueError(f"相机 {name} 必须输出非空 H×W×3 RGB uint8 图像")
    state = np.asarray(observation.state)
    if state.shape != (config.observation.state_dim,) or not np.isfinite(state).all():
        raise ValueError(f"机器人状态应为 {config.observation.state_dim} 维有限向量")
    if not isinstance(observation.instruction, str):
        raise ValueError("机器人 instruction 必须为字符串")


def validate_action_chunk(chunk: ActionChunk, action_dim: int, action_space: str) -> np.ndarray:
    actions = np.asarray(chunk.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise ValueError(f"策略动作形状 {actions.shape} 与机器人动作维度 {action_dim} 不兼容")
    if chunk.action_space != action_space:
        raise ValueError(f"策略动作空间 {chunk.action_space!r} 与机器人动作空间 {action_space!r} 不匹配")
    if not actions.size or not np.isfinite(actions).all():
        raise ValueError("策略返回空动作或非有限数值")
    return actions
