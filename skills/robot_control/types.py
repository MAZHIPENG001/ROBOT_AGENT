"""RobotControl 中与环境、策略无关的统一数据协议。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Observation:
    # 统一约定：图像是 RGB uint8 的 H×W×3；归一化/缩放由具体策略负责。
    images: dict[str, np.ndarray]
    state: np.ndarray
    instruction: str
    episode_id: str = ""
    step_index: int = 0
    done: bool = False
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionChunk:
    # 必须为 [T, action_dim]；action 的物理含义取决于机器人适配器。
    actions: np.ndarray
    action_space: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepFeedback:
    done: bool = False
    success: bool = False
    episode_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    """配置驱动构造约定：Driver(*, config: PolicyConfig)。只推理，不操作机器人。"""
    def predict(self, observation: Observation) -> ActionChunk: ...
    def close(self) -> None: ...


class RobotAdapter(Protocol):
    """配置驱动构造约定：Driver(*, config: RobotConfig)。资源生命周期由驱动负责。"""
    action_dim: int
    action_space: str
    def observe(self) -> Observation: ...
    def step(self, action: np.ndarray) -> StepFeedback: ...
    def close(self) -> None: ...
