"""从配置发现驱动并组装控制任务；不依赖具体仿真环境或策略实现。"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG_DIR, ControlDefaults, PolicyConfig, RobotConfig, read_config
from .validation import validate_action_chunk, validate_observation


def _load_driver(path: str):
    module, name = path.split(":")
    driver = getattr(importlib.import_module(module), name)
    if not callable(driver):
        raise TypeError(f"驱动不可调用：{path}")
    return driver


class ConfiguredRobot:
    """在驱动边界核对配置声明与实际数据。"""

    def __init__(self, driver, config: RobotConfig):
        self.driver = driver
        self.config = config
        self.action_dim = config.action.dim
        self.action_space = config.action.space

    def observe(self):
        observation = self.driver.observe()
        validate_observation(observation, self.config)
        return observation

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"机器人动作必须为 {self.action_dim} 维有限向量")
        return self.driver.step(action)

    def close(self):
        self.driver.close()


class ConfiguredPolicy:
    def __init__(self, driver, config: PolicyConfig):
        self.driver = driver
        self.config = config

    def predict(self, observation):
        inputs = self.config.inputs
        missing = inputs.images.keys() - observation.images.keys()
        if missing:
            raise ValueError(f"策略缺少相机：{sorted(missing)}")
        state = np.asarray(observation.state)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("策略输入状态必须为一维有限向量")
        selected = state if inputs.state_indices is None else state[inputs.state_indices]
        if selected.shape != (inputs.state_dim,):
            raise ValueError(f"策略输入状态必须为 {inputs.state_dim} 维")
        chunk = self.driver.predict(observation)
        validate_action_chunk(chunk, self.config.outputs.action_dim, self.config.outputs.action_space)
        return chunk

    def close(self):
        self.driver.close()


class ControlRegistry:
    def __init__(self, config_dir: str | Path | None = None):
        self.config_dir = Path(config_dir or os.getenv("ROBOT_CONFIG_DIR") or DEFAULT_CONFIG_DIR).expanduser().resolve()
        self.defaults = read_config(self.config_dir / "robot_control.yaml", ControlDefaults)
        self.robots = self._read_profiles("robots", RobotConfig)
        self.policies = self._read_profiles("policies", PolicyConfig)
        self.check_pair(self.defaults.robot, self.defaults.policy)

    def _read_profiles(self, directory, model):
        profiles = {}
        folder = self.config_dir / directory
        for path in sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")]):
            profile = read_config(path, model)
            if profile.name in profiles:
                raise ValueError(f"重复的配置名称：{profile.name}")
            profiles[profile.name] = profile
        return profiles

    def check_pair(self, robot: str, policy: str):
        if robot not in self.robots:
            raise ValueError(f"未知 robot={robot}；可用：{list(self.robots)}")
        if policy not in self.policies:
            raise ValueError(f"未知 policy={policy}；可用：{list(self.policies)}")
        r, p = self.robots[robot], self.policies[policy]
        if (r.action.space, r.action.dim) != (p.outputs.action_space, p.outputs.action_dim):
            raise ValueError(f"robot={robot} 与 policy={policy} 的动作空间或维度不兼容")
        missing = p.inputs.images.keys() - r.observation.images.keys()
        if missing:
            raise ValueError(f"robot={robot} 缺少策略相机：{sorted(missing)}")
        indices = p.inputs.state_indices
        if indices is None:
            if r.observation.state_dim != p.inputs.state_dim:
                raise ValueError("机器人与策略状态维度不兼容，请配置 state_indices 或使用匹配的策略")
        elif max(indices) >= r.observation.state_dim:
            raise ValueError("策略 state_indices 超出机器人状态范围")
        return r, p

    def create_robot(self, name: str):
        config = self.robots[name]
        driver = _load_driver(config.driver)(config=config)
        return ConfiguredRobot(driver, config)

    def create_policy(self, name: str):
        config = self.policies[name]
        driver = _load_driver(config.driver)(config=config)
        return ConfiguredPolicy(driver, config)

    def factories(self, robot: str, policy: str):
        self.check_pair(robot, policy)
        return lambda: self.create_robot(robot), lambda: self.create_policy(policy)

    def describe(self) -> dict:
        # options 可能含凭据，不暴露给模型。
        return {
            "defaults": self.defaults.model_dump(),
            "robots": [{"name": r.name, "kind": r.kind,
                        "observation": r.observation.model_dump(),
                        "action": r.action.model_dump(), "action_dim": r.action.dim}
                       for r in self.robots.values()],
            "policies": [{"name": p.name, "inputs": p.inputs.model_dump(),
                          "outputs": p.outputs.model_dump()} for p in self.policies.values()],
        }
