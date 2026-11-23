"""使用现有 LiberoEnvClient；不负责 reset 和启动 worker。"""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from skills.robot_control.types import Observation, StepFeedback
from skills.robot_control.config import RobotConfig
from .bridge import LiberoEnvClient


class LiberoAdapter:
    action_dim = 7
    action_space = "libero_delta_pose_axisangle_gripper_7d"

    def __init__(self, client: LiberoEnvClient | None = None, *, config: RobotConfig) -> None:
        # 底层 LIBERO 控制器的物理语义不能通过改 YAML 名称而改变。
        if config.action.dim != self.action_dim or config.action.space != self.action_space:
            raise ValueError("LIBERO 配置与底层控制器的 7 维动作协议不一致")
        self.config = config
        unknown = config.options.keys() - {"socket_path", "timeout"}
        if unknown:
            raise ValueError(f"未知 LIBERO 适配器参数：{sorted(unknown)}")
        self.client = client if client is not None else LiberoEnvClient(**config.options)

    @staticmethod
    def _decode_png(encoded: str) -> np.ndarray:
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))

    def observe(self) -> Observation:
        data = self.client.request("observe", include_images=True)
        if not data.get("scene_loaded"):
            raise RuntimeError("LIBERO 场景未加载：先调用 robot_libero_reset")
        # worker.observe() 已将两张原始 LIBERO 图像旋转 180°，此处不能再翻转。
        images = {name: self._decode_png(data[source])
                  for name, source in self.config.observation.images.items()}
        parts = []
        for field in self.config.observation.state:
            part = np.asarray(data[field.source], dtype=np.float32)
            if part.shape != (field.size,) or not np.isfinite(part).all():
                raise ValueError(f"LIBERO 状态字段 {field.source} 应为 {field.size} 维有限向量")
            parts.append(part)
        state = np.concatenate(parts)
        return Observation(
            images=images,
            state=state,
            instruction=str(data[self.config.observation.instruction]),
            episode_id=str(data.get("episode_uid") or ""),
            step_index=int(data.get("step_count", 0)),
            done=bool(data.get("done", False)),
            success=bool(data.get("success", False)),
            metadata={"source": "libero", "task_id": data.get("task_id")},
        )

    def step(self, action: np.ndarray) -> StepFeedback:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("LIBERO 动作必须为 7 个有限数值")
        result = self.client.request("step", action=action.tolist())
        return StepFeedback(
            done=bool(result.get("done", False)),
            success=bool(result.get("success", False)),
            episode_id=str(result.get("episode_uid") or ""),
            metadata={"step_count": result.get("step_count"), "reward": result.get("last_reward"),
                      "state_report": result.get("state_report")},
        )

    def close(self) -> None:
        # 不关闭共享 LIBERO 场景，场景由 robot_env Skill 管理。
        pass
