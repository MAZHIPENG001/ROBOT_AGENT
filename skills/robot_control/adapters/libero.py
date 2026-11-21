"""使用现有 LiberoEnvClient；不负责 reset 和启动 worker。"""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skills.robot_env.libero.bridge import LiberoEnvClient
from ..types import Observation, StepFeedback


class LiberoAdapter:
    action_dim = 7
    action_space = "libero_delta_pose_axisangle_gripper_7d"

    def __init__(self, client: LiberoEnvClient | None = None) -> None:
        if client is None:
            from skills.robot_env.libero.bridge import LiberoEnvClient
            client = LiberoEnvClient()
        self.client = client

    @staticmethod
    def _decode_png(encoded: str) -> np.ndarray:
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))

    def observe(self) -> Observation:
        data = self.client.request("observe", include_images=True)
        if not data.get("scene_loaded"):
            raise RuntimeError("LIBERO 场景未加载：先调用 robot_libero_reset")
        # worker.observe() 已将两张原始 LIBERO 图像旋转 180°，此处不能再翻转。
        images = {
            "primary": self._decode_png(data["agentview_png_base64"]),
            "wrist": self._decode_png(data["eye_in_hand_png_base64"]),
        }
        state = np.asarray(data["observation_state"], dtype=np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("LIBERO 返回的 observation_state 无效")
        return Observation(
            images=images,
            state=state,
            instruction=str(data["task_description"]),
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
            metadata={"step_count": result.get("step_count"), "reward": result.get("last_reward")},
        )

    def close(self) -> None:
        # 不关闭共享 LIBERO 场景，场景由 robot_env Skill 管理。
        pass
