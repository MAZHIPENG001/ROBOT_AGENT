"""DeepSeek multimodal language model as a LIBERO OSC policy.

Drop-in peer of OpenPiPolicy/MaasPolicy. It implements Observation -> ActionChunk
and NEVER directly advances or resets the simulator.

This is an experimental low-frequency policy, not a trained VLA model.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageOps

from ..config import PolicyConfig
from ..types import ActionChunk, Observation


SYSTEM_PROMPT = """You are an experimental robot action policy for a Franka arm in LIBERO.
You receive a task, current state and two RGB camera images.
Your only task is to choose ONE small 7-dimensional action for the next few
simulator steps. You are NOT given ground-truth object coordinates.

The LIBERO OSC_POSE action is [dx,dy,dz,drx,dry,drz,gripper].
The first six values are NORMALIZED controller inputs, NOT meters or radians.
The first 3 control translational motion; the next 3 control rotational motion.
The gripper command is -1 for open and +1 for closed.
The state is [eef_x,eef_y,eef_z,eef_ax,eef_ay,eef_az,finger_qpos_0,finger_qpos_1].
Use the observations to select short, cautious movements; do not assume pixel
positions are already 3D target coordinates. Do not invent robot geometry.
Do not output a trajectory, a plan, Markdown, or explanation.
Return ONLY valid JSON in this exact form:
{"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]}
If uncertain, use zero translation and rotation and keep the previous gripper
command (provided in the user observation). The action MUST have seven finite
JSON numbers; gripper MUST be exactly -1 or +1.
IMPORTANT: Never exceed the per-component bounds supplied in the observation.
"""


class DeepSeekVlmPolicy:
    """Direct VLM-to-7D action policy. All execution belongs to RobotController."""

    _ALLOWED = {
        "api_key", "base_url", "model", "timeout", "image_size",
        "jpeg_quality", "max_tokens", "repeat_steps", "max_translation",
        "max_rotation", "max_gripper", "request_path",
    }
    _SPACE = "libero_delta_pose_axisangle_gripper_7d"

    def __init__(self, *, config: PolicyConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        opts = config.options
        unknown = set(opts) - self._ALLOWED
        if unknown:
            raise ValueError(f"DeepSeek VLM 未知 options: {sorted(unknown)}")
        if config.outputs.action_dim != 7 or config.outputs.action_space != self._SPACE:
            raise ValueError("DeepSeek VLM 当前只支持 LIBERO OSC_POSE 7D 动作协议")
        if config.outputs.actions_key != "action":
            raise ValueError("DeepSeek VLM outputs.actions_key 必须是 action")

        self.api_key = str(opts.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or "")
        if not self.api_key:
            raise ValueError("DeepSeek VLM 缺少 DEEPSEEK_API_KEY")
        self.base_url = str(opts.get("base_url", "https://api.deepseek.com")).rstrip("/")
        if not self.base_url.startswith("https://"):
            raise ValueError("DeepSeek VLM base_url 必须使用 HTTPS")
        self.model = str(opts.get("model", "deepseek-flash")).strip()
        if not self.model:
            raise ValueError("DeepSeek VLM model 不能为空")
        self.timeout = float(opts.get("timeout", 120))
        self.image_size = int(opts.get("image_size", 384))
        self.jpeg_quality = int(opts.get("jpeg_quality", 85))
        # auto / empty = omit max_tokens from the HTTP request, use provider default.
        token_value = opts.get("max_tokens", "auto")
        if token_value is None or str(token_value).strip().lower() in {"", "auto", "none", "null"}:
            self.max_tokens = None
        else:
            try:
                self.max_tokens = int(token_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_tokens 应为整数或 auto") from exc
        self.repeat_steps = int(opts.get("repeat_steps", 2))
        self.max_translation = float(opts.get("max_translation", 0.20))
        self.max_rotation = float(opts.get("max_rotation", 0.12))
        self.max_gripper = float(opts.get("max_gripper", 1.0))
        self.request_path = str(opts.get("request_path", "/chat/completions"))
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 300:
            raise ValueError("DeepSeek VLM timeout 必须为 0~300 秒")
        if not 64 <= self.image_size <= 1024:
            raise ValueError("DeepSeek VLM image_size 必须为 64~1024")
        if not 50 <= self.jpeg_quality <= 100:
            raise ValueError("DeepSeek VLM jpeg_quality 必须为 50~100")
        if self.max_tokens is not None and not 64 <= self.max_tokens <= 16384:
            raise ValueError("DeepSeek VLM max_tokens 必须为 64~16384（实际可用上限取决于模型服务）")
        if not 1 <= self.repeat_steps <= 5:
            raise ValueError("DeepSeek VLM repeat_steps 必须为 1~5")
        if not 0 < self.max_translation <= 1 or not 0 < self.max_rotation <= 1:
            raise ValueError("DeepSeek VLM 位移和旋转限幅必须为 (0,1]")
        if self.max_gripper != 1.0:
            raise ValueError("LIBERO 夹爪必须使用 -1/+1 编码；max_gripper 必须为 1.0")
        if self.request_path != "/chat/completions":
            raise ValueError("目前仅支持 /chat/completions 接口")

        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            timeout=self.timeout,
            trust_env=True,
        )
        self._owns_client = client is None
        self._episode_id: str | None = None
        self._last_action = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)

    def _encode_image(self, image: np.ndarray) -> str:
        rgb = np.asarray(image)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.size:
            raise ValueError("DeepSeek VLM 图像必须为 H×W×3 uint8 RGB")
        pil = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        pil = ImageOps.pad(
            pil, (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR, color=(0, 0, 0),
        )
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _parse_content(content: str) -> dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek VLM 未返回文本动作（模型可能不支持当前图像格式）")
        raw = content.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.S | re.I)
        if match:
            raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek VLM 没有返回合法 JSON 动作") from exc
        if not isinstance(obj, dict):
            raise ValueError("DeepSeek VLM 返回的 JSON 顶层必须是对象")
        return obj

    def _validate_action(self, obj: dict[str, Any]) -> np.ndarray:
        raw = obj.get(self.config.outputs.actions_key)
        if not isinstance(raw, list) or len(raw) != 7:
            raise ValueError("DeepSeek VLM 必须返回一个 7 元素 action 列表")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in raw):
            raise ValueError("DeepSeek VLM 的 action 必须全部是有限 JSON 数字")
        action = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(action).all():
            raise ValueError("DeepSeek VLM 动作包含非有限 float32 数值")
        if (np.abs(action[:3]) > self.max_translation + 1e-6).any():
            raise ValueError("DeepSeek VLM 位移命令超出安全限幅，拒绝执行")
        if (np.abs(action[3:6]) > self.max_rotation + 1e-6).any():
            raise ValueError("DeepSeek VLM 旋转命令超出安全限幅，拒绝执行")
        if float(action[6]) not in (-1.0, 1.0):
            raise ValueError("DeepSeek VLM 夹爪必须精确为 -1 或 +1")
        return action

    def predict(self, observation: Observation) -> ActionChunk:
        if observation.done or observation.success:
            raise RuntimeError("LIBERO 任务已结束，请先 reset")
        if observation.episode_id != self._episode_id:
            self._episode_id = observation.episode_id
            self._last_action = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)

        state = np.asarray(observation.state, dtype=np.float32)
        indices = self.config.inputs.state_indices
        if indices is not None:
            state = state[indices]
        if state.shape != (self.config.inputs.state_dim,) or not np.isfinite(state).all():
            raise ValueError("DeepSeek VLM 状态维度不匹配或存在 NaN/Inf")

        camera_content = []
        for logical_name, label in self.config.inputs.images.items():
            if logical_name not in observation.images:
                raise ValueError(f"DeepSeek VLM 缺少相机 {logical_name}")
            camera_content.extend([
                {"type": "text", "text": f"Camera: {label} (RGB view)"},
                {"type": "image_url", "image_url": {
                    "url": self._encode_image(observation.images[logical_name])
                }},
            ])
        if not camera_content:
            raise ValueError("DeepSeek VLM 至少需要一个 RGB 相机")

        situation = {
            "task": observation.instruction,
            "episode_step": observation.step_index,
            self.config.inputs.state_key: state.tolist(),
            "previous_action": self._last_action.tolist(),
            "limits": {"translation_abs_max_each": self.max_translation,
                       "rotation_abs_max_each": self.max_rotation,
                       "gripper_allowed": [-1, 1]},
            "output_json_key": self.config.outputs.actions_key,
        }
        content = [{"type": "text", "text": json.dumps(situation, ensure_ascii=False)},
                   *camera_content]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        start = time.monotonic()
        try:
            response = self._client.post(self.request_path, json=payload)
        except httpx.RequestError as exc:
            # Never expose Authorization headers, base64 images or URLs in errors.
            raise RuntimeError(f"DeepSeek VLM 网络请求失败: {type(exc).__name__}") from exc
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"DeepSeek VLM HTTP {response.status_code}；请检查模型权限、请求格式和网关")
        try:
            data = response.json()
            choice = data["choices"][0]
            msg = choice["message"]
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ValueError("DeepSeek VLM 响应缺少 choices[0].message") from exc
        if choice.get("finish_reason") == "length":
            # Some reasoning-enabled services count internal reasoning tokens
            # toward max_tokens; never try to parse or execute truncated JSON.
            usage = data.get("usage") if isinstance(data, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            completion = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details")
            details = details if isinstance(details, dict) else {}
            reasoning = details.get("reasoning_tokens")
            extra = (
                f"; completion_tokens={completion}" if type(completion) is int else ""
            )
            if type(reasoning) is int:
                extra += f"; reasoning_tokens={reasoning}"
            budget = (f"请求 max_tokens={self.max_tokens}"
                      if self.max_tokens is not None
                      else "请求未指定 max_tokens（使用服务端默认限制）")
            raise ValueError(
                f"DeepSeek VLM 响应被截断（{budget}{extra}）；"
                "建议使用支持简短输出的非推理模式，或确认服务端默认输出限制"
            )
        obj = self._parse_content(msg.get("content"))
        action = self._validate_action(obj)
        self._last_action = action.copy()
        actions = np.repeat(action[np.newaxis, :], self.repeat_steps, axis=0)
        return ActionChunk(
            actions=actions,
            action_space=self.config.outputs.action_space,
            metadata={"policy": self.config.name, "model": self.model,
                      "inference_ms": round((time.monotonic()-start)*1000, 1),
                      "repeat_steps": self.repeat_steps},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()