"""DexOS MaaS VLA policy, independent of LIBERO and robot execution.

Reads the existing unified Observation and returns an ActionChunk. No SSH
forwarding or separate Python worker: MaaS is a remote HTTPS inference API.
"""
from __future__ import annotations

import base64
import io
import os
import time
from collections import deque
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageOps

from ..config import PolicyConfig
from ..types import ActionChunk, Observation


class MaasPolicy:
    def __init__(self, *, config: PolicyConfig) -> None:
        self.config = config
        opts = config.options
        allowed = {
            "base_url", "api_key", "model_id", "robot_id", "action_type",
            "image_size", "jpeg_quality", "timeout", "history_size",
            "history_fps", "verify_model", "right_wrist_fallback",
        }
        unknown = set(opts) - allowed
        if unknown:
            raise ValueError(f"未知 MaaS options: {sorted(unknown)}")
        self.base_url = str(opts.get("base_url", "https://maas.dexmal.com")).rstrip("/")
        if not self.base_url.startswith("https://"):
            raise ValueError("MaaS base_url 必须是 HTTPS")
        self.api_key = str(opts.get("api_key") or os.getenv("DEXOS_API_KEY") or "")
        if not self.api_key:
            raise ValueError("缺少 DEXOS_API_KEY")
        self.model_id = str(opts.get("model_id", "dm05"))
        self.robot_id = str(opts.get("robot_id", "franka"))
        self.action_type = str(opts.get("action_type", "JOINT")).upper()
        self.image_size = int(opts.get("image_size", 640))
        self.jpeg_quality = int(opts.get("jpeg_quality", 90))
        self.timeout = float(opts.get("timeout", 60))
        self.history_size = int(opts.get("history_size", 0))
        self.history_fps = float(opts.get("history_fps", 2))
        self.verify_model = str(opts.get("verify_model", "true")).lower() in ("true", "1", "yes")
        self.right_wrist_fallback = str(opts.get("right_wrist_fallback", "disabled")).strip().lower()
        if self.right_wrist_fallback not in {"disabled", "black"}:
            raise ValueError("right_wrist_fallback 仅支持 disabled 或 black")
        if not 64 <= self.image_size <= 2048:
            raise ValueError("image_size 应在 64-2048 之间")
        if not 50 <= self.jpeg_quality <= 100 or self.timeout <= 0:
            raise ValueError("jpeg_quality 或 timeout 无效")
        if not 0 <= self.history_size <= 32 or self.history_fps <= 0:
            raise ValueError("history_size 应为 0..32，history_fps > 0")
        if self.model_id == "dw05" and self.history_size:
            raise ValueError("dw05 不支持历史图像")
        if self.robot_id == "franka" and (self.action_type != "JOINT" or config.outputs.action_dim != 8):
            raise ValueError("MaaS Franka 基础模型要求 JOINT 与 8 维输出")
        if config.inputs.state_dim != 8 and self.robot_id == "franka":
            raise ValueError("MaaS Franka 状态必须是 8 维（7关节 + 夹爪）")
        self.session = requests.Session()
        self.session.trust_env = True
        self.session.headers.update({"X-API-Key": self.api_key, "Accept": "application/json"})
        self._history: deque[str] = deque(maxlen=max(1, self.history_size))
        self._episode = None
        self._previous_main: str | None = None
        self._previous_step: int | None = None
        self._last_hist_time = 0.0
        if self.verify_model:
            self._check_model()

    def _call(self, method: str, endpoint: str, *, payload: dict | None = None) -> dict:
        """Call MaaS and surface safe, actionable HTTP/business errors.

        Do not print headers, payloads or image base64 strings to Agent logs.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.request(method, url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"MaaS {endpoint} 网络请求失败: {type(exc).__name__}"
            ) from exc

        status = getattr(resp, "status_code", 200)
        try:
            body = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"MaaS {endpoint} HTTP {status}; 响应不是 JSON（可能是网关/代理返回的 HTML）"
            ) from exc

        if not isinstance(body, dict):
            raise RuntimeError(f"MaaS {endpoint} HTTP {status}; JSON 顶层不是对象")

        code = body.get("code")
        raw_message = body.get("message") or body.get("msg") or body.get("detail") or ""
        safe_message = " ".join(str(raw_message).split())
        if self.api_key:
            safe_message = safe_message.replace(self.api_key, "[REDACTED]")
        safe_message = safe_message[:300]

        if not (200 <= status < 300) or code != 2000:
            # The model listing reported 3 images; our current LIBERO profile
            # supplies 2. Include only counts, never camera content.
            context = ""
            if endpoint == "/api/v1/action" and isinstance(payload, dict):
                images = payload.get("images")
                state = payload.get("state")
                context = (
                    f"; model_id={self.model_id}; robot_id={self.robot_id}"
                    f"; action_type={self.action_type}"
                    f"; images_count={len(images) if isinstance(images, dict) else 'invalid'}"
                    f"; state_dim={len(state) if isinstance(state, list) else 'invalid'}"
                )
            raise RuntimeError(
                f"MaaS {endpoint} HTTP {status}; code={code}; "
                f"message={safe_message or '<empty>'}{context}"
            )
        return body

    def _check_model(self) -> None:
        body = self._call("GET", "/api/v1/model")
        models = body.get("data")
        if not isinstance(models, list):
            raise RuntimeError("MaaS /model 响应缺少模型列表")
        match = next((m for m in models if isinstance(m, dict) and m.get("id") == self.model_id), None)
        if match is None or match.get("status") != "success":
            raise RuntimeError(f"MaaS 模型 {self.model_id!r} 不可用")
        capabilities = match.get("capabilities", {})
        supported = capabilities.get("robot_id", [])
        if supported and self.robot_id not in supported:
            raise RuntimeError(f"模型 {self.model_id} 不支持 robot_id={self.robot_id}")
        # action_dim=14 denotes maximum capacity, not the Franka JOINT dimensionality.
        max_history = capabilities.get("max_history_images")
        if max_history is not None and self.history_size > int(max_history):
            raise RuntimeError(f"模型 {self.model_id} 最多支持 {max_history} 张历史图像")

    def _image(self, rgb: Any) -> str:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("MaaS 图像必须是 H×W×3 RGB uint8")
        pil = Image.fromarray(arr, "RGB")
        # Padding rather than distortion; MaaS advertised maximum image size 640×640.
        pil = ImageOps.pad(pil, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
        stream = io.BytesIO()
        pil.save(stream, format="JPEG", quality=self.jpeg_quality)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")

    def predict(self, observation: Observation) -> ActionChunk:
        indices = self.config.inputs.state_indices
        raw = np.asarray(observation.state, dtype=np.float32)
        state = raw if indices is None else raw[indices]
        if state.shape != (self.config.inputs.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"MaaS Franka 状态需要 {self.config.inputs.state_dim} 个有限数值")
        imgs = {}
        for logical_name, api_name in self.config.inputs.images.items():
            if logical_name not in observation.images:
                raise ValueError(f"缺少图像 {logical_name!r}")
            imgs[api_name] = self._image(observation.images[logical_name])
        if not imgs:
            raise ValueError("MaaS 至少需要一张图像")

        placeholder_cameras = []
        if not self.model_id.startswith("ft:"):
            required = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
            missing = required - imgs.keys()
            if missing == {"cam_right_wrist"} and self.right_wrist_fallback == "black":
                # Explicit compatibility probe only, not an actual right-wrist camera.
                black = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                imgs["cam_right_wrist"] = self._image(black)
                placeholder_cameras.append("cam_right_wrist")
                missing.clear()
            if missing:
                raise ValueError(
                    f"MaaS model {self.model_id} requires three camera slots; missing={sorted(missing)}. "
                    "For single-arm Franka, set right_wrist_fallback=black only for a test, "
                    "or supply a real correctly calibrated third view."
                )

        if observation.episode_id != self._episode:
            self._episode = observation.episode_id
            self._history.clear()
            self._previous_main = None
            self._previous_step = None
            self._last_hist_time = time.monotonic()
        main_image = imgs.get("cam_high")
        now = time.monotonic()
        # Historical images MUST be earlier observations, not copies of the current frame.
        if (self.history_size and self._previous_main is not None
                and self._previous_step is not None
                and observation.step_index != self._previous_step
                and now - self._last_hist_time >= 1.0 / self.history_fps):
            self._history.append(self._previous_main)
            self._last_hist_time = now
        self._previous_main = main_image
        self._previous_step = observation.step_index

        payload = {
            "model_id": self.model_id,
            "robot_id": self.robot_id,
            "action_type": self.action_type,
            "prompt": str(observation.instruction),
            "state": state.astype(float).tolist(),
            "images": imgs,
        }
        if len(payload["prompt"]) > 8192:
            raise ValueError("MaaS prompt 长度超过 8192")
        if self.history_size and self._history:
            payload["history_images"] = list(self._history)
        body = self._call("POST", "/api/v1/action", payload=payload)
        data = body.get("data")
        if not isinstance(data, dict) or "actions" not in data:
            raise RuntimeError("MaaS 响应缺少 data.actions")
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.config.outputs.action_dim or actions.shape[0] == 0:
            raise ValueError(f"MaaS 返回动作形状 {actions.shape}，期望 [T,{self.config.outputs.action_dim}]")
        if not np.isfinite(actions).all():
            raise ValueError("MaaS 动作包含 NaN/Inf，拒绝执行")
        return ActionChunk(
            actions=actions,
            action_space=self.config.outputs.action_space,
            metadata={"model_id": self.model_id, "request_id": data.get("id"),
                      "session_id": data.get("session_id"),
                      "placeholder_cameras": placeholder_cameras},
        )

    def close(self) -> None:
        self.session.close()
