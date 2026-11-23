"""运行在 LIBERO 的 Python 中：持有仿真状态、提供 UNIX socket JSON API。

此模块不依赖 Agent、LangChain、DeepSeek 或任何 VLA。
动作接口只有底层 step(action)，由控制适配器调用；本环境 Skill 不注册它。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("libero_env")
SUPPORTED_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
DUMMY_ACTION = [0.0] * 6 + [-1.0]
SOCKET_NAME = "libero.sock"
CAMERA_CHOICES = {"agentview", "eye_in_hand", "side_by_side"}


def _output_dir() -> Path:
    return Path(os.getenv("ROBOT_LIBERO_OUTPUT_DIR", "./cache/libero")).expanduser().resolve()


def _rotation_axis_angle(quat: Any) -> list[float]:
    """与原评测脚本保持一致，默认 LIBERO quat = (x,y,z,w)。"""
    import numpy as np
    quat = np.asarray(quat, dtype=np.float64).copy()
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return [0.0, 0.0, 0.0]
    return (quat[:3] * (2.0 * math.acos(w) / den)).tolist()


class LiberoEnvironment:
    """一份 LIBERO 场景状态。所有调用由单线程 socketserver 顺序执行。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.env = None
        self.obs = None
        self.suite_name: str | None = None
        self.task_id: int | None = None
        self.episode_idx: int | None = None
        self.description: str | None = None
        self.step_count = 0
        self.settle_steps = 0
        self.last_reward = 0.0
        self.done = False
        self.success = False
        self.episode_uid = ""
        self.control_mode = "osc_pose"
        self.report_after_step = False
        # 录像是环境数据采集，与 VLA 和动作规划完全解耦。
        self._video_writer = None
        self._video_path: Path | None = None
        self._video_frames = 0
        self._video_fps = 10
        self._video_camera = "agentview"
        self.last_video: dict[str, Any] | None = None
        self.record_error: str | None = None

        # 高 GUI 风险操作放在单独的 viewer.py 进程，崩溃不会带走 MuJoCo worker。
        self._viewer: subprocess.Popen | None = None
        self._viewer_camera = "agentview"
        self._live_frame = output_dir / "live_frame.jpg"
        self._viewer_ready = output_dir / "viewer.ready"
        self.display_error: str | None = None

    def _viewer_alive(self) -> bool:
        return self._viewer is not None and self._viewer.poll() is None

    def status(self) -> dict[str, Any]:
        return {
            "service": "libero_env", "running": True, "scene_loaded": self.env is not None,
            "pid": os.getpid(), "socket_path": str(self.output_dir / SOCKET_NAME),
            "task_suite_name": self.suite_name, "task_id": self.task_id,
            "episode_idx": self.episode_idx, "task_description": self.description,
            "episode_uid": self.episode_uid, "control_mode": self.control_mode,
            "step_count": self.step_count, "settle_steps": self.settle_steps,
            "last_reward": self.last_reward, "done": self.done, "success": self.success,
            "report_after_step": self.report_after_step,
            "recording": self._video_writer is not None,
            "recording_path": str(self._video_path) if self._video_path else None,
            "recording_frames": self._video_frames if self._video_writer is not None else 0,
            "last_video": self.last_video,
            "record_error": self.record_error,
            "display_active": self._viewer_alive(),
            "display_camera": self._viewer_camera,
            "display_error": self.display_error,
        }

    @staticmethod
    def _validate_camera(camera: str) -> None:
        if camera not in CAMERA_CHOICES:
            raise ValueError(f"camera 只能是 {sorted(CAMERA_CHOICES)}")

    def _frame(self, camera: str):
        """取 LIBERO 当前观测，按用户上传的评测代码旋转 180°。"""
        if self.obs is None:
            raise RuntimeError("请先初始化场景")
        import numpy as np

        main = np.ascontiguousarray(np.asarray(self.obs["agentview_image"])[::-1, ::-1], dtype=np.uint8)
        if camera == "agentview":
            return main
        wrist = np.ascontiguousarray(np.asarray(self.obs["robot0_eye_in_hand_image"])[::-1, ::-1], dtype=np.uint8)
        if camera == "eye_in_hand":
            return wrist
        if wrist.shape[0] != main.shape[0]:
            raise RuntimeError("两路相机高度不一致，无法拼接")
        return np.concatenate((main, wrist), axis=1)

    def _publish_frame(self) -> None:
        """将最新帧原子发布给 GUI 进程；只在启用 GUI 且有场景时写入。"""
        if not self._viewer_alive() or self.obs is None:
            return
        from PIL import Image

        temp = self.output_dir / f".live_{uuid.uuid4().hex}.jpg"
        try:
            Image.fromarray(self._frame(self._viewer_camera)).save(temp, format="JPEG", quality=85)
            os.replace(temp, self._live_frame)
        finally:
            temp.unlink(missing_ok=True)

    def _stop_viewer(self) -> None:
        process = self._viewer
        self._viewer = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self._viewer_ready.unlink(missing_ok=True)
        self._live_frame.unlink(missing_ok=True)

    def display(self, show_window: bool = False, camera: str = "agentview") -> dict[str, Any]:
        """启动/关闭实时观察窗；不会推进仿真，不会开始录像。"""
        if not isinstance(show_window, bool):
            raise ValueError("show_window 必须是布尔值")
        self._validate_camera(camera)
        if not show_window:
            self._stop_viewer()
            self.display_error = None
            return self.status()

        # Linux 图形桌面；SSH/headless 没有可用图形会话时明确报错。
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("没有 DISPLAY/WAYLAND_DISPLAY。可使用 show_window=False 无头运行，录像仍可使用")
        try:
            import importlib.util
            if importlib.util.find_spec("cv2") is None:
                raise RuntimeError("实时窗口依赖 LIBERO Python 环境中的 opencv-python（不能是仅 headless 版本）")
            if self._viewer_alive() and camera == self._viewer_camera:
                return self.status()
            self._stop_viewer()
            self._viewer_camera = camera
            self.display_error = None
            self._viewer = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("viewer.py")),
                 str(self._live_frame), str(self._viewer_ready)],
                stdin=subprocess.DEVNULL,
                # 继承 worker.log 的输出；GUI 启动失败不会使 worker 崩溃。
                stdout=None, stderr=None,
                env=os.environ.copy(),
                start_new_session=True,
            )
            self._publish_frame()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self._viewer_ready.is_file() and self._viewer_alive():
                    return self.status()
                if not self._viewer_alive():
                    raise RuntimeError(f"实时窗口进程启动失败（退出码 {self._viewer.returncode}），请查看 worker.log")
                time.sleep(0.05)
            raise TimeoutError("实时窗口未能在 5 秒内打开；请检查 DISPLAY、X11 权限和 opencv-python GUI 支持")
        except Exception as exc:
            self.display_error = str(exc)
            self._stop_viewer()
            raise

    def record_start(self, fps: int = 10, camera: str = "agentview") -> dict[str, Any]:
        """从当前观测开始录像，后续每次 step 自动采样一帧。"""
        if self.env is None or self.obs is None:
            raise RuntimeError("请先 robot_libero_reset 初始化场景，然后再开始录像")
        if self._video_writer is not None:
            raise RuntimeError("已在录像，请先 robot_libero_record_stop")
        if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 60:
            raise ValueError("fps 必须是 1~60 的整数")
        self._validate_camera(camera)

        import imageio.v2 as imageio

        videos = self.output_dir / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        name = f"{self.suite_name}_task{self.task_id}_ep{self.episode_idx}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
        path = videos / name
        # imageio-ffmpeg 通常用于 MP4 编码；与用户原始 LIBERO 评测示例的 imageio 一致。
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=16)
        try:
            writer.append_data(self._frame(camera))  # 起始时刻的一帧
        except Exception:
            writer.close()
            path.unlink(missing_ok=True)
            raise
        self._video_writer = writer
        self._video_path = path
        self._video_frames = 1
        self._video_fps = fps
        self._video_camera = camera
        self.record_error = None
        return {"recording": True, "video_path": str(path), "frames": 1,
                "fps": fps, "camera": camera}

    def _finish_record(self, reason: str = "manual") -> dict[str, Any]:
        if self._video_writer is None:
            return {"saved": False, "reason": "not_recording", "last_video": self.last_video}
        writer, path = self._video_writer, self._video_path
        frames, fps, camera = self._video_frames, self._video_fps, self._video_camera
        self._video_writer = None
        self._video_path = None
        self._video_frames = 0
        try:
            writer.close()  # 重要：完成 MP4 封装，保证文件可播放
        except Exception as exc:
            self.record_error = f"视频封装失败：{exc}"
            raise
        self.last_video = {"saved": True, "video_path": str(path), "frames": frames,
                           "fps": fps, "camera": camera, "duration_s": round(frames / fps, 3),
                           "reason": reason}
        return dict(self.last_video)

    def record_stop(self) -> dict[str, Any]:
        """手动结束录像并写出完整 MP4；动作 Skill 可在达到 rollout 上限时调用。"""
        return self._finish_record("manual")

    @staticmethod
    def _suite(name: str):
        if name not in SUPPORTED_SUITES:
            raise ValueError(f"不支持的任务集：{name}；可用值：{sorted(SUPPORTED_SUITES)}")
        from libero.libero import benchmark
        suites = benchmark.get_benchmark_dict()
        if name not in suites:
            raise RuntimeError(f"当前 LIBERO 安装版本不提供 {name}")
        return suites[name]()

    def list_tasks(self, task_suite_name: str = "libero_spatial", start: int = 0, limit: int = 20):
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("start 必须为非负整数")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit 必须为 1~100")
        suite = self._suite(task_suite_name)
        count = int(suite.n_tasks)
        entries = [{"task_id": i, "description": str(suite.get_task(i).language)}
                   for i in range(start, min(count, start + limit))]
        return {"task_suite_name": task_suite_name, "total": count, "tasks": entries}

    def close(self) -> dict[str, Any]:
        # 场景重置/关闭时也要确保已录制的 MP4 可以正常播放。
        saved_video = self._finish_record("scene_closed")
        old = self.env
        self.env = None
        self.obs = None
        self.suite_name = self.task_id = self.episode_idx = self.description = None
        self.step_count = self.settle_steps = 0
        self.last_reward = 0.0
        self.done = self.success = False
        self.episode_uid = ""
        self.control_mode = "osc_pose"
        self._live_frame.unlink(missing_ok=True)
        if old is not None:
            old.close()
        return {"scene_closed": True, "service_running": True, "video": saved_video}

    def _check_success(self, info: Any = None) -> bool:
        if self.env is None:
            return False
        checker = getattr(self.env, "check_success", None)
        if callable(checker):
            return bool(checker())
        if isinstance(info, dict):
            return bool(info.get("success", False))
        return False

    def reset(self, task_suite_name: str = "libero_spatial", task_id: int = 0,
              episode_idx: int = 0, settle_steps: int = 10, seed: int = 7, resolution: int = 256,
              control_mode: str = "osc_pose"):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if control_mode not in {"osc_pose", "joint_position"}:
            raise ValueError("control_mode 必须是 osc_pose 或 joint_position")

        for value, name in ((task_id, "task_id"), (episode_idx, "episode_idx"),
                            (settle_steps, "settle_steps"), (seed, "seed"), (resolution, "resolution")):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} 必须是整数")
        if not 0 <= settle_steps <= 100:
            raise ValueError("settle_steps 必须在 0~100 范围内")
        if not 64 <= resolution <= 1024:
            raise ValueError("resolution 必须在 64~1024 范围内")

        suite = self._suite(task_suite_name)
        if not 0 <= task_id < suite.n_tasks:
            raise IndexError(f"task_id 超出范围：0~{suite.n_tasks - 1}")
        # PyTorch 2.6+ 默认 torch.load(weights_only=True)，但 LIBERO 的旧版
        # init_states 使用 numpy 对象，无法通过受限反序列化加载。
        # 仅对可信 LIBERO 初始状态文件临时关闭 weights_only；不影响 Agent
        # 及此 worker 的其他加载操作，也无需改 LIBERO 安装源码。
        # 注意：如果 LIBERO 内部显式写了 weights_only=True，此兼容方式不生效。
        previous_force = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "").strip().lower() in ("1", "y", "yes", "true"):
            raise RuntimeError(
                "检测到 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1，与 LIBERO 旧初始状态加载不兼容。"
                "请检查 LIBERO Python 进程的环境变量，并仅对可信数据调整设置。"
            )
        try:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            initial_states = suite.get_task_init_states(task_id)
        finally:
            if previous_force is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_force
        if not 0 <= episode_idx < len(initial_states):
            raise IndexError(f"episode_idx 超出范围：0~{len(initial_states) - 1}")
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl.is_file():
            raise FileNotFoundError(f"找不到 LIBERO BDDL 文件：{bddl}")

        self.close()
        env_kwargs = {"bddl_file_name": str(bddl),
                      "camera_heights": resolution, "camera_widths": resolution}
        # MaaS JOINT_POSITION constructor injection (scoped).
        # LIBERO's task wrappers supply controller_configs themselves. For
        # robosuite builds where the constructed env does NOT expose a writable
        # env.controller_configs, change the config at ManipulationEnv.__init__
        # (where controller_configs is actually consumed), BEFORE robot setup.
        # The worker is single-threaded, and the monkeypatch is always restored.
        if control_mode == "joint_position":
            import inspect
            from copy import deepcopy
            from robosuite.controllers import load_controller_config
            from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

            config_joint = load_controller_config(default_controller="JOINT_POSITION")
            original_init = ManipulationEnv.__init__
            signature = inspect.signature(original_init)
            if "controller_configs" not in signature.parameters:
                raise RuntimeError(
                    "当前 robosuite ManipulationEnv.__init__ 无 controller_configs 参数；"
                    "请检查已安装版本的构造签名"
                )
            injected_calls = 0

            def _joint_init(instance, *args, **kwargs):
                nonlocal injected_calls
                bound = signature.bind_partial(instance, *args, **kwargs)
                # Handles both keyword and positional controller_configs.
                bound.arguments["controller_configs"] = deepcopy(config_joint)
                injected_calls += 1
                return original_init(*bound.args, **bound.kwargs)

            ManipulationEnv.__init__ = _joint_init
            try:
                env = OffScreenRenderEnv(**env_kwargs)
            finally:
                ManipulationEnv.__init__ = original_init
            if injected_calls != 1:
                env.close()
                raise RuntimeError(
                    f"LIBERO 关节控制器注入次数为 {injected_calls}；"
                    "可能没有经过 robosuite ManipulationEnv.__init__，已拒绝执行"
                )
            logger.info("JointPosition controller config injected during environment construction")
        else:
            env = OffScreenRenderEnv(**env_kwargs)
        try:
            env.seed(seed)
            env.reset()
            if control_mode == "joint_position":
                sim = getattr(env, "env", None)
                robots = getattr(sim, "robots", [])
                if len(robots) != 1 or sim.action_dim != 8:
                    raise RuntimeError("当前 LIBERO 未提供单臂 Franka 8D JOINT 控制器")
                ctrl = getattr(robots[0], "controller", None)
                if ctrl is None or "JointPosition" not in type(ctrl).__name__:
                    raise RuntimeError("JOINT_POSITION 参数未生效，拒绝继续")
                if getattr(ctrl, "input_type", "delta") != "delta":
                    raise RuntimeError("需要 delta-input JOINT_POSITION 控制器以保持目标关节转换可预测")
            obs = env.set_init_state(initial_states[episode_idx])
            reward = 0.0
            done = False
            info: dict[str, Any] = {}
            # 参考原评测脚本，用无移动动作等待物体落稳；不属于策略或规划动作。
            settle_action = DUMMY_ACTION if control_mode == "osc_pose" else ([0.0] * 7 + [-1.0])
            for _ in range(settle_steps):
                obs, reward, done, info = env.step(settle_action)
        except Exception:
            env.close()
            raise

        self.env = env
        self.control_mode = control_mode
        self.obs = obs
        self.suite_name = task_suite_name
        self.task_id = task_id
        self.episode_idx = episode_idx
        self.description = str(task.language)
        self.step_count = 0  # 不包含 settle_steps
        self.settle_steps = settle_steps
        self.last_reward = float(reward)
        self.done = bool(done)
        self.success = self._check_success(info)
        self.episode_uid = uuid.uuid4().hex[:10]
        self._publish_frame()
        return self.status()

    def _current_state(self) -> dict[str, Any]:
        if self.obs is None:
            raise RuntimeError("场景未初始化：请先调用 robot_libero_reset")
        import numpy as np
        obs = self.obs
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float).reshape(-1)
        quat = np.asarray(obs["robot0_eef_quat"], dtype=float).reshape(-1)
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=float).reshape(-1)
        axisangle = _rotation_axis_angle(quat)
        joints = np.asarray(obs.get("robot0_joint_pos", []), dtype=float).reshape(-1)
        if joints.size not in (0, 7):
            raise RuntimeError(f"LIBERO robot0_joint_pos unexpected size: {joints.size}")
        # Measured finger travel in meters; confirm the scalar definition expected by MaaS.
        opening = float(np.abs(gripper).sum())
        return {
            "robot0_joint_pos": joints.tolist(),
            "robot0_gripper_opening": [opening],
            "robot0_eef_pos": eef_pos.tolist(),
            "robot0_eef_quat": quat.tolist(),
            "robot0_eef_axisangle": axisangle,
            "robot0_gripper_qpos": gripper.tolist(),
            # 与上传的 LIBERO 示例一致，拼接末端 xyz + axisangle + gripper qpos
            "observation_state": eef_pos.tolist() + axisangle + gripper.tolist(),
        }

    def query_state(self, report_after_step: bool | None = None) -> dict[str, Any]:
        """查询数值状态，可切换后续 step 的状态报告；不推进仿真或生成图像。"""
        if report_after_step is not None:
            if not isinstance(report_after_step, bool):
                raise ValueError("report_after_step 必须是布尔值或 null")
            self.report_after_step = report_after_step
        data = self.status()
        if self.env is not None and self.obs is not None:
            data.update(self._current_state())
        return data

    def observe(self, save_images: bool = False, include_images: bool = False) -> dict[str, Any]:
        """Agent 推荐 save_images=True；其他 Skill 可用 include_images=True 获取 base64 PNG。"""
        if self.env is None or self.obs is None:
            raise RuntimeError("场景未初始化：请先调用 robot_libero_reset")
        if not isinstance(save_images, bool) or not isinstance(include_images, bool):
            raise ValueError("save_images 和 include_images 必须是布尔值")
        data = {**self.status(), **self._current_state()}
        if not (save_images or include_images):
            return data

        from PIL import Image

        cameras = ("agentview", "eye_in_hand")
        if save_images:
            directory = self.output_dir / "observations"
            directory.mkdir(parents=True, exist_ok=True)
        for label in cameras:
            rgb = self._frame(label)
            im = Image.fromarray(rgb)
            data[f"{label}_shape"] = list(rgb.shape)
            if save_images:
                path = directory / f"{self.episode_uid}_step{self.step_count:06d}_{label}.png"
                im.save(path)
                data[f"{label}_image_path"] = str(path)
            if include_images:
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                data[f"{label}_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return data

    def _joint_target_to_action(self, action: list[float]) -> list[float]:
        """Convert ABSOLUTE rad joint targets to JOINT_POSITION delta-controller inputs.

        The MaaS absolute joint target assumption must be verified before enabling
        execution. Gripper has already been converted to robosuite [-1,+1]
        convention by the robot adapter.
        """
        import numpy as np

        sim = getattr(self.env, "env", None)
        robot = sim.robots[0] if sim is not None and getattr(sim, "robots", None) else None
        controller = getattr(robot, "controller", None)
        if controller is None or "JointPosition" not in type(controller).__name__:
            raise RuntimeError("场景不在 JOINT_POSITION 控制模式，拒绝执行 8D 动作")
        if getattr(controller, "input_type", "delta") != "delta":
            raise RuntimeError("未处理的 JOINT_POSITION 控制器输入类型")
        current = np.asarray(self.obs["robot0_joint_pos"], dtype=float).reshape(-1)
        desired = np.asarray(action[:7], dtype=float)
        if current.shape != (7,):
            raise RuntimeError("Franka joint state 必须是 7D")
        # Reject physically invalid targets when joint limits are supplied by robosuite.
        limits = getattr(robot, "joint_limits", None)
        if limits is not None:
            lower = np.asarray(limits[0], dtype=float).reshape(-1)
            upper = np.asarray(limits[1], dtype=float).reshape(-1)
            if lower.shape == (7,) and upper.shape == (7,) and (
                (desired < lower).any() or (desired > upper).any()
            ):
                raise ValueError("MaaS target exceeds Franka joint limits")

        # Invert the controller's configured linear scale_action() mapping.
        def as7(value):
            return np.broadcast_to(np.asarray(value, dtype=float), (7,))

        try:
            out_lo, out_hi = as7(controller.output_min), as7(controller.output_max)
            in_lo, in_hi = as7(controller.input_min), as7(controller.input_max)
        except (AttributeError, ValueError) as exc:
            raise RuntimeError("Cannot infer JOINT_POSITION controller scale from installed robosuite") from exc
        if ((out_hi - out_lo) <= 0).any() or ((in_hi - in_lo) <= 0).any():
            raise RuntimeError("Invalid JOINT_POSITION scaling range")
        delta = desired - current
        control = in_lo + (delta - out_lo) * (in_hi - in_lo) / (out_hi - out_lo)
        control = np.clip(control, in_lo, in_hi)
        return np.concatenate((control, [float(action[7])])).tolist()

    def step(self, action: list[float]) -> dict[str, Any]:
        """供控制适配器调用；不会注册给 DeepSeek Agent。"""
        if self.env is None:
            raise RuntimeError("场景未初始化：先 reset")
        if self.done:
            raise RuntimeError("当前 episode 已结束；请先 reset")
        dim = 8 if self.control_mode == "joint_position" else 7
        if not isinstance(action, list) or len(action) != dim:
            raise ValueError(f"当前 {self.control_mode} 控制器需要 {dim} 维动作")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in action):
            raise ValueError("动作必须为有限数值")
        command = (self._joint_target_to_action(action) if dim == 8
                   else [float(x) for x in action])
        obs, reward, done, info = self.env.step(command)
        self.obs = obs
        self.last_reward = float(reward)
        self.done = bool(done)
        self.success = self._check_success(info)
        self.step_count += 1
        self._publish_frame()
        if self._video_writer is not None:
            try:
                self._video_writer.append_data(self._frame(self._video_camera))
                self._video_frames += 1
            except Exception as exc:
                logger.exception("录制一帧失败")
                self.record_error = f"录制中断：{exc}"
                self._finish_record("write_error")
        # LIBERO 环境判定成功或 episode 终止时自动保存视频。
        if (self.done or self.success) and self._video_writer is not None:
            self._finish_record("task_finished")
        result = self.status()
        if self.report_after_step:
            result["state_report"] = {**result, **self._current_state()}
        return result

    def dispatch(self, command: str, params: dict[str, Any]):
        if command == "status":
            return self.status()
        if command == "query_state":
            return self.query_state(**params)
        if command == "list_tasks":
            return self.list_tasks(**params)
        if command == "reset":
            return self.reset(**params)
        if command == "observe":
            return self.observe(**params)
        if command == "display":
            return self.display(**params)
        if command == "record_start":
            return self.record_start(**params)
        if command == "record_stop":
            return self.record_stop()
        if command == "step":
            return self.step(**params)
        if command == "close":
            return self.close()
        if command == "shutdown":
            self.close()
            self._stop_viewer()
            return {"stopped": True, "was_running": True}
        raise ValueError(f"不支持的环境命令：{command}")


class _Server(socketserver.UnixStreamServer):
    allow_reuse_address = False

    def __init__(self, address: str, env: LiberoEnvironment):
        self.simulation = env
        super().__init__(address, _Handler)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        cmd = ""
        try:
            line = self.rfile.readline(1_000_001)
            if not line or len(line) > 1_000_000 or not line.endswith(b"\n"):
                raise ValueError("JSON 请求为空或过大")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("JSON 请求必须是对象")
            cmd = request.get("command", "")
            params = request.get("params", {})
            if not isinstance(cmd, str) or not isinstance(params, dict):
                raise ValueError("命令或参数类型错误")
            result = self.server.simulation.dispatch(cmd, params)
            output = {"ok": True, "result": result}
        except Exception as exc:
            logger.exception("LIBERO 命令 %r 执行失败", cmd)
            output = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            self.wfile.write((json.dumps(output, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("调用方断开连接：%s", cmd)
        if cmd == "shutdown" and output.get("ok"):
            # shutdown 必须从另一个线程调用，否则会与 serve_forever 互相等待。
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    sock_path = output_dir / SOCKET_NAME
    # 不覆盖其他仍在运行的 LIBERO worker。
    if sock_path.exists():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            try:
                probe.settimeout(0.5)
                probe.connect(str(sock_path))
            except OSError:
                sock_path.unlink()  # 上次崩溃残留的 Unix socket
            else:
                raise RuntimeError(f"LIBERO worker 已在使用该 socket：{sock_path}")

    simulation = LiberoEnvironment(output_dir)
    server = _Server(str(sock_path), simulation)
    os.chmod(sock_path, 0o600)
    logger.info("LIBERO environment worker ready: %s", sock_path)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        try:
            simulation.close()
        finally:
            simulation._stop_viewer()
            sock_path.unlink(missing_ok=True)
        logger.info("LIBERO environment worker closed")


if __name__ == "__main__":
    main()
