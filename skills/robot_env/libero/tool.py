"""给 LangChain 注册环境生命周期工具；不暴露任何动作执行工具。"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from .bridge import LiberoBridge


class LiberoSkill:
    """LIBERO 环境管理 Skill。动作由其他 Skill 负责。"""

    def __init__(self) -> None:
        self.bridge = LiberoBridge()  # 仅创建桥接对象，不会马上打开仿真

    def _call(self, command: str, **params: Any) -> str:
        try:
            data = self.bridge.request(command, **params)
            return json.dumps({"ok": True, **data}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)

    def tools(self) -> list:
        @tool
        def robot_libero_start(show_window: bool = False, display_camera: str = "agentview") -> str:
            """启动 LIBERO 仿真服务，可选 show_window=True 在本机实时显示画面；不加载任务、不自动录像。display_camera 可用 agentview、eye_in_hand、side_by_side。"""
            try:
                data = self.bridge.start(show_window=show_window, display_camera=display_camera)
                return json.dumps({"ok": True, **data}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        @tool
        def robot_libero_display(show_window: bool = True, camera: str = "agentview") -> str:
            """开启或关闭本机 LIBERO 实时窗口，允许启动服务之后切换；只显示场景最新帧，不会执行动作或录像。"""
            return self._call("display", show_window=show_window, camera=camera)

        @tool
        def robot_libero_status() -> str:
            """查看 LIBERO 环境服务是否启动、当前任务、步数及任务成功状态。"""
            return self._call("status")

        @tool
        def robot_libero_list_tasks(task_suite_name: str = "libero_spatial", start: int = 0, limit: int = 20) -> str:
            """列出 LIBERO 任务集的任务 ID 和语言描述；支持 spatial/object/goal/10/90。"""
            return self._call("list_tasks", task_suite_name=task_suite_name, start=start, limit=limit)

        @tool
        def robot_libero_reset(
            task_suite_name: str = "libero_spatial",
            task_id: int = 0,
            episode_idx: int = 0,
            settle_steps: int = 10,
            control_mode: str = "osc_pose",
        ) -> str:
            """创建或重置 LIBERO 仿真场景，选择任务与初始状态；settle_steps 为物体稳定等待步数。不会调用 VLA。"""
            return self._call("reset", task_suite_name=task_suite_name, task_id=task_id,
                              episode_idx=episode_idx, settle_steps=settle_steps,
                              control_mode=control_mode)

        @tool
        def robot_libero_observe(save_images: bool = True) -> str:
            """读取当前 LIBERO 机器人状态、任务描述和成功判定，可将双摄像头观测保存为 PNG；不推理、不执行动作。"""
            return self._call("observe", save_images=save_images)

        @tool
        def robot_libero_record_start(fps: int = 10, camera: str = "agentview") -> str:
            """在 reset 后开始录制 LIBERO MP4。起始观测记录一帧，之后每次环境 step 记录一帧；fps 是回放帧率。camera 支持 agentview、eye_in_hand、side_by_side。"""
            return self._call("record_start", fps=fps, camera=camera)

        @tool
        def robot_libero_record_stop() -> str:
            """停止 LIBERO 录像并封装 MP4，返回真实文件路径。任务成功/结束或场景关闭时也会自动停止保存。"""
            return self._call("record_stop")

        @tool
        def robot_libero_close() -> str:
            """关闭当前 LIBERO 场景、释放 MuJoCo 资源，但保留环境服务以便下一次 reset。"""
            return self._call("close")

        @tool
        def robot_libero_stop() -> str:
            """完全停止 LIBERO 环境服务及其独立 Python 子进程。"""
            try:
                return json.dumps({"ok": True, **self.bridge.stop()}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

        return [robot_libero_start, robot_libero_display, robot_libero_status,
                robot_libero_list_tasks, robot_libero_reset, robot_libero_observe,
                robot_libero_record_start, robot_libero_record_stop,
                robot_libero_close, robot_libero_stop]
