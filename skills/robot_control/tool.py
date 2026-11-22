"""面向 Agent 的控制工具：只接受本地配置中已注册的机器人/策略名称。"""
from __future__ import annotations

import json
from pathlib import Path
from langchain_core.tools import tool

from .controller import RobotController
from .registry import ControlRegistry
from .validation import validate_action_chunk


class RobotControlSkill:
    def __init__(self, config_dir: str | Path | None = None) -> None:
        self.controller = RobotController()
        self.registry = ControlRegistry(config_dir)

    @staticmethod
    def _reply(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _resolve_pair(self, robot: str | None, policy: str | None) -> tuple[str, str]:
        """When robot was omitted, select the sole profile compatible with policy."""
        defaults = self.registry.defaults
        selected_policy = defaults.policy if policy is None else policy
        selected_robot = defaults.robot if robot is None else robot
        if robot is not None:
            # An explicit incompatible pair is a user error, not permission to switch robots.
            self.registry.check_pair(selected_robot, selected_policy)
            return selected_robot, selected_policy
        try:
            self.registry.check_pair(selected_robot, selected_policy)
            return selected_robot, selected_policy
        except ValueError:
            compatible = []
            for candidate in self.registry.robots:
                try:
                    self.registry.check_pair(candidate, selected_policy)
                except ValueError:
                    continue
                compatible.append(candidate)
            if len(compatible) != 1:
                # Unknown policy or multiple candidates: preserve original error.
                self.registry.check_pair(selected_robot, selected_policy)
                raise RuntimeError("自动配对未找到唯一机器人")
            return compatible[0], selected_policy

    def _auto_prepare_robot(self, robot: str) -> dict:
        """Switch already-loaded LIBERO scene to joint_position once before a run.

        Reset destroys current scene progress. Refuse to reset during recording.
        Never reset from the high-frequency controller / adapter observe() path.
        """
        if robot != "libero_joint":
            return {"auto_reset": False}
        if self.controller.running():
            raise RuntimeError("控制任务运行中，禁止自动重置 LIBERO 场景")
        from skills.robot_env.libero.bridge import LiberoEnvClient

        client = LiberoEnvClient()
        status = client.request("status")
        if not status.get("scene_loaded"):
            raise RuntimeError(
                "LIBERO 当前没有加载任务，无法知道应该自动重置哪一个任务；"
                "请先调用 robot_libero_reset 指定 task_suite_name/task_id，之后可自动切换控制模式"
            )
        if status.get("control_mode") == "joint_position":
            return {"auto_reset": False, "control_mode": "joint_position"}
        if status.get("recording"):
            raise RuntimeError("LIBERO 正在录像，自动 reset 会中断录像；请先结束录像再切换控制模式")
        suite = status.get("task_suite_name")
        task_id = status.get("task_id")
        episode_idx = status.get("episode_idx")
        if not isinstance(suite, str) or type(task_id) is not int or type(episode_idx) is not int:
            raise RuntimeError("LIBERO 状态缺少 task_suite_name/task_id/episode_idx，无法安全自动切换")
        old_mode = status.get("control_mode")
        reset_args = {
            "task_suite_name": suite,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "control_mode": "joint_position",
        }
        settle_steps = status.get("settle_steps")
        if type(settle_steps) is int and 0 <= settle_steps <= 100:
            reset_args["settle_steps"] = settle_steps
        result = client.request("reset", **reset_args)
        if not result.get("scene_loaded") or result.get("control_mode") != "joint_position":
            raise RuntimeError("自动 reset 后未确认 joint_position，禁止继续 MaaS 控制")
        return {
            "auto_reset": True,
            "from_mode": old_mode,
            "control_mode": "joint_position",
            "task_suite_name": suite,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "note": "已重新初始化同一任务，之前的场景进度已清空",
        }

    def tools(self) -> list:
        @tool
        def robot_control_list() -> str:
            """列出本地配置的机器人、策略、输入输出协议与控制默认值，不返回连接凭据。"""
            return self._reply({"ok": True, **self.registry.describe()})

        @tool
        def robot_control_start(
            robot: str | None = None, policy: str | None = None,
            max_steps: int | None = None, replan_steps: int | None = None,
        ) -> str:
            """启动控制循环并返回 run_id。指定 maas+libero_joint 时会在启动前检查控制模式；若已有 LIBERO 任务为 osc_pose，将自动重置同一任务为 joint_position（清空当前场景进度）。没加载任务、正在录像或正在运行控制任务时不自动重置。"""
            try:
                defaults = self.registry.defaults
                robot, policy = self._resolve_pair(robot, policy)
                if self.controller.running():
                    raise RuntimeError("已有控制任务正在运行，禁止自动重置场景")
                robot_factory, policy_factory = self.registry.factories(robot, policy)
                preparation = self._auto_prepare_robot(robot)
                return self._reply({"ok": True, "environment_preparation": preparation, **self.controller.start(
                    robot_factory, policy_factory,
                    robot_name=robot, policy_name=policy,
                    max_steps=defaults.max_steps if max_steps is None else max_steps,
                    replan_steps=defaults.replan_steps if replan_steps is None else replan_steps,
                )})
            except Exception as exc:
                return self._reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        @tool
        def robot_control_status() -> str:
            """查询机器人控制运行状态、执行步数、成功标志和错误信息。"""
            return self._reply({"ok": True, **self.controller.status()})

        @tool
        def robot_control_stop() -> str:
            """请求停止当前机器人控制循环；仅协作式停止，不是硬件急停。"""
            return self._reply({"ok": True, **self.controller.stop()})

        @tool
        def robot_control_predict_once(
            robot: str | None = None, policy: str | None = None, show_actions: int = 3,
        ) -> str:
            """推理一次且不执行动作；maas+libero_joint 若已有场景但模式不对，将自动重置同一任务为 joint_position（清空场景进度），再采集观测。"""
            predictor = None
            adapter = None
            try:
                if self.controller.running():
                    raise RuntimeError("控制循环运行中，不能同时发起独立推理")
                robot, policy = self._resolve_pair(robot, policy)
                robot_factory, policy_factory = self.registry.factories(robot, policy)
                if not isinstance(show_actions, int) or isinstance(show_actions, bool) or not 0 <= show_actions <= 10:
                    raise ValueError("show_actions 必须在 0~10 之间")
                preparation = self._auto_prepare_robot(robot)
                adapter = robot_factory()
                observation = adapter.observe()
                if observation.done or observation.success:
                    raise RuntimeError("环境任务已结束，请先 reset")
                predictor = policy_factory()
                chunk = predictor.predict(observation)
                actions = validate_action_chunk(chunk, adapter.action_dim, adapter.action_space)
                return self._reply({
                    "ok": True, "executed": False, "policy": policy,
                    "environment_preparation": preparation,
                    "action_shape": list(actions.shape),
                    "action_space": chunk.action_space,
                    "actions_preview": actions[:show_actions].tolist(),
                    "instruction": observation.instruction,
                })
            except Exception as exc:
                return self._reply({"ok": False, "executed": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
            finally:
                for obj in (predictor, adapter):
                    if obj is not None:
                        try:
                            obj.close()
                        except Exception:
                            pass

        return [robot_control_list, robot_control_start, robot_control_status,
                robot_control_stop, robot_control_predict_once]
