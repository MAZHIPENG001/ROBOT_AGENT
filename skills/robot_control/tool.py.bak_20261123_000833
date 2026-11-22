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
            """启动控制循环并立即返回 run_id。需先初始化对应仿真环境或真机；省略参数时读取 config 默认值。仅接受 robot_control_list 中的机器人/策略。"""
            try:
                defaults = self.registry.defaults
                robot = defaults.robot if robot is None else robot
                policy = defaults.policy if policy is None else policy
                robot_factory, policy_factory = self.registry.factories(robot, policy)
                return self._reply({"ok": True, **self.controller.start(
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
            """使用当前机器人观测推理一次并返回动作预览，不执行动作；省略名称时使用 config 默认组合。"""
            predictor = None
            adapter = None
            try:
                if self.controller.running():
                    raise RuntimeError("控制循环运行中，不能同时发起独立推理")
                robot = self.registry.defaults.robot if robot is None else robot
                policy = self.registry.defaults.policy if policy is None else policy
                robot_factory, policy_factory = self.registry.factories(robot, policy)
                if not isinstance(show_actions, int) or isinstance(show_actions, bool) or not 0 <= show_actions <= 10:
                    raise ValueError("show_actions 必须在 0~10 之间")
                adapter = robot_factory()
                observation = adapter.observe()
                if observation.done or observation.success:
                    raise RuntimeError("环境任务已结束，请先 reset")
                predictor = policy_factory()
                chunk = predictor.predict(observation)
                actions = validate_action_chunk(chunk, adapter.action_dim, adapter.action_space)
                return self._reply({
                    "ok": True, "executed": False, "policy": policy,
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
