"""面向 Agent 的高级控制工具：策略网络与解释器配置只从 .env 读取。"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
from langchain_core.tools import tool

from .adapters import LiberoAdapter
from .controller import RobotController
from .policies import OpenPiPolicy


class RobotControlSkill:
    def __init__(self) -> None:
        self.controller = RobotController()
        self.robot_factories: dict[str, Callable[[], Any]] = {
            "libero": LiberoAdapter,
        }
        # 未来可加入 "lingbot": LingBotPolicy、"rl": RlPolicy 等。
        self.policy_factories: dict[str, Callable[[], Any]] = {
            "openpi": OpenPiPolicy,
        }

    @staticmethod
    def _reply(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def tools(self) -> list:
        @tool
        def robot_control_start(
            robot: str = "libero", policy: str = "openpi",
            max_steps: int = 300, replan_steps: int = 5,
        ) -> str:
            """启动机器人控制循环并立即返回 run_id；先通过 robot_libero_reset 加载环境。模型地址、SSH 及 Python 环境由 .env 配置。此工具仅接受已注册 robot/policy。"""
            try:
                if robot not in self.robot_factories:
                    raise ValueError(f"未知 robot={robot}；可用：{list(self.robot_factories)}")
                if policy not in self.policy_factories:
                    raise ValueError(f"未知 policy={policy}；可用：{list(self.policy_factories)}")
                return self._reply({"ok": True, **self.controller.start(
                    self.robot_factories[robot], self.policy_factories[policy],
                    robot_name=robot, policy_name=policy,
                    max_steps=max_steps, replan_steps=replan_steps,
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
            robot: str = "libero", policy: str = "openpi", show_actions: int = 3,
        ) -> str:
            """使用当前环境观测调用一次指定策略，仅推理、绝不执行动作；地址和 SSH 参数均从 .env 读取。"""
            predictor = None
            adapter = None
            try:
                if self.controller.running():
                    raise RuntimeError("控制循环运行中，不能同时发起独立推理")
                if robot not in self.robot_factories or policy not in self.policy_factories:
                    raise ValueError(f"未知 robot/policy，可用：{list(self.robot_factories)} / {list(self.policy_factories)}")
                if not isinstance(show_actions, int) or isinstance(show_actions, bool) or not 0 <= show_actions <= 10:
                    raise ValueError("show_actions 必须在 0~10 之间")
                adapter = self.robot_factories[robot]()
                observation = adapter.observe()
                if observation.done or observation.success:
                    raise RuntimeError("环境任务已结束，请先 reset")
                predictor = self.policy_factories[policy]()
                chunk = predictor.predict(observation)
                actions = np.asarray(chunk.actions)
                if actions.ndim != 2 or actions.shape[1] != adapter.action_dim:
                    raise ValueError("策略动作维度与环境不匹配")
                if not len(actions) or not np.isfinite(actions).all():
                    raise ValueError("策略返回非法动作")
                if chunk.action_space != adapter.action_space:
                    raise ValueError("策略动作空间与机器人动作空间不匹配")
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

        return [robot_control_start, robot_control_status,
                robot_control_stop, robot_control_predict_once]
