"""通用动作执行引擎：不导入 LIBERO、OpenPI、LangChain。"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable

import numpy as np

from .types import Policy, RobotAdapter


class RobotController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict = {"state": "idle", "steps": 0}

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _update(self, **updates) -> None:
        with self._lock:
            self._status.update(updates)

    def start(
        self, robot_factory: Callable[[], RobotAdapter],
        policy_factory: Callable[[], Policy],
        *, robot_name: str, policy_name: str, max_steps: int, replan_steps: int,
    ) -> dict:
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 10000:
            raise ValueError("max_steps 必须在 1~10000 之间")
        if not isinstance(replan_steps, int) or not 1 <= replan_steps <= 100:
            raise ValueError("replan_steps 必须在 1~100 之间")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("已有控制任务运行中，请先 robot_control_stop")
            self._stop.clear()
            run_id = uuid.uuid4().hex[:10]
            self._status = {
                "run_id": run_id, "robot": robot_name, "policy": policy_name,
                "state": "starting", "steps": 0, "max_steps": max_steps,
                "replan_steps": replan_steps, "success": False, "error": None,
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(robot_factory, policy_factory, max_steps, replan_steps),
                name=f"robot-controller-{run_id}", daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def stop(self) -> dict:
        self._stop.set()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._status["state"] = "stopping"
            return dict(self._status)

    def _run(self, robot_factory, policy_factory, max_steps, replan_steps):
        robot = None
        policy = None
        steps = 0
        final_state = "finished"
        try:
            robot = robot_factory()
            # 先确认仿真已加载；避免后续网络连接成功后才发现没有场景。
            first_observation = robot.observe()
            if first_observation.done or first_observation.success:
                raise RuntimeError("任务已结束，请先重置环境")
            policy = policy_factory()
            self._update(state="running", episode_id=first_observation.episode_id)
            episode_id = first_observation.episode_id
            while steps < max_steps:
                if self._stop.is_set():
                    final_state = "stopped"
                    break
                observation = robot.observe()
                if observation.episode_id != episode_id:
                    raise RuntimeError("检测到场景已被重置，停止旧控制任务")
                if observation.done or observation.success:
                    final_state = "finished"
                    self._update(success=observation.success)
                    break
                prediction = policy.predict(observation)
                actions = np.asarray(prediction.actions, dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != robot.action_dim:
                    raise ValueError(
                        f"策略动作形状 {actions.shape} 与机器人动作维度 {robot.action_dim} 不兼容"
                    )
                if prediction.action_space != robot.action_space:
                    raise ValueError(
                        f"策略动作空间 {prediction.action_space!r} 与机器人动作空间 {robot.action_space!r} 不匹配"
                    )
                if not len(actions) or not np.isfinite(actions).all():
                    raise ValueError("策略返回空动作或非有限数值")
                n = min(len(actions), replan_steps, max_steps - steps)
                for action in actions[:n]:
                    if self._stop.is_set():
                        final_state = "stopped"
                        break
                    feedback = robot.step(action)
                    steps += 1
                    self._update(steps=steps, success=feedback.success,
                                 last_reward=feedback.metadata.get("reward"))
                    if feedback.episode_id and episode_id and feedback.episode_id != episode_id:
                        raise RuntimeError("环境 episode 在执行期间发生变化，已停止")
                    if feedback.done or feedback.success:
                        final_state = "finished"
                        break
                else:
                    continue
                break
            if self._stop.is_set() and final_state != "finished":
                final_state = "stopped"
            self._update(state=final_state)
        except Exception as exc:
            self._update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            for resource in (policy, robot):
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception as exc:
                    self._update(close_error=f"{type(exc).__name__}: {exc}")
            self._update(completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
