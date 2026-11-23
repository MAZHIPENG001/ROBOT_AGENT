"""离线验证状态查询及 worker → 适配器 → 控制循环的报告链路。"""
import json
from pathlib import Path

import numpy as np
import pytest

from skills.robot_control.config import RobotConfig, read_config
from skills.robot_control.controller import RobotController
from skills.robot_control.types import ActionChunk
from skills.robot_env.libero.adapter import LiberoAdapter
from skills.robot_env.libero.joint_adapter import LiberoJointAdapter
from skills.robot_env.libero.tool import LiberoSkill
from skills.robot_env.libero.worker import LiberoEnvironment


class Simulation:
    def __init__(self):
        self.steps = 0

    def observation(self):
        return {
            "robot0_eef_pos": [float(self.steps), 0.0, 0.3],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_gripper_qpos": [0.02, -0.02],
            "robot0_joint_pos": [self.steps * 0.01] * 7,
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
        }

    def step(self, action):
        self.steps += 1
        return self.observation(), float(self.steps), self.steps >= 3, {"success": self.steps >= 3}

    def close(self):
        pass


class LocalClient:
    def __init__(self, worker):
        self.worker = worker

    def request(self, command, **params):
        return self.worker.dispatch(command, params)


@pytest.fixture
def worker(tmp_path):
    env = LiberoEnvironment(tmp_path)
    env.env = Simulation()
    env.obs = env.env.observation()
    env.episode_uid = "test-episode"
    env.description = "test task"
    return env


def test_query_is_read_only_and_toggle_survives_close(worker):
    original_obs = worker.obs
    report = worker.dispatch("query_state", {})
    assert report["report_after_step"] is False
    assert report["robot0_eef_pos"] == [0.0, 0.0, 0.3]
    assert len(report["robot0_joint_pos"]) == 7
    assert report["robot0_gripper_opening"] == [0.04]
    assert len(report["observation_state"]) == 8
    assert worker.obs is original_obs
    assert worker.env.steps == worker.step_count == 0
    assert not list(worker.output_dir.iterdir())

    worker.query_state(report_after_step=True)
    assert worker.query_state()["report_after_step"] is True
    worker.close()
    unloaded = worker.query_state()
    assert unloaded["scene_loaded"] is False
    assert unloaded["report_after_step"] is True
    assert "robot0_eef_pos" not in unloaded
    assert worker.query_state(report_after_step=False)["report_after_step"] is False


@pytest.mark.parametrize("invalid", ["false", 0, 1, [], {}])
def test_invalid_toggle_does_not_change_setting(worker, invalid):
    worker.query_state(report_after_step=True)
    with pytest.raises(ValueError, match="report_after_step"):
        worker.query_state(report_after_step=invalid)
    assert worker.report_after_step is True
    assert worker.env.steps == 0


def test_reports_use_post_action_state_and_can_be_disabled(worker, monkeypatch):
    action = [0.0] * 6 + [-1.0]
    # 默认关闭时无需计算数值报告。
    with monkeypatch.context() as patch:
        patch.setattr(worker, "_current_state", lambda: pytest.fail("unexpected state report"))
        assert "state_report" not in worker.step(action)
    worker.query_state(report_after_step=True)
    result = worker.step(action)
    report = result["state_report"]
    assert report["step_count"] == 2
    assert report["robot0_eef_pos"][0] == 2.0
    assert report["last_reward"] == 2.0
    json.dumps(result)  # socket 响应必须可序列化。
    worker.query_state(report_after_step=False)
    assert "state_report" not in worker.step(action)
    assert report["step_count"] == 2  # 旧报告是该步的快照。


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("disable_after_first", [False, True])
def test_report_reaches_controller(worker, monkeypatch, joint, disable_after_first):
    client = LocalClient(worker)
    profile = "libero_joint" if joint else "libero"
    path = Path(__file__).resolve().parents[1] / "config/robot_env_cfg/libero" / f"{profile}.yaml"
    config = read_config(path, RobotConfig)
    if joint:
        worker.control_mode = "joint_position"
        monkeypatch.setattr(worker, "_joint_target_to_action", lambda action: action)
        monkeypatch.setattr("skills.robot_env.libero.joint_adapter.LiberoEnvClient", lambda **kw: client)
        config = config.model_copy(update={"options": {**config.options, "execution_enabled": True}})
        adapter = LiberoJointAdapter(config=config)
    else:
        adapter = LiberoAdapter(client=client, config=config)

    skill = LiberoSkill()
    monkeypatch.setattr(skill, "bridge", client)
    query = next(tool for tool in skill.tools() if tool.name == "robot_libero_query_state")
    assert json.loads(query.invoke({"report_after_step": True}))["report_after_step"] is True
    assert json.loads(query.invoke({}))["step_count"] == 0

    controller = RobotController()

    class Policy:
        def predict(self, observation):
            if observation.step_index == 1:
                assert controller.status()["last_state_report"]["step_count"] == 1
                if disable_after_first:
                    query.invoke({"report_after_step": False})
            action = np.zeros((1, adapter.action_dim))
            action[0, -1] = -1.0
            return ActionChunk(actions=action, action_space=adapter.action_space)

        def close(self):
            pass

    # 同步运行以确定性地验证每步反馈以及终止时报告。
    controller._run(lambda: adapter, Policy, 5, 1)
    status = controller.status()
    assert status.get("error") is None
    assert status["state"] == "finished"
    assert status["steps"] == 3
    assert status["success"] is True
    if disable_after_first:
        assert status["last_state_report"] is None
    else:
        report = status["last_state_report"]
        assert report["step_count"] == 3
        assert report["robot0_eef_pos"][0] == 3.0
        assert report["done"] is report["success"] is True
        assert report["episode_uid"] == "test-episode"
        assert report["control_mode"] == worker.control_mode
