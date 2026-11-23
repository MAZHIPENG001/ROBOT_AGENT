"""Batch prediction and execution checks, without API calls or simulator access."""
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from skills.robot_control.config import PolicyConfig, read_config
from skills.robot_control.policies.deepseek_vlm import DeepSeekVlmPolicy
from skills.robot_control.tool import RobotControlSkill
from skills.robot_control.types import Observation, StepFeedback


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def disable_notifications(monkeypatch):
    monkeypatch.setenv("ROBOT_VLM_FEISHU_NOTIFY", "false")


@pytest.fixture
def config():
    profile = read_config(
        CONFIG_DIR / "policy_cfg/deepseek_vlm/deepseek_vlm.yaml", PolicyConfig
    )
    return profile.model_copy(update={"options": {
        **profile.options, "api_key": "test-key", "image_size": 64,
        "max_tokens": "auto",
    }})


@pytest.fixture
def rows():
    return [[i / 1000, 0, 0, 0, 0, 0, -1 if i < 25 else 1] for i in range(50)]


class FakeRobot:
    action_dim = 7
    action_space = DeepSeekVlmPolicy._SPACE

    def __init__(self, finish_at=None):
        self.actions = []
        self.finish_at = finish_at
        self.closed = False

    def observe(self):
        return Observation(
            images={name: np.zeros((8, 8, 3), dtype=np.uint8)
                    for name in ("primary", "wrist")},
            state=np.zeros(8, dtype=np.float32), instruction="move cautiously",
            episode_id="episode-1", step_index=len(self.actions),
        )

    def step(self, action):
        self.actions.append(action.copy())
        finished = len(self.actions) == self.finish_at
        return StepFeedback(done=finished, success=finished, episode_id="episode-1")

    def close(self):
        self.closed = True


@pytest.fixture
def policy_factory(config):
    clients = []

    def make(rows, *, finish_reason="stop", options=None):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{
                "message": {"content": json.dumps({"action": rows})},
                "finish_reason": finish_reason,
            }]})

        client = httpx.Client(
            base_url="https://example.test", transport=httpx.MockTransport(respond)
        )
        clients.append(client)
        profile = config.model_copy(update={
            "options": {**config.options, **(options or {})}
        })
        return DeepSeekVlmPolicy(config=profile, client=client), requests

    yield make
    for client in clients:
        client.close()


def test_one_request_returns_50_distinct_actions(policy_factory, rows):
    policy, requests = policy_factory(rows)
    chunk = policy.predict(FakeRobot().observe())
    assert len(requests) == 1
    assert chunk.actions.shape == (50, 7)
    assert chunk.actions.dtype == np.float32
    np.testing.assert_allclose(chunk.actions, rows)
    prompt = requests[0]["messages"][0]["content"]
    assert "exactly 50" in prompt
    assert "{action_horizon}" not in prompt
    situation = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert situation["output_action_count"] == 50
    assert situation["simulator_steps_per_action"] == 1


@pytest.mark.parametrize("count", [0, 1, 49, 51])
def test_wrong_action_count_is_rejected(policy_factory, rows, count):
    policy, _ = policy_factory((rows + rows)[:count])
    with pytest.raises(ValueError, match="恰好 50"):
        policy.predict(FakeRobot().observe())
    assert policy._last_actions is None


@pytest.mark.parametrize("bad_row", [
    [0] * 6,
    [0.21, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0.13, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1.00000001],
    [True, 0, 0, 0, 0, 0, -1],
    ["0", 0, 0, 0, 0, 0, -1],
    [float("nan"), 0, 0, 0, 0, 0, -1],
    [float("inf"), 0, 0, 0, 0, 0, -1],
])
def test_invalid_last_action_rejects_entire_chunk(policy_factory, rows, bad_row):
    rows[-1] = bad_row
    policy, _ = policy_factory(rows)
    with pytest.raises(ValueError, match="第 50 组动作无效"):
        policy.predict(FakeRobot().observe())
    assert policy._last_actions is None


def test_truncation_is_rejected_even_with_parseable_json(policy_factory, rows):
    policy, _ = policy_factory(rows, finish_reason="length")
    with pytest.raises(ValueError, match="响应被截断"):
        policy.predict(FakeRobot().observe())


def test_repeat_steps_preserves_row_order(policy_factory, rows):
    policy, _ = policy_factory(rows, options={"repeat_steps": 2})
    chunk = policy.predict(FakeRobot().observe())
    np.testing.assert_allclose(chunk.actions, np.repeat(rows, 2, axis=0))


def test_previous_action_tracks_executed_prefix_and_episode(policy_factory, rows):
    policy, requests = policy_factory(rows)
    observation = FakeRobot().observe()
    policy.predict(observation)
    observation.step_index = 5
    policy.predict(observation)
    situation = json.loads(requests[-1]["messages"][1]["content"][0]["text"])
    np.testing.assert_allclose(situation["previous_action"], rows[4])
    observation.episode_id = "episode-2"
    observation.step_index = 0
    policy.predict(observation)
    situation = json.loads(requests[-1]["messages"][1]["content"][0]["text"])
    assert situation["previous_action"] == [0, 0, 0, 0, 0, 0, -1]


@pytest.mark.parametrize("horizon", [0, 101, 1.5, True, "invalid"])
def test_invalid_horizon_is_rejected(policy_factory, rows, horizon):
    with pytest.raises(ValueError, match="action_horizon"):
        policy_factory(rows, options={"action_horizon": horizon})


@pytest.mark.parametrize("max_steps, finish_at, expected_steps, expected_calls", [
    (100, None, 100, 2),
    (55, None, 55, 2),
    (100, 13, 13, 1),
])
def test_tool_executes_batches_and_respects_limits(
    monkeypatch, policy_factory, rows, max_steps, finish_at, expected_steps, expected_calls
):
    skill = RobotControlSkill(CONFIG_DIR)
    robot = FakeRobot(finish_at=finish_at)
    policy, requests = policy_factory(rows)
    monkeypatch.setattr(skill.registry, "create_robot", lambda name: robot)
    monkeypatch.setattr(skill.registry, "create_policy", lambda name: policy)
    start = next(tool for tool in skill.tools() if tool.name == "robot_control_start")
    result = json.loads(start.invoke({"policy": "deepseek_vlm", "max_steps": max_steps}))
    assert result["ok"]
    assert result["replan_steps"] == 50
    skill.controller._thread.join(timeout=5)
    assert not skill.controller.running()
    status = skill.controller.status()
    assert status["state"] == "finished", status
    assert status["steps"] == expected_steps
    assert len(requests) == expected_calls
    np.testing.assert_allclose(robot.actions, (rows * expected_calls)[:expected_steps])
    assert robot.closed


@pytest.mark.parametrize("policy, explicit_steps, expected", [
    ("deepseek_vlm", None, 50),
    ("deepseek_vlm", 5, 5),
    ("openpi", None, 25),
])
def test_replan_precedence(monkeypatch, policy, explicit_steps, expected):
    skill = RobotControlSkill(CONFIG_DIR)
    monkeypatch.setattr(skill.controller, "start", lambda *args, **kwargs: kwargs)
    start = next(tool for tool in skill.tools() if tool.name == "robot_control_start")
    args = {"policy": policy}
    if explicit_steps is not None:
        args["replan_steps"] = explicit_steps
    result = json.loads(start.invoke(args))
    assert result["ok"]
    assert result["replan_steps"] == expected
    described = {p["name"]: p for p in skill.registry.describe()["policies"]}
    assert described["deepseek_vlm"]["replan_steps"] == 50
    assert described["openpi"]["replan_steps"] == 25
