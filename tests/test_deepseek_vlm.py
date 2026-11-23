"""Offline tests: httpx.MockTransport means no network or simulator needed."""
import json

import httpx
import numpy as np
import pytest

from skills.robot_control.config import PolicyConfig
from skills.robot_control.policies.deepseek_vlm import DeepSeekVlmPolicy
from skills.robot_control.types import Observation


def config():
    return PolicyConfig.model_validate({
        "name": "deepseek_vlm",
        "driver": "skills.robot_control.policies.deepseek_vlm:DeepSeekVlmPolicy",
        "inputs": {"images": {"primary": "agentview", "wrist": "eye_in_hand"},
                   "state_key": "state", "instruction_key": "instruction", "state_dim": 8},
        "outputs": {"actions_key": "action", "action_dim": 7,
                    "action_space": "libero_delta_pose_axisangle_gripper_7d"},
        "options": {"api_key": "test-placeholder", "repeat_steps": 2},
    })


def obs():
    rgb = np.zeros((20, 24, 3), dtype=np.uint8)
    return Observation(images={"primary": rgb, "wrist": rgb},
                       state=np.array([0.4, 0.0, 0.3, 0, 0, 0, 0.02, 0.02], dtype=np.float32),
                       instruction="pick up red cube", episode_id="test", step_index=0)


def mocked_policy(reply):
    def handler(req):
        assert req.url.path.endswith("/chat/completions")
        body = json.loads(req.content)
        assert body["model"] == "deepseek-flash"
        pieces = body["messages"][1]["content"]
        assert sum(piece["type"] == "image_url" for piece in pieces) == 2
        return httpx.Response(200, json={"choices": [
            {"message": {"content": json.dumps(reply)}}
        ]})
    client = httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handler))
    return DeepSeekVlmPolicy(config=config(), client=client), client


def test_valid_action_chunk():
    policy, client = mocked_policy({"action": [0.1, 0, -0.1, 0, 0, 0, -1]})
    try:
        chunk = policy.predict(obs())
        assert chunk.actions.shape == (2, 7)
        assert chunk.action_space == "libero_delta_pose_axisangle_gripper_7d"
        np.testing.assert_array_equal(chunk.actions[0], chunk.actions[1])
    finally:
        policy.close()
        client.close()


@pytest.mark.parametrize("bad", [
    {"action": [0.5, 0, 0, 0, 0, 0, -1]},   # over translation limit
    {"action": [0, 0, 0, 0, 0, 0, 0]},      # gripper invalid
    {"action": [0, 0, 0, 0, 0, 0]},         # wrong shape
    {"action": [0, 0, 0, 0, 0, 0, "-1"]},  # text instead of number
])
def test_bad_action_is_rejected(bad):
    policy, client = mocked_policy(bad)
    try:
        with pytest.raises(ValueError):
            policy.predict(obs())
    finally:
        policy.close()
        client.close()
