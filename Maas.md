# API 接入示例
UR5 接入示例：结合机器人 SDK 调用 MaaS API

此示例仅适用于已部署且与 UR5 的 6 关节＋1 夹爪、JOINT 和两路相机匹配的后训练模型。MODEL_ID 须填写查询返回的、与本例配置匹配的后训练模型 ID。基础模型要求三路图像，不适用本两路相机示例。停止、到位判断与资源释放需接入实际 UR5 SDK，不能把 close 等同于机械臂已停止。

本示例使用与 UR5 训练布局匹配、且部署 ID 以 ft: 开头的后训练模型。示例只发送两路图；无此前缀时接口要求三路图。后训练模型不支持记忆，忽略查询返回的 history_fps、max_history_images，不启动历史采样，也不提交 history_images；每次使用当前 state 和 images 请求推理。图像上传规范见图像处理。

```python
# -*- coding: utf-8 -*-
"""UR5 后训练模型调用示例：必须先完成文末的硬件适配。

必填环境变量：MAAS_API_KEY、MAAS_MODEL_ID、MAAS_PROMPT。
MAAS_MODEL_ID 须填写与 UR5 训练布局匹配、且以 ft: 开头的后训练部署 ID；
本例只发送两路图，无此前缀时接口要求三路图。
MAAS_IMAGE_MAX_SIZE 为可选客户端缩放配置，格式为“宽,高”，默认“3840,3840”。
4K 以内的图像保留原尺寸，超限时报错。推荐上传长边 640 像素的图像；
如需发送前主动缩小，可配置为“640,640”。宽、高均不得小于 10 像素；
capabilities.image_size 展示模型配置尺寸。
模型须与此机器人、7 维 JOINT 布局、两路相机的训练配置一致。
本脚本不进行真实硬件安全验证；未完成适配时，会在创建机器人前退出。
"""

import base64
import math
import os
import time

import cv2
import requests
from ur5_sdk.Ur5Robot import Ur5

API_URL = "https://maas.dexmal.com/api/v1/action"
CAMERAS = ("cam_high", "cam_left_wrist")
TASK_SECONDS = 600
ACTION_SECONDS = 10


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("任务或动作已超时，停止下发动作")
    return seconds


def vector7(value, label):
    try:
        items = list(value)
        if len(items) != 7 or any(isinstance(x, bool) for x in items):
            raise ValueError
        numbers = [float(x) for x in items]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError
        return numbers
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须包含 7 个有限数值") from exc


def read_config():
    names = ("MAAS_API_KEY", "MAAS_MODEL_ID", "MAAS_PROMPT")
    values = [os.environ.get(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError("请配置必填环境变量：" + "、".join(missing))
    api_key, model_id, prompt = values
    if not model_id.startswith("ft:"):
        raise ValueError("两路相机示例需要 ft: 开头的后训练部署 ID")
    # 上传上限独立于模型配置尺寸；按需主动缩小至推荐尺寸。
    image_limit = os.environ.get("MAAS_IMAGE_MAX_SIZE", "3840,3840")
    try:
        width, height = (int(part.strip()) for part in image_limit.split(","))
    except ValueError as exc:
        raise ValueError("MAAS_IMAGE_MAX_SIZE 须填写宽,高") from exc
    if not (10 <= width <= 3840 and 10 <= height <= 3840):
        raise ValueError("缩放目标的宽和高均须不小于 10 像素且不超过 4K")
    return api_key, model_id, prompt, (width, height)


def encode_image(frame, image_max_size):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("相机须返回有效的三通道 BGR 图像")
    height, width = frame.shape[:2]
    if min(height, width) < 10:
        raise ValueError("图片宽高均不得小于 10 像素")
    if max(height, width) > 3840:
        raise ValueError("图片超过 4K 上限，请先调整采集或上传尺寸")
    max_width, max_height = image_max_size
    scale = min(1.0, max_width / width, max_height / height)
    size = (int(width * scale), int(height * scale))
    if min(size) < 10:
        raise ValueError("等比缩放后不足 10 像素，需调整相机采集画幅")
    if size != (width, height):
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def confirmed(callback, *args):
    result = callback(*args)
    if not isinstance(result, bool):
        raise TypeError("硬件完成判定回调必须返回 True 或 False")
    return result


def run(*, stop_robot, close_robot, action_finished, task_finished,
        hardware_adapters_ready=False):
    callbacks = (stop_robot, close_robot, action_finished, task_finished)
    if hardware_adapters_ready is not True or not all(callable(fn) for fn in callbacks):
        raise NotImplementedError(
            "请先接入并验证真实 SDK 的安全停止、资源释放、动作及任务完成判定；"
            "不能使用空回调，再显式设置 hardware_adapters_ready=True"
        )
    api_key, model_id, prompt, image_max_size = read_config()
    task_deadline = time.monotonic() + TASK_SECONDS
    robot = None
    operation_error = None
    cleanup_errors = []
    try:
        robot = Ur5()
        remaining(task_deadline)
        with requests.Session() as client:
            while True:
                remaining(task_deadline)
                done = confirmed(task_finished, robot)
                remaining(task_deadline)
                if done:
                    break

                state = vector7(robot.get_joint_with_gripper(), "state")
                remaining(task_deadline)
                frames = robot.get_imgs()
                remaining(task_deadline)
                if frames is None or len(frames) < len(CAMERAS):
                    raise ValueError("缺少两路有效相机图像")
                # 必须确认 SDK 返回顺序与这两个相机名称一致。
                images = {
                    name: encode_image(frames[index][1], image_max_size)
                    for index, name in enumerate(CAMERAS)
                }
                response = client.post(
                    API_URL,
                    headers={"X-API-Key": api_key},
                    json={"model_id": model_id, "prompt": prompt, "state": state,
                          "action_type": "JOINT", "images": images},
                    timeout=min(60.0, remaining(task_deadline)),
                )
                # requests 的 timeout 不是整个请求的硬截止时间；返回后仍须检查。
                remaining(task_deadline)
                response.raise_for_status()
                payload = response.json()
                remaining(task_deadline)
                if payload.get("code") != 2000:
                    raise RuntimeError(f"API 业务失败，code={payload.get('code')}")
                data = payload.get("data")
                actions = data.get("actions") if isinstance(data, dict) else None
                if not isinstance(actions, list) or not actions:
                    raise ValueError("API 未返回有效动作序列")
                actions = [vector7(action, "action") for action in actions]

                for action in actions:
                    remaining(task_deadline)
                    action_deadline = min(task_deadline, time.monotonic() + ACTION_SECONDS)
                    robot.go_joint(action[:6], action[6])
                    remaining(action_deadline)
                    while True:
                        remaining(action_deadline)
                        current = vector7(robot.get_joint_with_gripper(), "当前状态")
                        remaining(action_deadline)
                        joints_reached = max(abs(a - b) for a, b in zip(current[:6], action[:6])) <= 0.01
                        # 回调补充夹爪和其他实际硬件到位条件，关节到位不等于整步完成。
                        reached = joints_reached and confirmed(action_finished, robot, current, action)
                        remaining(action_deadline)
                        if reached:
                            break
                        time.sleep(min(0.02, remaining(action_deadline)))
                    done = confirmed(task_finished, robot)
                    remaining(task_deadline)
                    if done:
                        return
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if robot is not None:
            # 停止失败也尝试释放；释放资源本身不代表机器人已经停止。
            for label, callback in (("安全停止", stop_robot), ("资源释放", close_robot)):
                try:
                    callback(robot)
                except BaseException as exc:
                    cleanup_errors.append((label, exc))
            if cleanup_errors:
                details = "；".join(f"{label}失败：{type(exc).__name__}" for label, exc in cleanup_errors)
                if operation_error is not None:
                    print("清理异常：" + details)
                else:
                    raise RuntimeError(details) from cleanup_errors[0][1]


# 以下是本示例的适配函数，不是对 UR5 SDK 方法名的声明。
# 必须替换为项目实际接口并完成验证；不可改成 pass 或无效果的空回调。
def hardware_stop(robot):
    raise NotImplementedError("接入实际 SDK 的安全停止接口")


def hardware_close(robot):
    raise NotImplementedError("接入实际 SDK 的资源释放接口")


def hardware_action_finished(robot, current, target):
    raise NotImplementedError("按实际反馈确认夹爪及该步动作完成，返回 bool")


def hardware_task_finished(robot):
    raise NotImplementedError("按业务任务的实际完成条件返回 bool")


if __name__ == "__main__":
    run(stop_robot=hardware_stop, close_robot=hardware_close,
        action_finished=hardware_action_finished, task_finished=hardware_task_finished,
        hardware_adapters_ready=False)
```

# Dexos Lite 接入示例

config.py 模板在前，BaseRobot 模板在后。两者都只用于用 dexos-lite 快速验证模型推理效果的快速验证：config.py 给两台原生机型，BaseRobot 给其它机型自己实现（见 其它机型：自己实现 BaseRobot）。正式应用不需要实现 BaseRobot。

## config.py 模板
```python
# dexos-lite robot config --robot-id mirror --path ./config.py 生成。
# 文件里只有一个 robot 变量，CLI 靠这个名字加载；加 --force 才会覆盖已有文件。
# 基础模型必须提供下面三路真实图像；
# 后训练模型按训练配置提供 1–3 路图像，数量与视角均须匹配训练配置；
# serial_number 必须填真实序列号，空字符串会让 start() 找不到设备。
#
# 运行前先启动 airbot_server（端口 50051/50053）。
from dexos_lite.robot import MirrorRobot

robot = MirrorRobot(
    # Start the Mirror arm service before running inference.
    sensors={
        "cam_high": {"serial_number": ""},
        "cam_left_wrist": {"serial_number": ""},
        "cam_right_wrist": {"serial_number": ""},
    },
)
```
```python
# --robot-id cobot_magic 生成的内容，与上面只差类名。
# 运行前先激活左右 CAN（波特率 1 Mbps）。
from dexos_lite.robot import CobotMagicRobot

robot = CobotMagicRobot(
    sensors={
        "cam_high": {"serial_number": ""},
        "cam_left_wrist": {"serial_number": ""},
        "cam_right_wrist": {"serial_number": ""},
    },
)
```
```python
# cobot_magic 的 CAN 接口名不是默认值时，手动加上 controller 字段。
robot = CobotMagicRobot(
    controller={
        "left_can_port": "can0",
        "right_can_port": "can1",
    },
    # 三个相机键与上面的 cobot_magic 模板相同，照抄过来。
    sensors={
        "cam_high": {"serial_number": ""},
        "cam_left_wrist": {"serial_number": ""},
        "cam_right_wrist": {"serial_number": ""},
    },
)
```

## BaseRobot 模板
自定义 BaseRobot 可覆盖 robot_id 属性，例如 robot_id="arx5"。dexos lite 推理循环会读取该属性并传入请求，无需另外改写请求层。state、actions 和相机仍须符合目标模型规范。

BaseRobot 提供当前观测与动作执行。基础模型启用记忆时，客户端按循环与时序维护独立历史缓存；使用 SDK 前确认其具备dexos lite：一份用于快速验证的现成实现所述能力，不在 sensors 中追加历史图片。后训练模型不启用历史采样。
```python
# dexos lite 自定义 BaseRobot 模板：基础模型 arx5／JOINT，7 维、三路图像。
# 将所有硬件适配方法实现后再运行，不能把占位值用于实机。
from __future__ import annotations
from typing import Any, Dict
from dexos_lite import ActionType
from dexos_lite.robot import BaseRobot

class MyRobot(BaseRobot):
    robot_id = "arx5"  # dexos lite 会自动读取并传入推理请求。

    @property
    def action_type(self) -> ActionType:
        return ActionType.JOINT

    def start(self) -> None:
        # 连接机器人与相机；初始化失败时释放已经打开的资源。
        raise NotImplementedError("请接入机器人和相机 SDK")

    def get(self, *, sensor_timeout: float = 1.0) -> Dict[str, Any]:
        # 每轮取一次观测；sensor_timeout 是单次取观测的超时。
        # 返回的键是 sensors（值是 data URL），与 API 接入代码里适配层的 images 不是同一套。
        # 这里按单臂 7 维、单路相机示例：state 长度按模型的 action_dim 调，
        # 相机条目数按 image_number 调（dm05 是 14 维、3 路）。
        return {
            "state": [0.0] * 7,
            "sensors": {
                "cam_high": "data:image/jpeg;base64,...",
                "cam_left_wrist": "data:image/jpeg;base64,...",
                "cam_right_wrist": "data:image/jpeg;base64,...",
            },
        }

    def set(self, actions: Any) -> Dict[str, Any]:
        # 校验每步 7 维动作、单位和范围，按控制器节拍下发。
        # 返回不等于机械臂已到位；如需到位确认，加入状态检查与超时。
        raise NotImplementedError("请实现动作下发与执行节奏")

    def reset_to_home(self) -> Dict[str, Any]:
        # 启动时调用（--continue 跳过）；Ctrl+C、普通异常退出时尝试调用。
        # 达到时间上限时只 close；复位失败不代表已回到初始位置。
        raise NotImplementedError("请实现可超时的复位流程")

    def close(self) -> None:
        # 释放连接与相机资源；释放资源不能替代硬件停止。
        raise NotImplementedError("请实现资源释放")

robot = MyRobot()
```