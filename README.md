# Robot-Agent

机器人控制通过统一的 `Observation → Policy.predict → ActionChunk → RobotAdapter.step`
接口组合环境与策略。当前提供 LIBERO 和 OpenPI 驱动，新增环境、真机或策略可以通过独立驱动接入。

配置入口在 [`config/`](config/README.md)：

- [`robot_control.yaml`](config/robot_control.yaml)：默认机器人、策略与控制步数。
- [`robots/libero.yaml`](config/robots/libero.yaml)：相机、状态字段顺序、动作空间与适配器。
- [`policies/openpi.yaml`](config/policies/openpi.yaml)：模型输入输出映射、解释器和服务连接。

`.env` 中的现有连接参数继续通过 YAML 的环境变量引用生效。可使用
`robot_control_list` 查看已配置的机器人和策略；初始化环境后，先调用
`robot_control_predict_once` 检查推理，再按需调用 `robot_control_start`。

运行接口测试：`.venv/bin/python -m pytest -q`。
