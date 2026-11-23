# RobotControlSkill — 统一策略推理与机器人控制

该 Skill 负责将 `Policy.predict(Observation) -> ActionChunk` 和 `RobotAdapter.step(action)` 组合为控制循环。策略与机器人通过配置中的驱动加载。环境启停、场景初始化、预览和录像由 `robot_env` 管理。**VLA 本身只推理，不能直接执行机器人动作。**

## 已实现策略与机器人

- `policy="openpi"`：通过独立 Python 进程运行 `openpi_client`，使用 OpenPI WebSocket 协议远程推理。输入输出映射和连接参数读取 `config/policy_cfg/openpi/openpi.yaml`，其中可引用根目录 `.env` 的 `ROBOT_OPENPI_*` 变量。**不接受用户聊天消息指定 SSH 用户、服务器或密钥。**
- `robot="libero"`：使用 `robot_env/libero/adapter.py` 读取已有 worker 的状态/双相机图像，通过 Unix socket 发送 `step(action)`；不会新建或重置场景。I/O 配置读取 `config/robot_env_cfg/libero/libero.yaml`。

机器人和策略从 `config/robot_env_cfg/`、`config/policy_cfg/` 及其子目录中的 `.yaml` / `.yml` 自动发现，配置按环境或策略放入对应子目录。上面是当前内置实现；
先调用 `robot_control_list` 获取实际可用名称与 I/O 协议。省略名称和步数时使用
`config/robot_control.yaml` 中的默认值。配置目录可由 `ROBOT_CONFIG_DIR` 指定。
新增驱动与配置的说明见 `config/README.md`。

## Agent 工具

- `robot_control_list()`：列出已配置机器人、策略、输入输出与默认值，不返回连接凭据。
- `robot_control_predict_once(robot="libero", policy="openpi", show_actions=3)`：只测试策略推理，返回动作形状，不执行动作。建议第一次使用先调用。
- `robot_control_start(robot="libero", policy="openpi", max_steps=300, replan_steps=5)`：启动同步控制循环，异步返回 run_id，**不表示执行成功**。
- `robot_control_status()`：查询当前执行状态、步数、错误和成功标记。通过 `robot_libero_query_state(report_after_step=True)` 开启报告后，`last_state_report` 返回最近一步动作执行后的机器人与任务状态；关闭后下一步清空该报告。
- `robot_control_stop()`：请求停止控制线程，不等于物理急停，不能中断正在阻塞的推理。

## 正常使用

1. 查看配置并通过 `robot_env` 初始化对应环境；LIBERO 需先加载场景。
2. 确认服务器侧模型服务已启动；OpenPI 驱动可以按配置建立 SSH 端口转发，但**不负责通过 SSH 启动或训练云端模型**。
3. 先调用 `robot_control_predict_once`，确认推理成功和动作维度/空间相符。
4. 用户明确请求执行时，调用 `robot_control_start`，之后用 `robot_control_status` 查询结果。
5. 必要时使用 `robot_control_stop`；录像由 robot_env Skill 负责。

## 重要限制

- 当前 LIBERO 默认 action_space 是 `libero_delta_pose_axisangle_gripper_7d`；即使同样是 7 维，真机也不代表可直接执行这些动作。
- 创建驱动前检查机器人/策略的动作空间、动作维度、相机及状态维度；运行中检查实际数据格式。模型训练、状态含义和动作单位必须与目标机器人一致，配置不会自动转换控制语义。
- 推理返回完整 action chunk，Controller 执行前 `replan_steps` 个动作后重新观测并推理。
- 不支持 RTC 或在云端推理期间持续执行旧动作；这是同步版本。
- 运行中禁止并行 reset/step，不允许把 agent 对话作为高速控制循环。真机必须配备独立急停、关节/末端限幅、看门狗与控制所有权机制。


## MaaS Franka Joint 适配（新增）

- MaaS 是 `policy=maas`，通过 HTTPS POST `/api/v1/action` 推理；无需 SSH。
- 仅与 `robot=libero_joint` 配对；先在环境工具执行 `robot_libero_reset(control_mode="joint_position")`。
- 优先 `robot_control_predict_once(robot="libero_joint", policy="maas")` 验证；初始禁用动作执行。
- 输出 8D joint 控制与旧 LIBERO OSC 7D 动作不兼容。启用前须确认关节弧度、夹爪约定、第三路相机要求及 MaaS 状态输入单位。


## MaaS 自动切换到关节模式

- 当用户选择 `policy="maas"` 且未指定 robot 时，控制工具会自动寻找唯一与 MaaS I/O 契约兼容的机器人配置；明确指定的错误组合不会被偷偷改掉。
- 对 `robot="libero_joint"`，`robot_control_predict_once` 和 `robot_control_start` 在开始前先查询 LIBERO worker。若已加载任务但 `control_mode != "joint_position"`，将自动使用相同 `task_suite_name`、`task_id`、`episode_idx` 重置为 `joint_position`。
- **自动重置会清空场景进度**。若正在录像、控制任务运行中或没有已加载的任务，不自动重置，返回清晰错误。未加载任务时仍需用户或 Agent 通过 `robot_libero_reset` 先选择具体任务。
- `predict_once` 不执行机械臂动作，但可能触发一次场景重置。Agent 不需要再向用户询问是否允许切换关节模式（用户已经选择让其自动切换），但应明确告知实际是否发生重置。
- 该功能不解除 `execution_enabled` 安全锁，不会转换动作语义。MaaS 的 8D 关节动作、夹爪编码和第三路相机占位问题必须独立验证。
