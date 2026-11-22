# 机器人与策略配置

`ControlRegistry` 在启动时读取本目录，自动发现 `robots/*.yaml`、`robots/*.yml`、
`policies/*.yaml` 和 `policies/*.yml`。配置的 `name` 是工具使用的名称，不依赖文件名。
修改后重启 Agent 生效。也可通过 `.env` 的 `ROBOT_CONFIG_DIR` 指定另一个目录。

## 现有配置

| 文件 | 作用 |
| --- | --- |
| `robot_control.yaml` | 默认机器人、策略、最大执行步数、每次推理后执行步数 |
| `robots/libero.yaml` | LIBERO 驱动、相机源字段、状态拼接顺序、动作字段与空间 |
| `policies/openpi.yaml` | 当前 OpenPI 服务的模型输入输出及连接参数 |

现有 LIBERO 生命周期仍由 `robot_libero_start/reset/stop` 等环境工具负责。
控制适配器连接已经初始化的场景，不会隐式 reset。新增环境或真机的初始化由对应驱动或管理工具负责。

## 统一输入输出

- `Observation.images`：逻辑相机名到正向 RGB `uint8[H,W,3]` 的字典；相机数量和名称由配置决定。
- `Observation.state`：有限的一维状态向量，按机器人配置 `observation.state` 的顺序拼接。
- `Observation.instruction`：任务文字；`episode_id` 用于发现执行期间的任务切换。
- `ActionChunk.actions`：有限的 `float32[T, action_dim]` 动作块。
- `ActionChunk.action_space`：动作物理语义的明确标识；机器人与策略必须声明相同的空间和维度。
- `StepFeedback`：单步后的结束、成功与 episode 信息。

状态每个字段声明名称、源字段、长度、单位；动作每个字段声明名称、长度、单位。
状态与动作维度分别由字段长度求和。例如当前 LIBERO 状态为 `3 + 3 + 2 = 8`，
动作为 `3 + 3 + 1 = 7`。图像的旋转与坐标转换由机器人驱动完成。

配置配对会在创建驱动前检查动作空间/维度、必需相机和状态维度。
运行时继续检查实际观测格式、动作维度及有限值。
`action_space` 的名称应明确坐标系、旋转表示、绝对/增量控制和夹爪约定。
这里不会自动转换动作单位或坐标系；相同维度、相同接口也不能保证模型适用于另一个机器人，
必须与服务端模型的训练数据、归一化和控制器约定一致。

## 接入另一个 OpenPI 模型服务

复制 `policies/openpi.yaml` 为新文件，修改 `name`、模型输入输出和 `options`。
例如单相机、关节状态模型可以配置为下面的输入映射（维度需与实际模型一致）：

```yaml
inputs:
  images:
    overhead: observation/front_rgb
  state_key: observation/joints
  instruction_key: language_instruction
  state_dim: 6
  state_indices: [0, 1, 2, 3, 4, 5]
outputs:
  actions_key: actions
  action_dim: 6
  action_space: my_robot_joint_delta_rad_6d
```

机器人配置需提供 `overhead` 相机和匹配的动作协议。`state_indices` 可省略；
指定时按索引选择/重排机器人状态，再传给模型。相机数量也可以为零（`images: {}`）。
模型字段映射按字面键名发送；`observation/front_rgb` 是单个键，不会自动构造嵌套字典。
OpenPI 负责图像缩放与 WebSocket 推理，不包含 LIBERO 字段或状态拼接规则。

## 接入新环境、真机或策略

1. 实现独立 Python 驱动，并接受关键字构造参数 `config`。
2. 为驱动添加机器人或策略 YAML，设置 `driver: 包.模块:类名`。
3. 修改 `robot_control.yaml` 的默认组合，或在工具中指定对应 `robot` / `policy` 名称。

机器人驱动接收 `RobotConfig`，实现以下协议：

```python
class MyRobotAdapter:
    def __init__(self, *, config): ...
    def observe(self) -> Observation: ...
    def step(self, action: np.ndarray) -> StepFeedback: ...
    def close(self) -> None: ...
```

机器人驱动应按 `config.observation` 读取、解码和拼接源数据；
按 `config.action` 对接底层控制器，不能通过改配置名称改变底层动作语义。
`kind` 填 `simulation` 或 `hardware`，`options` 存放该驱动的连接和运行参数。

策略驱动接收 `PolicyConfig`，实现以下协议：

```python
class MyPolicy:
    def __init__(self, *, config): ...
    def predict(self, observation: Observation) -> ActionChunk: ...
    def close(self) -> None: ...
```

策略驱动负责按 `config.inputs` 选取图像、执行可选状态索引、构造模型请求，
并按 `config.outputs` 返回动作块；控制器始终只调用统一接口。
无需修改控制器或工具注册代码。驱动地址只从本地配置读取，聊天工具只接受已配置的名称。
硬件驱动需要自行实现控制频率、限幅、看门狗和急停；`robot_control_stop` 是协作式停止。

## 连接信息与环境变量

配置支持 `${NAME}` 和 `${NAME:-default}`。前者缺失时报错；后者在变量缺失或空值时使用默认值。
先解析 YAML 再替换值，因此密码中的引号、冒号和换行不会变成 YAML 语法。
凭据建议继续放在未提交的 `.env` 中；Agent 启动入口会加载它。

当前 OpenPI 配置引用既有 `ROBOT_OPENPI_*` 连接变量。图像映射、状态维度和动作空间以
YAML 为准（原 `ROBOT_OPENPI_ACTION_SPACE` 不再控制动作协议）。
同一策略的不同实例可以配置不同的服务器、解释器和字段映射。
`robot_control_list` 不返回 `options` 中的连接信息和凭据。
