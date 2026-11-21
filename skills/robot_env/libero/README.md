# LIBERO Environment Skill — optional live preview + MP4 recording

## 放置路径

```text
/home/ma/robot-agent/skills/
└── robot_env/
    ├── __init__.py           # 让现有 SkillRegistry 扫描到 LiberoSkill
    └── libero/
        ├── __init__.py
        ├── SKILL.md          # Agent 的技能说明
        ├── tool.py           # LangChain Tools，只有环境功能
        ├── bridge.py         # 独立 Python worker 的启动与 Unix socket 客户端
        ├── worker.py         # 真正的 LIBERO 仿真、观测、录像
        ├── viewer.py         # 独立 GUI 子进程，只显示最新仿真帧
        └── README.md
```

无需修改项目的 `agent.py`、`runtime.py`、`main.py` 或 `skills/registry.py`。Agent Python 和 LIBERO Python 完全分离。

## 环境准备

在 `robot-agent/.env` 配置：

```dotenv
ROBOT_LIBERO_PYTHON=/path/to/libero_env/bin/python
ROBOT_LIBERO_WORKDIR=/path/to/libero/project
# 当 editable 安装无法识别源码时，显式指定可导入 libero 的 Python 搜索根目录。
# 未配置时默认使用 ROBOT_LIBERO_WORKDIR。
ROBOT_LIBERO_SOURCE_ROOT=/path/to/libero/project
ROBOT_LIBERO_OUTPUT_DIR=/home/ma/robot-agent/cache/libero
ROBOT_LIBERO_TIMEOUT=120
ROBOT_LIBERO_START_TIMEOUT=20
```

启动 worker 时会清理 Agent 侧的 `PYTHONPATH`，然后仅在 LIBERO worker 子进程中设置 `ROBOT_LIBERO_SOURCE_ROOT`（默认 `ROBOT_LIBERO_WORKDIR`）作为 `PYTHONPATH`。例如用户仓库在 `/home/ma/GithubDoc/LIBERO`，则两个变量均可设置到该路径。这可修复只在显式设置 `PYTHONPATH` 时才可导入 `libero` 的 editable 安装问题。

独立 LIBERO Python 应能导入 `libero`、`numpy`、`PIL`、`imageio`，能够执行 MuJoCo offscreen 渲染。录 MP4 还应安装 imageio 的 ffmpeg 后端：

```bash
/path/to/libero_env/bin/python -m pip install 'imageio[ffmpeg]'
```

需要实时图形窗口时在 **LIBERO 的 Python** 中安装支持 GUI 的 OpenCV（仅 `opencv-python-headless` 不够），同时本机必须有可用的图形显示会话：

```bash
/path/to/libero_env/bin/python -m pip install opencv-python
```

注意已经装有 `opencv-python-headless` 的环境不应无脑混装；需要先确认 OpenCV 包安装情况，然后只保留符合需求的版本。Linux 上没有 `DISPLAY` / `WAYLAND_DISPLAY` 的情况下无法直接弹出窗口，但无窗口录像仍然可用。`MUJOCO_GL=egl` 可用于配置兼容的离屏渲染环境，与是否显示 OpenCV 观察窗是不同的设置。

## 使用示例（通过 Agent / 飞书自然语言）

1. “启动 LIBERO 仿真环境，**实时显示** agentview 画面。”（`robot_libero_start(show_window=True)`）
2. “列出 libero_spatial 的任务，并初始化 task 0，初始状态 0。”（list_tasks + reset）
3. “开始录像，使用两个相机并排显示，回放帧率 10。”（`robot_libero_record_start(fps=10, camera="side_by_side")`）
4. **未来的动作 Skill 负责执行**（每个 `step` 自动产生新一帧供录像和预览使用）。
5. “结束录像，返回 MP4 文件路径。”（`robot_libero_record_stop`）
6. “关闭实时显示，但保留场景。”（`robot_libero_display(show_window=False)`）
7. “关闭 LIBERO 环境。”（close 或 stop）

也可以不打开窗口直接使用录像：`robot_libero_start(show_window=False)`。

## 自动保存策略

| 事件 | 录像结果 |
| --- | --- |
| 启动环境服务 `start` | **不开始**录像 |
| 重置场景 `reset` | **不开始**录像；旧录像如有则自动结束并保存 |
| 开始录像 `record_start` | 记录当前帧；以后每次 step 记录一帧 |
| LIBERO 报告 success/done | **自动结束并保存 MP4** |
| 手动 `record_stop` | **保存 MP4** |
| `close`/`stop` | **保存正在录制的 MP4** |
| 未发生动作 `step` | 无新仿真帧，因此录像可能只有一帧 |

MP4 输出位置：`ROBOT_LIBERO_OUTPUT_DIR/videos/*.mp4`，不会覆盖旧文件。结果中包含 `video_path`、`frames`、`fps`、`duration_s`。回放时长 `frames/fps` 不代表真实墙钟耗时；仿真任务的真实执行速度由动作调用频率决定。

## 实时显示是什么

本 Skill 使用 `OffScreenRenderEnv` 获取图像，再由**单独的 viewer.py 图形子进程**显示 `agentview`、`eye_in_hand` 或拼接画面。只在 `reset` 和每次 `step` 后更新画面；不自动推进仿真、不自动录像。这样即使 OpenCV 高 GUI 组件崩溃，也不会直接终止 LIBERO 的 MuJoCo worker。

使用窗口需要本机 Linux 图形会话；直接通过 SSH 在无桌面服务器上启动 Agent 无法在用户手机飞书里显示实时流，这个功能**不是网络视频直播**。可以通过保存 PNG / MP4 查看画面。

通过 `robot_libero_display(show_window=True/False)` 可在同一个任务内切换窗口。`status` 返回 `display_active`、`display_camera`、`display_error`，录像字段包含 `recording`、`recording_path`、`recording_frames`、`last_video`。

## 下层动作 Skill 的接口

Unix socket 路径：`ROBOT_LIBERO_OUTPUT_DIR/libero.sock`。一条请求是一行 JSON：

```json
{"command":"observe","params":{"include_images":true}}
```

响应：`{"ok":true,"result":{...}}`；出错时：`{"ok":false,"error":"..."}`。

新增环境协议：

| Command | 参数 | 说明 |
| --- | --- | --- |
| `display` | `show_window`, `camera` | 控制 GUI 预览窗口 |
| `record_start` | `fps`, `camera` | 开始录像，仅在初始化后可用 |
| `record_stop` | 无 | 停止并保存 MP4 |

原有协议依然保留：`status`、`list_tasks`、`reset`、`observe`、`step`、`close`、`shutdown`。

```python
from skills.robot_env.libero.bridge import LiberoEnvClient
client = LiberoEnvClient()
obs = client.request("observe", include_images=True)
# 在另一个 Skill 中计算 action；不要把此接口直接注册给 Agent
result = client.request("step", action=[0, 0, 0, 0, 0, 0, -1])
```

`step` 接口只在底层存在，**不属于本环境 Skill 暴露的 Agent Tools**。观测图片仍按用户原始评测脚本旋转 180°，没有 VLA 专用 resize 或动作预处理逻辑。

## 故障排查

日志：`cache/libero/worker.log`。

- `show_window=True` 启动失败但环境仍可用：检查 `DISPLAY`、桌面权限、OpenCV 是否支持 GUI。关闭显示后继续无头仿真。
- 提示 ffmpeg 缺失：在 LIBERO Python 环境安装 `imageio[ffmpeg]`。
- reset 报缺少 BDDL / MuJoCo 渲染错误：检查 LIBERO 数据集配置、`MUJOCO_GL`、图形驱动。
- 如果视频只有一帧：检查是否有动作 Skill 调用底层 `step`；`observe` 只读取不推进物理。

本 Skill 仅使用本机 Unix socket（输出目录 0700、socket 0600），默认不暴露远程 HTTP 端口。
