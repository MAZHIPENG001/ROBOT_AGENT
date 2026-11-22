# LIBERO 环境准备与排障

Agent 操作与录像规则统一见 [robot_env/SKILL.md](../SKILL.md)。
控制适配器 `adapter.py` 通过 `LiberoEnvClient` 共享 `worker.py` 的场景；
`bridge.py` 管理独立 Python worker，`viewer.py` 负责可选的本机窗口。

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

启动 worker 时会清理 Agent 侧的 `PYTHONPATH`，然后仅在 LIBERO worker 子进程中设置 `ROBOT_LIBERO_SOURCE_ROOT`（默认 `ROBOT_LIBERO_WORKDIR`）作为 `PYTHONPATH`。这可修复 editable 安装无法自动找到 `libero` 源码的问题。

独立 LIBERO Python 应能导入 `libero`、`numpy`、`PIL`、`imageio`，能够执行 MuJoCo offscreen 渲染。录 MP4 还应安装 imageio 的 ffmpeg 后端：

```bash
/path/to/libero_env/bin/python -m pip install 'imageio[ffmpeg]'
```

需要实时图形窗口时在 **LIBERO 的 Python** 中安装支持 GUI 的 OpenCV（仅 `opencv-python-headless` 不够），同时本机必须有可用的图形显示会话：

```bash
/path/to/libero_env/bin/python -m pip install opencv-python
```

注意已经装有 `opencv-python-headless` 的环境不应无脑混装；需要先确认 OpenCV 包安装情况，然后只保留符合需求的版本。Linux 上没有 `DISPLAY` / `WAYLAND_DISPLAY` 的情况下无法直接弹出窗口，但无窗口录像仍然可用。`MUJOCO_GL=egl` 可用于配置兼容的离屏渲染环境，与是否显示 OpenCV 观察窗是不同的设置。

## 故障排查

日志：`cache/libero/worker.log`。

- `show_window=True` 启动失败但环境仍可用：检查 `DISPLAY`、桌面权限、OpenCV 是否支持 GUI。关闭显示后继续无头仿真。
- 提示 ffmpeg 缺失：在 LIBERO Python 环境安装 `imageio[ffmpeg]`。
- reset 报缺少 BDDL / MuJoCo 渲染错误：检查 LIBERO 数据集配置、`MUJOCO_GL`、图形驱动。
- 如果视频只有一帧：检查是否有动作 Skill 调用底层 `step`；`observe` 只读取不推进物理。

本 Skill 仅使用本机 Unix socket（输出目录 0700、socket 0600），默认不暴露远程 HTTP 端口。
