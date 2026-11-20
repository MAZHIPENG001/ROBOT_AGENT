# 两个独立 Camera Skills

把压缩包中的 `skills/laptop_camera/` 与 `skills/realsense_camera/` 复制到**已有项目**的 `skills/` 下，不替换已有的 `agent.py`、`main.py`、`registry.py`。

```text
skills/
├── registry.py                 # 仍然使用已有自动发现版
├── calculator/
├── feishu/
├── laptop_camera/
│   ├── __init__.py
│   ├── SKILL.md
│   ├── camera.py             # 原 laptop_camera.py 驱动
│   └── tool.py               # LaptopCameraSkill
└── realsense_camera/
    ├── __init__.py
    ├── SKILL.md
    ├── camera.py             # 改造的 RealSenseCamera 驱动
    └── tool.py               # RealSenseCameraSkill
```

原来的自动发现 `SkillRegistry` 会分别导入 `LaptopCameraSkill` 和 `RealSenseCameraSkill`。扫描技能阶段不会访问硬件，也不会因为没插 D435 而失败；检查/拍摄时才访问设备。`main.py` 继续调用 `registry.get("feishu").start(on_text)`。

## 安装

```bash
uv add opencv-python numpy
# 仅需要用 D435 的机器安装：
uv add pyrealsense2
```

`pyrealsense2` 安装和使用依赖操作系统、Python 版本、设备驱动及 USB 权限，需在自己的电脑上确认。

## 可选 .env

```dotenv
LAPTOP_CAMERA_INDEX=0
LAPTOP_CAMERA_WIDTH=640
LAPTOP_CAMERA_HEIGHT=480
LAPTOP_CAMERA_FPS=30
REALSENSE_WIDTH=640
REALSENSE_HEIGHT=480
REALSENSE_FPS=30
# REALSENSE_SERIAL_NUMBER=...
CAMERA_OUTPUT_DIR=captures
```

## Agent 工具

- `laptop_camera_status`：检查笔记本摄像头
- `laptop_camera_capture`：拍一张本地照片
- `realsense_camera_status`：查询 RealSense 设备列表
- `realsense_camera_capture(include_depth=True)`：拍照并按需保存对齐深度图

## 本地开发验证

`python -m skills.laptop_camera.camera` 可以本地预览，按 Q 退出。RealSense 直接测试：

```python
from skills.realsense_camera.camera import RealSenseCamera

with RealSenseCamera(width=640, height=480, fps=30) as camera:
    color, depth, metadata = camera.get_images(return_metadata=True)
    print(color.shape, depth.shape, metadata["capture_time_ns"])
```

## 限制与安全

相机打开、图片保存均发生在**运行 Agent 的机器上**。返回给飞书的是图片本地路径，不是图片文件，更不是物体识别结果。要在飞书直接看图片，需要另做上传图片/发送图片的 Feishu Tool。`deepseek-chat` 文本模型不能直接理解图片，要让 Agent 报告画面中物体，需接入支持视觉输入的模型或视觉推理服务。

上传前确认本地隐私/授权需求；不要为 Agent 增加可由用户自由指定的写入路径。
