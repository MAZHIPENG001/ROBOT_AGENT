"""笔记本相机 LangChain Tools。仅在调用时访问本机硬件。"""

import os
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


class LaptopCameraSkill:
    def __init__(self) -> None:
        self.index = int(os.getenv("LAPTOP_CAMERA_INDEX", "0"))
        self.width = int(os.getenv("LAPTOP_CAMERA_WIDTH", "640"))
        self.height = int(os.getenv("LAPTOP_CAMERA_HEIGHT", "480"))
        self.fps = int(os.getenv("LAPTOP_CAMERA_FPS", "30"))
        self.output_dir = Path(os.getenv("CAMERA_OUTPUT_DIR", "captures")) / "laptop"

    def tools(self) -> list:
        @tool
        def laptop_camera_status() -> str:
            """检查当前运行 Agent 的笔记本电脑内置/USB 普通摄像头是否可打开；不拍照、不识别画面。"""
            try:
                from .camera import LaptopCamera

                with LaptopCamera(index=self.index, width=self.width,
                                  height=self.height, fps=self.fps):
                    return f"笔记本摄像头可用，index={self.index}。"
            except Exception as exc:
                return f"笔记本摄像头不可用：{type(exc).__name__}: {exc}"

        @tool
        def laptop_camera_capture() -> str:
            """经用户要求，从笔记本电脑摄像头采集当前的一张原始彩色照片，保存到运行 Agent 的电脑本地；不识别物体、不发送图片到飞书。"""
            try:
                import cv2
                from .camera import LaptopCamera

                with LaptopCamera(index=self.index, width=self.width,
                                  height=self.height, fps=self.fps) as camera:
                    frame = camera.read()  # OpenCV BGR 原始帧

                self.output_dir.mkdir(parents=True, exist_ok=True)
                filename = f"laptop_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.png"
                path = self.output_dir / filename
                if not cv2.imwrite(str(path), frame):
                    return "笔记本摄像头采集成功，但保存图片失败。"
                return f"笔记本摄像头拍照成功。本地图片：{path.resolve()}；分辨率：{frame.shape[1]}x{frame.shape[0]}。"
            except Exception as exc:
                return f"笔记本摄像头拍照失败：{type(exc).__name__}: {exc}"

        return [laptop_camera_status, laptop_camera_capture]
