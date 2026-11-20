"""RealSense D435 的 LangChain Tools。未连接设备不影响 Agent 正常启动。"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


class RealSenseCameraSkill:
    def __init__(self):
        self.width = int(os.getenv("REALSENSE_WIDTH", "640"))
        self.height = int(os.getenv("REALSENSE_HEIGHT", "480"))
        self.fps = int(os.getenv("REALSENSE_FPS", "30"))
        self.serial_number = os.getenv("REALSENSE_SERIAL_NUMBER") or None
        self.output_dir = Path(os.getenv("CAMERA_OUTPUT_DIR", "captures")) / "realsense"

    def tools(self) -> list:
        @tool
        def realsense_camera_status() -> str:
            """查询本机是否连接 Intel RealSense（如 D435）深度相机，不启动采集。"""
            try:
                from .camera import RealSenseCamera
                devices = RealSenseCamera.list_devices()
                if not devices:
                    return "RealSense 未连接：当前未检测到任何 RealSense 相机。"
                if self.serial_number and not any(
                    item["serial_number"] == self.serial_number for item in devices
                ):
                    return f"RealSense 未连接：未找到指定序列号 {self.serial_number}；其他设备：{devices}"
                return f"检测到 RealSense 相机：{json.dumps(devices, ensure_ascii=False)}"
            except (ImportError, RuntimeError, OSError) as exc:
                return f"RealSense 不可用（可能未安装 pyrealsense2 或设备不可访问）：{exc}"

        @tool
        def realsense_camera_capture(include_depth: bool = True) -> str:
            """按用户要求从 RealSense D435 等相机拍一张彩色原图；可同时保存对齐的16位原始深度图和深度标尺。未插入设备时返回明确提示。不会识别图像或发送飞书图片。"""
            try:
                import cv2
                from .camera import RealSenseCamera

                with RealSenseCamera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    serial_number=self.serial_number,
                ) as camera:
                    color, depth, metadata = camera.get_images(return_metadata=True)
                    if color is None or depth is None:
                        return "RealSense 已连接但未取得有效帧，请重试。"
                    color = color.copy()
                    depth = depth.copy()
                    # 深度已对齐到 color stream，对齐后内参必须取 metadata。
                    depth_intrinsics = metadata["depth_intrinsics"]
                    data = {
                        "width": int(color.shape[1]),
                        "height": int(color.shape[0]),
                        "capture_time_ns": metadata["capture_time_ns"],
                        "depth_scale_m_per_unit": camera.depth_scale,
                        "aligned_depth_intrinsics": camera._intrinsics_as_dict(depth_intrinsics),
                        "color_intrinsics": camera._intrinsics_as_dict(camera.color_intrinsics),
                    }

                self.output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"realsense_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
                color_path = self.output_dir / f"{stem}_color.png"
                if not cv2.imwrite(str(color_path), color):
                    return "RealSense 彩色帧已采集但保存失败。"

                if not include_depth:
                    return f"RealSense 彩色图片已保存：{color_path.resolve()}"

                depth_path = self.output_dir / f"{stem}_depth.png"
                meta_path = self.output_dir / f"{stem}_metadata.json"
                if not cv2.imwrite(str(depth_path), depth):
                    return f"彩色图片：{color_path.resolve()}；但深度图保存失败。"
                meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return (
                    f"RealSense 拍照成功。彩色图：{color_path.resolve()}；"
                    f"对齐16位深度图：{depth_path.resolve()}；"
                    f"深度标尺及内参：{meta_path.resolve()}。"
                )
            except Exception as exc:
                # 未插入设备/驱动缺失/USB权限问题属于预期可恢复错误，不影响其他 Skill。
                return f"RealSense 拍照失败：{type(exc).__name__}: {exc}"

        return [realsense_camera_status, realsense_camera_capture]
