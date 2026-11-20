"""Intel RealSense RGB-D 相机驱动。

与原始驱动兼容的主要接口：start/stop/get_frames/get_frame_bundle/
get_images/get_real_position/get_masked_point_cloud/get_point_cloud 等。

关键变化：实例化不连接 USB、不打开 pipeline；SDK 仅在真正需要时加载。
默认 640x480@30，可在构造函数中配置。设备不在时 start() 返回 False。
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial_number=None):
        if min(width, height, fps) <= 0:
            raise ValueError("width、height、fps 必须为正数")
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_number = serial_number

        # 不在 __init__ 里 import SDK / start pipeline；拔掉相机不影响 Agent 启动。
        self._rs = None
        self.pipeline = None
        self.profile = None
        self.align = None
        self.spatial_filter = None
        self.depth_scale = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.camera_config = None
        self.last_error = None
        self.lock = threading.Lock()
        self.thread = None
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_capture_time_ns = None

    @staticmethod
    def list_devices() -> list[dict]:
        """仅查询设备，不启动相机。SDK 缺失时抛出 ImportError。"""
        import pyrealsense2 as rs
        devices = rs.context().query_devices()
        return [
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial_number": dev.get_info(rs.camera_info.serial_number),
            }
            for dev in devices
        ]

    def start(self) -> bool:
        """按需启动，获取 RGB+对齐深度；失败返回 False，原因在 last_error。"""
        if self.pipeline is not None and self.profile is not None:
            return True
        self.last_error = None
        try:
            import pyrealsense2 as rs
            self._rs = rs
            devices = self.list_devices()
            if not devices:
                raise RuntimeError("未发现 RealSense 设备，请检查 D435 是否已连接")
            serials = {d["serial_number"] for d in devices}
            if self.serial_number and self.serial_number not in serials:
                raise RuntimeError(f"未找到序列号为 {self.serial_number} 的 RealSense 设备")

            self.pipeline = rs.pipeline()
            config = rs.config()
            if self.serial_number:
                config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.profile = self.pipeline.start(config)

            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            self.depth_intrinsics = self.profile.get_stream(
                rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = self.profile.get_stream(
                rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.align = rs.align(rs.stream.color)
            self.spatial_filter = rs.spatial_filter()
            self.camera_config = {
                "intrinsics": {
                    "color": self._intrinsics_as_dict(self.color_intrinsics),
                    "depth": self._intrinsics_as_dict(self.depth_intrinsics),
                    "depth_scale": self.depth_scale,
                }
            }

            self._stop_event.clear()
            self._first_frame_event.clear()
            with self.lock:
                self.latest_color_frame = None
                self.latest_depth_frame = None
                self.latest_capture_time_ns = None
            self.thread = threading.Thread(target=self._update_frames, daemon=True)
            self.thread.start()
            if not self._first_frame_event.wait(timeout=5):
                raise RuntimeError(self.last_error or "启动后未能在 5 秒内读到 RGB-D 帧")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("RealSense start failed: %s", exc)
            self.stop()
            return False

    @staticmethod
    def _intrinsics_as_dict(intr):
        return {
            "fx": intr.fx, "fy": intr.fy,
            "ppx": intr.ppx, "ppy": intr.ppy,
            "width": intr.width, "height": intr.height,
            "coeffs": list(intr.coeffs),
        }

    def _update_frames(self):
        failures = 0
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1200)
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                filtered = self.spatial_filter.process(depth_frame)
                depth_frame = filtered.as_depth_frame()
                color_frame.keep()
                depth_frame.keep()
                with self.lock:
                    self.latest_color_frame = color_frame
                    self.latest_depth_frame = depth_frame
                    self.latest_capture_time_ns = time.time_ns()
                failures = 0
                self._first_frame_event.set()
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                failures += 1
                self.last_error = str(exc)
                if failures >= 3:
                    logger.warning("RealSense frame read stopped: %s", exc)
                    break
                time.sleep(0.1)

    def get_frames(self):
        with self.lock:
            return self.latest_color_frame, self.latest_depth_frame

    def get_frame_bundle(self):
        with self.lock:
            return (self.latest_color_frame,
                    self.latest_depth_frame,
                    self.latest_capture_time_ns)

    def get_images(self, return_metadata=False):
        """返回彩色 BGR np.ndarray、对齐的 uint16 深度图；metadata 含时间与对齐深度内参。"""
        import numpy as np
        color_frame, depth_frame, timestamp_ns = self.get_frame_bundle()
        if color_frame is None or depth_frame is None:
            return (None, None, None) if return_metadata else (None, None)
        # 拷贝，避免在下一帧到达后底层缓冲被复用。
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        if not return_metadata:
            return color, depth
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        metadata = {
            "depth_frame": depth_frame,
            "depth_intrinsics": depth_intrinsics,
            "capture_time_ns": timestamp_ns,
        }
        return color, depth, metadata

    def get_real_position(self, u, v, window_size=9, depth_frame=None,
                          intrinsics=None, mask=None):
        """对齐深度图像素坐标 -> 相机坐标系米制 3D 坐标；采用邻域稳健中位数。"""
        import numpy as np
        if depth_frame is None:
            _, depth_frame = self.get_frames()
        if depth_frame is None:
            return None, None, None
        half = window_size // 2
        values = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px, py = int(u) + dx, int(v) + dy
                if not (0 <= px < self.width and 0 <= py < self.height):
                    continue
                if mask is not None and not mask[py, px]:
                    continue
                distance = depth_frame.get_distance(px, py)
                if 0.05 < distance < 10.0:
                    values.append(distance)
        if not values:
            return None, None, None
        values = np.asarray(values, dtype=np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad > 1e-6:
            inliers = values[np.abs(values - median) <= 2.5 * 1.4826 * mad]
            if len(inliers) >= max(5, len(values) // 3):
                values = inliers
        if intrinsics is None:
            intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        point = self._rs.rs2_deproject_pixel_to_point(
            intrinsics, [float(u), float(v)], float(np.median(values)))
        return tuple(point)

    def deproject_to_3d(self, u, v, depth_m):
        if depth_m <= 0 or self.color_intrinsics is None or self._rs is None:
            return None, None, None
        return tuple(self._rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, [float(u), float(v)], float(depth_m)))

    def get_masked_point_cloud(self, depth_image, mask, intrinsics,
                               max_points=2500, min_depth_m=0.05, max_depth_m=10.0):
        import numpy as np
        if depth_image is None or mask is None or intrinsics is None:
            return np.empty((0, 3), dtype=np.float64)
        depth_image = np.asarray(depth_image)
        mask = np.asarray(mask, dtype=bool)
        if depth_image.shape[:2] != mask.shape[:2]:
            raise ValueError("深度图与掩码尺寸不一致")
        depth_m = depth_image.astype(np.float64) * self.depth_scale
        valid = mask & np.isfinite(depth_m) & (depth_m > min_depth_m) & (depth_m < max_depth_m)
        rows, cols = np.nonzero(valid)
        if len(rows) == 0:
            return np.empty((0, 3), dtype=np.float64)
        if len(rows) > max_points:
            idx = np.linspace(0, len(rows)-1, max_points, dtype=np.int64)
            rows, cols = rows[idx], cols[idx]
        z = depth_m[rows, cols]
        coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)
        if np.all(np.abs(coeffs) < 1e-12):
            x = (cols - intrinsics.ppx) * z / intrinsics.fx
            y = (rows - intrinsics.ppy) * z / intrinsics.fy
            return np.column_stack((x, y, z))
        points = [self._rs.rs2_deproject_pixel_to_point(
            intrinsics, [float(c), float(r)], float(d))
            for r, c, d in zip(rows, cols, z)]
        return np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def get_point_cloud(self, depth_frame=None):
        if depth_frame is None:
            _, depth_frame = self.get_frames()
        if depth_frame is None:
            return None
        pc = self._rs.pointcloud()
        return pc.calculate(depth_frame)

    def get_color_intrinsics(self):
        intr = self.color_intrinsics
        if intr is None:
            raise RuntimeError("相机尚未启动")
        return intr.fx, intr.fy, intr.ppx, intr.ppy

    def get_depth_intrinsics(self):
        intr = self.depth_intrinsics
        if intr is None:
            raise RuntimeError("相机尚未启动")
        return intr.fx, intr.fy, intr.ppx, intr.ppy

    def get_color_intrinsic_matrix(self):
        import numpy as np
        fx, fy, cx, cy = self.get_color_intrinsics()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_color_distortion_coeffs(self):
        import numpy as np
        return np.asarray(self.color_intrinsics.coeffs, dtype=np.float64).reshape(1, 5)

    def save_images(self, save_path):
        """保存最近的彩色帧到本地文件，保留原驱动的方法名称。"""
        import cv2
        color, _ = self.get_images()
        if color is None:
            raise RuntimeError("尚无可保存的彩色帧")
        if not cv2.imwrite(str(save_path), color):
            raise RuntimeError(f"无法保存图片：{save_path}")

    def display_images(self):
        """本地桌面实时预览（不会通过飞书传输）；按 Q 退出。"""
        import cv2
        try:
            while True:
                color, depth = self.get_images()
                if color is not None and depth is not None:
                    depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                    cv2.imshow("RealSense - Color", color)
                    cv2.imshow("RealSense - Depth", depth_vis)
                if (cv2.waitKey(10) & 0xFF) in (ord("q"), 27):
                    break
                if self._stop_event.is_set():
                    break
        finally:
            cv2.destroyAllWindows()

    def stop(self):
        self._stop_event.set()
        worker = self.thread
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.5)
        self.thread = None
        if self.pipeline is not None and self.profile is not None:
            try:
                self.pipeline.stop()
            except Exception:
                logger.debug("忽略重复/异常关闭 RealSense pipeline", exc_info=True)
        self.profile = None
        self.pipeline = None
        with self.lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_capture_time_ns = None

    def __enter__(self):
        if not self.start():
            raise RuntimeError(self.last_error or "RealSense 启动失败")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()