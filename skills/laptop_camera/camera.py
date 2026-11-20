"""笔记本摄像头驱动，支持单帧采集与本地预览。

单独运行预览：python -m skills.laptop_camera.camera
"""

import argparse
import sys

import cv2


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def camera_index(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("摄像头编号不能小于 0")
    return number


class LaptopCamera:
    """笔记本摄像头，支持独立读取帧、窗口预览和 with 自动释放。"""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = False,
    ):
        if index < 0 or width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("摄像头编号不能小于 0，宽度、高度和帧率必须大于 0")
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self._capture = None

    def open(self) -> "LaptopCamera":
        """打开摄像头；重复调用不会重复打开设备。"""
        if self._capture is not None and self._capture.isOpened():
            return self
        self.close()
        self._capture = cv2.VideoCapture(self.index)
        try:
            if not self._capture.isOpened():
                raise RuntimeError(
                    f"无法打开摄像头 {self.index}。请检查摄像头权限、设备是否被占用，"
                    "或用 --index 1 尝试其他摄像头。"
                )
            # 摄像头会根据硬件支持情况选择实际分辨率和帧率。
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        """返回一帧 BGR 图像（numpy 数组）；未打开或读取失败时抛出异常。"""
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("摄像头尚未打开，请先调用 open() 或使用 with 语句。")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("读取摄像头画面失败，请检查设备连接。")
        return cv2.flip(frame, 1) if self.mirror else frame

    def close(self) -> None:
        """释放摄像头；允许重复调用。"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "LaptopCamera":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def preview(self) -> None:
        """打开摄像头并预览，退出或发生异常时自动释放设备和预览窗口。"""
        window_name = "Laptop Camera - Q / Esc to quit"
        window_created = False
        try:
            self.open()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
            print("摄像头已打开，按 Q、Esc 或关闭窗口退出；Ctrl+C 也可退出。")

            while True:
                cv2.imshow(window_name, self.read())
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.close()
            if window_created:
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    # 用户通过窗口关闭按钮退出时，窗口可能已被后端销毁。
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 OpenCV 显示笔记本摄像头画面")
    parser.add_argument("--index", type=camera_index, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--width", type=positive_int, default=640, help="画面宽度，默认 640")
    parser.add_argument("--height", type=positive_int, default=480, help="画面高度，默认 480")
    parser.add_argument("--fps", type=positive_int, default=30, help="请求帧率，默认 30")
    parser.add_argument("--mirror", action="store_true", help="水平镜像显示")
    args = parser.parse_args()

    try:
        camera = LaptopCamera(args.index, args.width, args.height, args.fps, args.mirror)
        camera.preview()
        return 0
    except KeyboardInterrupt:
        return 0
    except cv2.error as exc:
        print(f"OpenCV 错误：{exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())