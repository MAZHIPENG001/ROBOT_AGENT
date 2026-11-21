"""LIBERO 可选实时观察窗口（独立进程）。

只展示最新的观察帧，不控制 MuJoCo，也不推进仿真时间。
窗口崩溃不会影响环境 worker；GUI 需要本机桌面 DISPLAY。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法：viewer.py <live_frame.jpg> <ready.marker>")

    import cv2
    import numpy as np

    frame_path = Path(sys.argv[1])
    ready_path = Path(sys.argv[2])
    title = "LIBERO live view (close to hide)"
    placeholder = np.zeros((256, 480, 3), dtype=np.uint8)
    cv2.putText(placeholder, "LIBERO - waiting for reset", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
    latest = placeholder
    last_mtime = None

    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, latest)
        cv2.waitKey(1)
        # READY 文件只在 GUI 真正打开后创建；worker 检查这个握手。
        ready_path.write_text(str(os.getpid()), encoding="utf-8")

        while True:
            if frame_path.is_file():
                try:
                    mtime = frame_path.stat().st_mtime_ns
                    if mtime != last_mtime:
                        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                        if frame is not None:
                            latest = frame
                            last_mtime = mtime
                except OSError:
                    pass
            else:
                latest = placeholder
                last_mtime = None

            cv2.imshow(title, latest)
            # waitKey 必须持续执行，才能处理窗口事件。
            if cv2.waitKey(30) in (27, ord("q")):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        ready_path.unlink(missing_ok=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
