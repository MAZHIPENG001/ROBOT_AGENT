# LaptopCameraSkill

## 功能

- `laptop_camera_status`：检查笔记本内置/普通 USB 摄像头是否可打开。
- `laptop_camera_capture`：按用户要求拍摄一张当前原始彩色图（BGR 数据保存为 PNG），保存到 Agent 所在电脑。

## 使用范围

- 使用 OpenCV `VideoCapture`，默认索引 `0`；可通过 `LAPTOP_CAMERA_INDEX` 修改。
- 不提供物体识别、录像、远程视频流或发送飞书图片能力。
- 只有工具被实际调用时才会打开摄像头；采集完成立即释放。
- 本地存储路径在 `CAMERA_OUTPUT_DIR/laptop/`。
