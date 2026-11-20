# RealSenseCameraSkill

## 功能

- `realsense_camera_status`：查询 D435 或其他 RealSense 设备是否接入。
- `realsense_camera_capture`：按用户要求启动 RGB-D 流，抓取当前彩色 BGR 帧和对齐后的 16 位深度帧，可选只保留彩色图；完成后关闭设备。
- 深度图与彩色图对齐，另外保存 `depth_scale_m_per_unit` 和对齐深度内参 JSON；深度图 PNG 像素不是米，须乘深度标尺。

## 使用范围

- `pyrealsense2` 只有实际使用相机时才需要；未安装或未插入设备不影响 Agent 启动，也不影响其他 Skill。
- 分辨率和帧率默认 `640x480@30`，可通过环境变量配置；设备不支持该组合时会明确返回失败。
- 图片仅存本地 `CAMERA_OUTPUT_DIR/realsense/`，不自动识别物体、不自动发图片到飞书。
- 其他程序可从 `skills.realsense_camera.camera` 导入 `RealSenseCamera`，使用 `start/get_images/stop`、`get_real_position` 和点云接口。
