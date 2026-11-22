# robot_env — 环境生命周期、观测与录像

当前实现为 LIBERO，使用 `ROBOT_LIBERO_PYTHON` 指定的独立 Python 环境。
本 Skill 管理场景、观测、实时预览和 MP4；策略推理与任务执行交给 `robot_control`。
底层 `step` 由控制适配器调用，不注册为 Agent 工具。

## LIBERO 工具

- `robot_libero_start(show_window=False, display_camera="agentview")`：启动 worker，不加载场景、不自动录像。
- `robot_libero_list_tasks(task_suite_name="libero_spatial", start=0, limit=20)`：列出任务描述与 ID；任务集支持 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90`。
- `robot_libero_reset(task_suite_name="libero_spatial", task_id=0, episode_idx=0, settle_steps=10)`：加载任务并等待物体稳定；若正在录制，先保存旧视频。
- `robot_libero_status()`：查看服务、任务、步数、成功标记、窗口和录像状态。
- `robot_libero_observe(save_images=True)`：读取状态，按需保存双摄像头 PNG，不推进仿真。
- `robot_libero_display(show_window=True, camera="agentview")`：切换本机预览窗口，不改变仿真状态、不开始录像。
- `robot_libero_record_start(fps=10, camera="agentview")`：reset 后开始录制 MP4。
- `robot_libero_record_stop()`：结束录像并返回保存结果。
- `robot_libero_close()`：关闭当前场景，保留 worker。
- `robot_libero_stop()`：停止 worker 和窗口。

## 使用约束

- 用户要求实时显示时才打开窗口，要求录像时才在 reset 后开始录制。相机支持 `agentview`、`eye_in_hand`、`side_by_side`（双相机并排）。
- 窗口需要本机图形桌面与支持 GUI 的 OpenCV；无头或 SSH 环境可保存 PNG / MP4。窗口显示最新观察帧，不提供网络直播，也不会自动推进仿真。
- 录像开始时保存当前帧，此后每次底层 step 增加一帧；没有动作时可能只有一帧。`fps` 是回放帧率，不是环境执行频率。
- 环境报告 done/success 或 close/reset/stop 时自动保存录像；控制循环达到最大步数或被人为停止后，仍需调用 `robot_libero_record_stop`。视频默认位于 `cache/libero/videos/`。
- 控制运行期间禁止 reset 或并行 step；切换场景前先停止控制并确认线程已结束。成功与终止分别以实际返回的 `success`、`done` 为准。

环境安装与排障见 [LIBERO 环境准备](libero/README.md)。
