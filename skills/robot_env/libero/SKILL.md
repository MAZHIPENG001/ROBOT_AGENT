# LiberoSkill — LIBERO 仿真环境、画面预览与录像

## 职责和边界

本 Skill 只负责 **LIBERO 仿真环境**，包括启动服务、选择任务、初始化场景、读取观察、可选实时显示和录制 MP4。**不负责 VLA 推理、轨迹生成、动作规划或动作自动执行。**

LIBERO 使用独立的 `ROBOT_LIBERO_PYTHON` Python 环境；Agent 的 Python 不需要安装 LIBERO、MuJoCo 或 VLA。

## Agent 可调用的工具

- `robot_libero_start(show_window=False, display_camera="agentview")`：启动环境服务，不加载场景、不自动录视频。`show_window=True` 时尝试打开本机实时画面窗口，画面在后续 reset/step 时更新。
- `robot_libero_display(show_window=True, camera="agentview")`：在启动后打开/关闭本机实时窗口，不会改变仿真状态或启动录像。
- `robot_libero_status()`：查看服务状态、场景、步数、任务是否成功、窗口是否存活、录像中/最近视频路径。
- `robot_libero_list_tasks(task_suite_name="libero_spatial", start=0, limit=20)`：列出任务描述和 task_id。
- `robot_libero_reset(task_suite_name="libero_spatial", task_id=0, episode_idx=0, settle_steps=10)`：加载/重置场景，包含初始物体稳定步骤。若之前在录制会先保存旧视频。
- `robot_libero_observe(save_images=True)`：获得当前状态，按需保存双摄像头 PNG。**不会推进仿真**。
- `robot_libero_record_start(fps=10, camera="agentview")`：必须在 reset 后调用，开始录制 MP4；保存当前帧，之后每次底层 `step` 自动增加一帧。
- `robot_libero_record_stop()`：手动结束录像并保存 MP4，返回路径；成功/终止时可自动保存。
- `robot_libero_close()`：关闭当前场景，不停止 worker；如正在录像，先保存 MP4。
- `robot_libero_stop()`：停止 worker 及窗口进程；如正在录像，先保存 MP4。

相机选项：`agentview`、`eye_in_hand`、`side_by_side`（两个相机左右拼接）。录像文件默认输出到 `cache/libero/videos/`，帧率参数为**视频回放帧率**，不是强制机器人或环境按此速度执行。

## 重要使用规则

1. 启动时**默认不打开窗口、不录像**；用户明确要求实时显示时将 `show_window=True`，用户明确要求录像时先 reset 后调用 `robot_libero_record_start`。
2. 用户要求移动机器人、执行抓取、VLA 推理或生成动作时，交给**其他动作 Skill**；不要声称当前环境 Skill 可做动作任务。
3. `robot_libero_record_start` 之后不会凭空产生连续运动：帧只在初始化时和**每次底层 step(action) 后**采集。若没有动作 Skill 驱动环境，视频可能只有一帧。
4. 环境判定 `done=True` 或 `success=True` 时自动完成 MP4 保存；若动作 Skill 因最大步数/人为原因结束 rollout，必须主动调用 `robot_libero_record_stop`（或 close/reset）。
5. 无头机器/SSH 无有效本机图形显示时，使用 `show_window=False`；**无窗口仍支持保存录像**。实时窗口使用独立进程和 OpenCV GUI，需要可用的图形桌面。
6. 实时窗口展示的是**最新仿真观察帧**，不会让仿真自动运行或改变物理时间。保存 PNG 或实时显示与录像是三个独立功能。
7. `success` 与 `done` 分别记录，不能将任一未确认结果伪装成已成功执行任务；窗口、编码器或数据缺失时报告真实错误。

## 与后续动作 Skill 的接口（不注册给 DeepSeek）

```python
from skills.robot_env.libero.bridge import LiberoEnvClient

client = LiberoEnvClient()       # Unix socket 与 LIBERO worker 共享场景
obs = client.request("observe", include_images=True)
# action 由未来单独的 VLA/运动规划 Skill 计算
result = client.request("step", action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
```

此处 7 维动作只是接口示意，并非可完成任务的策略。`step` 只有底层 JSON API，不会被注册为环境 Skill 的 LLM Tool；当录像激活时，每次 `step` 会自动记录画面，因此动作 Skill 无需处理视频帧。
