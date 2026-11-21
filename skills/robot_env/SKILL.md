# robot_env — 仿真环境/真机管理入口

当前已注册的环境实现：`robot_env/libero/`（导出 `LiberoSkill`）。这个一级目录文档供 `agent.py` 的 `load_skill_instructions()` 读取；原先 `libero/SKILL.md` 位于二级目录，因此不会被当前扫描逻辑自动包含。

## 已注册工具（LIBERO）

- `robot_libero_start` / `robot_libero_stop`：管理独立 LIBERO worker。
- `robot_libero_list_tasks`：列出任务描述和任务 ID。
- `robot_libero_reset`：加载任务并完成初始稳定步骤。
- `robot_libero_status` / `robot_libero_observe`：获取状态与 PNG 观测。
- `robot_libero_display`：打开本地预览窗口。
- `robot_libero_record_start` / `robot_libero_record_stop`：可选 MP4 录制。
- `robot_libero_close`：关闭当前仿真场景。

## 与 robot_control 的分工

这里只负责环境生命周期、观测、视频和底层动作执行；**Agent 不直接调用底层 step**。执行任务时先 reset，再调用 `robot_control_predict_once` 测试策略连通性，最后调用 `robot_control_start`。任务执行状态通过 `robot_control_status` 查看；停止通过 `robot_control_stop`，终止或结束后由环境 Skill 按需保存录像。不要在任务运行期间重置场景。

详细的 LIBERO 使用约束和参数说明仍位于 `robot_env/libero/SKILL.md`。