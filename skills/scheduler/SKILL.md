# Scheduler Skill

## 技能描述

独立定时任务技能。使用 APScheduler 3.x 和 SQLite 持久化任务。
负责创建任务、定时触发、列出任务以及取消任务；不负责执行任务内容，也不直接依赖飞书、聊天或 Agent。

## 工具

- `schedule_once(instruction, delay_seconds=0, run_at="")`：一次性任务。`delay_seconds` 和 `run_at` 二选一；最短延迟 1 秒。`run_at` 使用 `YYYY-MM-DD HH:MM:SS` 或 ISO8601。
- `schedule_interval(instruction, every_seconds, duration_seconds=0)`：每隔指定秒数执行；最短间隔 1 秒。首次执行发生在创建一个周期之后。`duration_seconds=0` 代表无限重复；非零表示在指定时间窗口内自动结束。
- `schedule_daily(instruction, hour, minute=0, second=0)`：每日固定时分秒触发。
- `schedule_list()`：查看全部未完成/未取消任务及下次运行时间。
- `schedule_cancel(job_id)`：取消指定任务；已开始执行的任务不会被中断。

## 使用规则

1. 用户请求定时提醒、延后执行或者周期任务时，使用上述工具。
2. 将要执行的具体动作写入 `instruction`，不提前伪造已经执行的结果。
3. 所有时间参数单位都是**秒**。
4. 例如“未来 5 分钟，每隔 1 分钟提醒喝水”应调用 `schedule_interval(instruction="提醒喝水，附带执行时的当前时间", every_seconds=60, duration_seconds=300)`。
5. SchedulerSkill 不发送消息。真正执行时由主程序注入 `on_execute(job_id, instruction)` 回调。
6. 实际执行可能受机器休眠、网络、模型推理耗时影响；一秒是调度间隔最小粒度，不保证 Agent 每秒完成一次任务。
7. 取消和查询任务不需要飞书 `chat_id` 参数。
8. 定时任务默认不具备删除文件、执行任意命令或控制机器人的权限；主程序负责限制后台 Agent 可用工具。

## 持久化与局限

- 任务保存到 `SCHEDULER_DB_PATH`，默认 `./cache/scheduler/jobs.sqlite3`（路径相对于项目根目录）。
- 重启时恢复尚未到期任务；程序关闭期间无法实时触发。
- APScheduler 3.x 不支持多个独立进程安全共享同一个持久化 job store；同一数据库只能运行一个 Scheduler 实例。
- SchedulerSkill 不知道用户身份、飞书会话和通知地址；这些由主程序管理。