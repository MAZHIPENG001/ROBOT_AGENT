"""Independent, persistent SchedulerSkill for LangChain (APScheduler 3.x).

Responsibilities: create, store, fire, list, and cancel scheduled tasks.
It knows nothing about Feishu, chats, channels, or the Agent itself.

A module-level job entry point is required by APScheduler's persistent job store.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor as SchedulerExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "cache" / "scheduler" / "jobs.sqlite3"
MAX_JOBS = 100
MAX_INSTRUCTION_LENGTH = 1000
MAX_SECONDS = 365 * 24 * 3600
MAX_INTERVAL_SECONDS = 7 * 24 * 3600

# A persistent job must refer to an importable top-level function, NOT a
# nested closure or bound instance method. main.py binds its callback at start.
_execute_handler: Callable[[str, str], None] | None = None
_execute_lock = Lock()


def run_scheduled_job(job_id: str, instruction: str, **_legacy_kwargs) -> None:
    """APScheduler job entry point: execute a task via the registered callback.

    Ignore legacy job kwargs when loading older persisted APScheduler jobs.
    This does NOT preserve their original notification destination.
    """
    with _execute_lock:
        callback = _execute_handler
    if callback is None:
        logger.error("定时任务 %s 未执行：尚未注册执行回调", job_id)
        return
    try:
        callback(job_id, instruction)
    except Exception:
        logger.exception("定时任务 %s 执行异常", job_id)


class SchedulerSkill:
    """Independent scheduled task manager. Minimum interval/delay: 1 second."""

    def __init__(self) -> None:
        self.timezone = ZoneInfo(os.getenv("SCHEDULER_TIMEZONE", "Asia/Shanghai"))
        db_path = Path(os.getenv("SCHEDULER_DB_PATH", str(DEFAULT_DB))).expanduser()
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scheduler = BackgroundScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{self.db_path}")},
            executors={"default": SchedulerExecutor(max_workers=4)},
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 30,
            },
            timezone=self.timezone,
        )

    def start(self, on_execute: Callable[[str, str], None]) -> None:
        """Start scheduler and bind callback(job_id, instruction)."""
        if self.scheduler.running:
            raise RuntimeError("SchedulerSkill 已启动")
        if not callable(on_execute):
            raise TypeError("on_execute 必须是可调用函数")
        global _execute_handler
        with _execute_lock:
            _execute_handler = on_execute
        try:
            self.scheduler.start()
        except Exception:
            with _execute_lock:
                _execute_handler = None
            raise
        logger.info("定时器启动：%s（时区 %s）", self.db_path, self.timezone)

    def stop(self) -> None:
        """Stop scheduling (persisted pending jobs remain in SQLite)."""
        global _execute_handler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        with _execute_lock:
            _execute_handler = None

    def _create(self, instruction: str, trigger) -> str:
        if not self.scheduler.running:
            return "创建失败：调度器未启动，请先在 main.py 调用 scheduler.start(on_execute)。"
        if not isinstance(instruction, str) or not (1 <= len(instruction.strip()) <= MAX_INSTRUCTION_LENGTH):
            return "创建失败：任务内容必须为 1～1000 个字符。"
        if len(self.scheduler.get_jobs()) >= MAX_JOBS:
            return f"创建失败：最多允许 {MAX_JOBS} 个定时任务。"
        job_id = "task_" + uuid.uuid4().hex[:12]
        try:
            job = self.scheduler.add_job(
                run_scheduled_job,
                trigger=trigger,
                id=job_id,
                name=instruction.strip()[:64],
                kwargs={"job_id": job_id, "instruction": instruction.strip()},
                replace_existing=False,
            )
            when = job.next_run_time.astimezone(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
            return f"任务创建成功：ID={job_id}；下次执行={when}；内容={instruction.strip()}"
        except Exception as exc:
            logger.exception("创建定时任务失败")
            return f"创建失败：{type(exc).__name__}: {exc}"

    def tools(self) -> list:
        """Expose scheduler operations to LangChain Agent."""

        @tool
        def schedule_once(
            instruction: str,
            delay_seconds: int = 0,
            run_at: str = "",
        ) -> str:
            """创建一次性任务：delay_seconds=多少秒后（至少1秒），或 run_at=当地时间 YYYY-MM-DD HH:MM:SS / ISO8601；二选一。instruction 是到点执行的具体任务。"""
            has_delay = delay_seconds != 0
            has_time = bool(run_at.strip())
            if has_delay == has_time:
                return "创建失败：delay_seconds 与 run_at 必须二选一。"
            now = datetime.now(self.timezone)
            if has_delay:
                if not 1 <= delay_seconds <= MAX_SECONDS:
                    return "创建失败：delay_seconds 必须在 1 秒到 365 天之间。"
                target = now + timedelta(seconds=delay_seconds)
            else:
                try:
                    target = datetime.fromisoformat(run_at.strip())
                except ValueError:
                    return "创建失败：run_at 必须为 YYYY-MM-DD HH:MM:SS 或 ISO8601。"
                if target.tzinfo is None:
                    target = target.replace(tzinfo=self.timezone)
                else:
                    target = target.astimezone(self.timezone)
                if not (now < target <= now + timedelta(seconds=MAX_SECONDS)):
                    return "创建失败：执行时间必须在未来一年内。"
            return self._create(instruction, DateTrigger(run_date=target, timezone=self.timezone))

        @tool
        def schedule_interval(
            instruction: str,
            every_seconds: int,
            duration_seconds: int = 0,
        ) -> str:
            """创建秒级周期任务。every_seconds 是执行间隔（最短1秒），首次在一个间隔后执行。duration_seconds 是总持续秒数；0 为无限重复。例：未来5分钟每分钟提醒：every_seconds=60,duration_seconds=300。"""
            if not 1 <= every_seconds <= MAX_INTERVAL_SECONDS:
                return f"创建失败：every_seconds 必须在 1～{MAX_INTERVAL_SECONDS} 秒之间。"
            if not 0 <= duration_seconds <= MAX_SECONDS:
                return "创建失败：duration_seconds 必须在 0～31536000 秒之间。"
            if duration_seconds and duration_seconds < every_seconds:
                return "创建失败：总持续时间不能小于执行间隔。"

            now = datetime.now(self.timezone)
            first = now + timedelta(seconds=every_seconds)
            end = now + timedelta(seconds=duration_seconds) if duration_seconds else None
            trigger = IntervalTrigger(
                seconds=every_seconds,
                start_date=first,
                end_date=end,
                timezone=self.timezone,
            )
            result = self._create(instruction, trigger)
            if result.startswith("任务创建成功"):
                if duration_seconds:
                    result += f"；持续={duration_seconds}秒；计划触发={duration_seconds // every_seconds}次，到期自动结束"
                else:
                    result += "；无限重复，需要主动取消"
            return result

        @tool
        def schedule_daily(
            instruction: str,
            hour: int,
            minute: int = 0,
            second: int = 0,
        ) -> str:
            """每天当地时间 hour:minute:second 执行任务；hour 0~23、minute/second 0~59，支持秒精度。"""
            if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                return "创建失败：hour、minute 或 second 不合法。"
            return self._create(
                instruction,
                CronTrigger(hour=hour, minute=minute, second=second, timezone=self.timezone),
            )

        @tool
        def schedule_list() -> str:
            """列出当前系统保存的定时任务，包含任务 ID、下次运行时间和任务内容。"""
            if not self.scheduler.running:
                return "查询失败：调度器未启动。"
            jobs = sorted(
                self.scheduler.get_jobs(),
                key=lambda job: job.next_run_time or datetime.max.replace(tzinfo=self.timezone),
            )
            if not jobs:
                return "目前没有定时任务。"
            lines = [f"当前共 {len(jobs)} 个任务（时区：{self.timezone}）："]
            for job in jobs:
                when = (
                    job.next_run_time.astimezone(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
                    if job.next_run_time else "暂停"
                )
                lines.append(f"{job.id} | 下次执行：{when} | {job.name}")
            return "\n".join(lines)

        @tool
        def schedule_cancel(job_id: str) -> str:
            """取消指定 ID 的定时任务；例如 task_123456789abc。已开始执行的任务不会被中断。"""
            if not self.scheduler.running:
                return "取消失败：调度器未启动。"
            try:
                job = self.scheduler.get_job(job_id)
                if job is None:
                    return f"取消失败：找不到任务 {job_id}。"
                self.scheduler.remove_job(job_id)
                return f"已取消任务：{job_id}"
            except Exception as exc:
                logger.exception("取消定时任务失败")
                return f"取消失败：{type(exc).__name__}: {exc}"

        return [schedule_once, schedule_interval, schedule_daily, schedule_list, schedule_cancel]