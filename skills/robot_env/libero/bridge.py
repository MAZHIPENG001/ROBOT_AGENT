"""LIBERO 本地 JSON-over-UNIX-socket 客户端与独立 Python worker 生命周期。

这个模块只依赖 Python 标准库，供环境工具及控制适配器共用：
    from skills.robot_env.libero.bridge import LiberoEnvClient
    client = LiberoEnvClient()
    obs = client.request("observe", include_images=True)
    result = client.request("step", action=[0, 0, 0, 0, 0, 0, -1])

注意：底层 step 接口不属于本 Skill 注册给 LLM 的 Tools。
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("worker.py")


def _output_dir() -> Path:
    return Path(os.getenv("ROBOT_LIBERO_OUTPUT_DIR", str(PROJECT_ROOT / "cache" / "libero"))).expanduser().resolve()


class LiberoEnvClient:
    """跨 Python 环境连接 LIBERO；只要知道 socket 路径即可共享同一场景。"""

    def __init__(self, socket_path: str | Path | None = None, timeout: float | None = None):
        self.socket_path = Path(socket_path) if socket_path else _output_dir() / "libero.sock"
        self.timeout = float(timeout if timeout is not None else os.getenv("ROBOT_LIBERO_TIMEOUT", "120"))
        if self.timeout <= 0:
            raise ValueError("LIBERO 请求超时时间必须大于 0")

    def request(self, command: str, **params: Any) -> dict[str, Any]:
        """使用一条 JSON 请求执行一个命令，并获得字典结果。"""
        payload = (json.dumps({"command": command, "params": params}, ensure_ascii=False) + "\n").encode("utf-8")
        if len(payload) > 1_000_000:
            raise ValueError("LIBERO 请求过大")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(self.timeout)
                conn.connect(str(self.socket_path))
                conn.sendall(payload)
                with conn.makefile("rb") as inp:
                    line = inp.readline(25_000_000)
        except (OSError, TimeoutError) as exc:
            raise RuntimeError(f"LIBERO worker 不可用或请求超时（socket={self.socket_path}）：{exc}") from exc
        if not line or not line.endswith(b"\n"):
            raise RuntimeError("LIBERO worker 无有效 JSON 响应；请查看 worker.log")
        try:
            message = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("LIBERO worker 返回非 JSON 数据") from exc
        if not isinstance(message, dict) or "ok" not in message:
            raise RuntimeError("LIBERO worker 响应格式错误")
        if not message["ok"]:
            raise RuntimeError(str(message.get("error", "未知仿真异常")))
        result = message.get("result", {})
        return result if isinstance(result, dict) else {"data": result}

    def ping(self) -> dict[str, Any] | None:
        try:
            result = self.request("status")
            return result if result.get("service") == "libero_env" else None
        except RuntimeError:
            return None


class LiberoBridge:
    """仅负责 worker 启停与基础请求；Agent 的 Python 不会 import LIBERO。"""

    def __init__(self):
        self.python = os.getenv("ROBOT_LIBERO_PYTHON", "").strip()
        self.cwd = os.getenv("ROBOT_LIBERO_WORKDIR", "").strip()
        self.output_dir = _output_dir()
        self.client = LiberoEnvClient()
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        atexit.register(self._stop_own_process)

    def is_running(self) -> bool:
        return self.client.ping() is not None

    def start(self, show_window: bool = False, display_camera: str = "agentview") -> dict[str, Any]:
        """启动服务，并按需打开本机实时观察窗口；不会自动加载任务或录像。"""
        with self._lock:
            old = self.client.ping()
            if old is not None:
                display_status = self.client.request("display", show_window=show_window, camera=display_camera)
                return {"started": False, "reused": True, **display_status}
            if not self.python:
                raise RuntimeError("未配置 ROBOT_LIBERO_PYTHON，请设置为 LIBERO 环境的 python 绝对路径")
            
            python = Path(self.python).expanduser().absolute()
            if not python.is_file():
                raise FileNotFoundError(f"LIBERO Python 不存在：{python}")
            if self.cwd and not Path(self.cwd).expanduser().is_dir():
                raise FileNotFoundError(f"ROBOT_LIBERO_WORKDIR 不存在：{self.cwd}")

            self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.output_dir, 0o700)
            env = os.environ.copy()
            # 隔离 agent venv；不改变指定 LIBERO Python 本身的 site-packages。
            for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
                env.pop(key, None)

            # 有些 LIBERO editable 安装只写入元数据，无法自动找到源码。
            # 在 LIBERO worker 中单独设置源码根目录，不污染 Agent 的 Python 环境。
            # 默认复用 ROBOT_LIBERO_WORKDIR；如工作目录不是源码根目录，可单独配置。
            source_root = os.getenv("ROBOT_LIBERO_SOURCE_ROOT", "").strip() or self.cwd
            if source_root:
                source_path = Path(source_root).expanduser().resolve()
                if not source_path.is_dir():
                    raise FileNotFoundError(f"LIBERO 源码目录不存在：{source_path}")
                env["PYTHONPATH"] = str(source_path)
                env["ROBOT_LIBERO_SOURCE_ROOT"] = str(source_path)

            env["ROBOT_LIBERO_OUTPUT_DIR"] = str(self.output_dir)
            env["PYTHONUNBUFFERED"] = "1"

            log_path = self.output_dir / "worker.log"
            with log_path.open("ab") as log:
                self._process = subprocess.Popen(
                    [str(python), "-u", str(WORKER)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    cwd=str(Path(self.cwd).expanduser()) if self.cwd else str(PROJECT_ROOT),
                    env=env,
                    start_new_session=True,
                )
            # worker 运行准备时间与单次仿真请求超时是两个不同的参数。
            startup_timeout = float(os.getenv("ROBOT_LIBERO_START_TIMEOUT", "20"))
            if startup_timeout <= 0:
                raise ValueError("ROBOT_LIBERO_START_TIMEOUT 必须大于 0")
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(f"LIBERO worker 退出（code={self._process.returncode}）；请查看 {log_path}")
                ready = self.client.ping()
                if ready is not None:
                    if show_window:
                        ready = self.client.request("display", show_window=True, camera=display_camera)
                    return {"started": True, "reused": False, **ready}
                time.sleep(0.1)
            self._stop_own_process()
            raise TimeoutError(f"等待 LIBERO worker 启动超时，请查看 {log_path}")

    def request(self, command: str, **params: Any) -> dict[str, Any]:
        # 非状态查询操作默认启动 worker，然后再执行命令。
        if not self.is_running():
            if command in ("status", "close"):
                return {"service": "libero_env", "running": False, "scene_loaded": False}
            self.start()
        return self.client.request(command, **params)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self.is_running():
                self._stop_own_process()
                return {"stopped": True, "was_running": False}
            result = self.client.request("shutdown")
            if self._process is not None:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._stop_own_process()
            self._process = None
            return result

    def _stop_own_process(self):
        # 只清理这个 Bridge 自己创建的子进程，避免误杀其他 Bridge 的 worker。
        with self._lock:
            process = self._process
            self._process = None
            if process is None or process.poll() is not None:
                return
            try:
                self.client.request("shutdown")
                process.wait(timeout=3)
            except (RuntimeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
