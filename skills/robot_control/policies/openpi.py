"""OpenPI 策略：单文件双运行模式。

Agent 导入 OpenPiPolicy：负责统一策略接口和子进程 JSON 通信。
指定的 ROBOT_OPENPI_PYTHON 执行本文件：负责 SSH 隧道及 OpenPI WebSocket 推理。
进程/环境依旧隔离，但无需另一个 openpi_worker.py 文件。
这里仅做 Observation -> ActionChunk，不操作机器人。
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# 直接运行本文件时，它是独立 worker 而非 Python 包；不能做相对导入。
if __name__ != "__main__":
    from ..types import ActionChunk, Observation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKER_SCRIPT = Path(__file__).resolve()


class OpenPiPolicy:
    """同一控制任务使用一个持久的独立进程，不每步重启解释器。"""

    def __init__(self) -> None:
        python_raw = os.getenv("ROBOT_OPENPI_PYTHON", "").strip()
        python = Path(python_raw).expanduser().resolve() if python_raw else Path(sys.executable)
        if not python.is_file():
            raise FileNotFoundError(f"OpenPI Python 不存在：{python}")

        self.action_space = os.getenv(
            "ROBOT_OPENPI_ACTION_SPACE", "libero_delta_pose_axisangle_gripper_7d"
        ).strip()
        if not self.action_space:
            raise ValueError("ROBOT_OPENPI_ACTION_SPACE 不能为空")
        self.request_timeout = float(os.getenv("ROBOT_OPENPI_REQUEST_TIMEOUT", "60"))
        self.start_timeout = float(os.getenv("ROBOT_OPENPI_START_TIMEOUT", "20"))
        if self.request_timeout <= 0 or self.start_timeout <= 0:
            raise ValueError("OpenPI 超时时间必须大于 0")

        log_dir = Path(os.getenv("ROBOT_OPENPI_OUTPUT_DIR", str(PROJECT_ROOT / "cache/openpi")))
        log_dir = log_dir.expanduser()
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_path = log_dir / "worker.log"
        self._lock = threading.RLock()
        self._stdout_buffer = bytearray()
        self._proc: subprocess.Popen | None = None

        # 避免 Agent 的 venv/conda 环境变量污染独立解释器，保留 OPENPI_* 配置。
        env = os.environ.copy()
        for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
            env.pop(key, None)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("ab") as log:
            self._proc = subprocess.Popen(
                [str(python), "-u", str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        try:
            info = self._request("init", timeout=self.start_timeout)
            if info.get("action_space") != self.action_space:
                raise RuntimeError("OpenPI worker 动作空间声明不一致")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _encode_image(array: np.ndarray) -> str:
        rgb = np.asarray(array)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("OpenPI 输入图像应为 H×W×3 RGB")
        if rgb.dtype != np.uint8:
            raise ValueError("OpenPI 输入图像应为 uint8")
        buffer = BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _readline(self, timeout: float) -> bytes:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("OpenPI worker 未运行")
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" in self._stdout_buffer:
                line, _, remaining = self._stdout_buffer.partition(b"\n")
                self._stdout_buffer = bytearray(remaining)
                return bytes(line)
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise TimeoutError(f"OpenPI worker 响应超时（{timeout}s）")
            readable, _, _ = select.select([proc.stdout.fileno()], [], [], remaining_time)
            if not readable:
                raise TimeoutError(f"OpenPI worker 响应超时（{timeout}s）")
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(
                    f"OpenPI worker 已退出（exit={proc.poll()}）；请查看 cache/openpi/worker.log"
                )
            self._stdout_buffer.extend(chunk)
            if len(self._stdout_buffer) > 30_000_000:
                raise RuntimeError("OpenPI worker 响应过大")

    def _request(self, command: str, *, timeout: float, **kwargs) -> dict:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("OpenPI worker 未运行")
        with self._lock:
            if proc.poll() is not None:
                raise RuntimeError(f"OpenPI worker 已退出（exit={proc.returncode}）")
            payload = json.dumps({"command": command, **kwargs}, ensure_ascii=False)
            try:
                proc.stdin.write((payload + "\n").encode("utf-8"))
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("无法向 OpenPI worker 发送请求") from exc
            raw = self._readline(timeout)
            try:
                result = json.loads(raw)
            except ValueError as exc:
                raise RuntimeError("OpenPI worker 返回无效 JSON（检查是否向 stdout 写入了调试日志）") from exc
            if not isinstance(result, dict) or "ok" not in result:
                raise RuntimeError("OpenPI worker 响应格式错误")
            if not result["ok"]:
                raise RuntimeError(str(result.get("error", "OpenPI worker 执行失败")))
            return result

    def predict(self, observation: Observation) -> ActionChunk:
        images = observation.images
        if "primary" not in images or "wrist" not in images:
            raise ValueError("OpenPI 需要 primary 和 wrist 两路 RGB 图像")
        state = np.asarray(observation.state, dtype=np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("OpenPI observation.state 必须是有限的一维数组")
        result = self._request(
            "predict", timeout=self.request_timeout,
            observation={
                "images": {
                    "primary": self._encode_image(images["primary"]),
                    "wrist": self._encode_image(images["wrist"]),
                },
                "state": state.tolist(),
                "instruction": observation.instruction,
            },
        )
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError(f"OpenPI 返回无效动作块，shape={actions.shape}")
        return ActionChunk(
            actions=actions,
            action_space=self.action_space,
            metadata={"policy": "openpi", "horizon": len(actions)},
        )

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self._request("close", timeout=2)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
        finally:
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    stream.close()
            self._proc = None


# ---------------------------------------------------------------
# 以下为独立 Python 执行本文件时的推理进程，OpenPiPolicy 不直接调用。
# ---------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    text = os.getenv(name, "true" if default else "false").strip().lower()
    if text in ("true", "yes", "1", "on"):
        return True
    if text in ("false", "no", "0", "off"):
        return False
    raise ValueError(f"{name} 必须为 true/false")


def _port_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} 必须在 1~65535 范围内")
    return value


@dataclass(frozen=True)
class OpenPiConfig:
    host: str
    port: int
    resize_size: int
    action_space: str
    ssh_enabled: bool
    ssh_host: str
    ssh_user: str
    ssh_port: int
    ssh_identity_file: str
    ssh_remote_host: str
    ssh_remote_port: int
    ssh_password: str = field(default="", repr=False)
    ssh_auth_method: str = "auto"
    ssh_host_key_policy: str = "strict"

    @classmethod
    def from_env(cls) -> "OpenPiConfig":
        host = os.getenv("ROBOT_OPENPI_HOST", "127.0.0.1").strip()
        port = _port_env("ROBOT_OPENPI_PORT", 8000)
        resize_size = int(os.getenv("ROBOT_OPENPI_RESIZE_SIZE", "224"))
        if not 32 <= resize_size <= 1024:
            raise ValueError("ROBOT_OPENPI_RESIZE_SIZE 必须在 32~1024 范围内")
        if not host:
            raise ValueError("ROBOT_OPENPI_HOST 不能为空")
        ssh_enabled = _bool_env("ROBOT_OPENPI_SSH_ENABLED")
        if ssh_enabled and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("自动 SSH 隧道模式要求 ROBOT_OPENPI_HOST 为回环地址")
        ssh_host = os.getenv("ROBOT_OPENPI_SSH_HOST", "").strip()
        ssh_user = os.getenv("ROBOT_OPENPI_SSH_USER", "").strip()
        # 限制 ssh 目标字符串，防止配置错误变成命令行选项。
        if ssh_enabled:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ssh_host):
                raise ValueError("ROBOT_OPENPI_SSH_HOST 必须是有效的 IPv4/主机名")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", ssh_user):
                raise ValueError("ROBOT_OPENPI_SSH_USER 格式不正确")
        action_space = os.getenv("ROBOT_OPENPI_ACTION_SPACE", "libero_delta_pose_axisangle_gripper_7d").strip()
        if not action_space:
            raise ValueError("ROBOT_OPENPI_ACTION_SPACE 不能为空")
        remote_host = os.getenv("ROBOT_OPENPI_SSH_REMOTE_HOST", "127.0.0.1").strip()
        if not remote_host or any(ch.isspace() for ch in remote_host):
            raise ValueError("ROBOT_OPENPI_SSH_REMOTE_HOST 无效")
        auth_method = os.getenv("ROBOT_OPENPI_SSH_AUTH_METHOD", "auto").strip().lower()
        if auth_method not in {"auto", "password", "key"}:
            raise ValueError("ROBOT_OPENPI_SSH_AUTH_METHOD 应为 auto/password/key")
        password = os.getenv("ROBOT_OPENPI_SSH_PASSWORD", "")  # 保留密码原有的空格字符
        if ssh_enabled and auth_method == "password" and not password:
            raise ValueError("密码认证需要设置 ROBOT_OPENPI_SSH_PASSWORD")
        host_key_policy = os.getenv("ROBOT_OPENPI_SSH_HOST_KEY_POLICY", "strict").strip().lower()
        if host_key_policy not in {"strict", "accept-new"}:
            raise ValueError("ROBOT_OPENPI_SSH_HOST_KEY_POLICY 应为 strict/accept-new")
        return cls(
            host=host, port=port, resize_size=resize_size,
            action_space=action_space, ssh_enabled=ssh_enabled,
            ssh_host=ssh_host, ssh_user=ssh_user,
            ssh_port=_port_env("ROBOT_OPENPI_SSH_PORT", 22),
            ssh_identity_file=os.getenv("ROBOT_OPENPI_SSH_IDENTITY_FILE", "").strip(),
            ssh_remote_host=remote_host,
            ssh_remote_port=_port_env("ROBOT_OPENPI_SSH_REMOTE_PORT", 8000),
            ssh_password=password, ssh_auth_method=auth_method,
            ssh_host_key_policy=host_key_policy,
        )


class OpenPiWorker:
    def __init__(self, cfg: OpenPiConfig) -> None:
        self.cfg = cfg
        self.ssh = None  # paramiko.SSHClient（仅在独立 worker 中导入）
        self.forward_server = None  # 监听本机 loopback 的 TCP 转发服务
        self.forward_thread = None
        self.client = None
        self.image_tools = None

    def _open_tunnel(self) -> None:
        if not self.cfg.ssh_enabled:
            return  # 使用现有通道或直连

        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI Python 环境缺少 paramiko，请安装："
                "uv pip install --python <ROBOT_OPENPI_PYTHON> paramiko"
            ) from exc

        cfg = self.cfg
        ssh = paramiko.SSHClient()
        # 默认采用严格检查；accept-new 模式仅首次信任新的主机密钥，后续变更会拒绝。
        ssh.load_system_host_keys()
        known_hosts = Path.home() / ".ssh" / "known_hosts"
        if cfg.ssh_host_key_policy == "accept-new":
            known_hosts.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not known_hosts.exists():
                fd = os.open(str(known_hosts), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
        if known_hosts.exists():
            ssh.load_host_keys(str(known_hosts))
        if cfg.ssh_host_key_policy == "accept-new":
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            ssh.set_missing_host_key_policy(paramiko.RejectPolicy())

        auth_method = cfg.ssh_auth_method
        if auth_method == "auto":
            auth_method = "password" if cfg.ssh_password else "key"
        options = {
            "hostname": cfg.ssh_host,
            "port": cfg.ssh_port,
            "username": cfg.ssh_user,
            "timeout": 10,
            "auth_timeout": 15,
            "banner_timeout": 10,
        }
        if auth_method == "password":
            if not cfg.ssh_password:
                raise ValueError("SSH 密码为空，请检查 ROBOT_OPENPI_SSH_PASSWORD")
            options.update(password=cfg.ssh_password, look_for_keys=False, allow_agent=False)
        else:
            options.update(look_for_keys=True, allow_agent=True)
            if cfg.ssh_identity_file:
                keyfile = Path(cfg.ssh_identity_file).expanduser()
                if not keyfile.is_file():
                    raise FileNotFoundError(f"SSH 密钥文件不存在：{keyfile}")
                options["key_filename"] = str(keyfile)

        try:
            ssh.connect(**options)
            transport = ssh.get_transport()
            if transport is None or not transport.is_active():
                raise RuntimeError("SSH 登录后传输连接不可用")
            transport.set_keepalive(15)

            remote = (cfg.ssh_remote_host, cfg.ssh_remote_port)
            # 基于 direct-tcpip 通道双向转发 WebSocket 数据。
            class Handler(socketserver.BaseRequestHandler):
                def handle(self):
                    channel = None
                    try:
                        if not transport.is_active():
                            return
                        channel = transport.open_channel(
                            "direct-tcpip", remote, self.request.getpeername(), timeout=10
                        )
                        if channel is None:
                            return
                        while transport.is_active():
                            readable, _, _ = select.select([self.request, channel], [], [], 1)
                            if self.request in readable:
                                data = self.request.recv(65536)
                                if not data:
                                    break
                                channel.sendall(data)
                            if channel in readable:
                                data = channel.recv(65536)
                                if not data:
                                    break
                                self.request.sendall(data)
                    except (OSError, EOFError, TimeoutError) as exc:
                        print(f"[openpi_ssh] 转发中断：{type(exc).__name__}: {exc}",
                              file=sys.stderr, flush=True)
                    finally:
                        if channel is not None:
                            channel.close()

            class LocalForwarder(socketserver.ThreadingTCPServer):
                allow_reuse_address = False
                daemon_threads = True

            # 先绑定本地端口，如果被现有 SSH 通道占用则明确报错。
            self.forward_server = LocalForwarder((cfg.host, cfg.port), Handler)
            self.ssh = ssh
            self.forward_thread = threading.Thread(
                target=self.forward_server.serve_forever,
                kwargs={"poll_interval": 0.1},
                daemon=True,
                name="openpi-ssh-forwarder",
            )
            self.forward_thread.start()
            if not self.forward_thread.is_alive():
                raise RuntimeError("SSH 本地端口转发线程启动失败")
        except Exception:
            if self.forward_server is not None:
                self.forward_server.server_close()
                self.forward_server = None
            ssh.close()
            self.ssh = None
            raise

    def initialize(self) -> dict:
        if self.client is not None:
            return {"ready": True, "action_space": self.cfg.action_space}
        try:
            self._open_tunnel()
            # 此处才导入 openpi_client，允许 Agent 使用另一 Python 版本运行。
            from openpi_client import image_tools
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            self.image_tools = image_tools
            self.client = WebsocketClientPolicy(self.cfg.host, self.cfg.port)
            return {"ready": True, "action_space": self.cfg.action_space}
        except Exception:
            self.close()
            raise

    @staticmethod
    def _decode_image(encoded: str):
        import numpy as np
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as img:
            return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))

    def _image(self, encoded: str):
        assert self.image_tools is not None
        rgb = self._decode_image(encoded)
        return self.image_tools.convert_to_uint8(
            self.image_tools.resize_with_pad(rgb, self.cfg.resize_size, self.cfg.resize_size)
        )

    def predict(self, observation: dict) -> dict:
        import numpy as np
        if self.client is None:
            raise RuntimeError("OpenPI worker 尚未初始化")
        if self.cfg.ssh_enabled:
            transport = self.ssh.get_transport() if self.ssh is not None else None
            if transport is None or not transport.is_active():
                raise ConnectionError("SSH 隧道已断开，请重新启动 OpenPI 控制任务")
        images = observation["images"]
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("非法机器人状态")
        # LIBERO 图像已经由 LiberoAdapter 正确旋转，不能再翻转。
        payload = {
            "observation/image": self._image(images["primary"]),
            "observation/wrist_image": self._image(images["wrist"]),
            "observation/state": state,
            "prompt": str(observation["instruction"]),
        }
        result = self.client.infer(payload)
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError("OpenPI 服务没有返回 actions")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or not actions.size or not np.isfinite(actions).all():
            raise ValueError(f"OpenPI 返回无效动作形状：{actions.shape}")
        return {"actions": actions.tolist()}

    def close(self) -> None:
        try:
            if self.client is not None:
                closer = getattr(self.client, "close", None)
                if callable(closer):
                    closer()
                self.client = None
        finally:
            if self.forward_server is not None:
                server, self.forward_server = self.forward_server, None
                if self.forward_thread is not None and self.forward_thread.is_alive():
                    server.shutdown()
                server.server_close()
            if self.forward_thread is not None:
                self.forward_thread.join(timeout=2)
                self.forward_thread = None
            if self.ssh is not None:
                ssh, self.ssh = self.ssh, None
                ssh.close()


def main() -> None:
    # stdout 专供 JSON 行协议。异常和第三方库日志应输出至 stderr。
    worker = None
    try:
        cfg = OpenPiConfig.from_env()
        worker = OpenPiWorker(cfg)
        for line in sys.stdin:
            cmd = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("请求必须为 JSON 对象")
                cmd = request.get("command")
                # 第三方 client 的 print 不能污染 stdout 的 JSON 行协议。
                with contextlib.redirect_stdout(sys.stderr):
                    if cmd == "init":
                        body = worker.initialize()
                    elif cmd == "predict":
                        body = worker.predict(request["observation"])
                    elif cmd == "close":
                        worker.close()
                        body = {"closed": True}
                    else:
                        raise ValueError(f"未知 OpenPI worker 命令：{cmd}")
                response = {"ok": True, **body}
            except Exception as exc:
                print(f"[openpi_worker] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(response, ensure_ascii=False), flush=True)
            if cmd == "close" and response.get("ok"):
                break
    except Exception as exc:
        print(f"[openpi_worker] 启动失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        # 即使配置有误也响应一次 init，便于 Agent 得到具体错误。
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()