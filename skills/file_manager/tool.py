"""文件管理：读取可访问文件，变更文件仅允许在项目 cache/ 内。"""

import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "cache"
MAX_WRITE_CHARS = 100_000
MAX_COPY_BYTES = 100 * 1024 * 1024


class FileManagerSkill:
    """文件系统 Skill。所有读操作遵守 OS 权限；变更仅允许 cache/。"""

    def __init__(self):
        if CACHE_ROOT.is_symlink():
            raise RuntimeError("cache 目录不能是符号链接")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.cache_root = CACHE_ROOT.resolve(strict=True)

    @staticmethod
    def _path(value: str) -> Path:
        """相对路径统一相对项目根目录，而非启动工作目录。"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("路径不能为空")
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _read_path(self, value: str) -> Path:
        """读取范围不限目录，但只能访问进程本身有权限的路径。"""
        return self._path(value).resolve(strict=True)

    def _write_path(self, value: str) -> Path:
        """验证目标及父目录，不允许绕过 cache/，拒绝路径中的符号链接。"""
        raw = self._path(value)
        path = raw.resolve(strict=False)
        if path == self.cache_root or not path.is_relative_to(self.cache_root):
            raise PermissionError("写入、移动或删除仅允许在项目 ./cache/ 内")

        # 所有父目录（包括目标本身）不得是符号链接，避免链接跳转。
        relative = raw.relative_to(PROJECT_ROOT) if raw.is_relative_to(PROJECT_ROOT) else None
        if relative is not None:
            probe = PROJECT_ROOT
            for part in relative.parts:
                probe /= part
                if probe.is_symlink():
                    raise PermissionError("不允许通过符号链接修改文件")
        # 对绝对路径从 cache 逐级检查，同样拒绝符号链接。
        for ancestor in [raw, *raw.parents]:
            if ancestor == self.cache_root or ancestor == CACHE_ROOT:
                break
            if ancestor.is_symlink():
                raise PermissionError("不允许通过符号链接修改文件")
        return path

    def tools(self) -> list:
        @tool
        def file_list(directory: str = ".") -> str:
            """列出任意可访问目录中的文件和子目录，默认项目根目录，最多 100 项。"""
            try:
                path = self._read_path(directory)
                if not path.is_dir():
                    return "不是目录"
                items = []
                for entry in sorted(path.iterdir(), key=lambda p: p.name.lower()):
                    kind = "目录" if entry.is_dir() else "文件"
                    items.append(f"[{kind}] {entry.name}")
                    if len(items) == 100:
                        break
                return f"目录：{path}\n" + ("\n".join(items) or "（空目录）")
            except (OSError, ValueError) as exc:
                return f"列出目录失败：{exc}"

        @tool
        def file_search(keyword: str, directory: str = ".", max_results: int = 20) -> str:
            """按文件名在指定目录递归搜索。可指定绝对路径，默认项目目录；最多扫描 5000 个文件。"""
            try:
                if not keyword or not keyword.strip():
                    return "请输入文件名关键词"
                root = self._read_path(directory)
                if not root.is_dir():
                    return "搜索路径不是目录"
                limit = max(1, min(int(max_results), 50))
                matches = []
                visited = 0
                truncated = False
                for current, dirs, files in os.walk(root, followlinks=False):
                    dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
                    for name in files:
                        visited += 1
                        if keyword.lower() in name.lower():
                            matches.append(str(Path(current) / name))
                            if len(matches) >= limit:
                                truncated = True
                                break
                        if visited >= 5000:
                            truncated = True
                            break
                    if truncated:
                        break
                message = "\n".join(matches) if matches else "没有找到匹配文件"
                if truncated:
                    message += "\n（达到数量上限，请缩小搜索目录）"
                return message
            except (OSError, ValueError) as exc:
                return f"搜索失败：{exc}"

        @tool
        def file_info(path: str) -> str:
            """查看可访问文件/目录的大小、修改时间和绝对路径，不改变文件。"""
            try:
                target = self._read_path(path)
                info = target.stat()
                return (
                    f"路径：{target}\n"
                    f"类型：{'目录' if target.is_dir() else '文件'}\n"
                    f"大小：{info.st_size} bytes\n"
                    f"修改时间：{datetime.fromtimestamp(info.st_mtime):%Y-%m-%d %H:%M:%S}"
                )
            except (OSError, ValueError) as exc:
                return f"获取信息失败：{exc}"

        @tool
        def file_read(path: str, max_chars: int = 6000) -> str:
            """读取任意路径的 UTF-8 文本文件，最多 12000 字符；不执行文件内容。"""
            try:
                target = self._read_path(path)
                if not target.is_file() or not stat.S_ISREG(target.stat().st_mode):
                    return "只能读取普通文件"
                limit = max(1, min(int(max_chars), 12000))
                with target.open("r", encoding="utf-8") as f:
                    content = f.read(limit + 1)
                if "\x00" in content:
                    return "文件疑似二进制内容，无法作为 UTF-8 文本直接读取"
                truncated = len(content) > limit
                return f"文件：{target}\n\n{content[:limit]}" + ("\n[内容已截断]" if truncated else "")
            except (OSError, ValueError, UnicodeError) as exc:
                return f"读取失败：{exc}"

        @tool
        def file_write(path: str, content: str, overwrite: bool = False) -> str:
            """在项目 ./cache/ 中创建或修改 UTF-8 文本文件；默认不覆盖已有文件。"""
            try:
                target = self._write_path(path)
                if not isinstance(content, str) or len(content) > MAX_WRITE_CHARS:
                    return f"写入失败：文本不得超过 {MAX_WRITE_CHARS} 字符"
                if target.exists() and not target.is_file():
                    return "写入失败：目标不是普通文件"
                if target.exists() and not overwrite:
                    return "写入失败：文件已存在。若确需修改，请明确要求覆盖"
                target.parent.mkdir(parents=True, exist_ok=True)
                # 再次校验路径（针对父目录刚创建的情况）
                target = self._write_path(str(target))
                # 覆盖时通过同目录临时文件 + os.replace 避免修改硬链接目标。
                if overwrite:
                    fd, temp_path = tempfile.mkstemp(prefix=".filemanager-", dir=target.parent)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(content)
                        os.replace(temp_path, target)
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    with target.open("x", encoding="utf-8") as f:
                        f.write(content)
                return f"写入成功：{target}"
            except (OSError, ValueError, PermissionError) as exc:
                return f"写入失败：{exc}"

        @tool
        def file_copy(source: str, destination: str, overwrite: bool = False) -> str:
            """将任何可读取的普通文件复制进项目 ./cache/；不会修改源文件。"""
            try:
                src = self._read_path(source)
                dst = self._write_path(destination)
                if not src.is_file() or not stat.S_ISREG(src.stat().st_mode):
                    return "复制失败：源路径不是普通文件"
                if src.stat().st_size > MAX_COPY_BYTES:
                    return "复制失败：源文件超过 100 MiB"
                if src == dst:
                    return "复制失败：源文件和目标路径相同"
                if dst.exists() and not overwrite:
                    return "复制失败：目标文件已存在"
                if dst.exists() and not dst.is_file():
                    return "复制失败：目标不是文件"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst = self._write_path(str(dst))
                if overwrite:
                    fd, temp_path = tempfile.mkstemp(prefix=".filemanager-", dir=dst.parent)
                    try:
                        with src.open("rb") as reader, os.fdopen(fd, "wb") as writer:
                            shutil.copyfileobj(reader, writer, length=1024 * 1024)
                        os.replace(temp_path, dst)
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                else:
                    with src.open("rb") as reader, dst.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                return f"复制成功：{src} -> {dst}"
            except (OSError, ValueError, PermissionError) as exc:
                return f"复制失败：{exc}"

        @tool
        def file_delete(path: str, confirm: bool = False) -> str:
            """删除项目 ./cache/ 内的单个文件。需同时配置 FILE_ALLOW_DELETE=true 并明确确认；不能删除目录。"""
            if os.getenv("FILE_ALLOW_DELETE", "false").strip().lower() != "true":
                return "删除已禁用：如确需使用，先在 .env 设置 FILE_ALLOW_DELETE=true"
            if not confirm:
                return "删除未执行：需要明确确认"
            try:
                target = self._write_path(path)
                if not target.is_file() or target.is_symlink():
                    return "删除失败：只支持删除 cache 内的普通文件"
                target.unlink()
                return f"删除成功：{target}"
            except (OSError, ValueError, PermissionError) as exc:
                return f"删除失败：{exc}"

        return [file_list, file_search, file_info, file_read, file_write, file_copy, file_delete]