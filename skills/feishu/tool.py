"""飞书主动发送工具：文本、文件附件、聊天图片。"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import lark_oapi as lark
from langchain_core.tools import tool
from lark_oapi.api.im.v1 import (
    CreateFileRequest,
    CreateFileRequestBody,
    CreateImageRequest,
    CreateImageRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
)

logger = logging.getLogger(__name__)

# tool.py 位于项目根目录下的 skills/feishu/，从而不依赖启动命令所在的目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif"}
FILE_TYPES = {
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "doc",
    ".xls": "xls",
    ".xlsx": "xls",
    ".ppt": "ppt",
    ".pptx": "ppt",
    ".mp4": "mp4",
    ".opus": "opus",
}


def _validate_local_file(file_path: str, max_mb: int) -> Path:
    """允许发送进程可读取的任意普通文件，保留大小与类型校验。"""
    if not file_path or not file_path.strip():
        raise ValueError("文件路径为空")

    path = Path(file_path.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=True)

    if not path.is_file():
        raise ValueError("路径不是普通文件")

    size = path.stat().st_size
    if size == 0:
        raise ValueError("不允许发送空文件")
    if size > max_mb * 1024 * 1024:
        raise ValueError(f"文件超过本工具的 {max_mb} MiB 限制")
    return path


def feishu_tools(
    client: lark.Client,
    is_chat_allowed: Callable[[str], bool],
):
    """创建飞书 LangChain Tools；不更改 FeishuSkill 的公共接口。"""

    def _send(chat_id: str, msg_type: str, content: dict) -> str:
        """调用飞书发送消息接口，返回内部状态；调用前必须完成会话授权校验。"""
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type(msg_type)
                .content(json.dumps(content, ensure_ascii=False))
                .build()
            )
            .build()
        )
        response = client.im.v1.message.create(request)
        if response.success():
            return "发送成功"
        return f"发送失败：飞书 API {response.code} {response.msg}"

    @tool
    def feishu_send_message(chat_id: str, text: str) -> str:
        """主动向已授权的飞书会话发送额外的文本消息。普通聊天回复由监听器自动完成，不要调用此工具。"""
        if not chat_id or not text.strip():
            return "发送失败：chat_id 或消息内容为空"
        if len(text) > 4000:
            return "发送失败：文本超过 4000 字符"
        if not is_chat_allowed(chat_id):
            return "发送失败：会话未授权"
        try:
            return f"文本消息{_send(chat_id, 'text', {'text': text})}"
        except Exception as exc:
            logger.exception("飞书发送文本异常")
            return f"发送异常：{type(exc).__name__}"

    @tool
    def feishu_send_file(chat_id: str, file_path: str) -> str:
        """将本地文件作为【文件附件】发送到已授权的飞书会话。file_path 是相机或其他技能生成的本地路径；可发送本机当前进程有权限读取的文件；用户必须明确提出发送请求。图片如需直接在聊天中预览，优先使用 feishu_send_image。"""
        if not chat_id or not is_chat_allowed(chat_id):
            return "发送失败：会话未授权或 chat_id 为空"
        try:
            path = _validate_local_file(file_path, max_mb=20)
            file_type = FILE_TYPES.get(path.suffix.lower(), "stream")

            # 第一步：上传文件，飞书返回 file_key。
            with path.open("rb") as stream:
                upload_request = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(file_type)
                        .file_name(path.name)
                        .file(stream)
                        .build()
                    )
                    .build()
                )
                upload = client.im.v1.file.create(upload_request)

            if not upload.success():
                return f"文件上传失败：飞书 API {upload.code} {upload.msg}"

            file_key = getattr(getattr(upload, "data", None), "file_key", None)
            if not file_key:
                return "文件上传失败：飞书没有返回 file_key"

            # 第二步：发送文件消息。
            result = _send(chat_id, "file", {"file_key": file_key})
            return f"文件 {path.name}：{result}"
        except (OSError, ValueError) as exc:
            return f"文件发送失败：{exc}"
        except Exception as exc:
            logger.exception("飞书发送文件异常")
            return f"文件发送异常：{type(exc).__name__}"

    @tool
    def feishu_send_image(chat_id: str, image_path: str) -> str:
        """将本地 JPG/JPEG/PNG/GIF 图片直接发送到已授权的飞书会话，适合发送摄像头拍摄的照片；可发送当前进程能够读取的本地图片，但用户必须明确提出发送请求。"""
        if not chat_id or not is_chat_allowed(chat_id):
            return "发送失败：会话未授权或 chat_id 为空"
        try:
            path = _validate_local_file(image_path, max_mb=8)
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                return "图片发送失败：仅支持 JPG、JPEG、PNG、GIF"

            # 第一步：上传聊天图片，取得 image_key。
            with path.open("rb") as stream:
                upload_request = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder()
                        .image_type("message")
                        .image(stream)
                        .build()
                    )
                    .build()
                )
                upload = client.im.v1.image.create(upload_request)

            if not upload.success():
                return f"图片上传失败：飞书 API {upload.code} {upload.msg}"

            image_key = getattr(getattr(upload, "data", None), "image_key", None)
            if not image_key:
                return "图片上传失败：飞书没有返回 image_key"

            # 第二步：发送图片消息。
            result = _send(chat_id, "image", {"image_key": image_key})
            return f"图片 {path.name}：{result}"
        except (OSError, ValueError) as exc:
            return f"图片发送失败：{exc}"
        except Exception as exc:
            logger.exception("飞书发送图片异常")
            return f"图片发送异常：{type(exc).__name__}"

    return [feishu_send_message, feishu_send_file, feishu_send_image]