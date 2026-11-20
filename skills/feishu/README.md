# Feishu Skill：文本 + 文件附件 + 图片发送

## 安装

将 `skills/feishu/` 下的四个文件覆盖到 `/home/ma/robot-agent/skills/feishu/`：
`__init__.py`、`channel.py`、`SKILL.md`、`tool.py`。
`agent.py`、`main.py` 和自动扫描版本的 `registry.py` 无需修改。

## 环境变量

在 `.env` 设置 `FEISHU_APP_ID`、`FEISHU_APP_SECRET`；建议配置
`FEISHU_ALLOWED_OPEN_IDS` / `FEISHU_ALLOWED_CHAT_IDS`。默认可发送文件仅来自
`/home/ma/robot-agent/captures/`。如需允许其他目录，追加：

```
FEISHU_ALLOWED_FILE_DIRS=/home/ma/robot-agent/outputs,/home/ma/robot-agent/reports
```

## 工具

- `feishu_send_message(chat_id, text)`：发送文本
- `feishu_send_file(chat_id, file_path)`：上传并发送文件附件（本工具限制 20 MiB）
- `feishu_send_image(chat_id, image_path)`：上传并发送图片消息（本工具限制 8 MiB）

文件必须真实存在、非空，且位于允许目录内。图片消息支持 jpg/jpeg/png/gif。
`feishu_send_file` 对其他格式按 `stream` 类型上传，具体格式须以飞书服务端的接受结果为准。

## 调用示例

飞书用户：“用笔记本摄像头拍张照片发给我”。
Agent：调用 `laptop_camera_capture`，拿到 `captures/laptop/photo.jpg`；
再调用 `feishu_send_image`，将照片发送到本次已授权的 `chat_id`。
发送结果由工具返回，消息接收器随后自动回复执行情况。

> 注意：没有在当前环境中连接实际的飞书应用、相机或飞书 SDK 服务端；请在本机执行联调。
