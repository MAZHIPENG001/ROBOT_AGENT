# Feishu Skill

## 功能

通过 WebSocket 长连接收取已授权用户的文本消息。向已授权会话发送文本、附件文件与图片。

## 工具

- `feishu_send_message(chat_id, text)`：发送文本。
- `feishu_send_file(chat_id, file_path)`：发送本机可读取的任意普通文件，最大 20 MiB。
- `feishu_send_image(chat_id, image_path)`：发送本机可读取的 JPG/PNG/GIF 图片，最大 8 MiB。

## 权限规则

- 文件发送与 FileManagerSkill 的文件读取具有相同的目录范围：**不限路径**，受操作系统当前账户的文件读取权限约束。
- 发送**必须是用户明确请求**，不能看到文件就自动上传。
- 远程入口要求在 `.env` 中配置 `FEISHU_ALLOWED_OPEN_IDS`，控制谁能使用 Agent；同时推荐配置 `FEISHU_ALLOWED_CHAT_IDS`，仅允许私聊会话。
- 未授权会话的发送请求将被拒绝。
- 发送任何账号凭证、SSH 私钥、Token 或其他敏感材料具有信息泄露风险，必须单独核验并征得用户明确同意。
- 普通对话回复由 channel 自动完成，不要重复调用主动发送工具。

## 配置

```dotenv
FEISHU_ALLOWED_OPEN_IDS=ou_你的OpenID
FEISHU_ALLOWED_CHAT_IDS=oc_你的私人会话ID
CAMERA_OUTPUT_DIR=./cache
```

已不需要 `FEISHU_ALLOWED_FILE_DIRS`。相对路径从项目根目录解析。