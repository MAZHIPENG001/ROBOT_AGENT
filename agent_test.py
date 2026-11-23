
import os
import sys
import base64
import mimetypes
from pathlib import Path
import json

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
# MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


if not API_KEY:
    raise RuntimeError("请在 .env 中配置 DEEPSEEK_API_KEY")

client = httpx.Client(
    base_url=BASE_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    timeout=120,
)


def check_response(response):
    print("HTTP 状态码:", response.status_code)
    if response.status_code != 200:
        print("错误信息:", response.text)
        return None
    return response.json()


def inference(messages):
    response = client.post(
        "/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 2048,
        },
    )

    data = check_response(response)

    if not data:
        return None

    choice = data["choices"][0]
    message = choice["message"]

    print("\n========== 完整返回 ==========")
    print(json.dumps(message, ensure_ascii=False, indent=2))

    print("\n========== 推理信息 ==========")
    print("finish_reason:", choice.get("finish_reason"))
    print("usage:", json.dumps(
        data.get("usage", {}),
        ensure_ascii=False,
        indent=2
    ))

    print("\n========== 最终回复 ==========")
    print("content:", repr(message.get("content")))
    print("reasoning_content:",
          repr(message.get("reasoning_content"))[:2000])

    return data


try:
    # 1. 查询模型列表
    print("\n========== 1. 可用模型 ==========")
    response = client.get("/models")
    data = check_response(response)

    if data:
        model_ids = [m["id"] for m in data.get("data", [])]
        for model_id in model_ids:
            print(" -", model_id)

        for model_id in model_ids:
            if model_id.lower() == MODEL.lower():
                MODEL = model_id
                break
        else:
            print(f"注意：模型列表中没有找到 {MODEL}")

    # 2. 测试文本
    print("\n========== 2. 文本测试 ==========")
    print("当前模型:", MODEL)

    inference([
        {
            "role": "user",
            "content": "你好，请用一句话介绍你自己。",
        }
    ])

    # 3. 测试多模态图片
    print("\n========== 3. 图片测试 ==========")

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.jpg")

    if not image_path.is_file():
        print(f"图片不存在: {image_path}")
        sys.exit(1)

    mime_type = mimetypes.guess_type(image_path.name)[0]
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError(f"不支持的图片类型: {mime_type}")

    image_base64 = base64.b64encode(image_path.read_bytes()).decode()

    inference([
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这张图片中的主要物体。",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                },
            ],
        }
    ])

finally:
    client.close()
