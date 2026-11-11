"""
火山引擎方舟 OpenAI 兼容接口封装（类名沿用 MassAPIWrapperHonor）。
配置见 config.ARK_OPENAI_*，可用环境变量覆盖；密钥请用 ARK_OPENAI_API_KEY。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import httpx
except ImportError as e:  # pragma: no cover
    raise ImportError("请安装: pip install httpx") from e


class MassAPIWrapperHonor:
    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
    ):
        default_id = model or model_name or config.ARK_OPENAI_MODEL
        self.model = default_id
        self.model_name = model_name or model or config.ARK_OPENAI_MODEL
        self.proxy_url = (
            config.ARK_OPENAI_PROXY_URL if proxy_url is None else proxy_url
        )
        self.api_key = api_key if api_key is not None else config.ARK_OPENAI_API_KEY
        self.base_url = (base_url or config.ARK_OPENAI_BASE_URL).rstrip("/")
        self.max_completion_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else config.ARK_OPENAI_MAX_COMPLETION_TOKENS
        )
        if not (self.api_key or "").strip():
            raise ValueError(
                "ARK_OPENAI_API_KEY 未设置：请在环境变量中配置或在构造函数传入 api_key"
            )

    def chat(
        self,
        query: str,
        imgs: Optional[Union[str, List[str]]] = None,
        max_entries: int = 100,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        compress_quality: Optional[float] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        use_model = model_name if model_name is not None else self.model_name

        # trust_env=False：避免环境代理变量与显式 proxy 叠加，防止非 ASCII header 编码错误
        client_kw: dict = {"trust_env": False, "timeout": httpx.Timeout(300.0)}
        if self.proxy_url:
            client_kw["proxy"] = self.proxy_url
            client_kw["verify"] = False

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if imgs:
            from utils.ask_qwen import _image_to_base64

            if isinstance(imgs, str):
                imgs = [imgs]
            user_content = []
            if query:
                user_content.append({"type": "text", "text": query})
            for img_path in imgs:
                base64_image = _image_to_base64(img_path, compress_quality=compress_quality)
                if not base64_image:
                    continue
                mime = (
                    "image/jpeg"
                    if compress_quality is not None
                    else "image/png"
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64_image}"
                        },
                    }
                )
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": query})

        # 直接用 httpx POST，避免 openai SDK 在某些环境下对 header 使用非 ASCII 编码路径
        payload = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": self.max_completion_tokens,
        }
        body = json.dumps(payload, ensure_ascii=True).encode("ascii")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        url = f"{self.base_url}/chat/completions"

        with httpx.Client(**client_kw) as http_client:
            try:
                resp = http_client.post(url, content=body, headers=headers)
                try:
                    data = resp.json()
                except Exception as je:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}, invalid JSON: {resp.text[:500]}"
                    ) from je
                if resp.status_code == 413:
                    raise RuntimeError(
                        "HTTP 413 Request Entity Too Large"
                    )
                if resp.status_code != 200:
                    err = data.get("error", data) if isinstance(data, dict) else data
                    raise RuntimeError(f"API error HTTP {resp.status_code}: {err}")
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"empty choices: {repr(data)[:500]}")
                msg = (choices[0].get("message") or {})
                content = msg.get("content")
                if content is None:
                    raise RuntimeError(f"no message.content: {repr(data)[:500]}")
                usage = data.get("usage") if isinstance(data, dict) else None
                if usage is not None and not isinstance(usage, dict):
                    usage = None
                return content, usage
            except Exception as e:
                err_s = str(e).lower()
                if (
                    "413" in str(e)
                    or "too large" in err_s
                    or "entity too large" in err_s
                    or "max message tokens" in err_s
                    or "total tokens of image and text exceed" in err_s
                ):
                    raise
                if max_entries > 1:
                    try:
                        print(f"Error: {e}. 等待3秒后重试...")
                    except UnicodeEncodeError:
                        print(f"Error: {type(e).__name__}. Retry in 3s...")
                    time.sleep(3)
                    return self.chat(
                        query=query,
                        imgs=imgs,
                        max_entries=max_entries - 1,
                        system_prompt=system_prompt,
                        model_name=model_name,
                        compress_quality=compress_quality,
                    )
                raise RuntimeError(
                    f"Error: {e}. 达到最大重试次数，停止尝试。"
                ) from e


if __name__ == "__main__":
    w = MassAPIWrapperHonor()
    _q = os.environ.get("ARK_SMOKE_QUERY", "Say hello in one short English sentence.")
    text, usage = w.chat(query=_q)
    print(text)
    if usage:
        print("usage:", usage)
