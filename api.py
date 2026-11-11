# -*- coding: utf-8 -*-
"""Unified text/image chat API wrapper selected by model_name."""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import requests
import urllib3


DEFAULT_TIMEOUT = 180
HONOR_HOST = "https://model-aigw-b.digital.hihonor.com"

ContentOrder = Literal["text_first", "image_first"]
EndpointType = Literal["chat", "responses"]


@dataclass(frozen=True)
class ModelConfig:
    url: str
    request_model: str | None = None
    default_api_key: str | None = None
    endpoint_type: EndpointType = "chat"
    content_order: ContentOrder = "text_first"
    max_tokens_field: str | None = "max_tokens"
    max_tokens: int | None = None
    verify: bool = False
    trust_env: bool = False
    text_only: bool = False
    temperature: float | None = None
    extra_payload: dict[str, Any] | None = None


HONOR_LLM_CHAT_URL = f"{HONOR_HOST}/honor-llm/v1/chat/completions"

MODEL_CONFIGS: dict[str, ModelConfig] = {
    "Kimi-K2.6-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024),
    "Kimi-K2.5-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024),
    "Kimi-K3": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024),
    "GPT-5.6-Sol": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "GPT-5.6-Luna": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "GPT-5.6-Terra": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "GPT-5.5-SaaS": ModelConfig(
        HONOR_LLM_CHAT_URL,
        default_api_key="",
        max_tokens_field="max_completion_tokens",
        max_tokens=4096,
        extra_payload={"reasoning_effort": "low"},
    ),
    "GPT-5.4-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "GPT-5.3-Chat-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "GLM-5.2-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024),
    "Doubao-Seed-2.1-Pro-Preview": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024, temperature=0),
    "Doubao-Seed-1.8-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024, temperature=0),
    "DouBao-Seed-2.0-Pro": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024, temperature=0),
    "Qwen3.7-Plus": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", content_order="image_first", max_tokens=1024),
    "Qwen3.7-Max": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", content_order="image_first", max_tokens=1024),
    "MiniMax-M2.7-SaaS": ModelConfig(HONOR_LLM_CHAT_URL, default_api_key="", max_tokens=1024, text_only=True),
    "GPT-5.2": ModelConfig(f"{HONOR_HOST}/gpt-5.2/v1/chat/completions", default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "GPT-5.1": ModelConfig(f"{HONOR_HOST}/gpt-5.1/v1/chat/completions", default_api_key="", max_tokens_field="max_completion_tokens", max_tokens=4096),
    "Qwen3.6-Plus": ModelConfig(f"{HONOR_HOST}/qwen3.6-plus/v1/chat/completions", default_api_key="", content_order="image_first", max_tokens=1024),
    "qwen3.6-plus": ModelConfig(f"{HONOR_HOST}/qwen3.6-plus/v1/chat/completions", default_api_key="", content_order="image_first", max_tokens=1024),
    "Gemini-3.1-Pro-Preview": ModelConfig(
        f"{HONOR_HOST}/gemini-pro-preview/v1/chat/completions",
        request_model="google/gemini-3.1-pro-preview",
        default_api_key="",
        max_tokens=1024,
    ),
    "DeepSeek-V4-Flash": ModelConfig(f"{HONOR_HOST}/deepseek-v4-flash/v1/chat/completions", default_api_key="", max_tokens=1024),
    "DeepSeek-V4-Pro": ModelConfig(f"{HONOR_HOST}/deepseek-v4-pro/v1/chat/completions", default_api_key="", max_tokens=1024, text_only=True),
    "GPT-5.5-Response": ModelConfig(
        f"{HONOR_HOST}/honor-llm/v1/responses",
        default_api_key="",
        endpoint_type="responses",
        max_tokens_field="max_output_tokens",
        max_tokens=4096,
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "qwen": "Qwen3.6-Plus",
    "qwen_plus": "Qwen3.6-Plus",
    "qwen3.6-plus": "Qwen3.6-Plus",
}


class ApiWrapper:
    """Single model-name driven wrapper for OpenAI-compatible and Honor AIGW APIs."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        debug: bool = False,
        max_tokens: Optional[int] = None,
        verify: bool | None = None,
        trust_env: bool | None = None,
        content_order: ContentOrder | None = None,
    ) -> None:
        resolved_model_name = self._resolve_model_name(model_name)
        config = MODEL_CONFIGS.get(resolved_model_name)
        if config is None and base_url is None:
            known = ", ".join(sorted(MODEL_CONFIGS))
            raise ValueError(f"未知模型: {model_name}. 请使用已注册 model_name 或传入 base_url。已注册模型: {known}")

        self.model_name = resolved_model_name
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.last_response: dict[str, Any] | None = None

        self.config = config or self._openai_compatible_config(self.base_url or "")
        if base_url:
            self.config = self._openai_compatible_config(self.base_url)

        self.api_key = api_key if api_key is not None else (self.config.default_api_key or "EMPTY")
        self.request_model = self.config.request_model or self.model_name
        self.url = self.config.url
        self.endpoint_type = self.config.endpoint_type
        self.content_order = content_order or self.config.content_order
        self.max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        self.verify = self.config.verify if verify is None else verify
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.session = requests.Session()
        self.session.trust_env = self.config.trust_env if trust_env is None else trust_env

        if not self.verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        return MODEL_ALIASES.get(model_name, model_name)

    @staticmethod
    def _openai_compatible_config(base_url: str) -> ModelConfig:
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        return ModelConfig(url=url, verify=False, trust_env=False)

    @staticmethod
    def _guess_image_mime(image_path: str) -> str:
        mime, _ = mimetypes.guess_type(image_path)
        if mime and mime.startswith("image/"):
            return mime
        suffix = Path(image_path).suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "image/png"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{self._guess_image_mime(image_path)};base64,{image_b64}"

    def _normalize_image_url(self, image_url: str) -> str:
        if image_url.startswith(("http://", "https://", "data:")):
            return image_url
        image_path = image_url.replace("file://", "", 1)
        if Path(image_path).exists():
            return self._encode_image(image_path)
        return image_url

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        prepared_messages = copy.deepcopy(messages)
        for message in prepared_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    image_url["url"] = self._normalize_image_url(image_url["url"])
        return prepared_messages

    def _build_chat_user_content(self, prompt: str, image_path: str | None) -> str | list[dict[str, Any]]:
        if not image_path or self.config.text_only:
            if image_path and self.config.text_only:
                print(f"警告 {self.model_name} 在本网关仅支持纯文本，已忽略图片 {image_path}")
            return prompt

        text_part = {"type": "text", "text": prompt}
        image_part = {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}}
        if self.content_order == "image_first":
            return [image_part, text_part]
        return [text_part, image_part]

    def _build_responses_input(self, prompt: str, image_path: str | None) -> str | list[dict[str, Any]]:
        if not image_path or self.config.text_only:
            return prompt
        text_part = {"type": "input_text", "text": prompt}
        image_part = {"type": "input_image", "image_url": self._encode_image(image_path), "detail": "auto"}
        content = [image_part, text_part] if self.content_order == "image_first" else [text_part, image_part]
        return [{"role": "user", "content": content}]

    def _build_messages(
        self,
        query: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self._build_chat_user_content(query, image_path)})
        return messages

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self.endpoint_type == "responses":
            prompt = "\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "user")
            payload: dict[str, Any] = {"model": self.request_model, "input": prompt}
        else:
            payload = {"model": self.request_model, "messages": messages, "stream": False}

        if self.max_tokens is not None and self.config.max_tokens_field:
            payload[self.config.max_tokens_field] = self.max_tokens
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.extra_payload:
            payload.update(self.config.extra_payload)
        return payload

    def _extract_chat_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts = [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
            if parts:
                return "\n".join(parts)
        for key in ("reasoning_content", "reasoning", "reasoning_text", "thinking", "output_text"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return json.dumps(tool_calls, ensure_ascii=False)
        return ""

    @staticmethod
    def _extract_responses_text(data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if output_text:
            return str(output_text)
        parts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    parts.append(str(content.get("text") or ""))
        return "".join(parts)

    def _extract_message_text(self, data: dict[str, Any]) -> str:
        if self.endpoint_type == "responses":
            return self._extract_responses_text(data)
        return self._extract_chat_text(data)

    def chat(
        self,
        query: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if self.endpoint_type == "responses":
            payload = {
                "model": self.request_model,
                "input": self._build_responses_input(query, image_path),
            }
            if self.max_tokens is not None and self.config.max_tokens_field:
                payload[self.config.max_tokens_field] = self.max_tokens
            if self.config.extra_payload:
                payload.update(self.config.extra_payload)
            return self._post_payload(payload)

        messages = self._build_messages(query=query, image_path=image_path, system_prompt=system_prompt)
        return self.chat_messages(messages)

    def chat_messages(self, messages: list[dict[str, Any]]) -> str:
        payload = self._build_payload(self._prepare_messages(messages))
        return self._post_payload(payload)

    def _post_payload(self, payload: dict[str, Any]) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    url=self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify,
                )
                body = response.json()
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {body}")
                self.last_response = body
                if self.debug:
                    print("\n===== 原始响应 =====")
                    print(json.dumps(body, ensure_ascii=False, indent=2))
                return self._extract_message_text(body)
            except Exception as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"请求失败，达到最大重试次数: {exc}") from exc
                print(f"第 {attempt} 次请求失败: {exc}，3 秒后重试")
                time.sleep(3)
        return ""


def main(
    api_key: str,
    model: str,
    prompt: str,
    *,
    image_path: str | None = None,
    content_order: ContentOrder | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Convenience helper for one-shot model calls."""

    return ApiWrapper(
        api_key=api_key,
        model_name=model,
        timeout=timeout,
        content_order=content_order,
    ).chat(prompt, image_path=image_path)
