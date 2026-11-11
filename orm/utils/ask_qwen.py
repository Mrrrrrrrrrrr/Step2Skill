"""
ask_qwen 核心能力（恢复版）

覆盖主流程依赖：
- call_mass_api / call_honor_cloud_api / call_vllm_api
- call_llm_api / call_vlm_api
- compress_image
"""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import struct
import sys
import time
import warnings
from io import BytesIO
from typing import List, Optional, Union
from urllib.parse import urlparse

import config
import requests

try:
    import urllib3
except Exception:  # pragma: no cover
    urllib3 = None

try:
    from PIL import Image
except Exception:
    Image = None


MIN_API_IMAGE_DIMENSION = 14
IMAGE_DOWNLOAD_RETRIES = int(os.environ.get("ORM_IMAGE_DOWNLOAD_RETRIES", "3"))
IMAGE_DOWNLOAD_CONNECT_TIMEOUT = float(os.environ.get("ORM_IMAGE_DOWNLOAD_CONNECT_TIMEOUT", "5"))
IMAGE_DOWNLOAD_READ_TIMEOUT = float(os.environ.get("ORM_IMAGE_DOWNLOAD_READ_TIMEOUT", "60"))


def _ensure_project_root_on_path() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _is_private_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").strip("[]")
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local")
    return ip.is_private or ip.is_loopback or ip.is_link_local


def _download_image_bytes(image_path: str) -> bytes:
    timeout = (IMAGE_DOWNLOAD_CONNECT_TIMEOUT, IMAGE_DOWNLOAD_READ_TIMEOUT)
    retries = max(1, IMAGE_DOWNLOAD_RETRIES)
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            if _is_private_url(image_path):
                # Internal rollout/minio URLs should not detour through env proxies.
                with requests.Session() as session:
                    session.trust_env = False
                    resp = session.get(image_path, timeout=timeout)
            else:
                resp = requests.get(image_path, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"图片下载失败，重试 {attempt}/{retries}: {image_path} error={exc}")
            time.sleep(min(2 ** (attempt - 1), 5))

    assert last_error is not None
    raise last_error


def compress_image(image_path: str, quality: float, output_path: str) -> Optional[str]:
    """按质量压缩图片到指定路径。quality 取值 [0,1]。"""
    if Image is None:
        return None
    try:
        q = max(0.1, min(1.0, float(quality)))
        jpeg_q = int(q * 95)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(output_path, format="JPEG", quality=jpeg_q, optimize=True)
        return output_path
    except Exception:
        return None


def _warn_skip_small_image(image_path: str, width: int, height: int) -> None:
    print(
        f"警告: 跳过过小图片 {image_path} "
        f"({width}x{height}，最小允许 {MIN_API_IMAGE_DIMENSION}x{MIN_API_IMAGE_DIMENSION})"
    )


def _is_too_small_dimension(width: int, height: int) -> bool:
    return width < MIN_API_IMAGE_DIMENSION or height < MIN_API_IMAGE_DIMENSION


def _is_too_small_image(img) -> bool:
    width, height = img.size
    return _is_too_small_dimension(width, height)


def _get_image_size_from_bytes(image_bytes: bytes) -> Optional[tuple[int, int]]:
    """Read common image dimensions from headers without requiring Pillow."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
        return struct.unpack(">II", image_bytes[16:24])

    if image_bytes.startswith((b"GIF87a", b"GIF89a")) and len(image_bytes) >= 10:
        return struct.unpack("<HH", image_bytes[6:10])

    if image_bytes.startswith(b"\xff\xd8"):
        i = 2
        size = len(image_bytes)
        while i + 9 < size:
            if image_bytes[i] != 0xFF:
                i += 1
                continue
            while i < size and image_bytes[i] == 0xFF:
                i += 1
            if i >= size:
                break
            marker = image_bytes[i]
            i += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > size:
                break
            segment_len = struct.unpack(">H", image_bytes[i:i + 2])[0]
            if segment_len < 2 or i + segment_len > size:
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                height = struct.unpack(">H", image_bytes[i + 3:i + 5])[0]
                width = struct.unpack(">H", image_bytes[i + 5:i + 7])[0]
                return width, height
            i += segment_len
    return None


def _skip_if_too_small_bytes(image_path: str, image_bytes: bytes) -> bool:
    size = _get_image_size_from_bytes(image_bytes)
    if not size:
        return False
    width, height = size
    if _is_too_small_dimension(width, height):
        _warn_skip_small_image(image_path, width, height)
        return True
    return False


def _image_to_base64(image_path: str, compress_quality: Optional[float] = None) -> Optional[str]:
    is_url = image_path.lower().startswith(("http://", "https://"))

    # URL: 直接下载图片字节，必要时在内存中压缩后编码
    if is_url:
        image_bytes = _download_image_bytes(image_path)

        if _skip_if_too_small_bytes(image_path, image_bytes):
            return None

        if Image is not None:
            try:
                with Image.open(BytesIO(image_bytes)) as img:
                    if _is_too_small_image(img):
                        _warn_skip_small_image(image_path, img.size[0], img.size[1])
                        return None
                    if compress_quality is None:
                        return base64.b64encode(image_bytes).decode("utf-8")
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    output = BytesIO()
                    q = max(0.1, min(1.0, float(compress_quality)))
                    jpeg_q = int(q * 95)
                    img.save(output, format="JPEG", quality=jpeg_q, optimize=True)
                    image_bytes = output.getvalue()
            except Exception:
                # URL 压缩失败时回退原图字节，避免中断主流程
                pass
        return base64.b64encode(image_bytes).decode("utf-8")

    # 本地路径：沿用已有逻辑（压缩后读取）
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    data_path = image_path
    temp_path = None
    if Image is not None:
        try:
            with Image.open(image_path) as img:
                if _is_too_small_image(img):
                    _warn_skip_small_image(image_path, img.size[0], img.size[1])
                    return None
        except Exception:
            pass
    else:
        try:
            with open(image_path, "rb") as f:
                if _skip_if_too_small_bytes(image_path, f.read()):
                    return None
        except Exception:
            pass
    if compress_quality is not None:
        temp_path = f"{image_path}_temp_compressed_{int(compress_quality*100)}.jpg"
        compressed = compress_image(image_path, compress_quality, temp_path)
        if compressed:
            data_path = compressed
    try:
        with open(data_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _normalize_imgs(imgs: Optional[Union[str, List[str]]]) -> List[str]:
    if not imgs:
        return []
    if isinstance(imgs, str):
        return [imgs]
    return [x for x in imgs if x]


def _guess_mime_type(image_path: str) -> str:
    lower = image_path.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _build_mass_messages(query: str, imgs: Optional[Union[str, List[str]]], compress_quality: Optional[float]):
    content = [{"type": "text", "text": query}]
    for img in _normalize_imgs(imgs):
        image_base64 = _image_to_base64(img, compress_quality=compress_quality)
        if not image_base64:
            continue
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
    return [{"role": "user", "content": content}]


def call_root_api(
    query: str,
    imgs: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
    max_tokens: int = 2000,
    compress_quality: Optional[float] = None,
):
    """Call the workspace root api.py ApiWrapper for ORM judge aliases."""
    _ensure_project_root_on_path()
    from api import ApiWrapper  # type: ignore

    target_model = config.get_model_name(model_name or "seed20")
    wrapper = ApiWrapper(model_name=target_model, max_tokens=max_tokens)
    content = [{"type": "text", "text": query}]
    for img in _normalize_imgs(imgs):
        image_base64 = _image_to_base64(img, compress_quality=compress_quality)
        if not image_base64:
            continue
        mime_type = _guess_mime_type(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            }
        )
    return wrapper.chat_messages([{"role": "user", "content": content}])


def call_mass_api(
    query: str,
    imgs: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    compress_quality: Optional[float] = None,
    proxy_url: Optional[str] = None,
):
    """
    调用 MaSS API。失败返回 None，并保留与历史一致的重试打印行为。
    """
    try:
        import httpx
        from openai import OpenAI
    except Exception:
        print("错误: MaSS API 依赖未安装，请安装: pip install httpx openai")
        return None

    if not model_name:
        print("错误: 未指定模型名称")
        return None

    base_url = config.MASS_API_BASE_URL or os.environ.get("MASS_API_BASE_URL", "")
    api_key = (
        os.environ.get("MASS_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or getattr(config, "MASS_API_KEY", "")
    )
    mass_model_name = config.get_mass_model_name(model_name)
    if not base_url:
        print("错误: MASS_API_BASE_URL 未配置")
        return None
    if not api_key:
        print("错误: MASS_API_KEY/OPENAI_API_KEY 未配置")
        return None

    proxy = proxy_url or os.environ.get("MASS_API_PROXY_URL") or os.environ.get("HTTPS_PROXY")
    client_kwargs = {"timeout": 180}
    if proxy:
        client_kwargs["proxy"] = proxy
    http_client = httpx.Client(**client_kwargs)
    client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    messages = _build_mass_messages(query, imgs, compress_quality)
    max_ret = int(getattr(config, "MASS_API_MAX_RETRIES", 3))
    last_error = None
    for attempt in range(max_ret):
        try:
            response = client.chat.completions.create(
                model=getattr(config, "MASS_API_MODEL", "pre-ah_agent_server"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"model_name": mass_model_name},
            )
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            last_error = "API 返回空响应"
            print(f"错误: {last_error}")
            if attempt < max_ret - 1:
                print(f"等待3秒后重试 ({attempt + 1}/{max_ret})...")
                time.sleep(3)
        except Exception as e:
            last_error = f"请求异常: {e}"
            print(f"错误: {last_error}")
            if attempt < max_ret - 1:
                print(f"等待3秒后重试 ({attempt + 1}/{max_ret})...")
                time.sleep(3)
    print("达到最大重试次数，停止尝试。")
    return None


def call_honor_cloud_api(
    query: str,
    imgs: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    system_prompt: Optional[str] = None,
    max_retries: int = 3,
    compress_quality: Optional[float] = None,
):
    # new 命名模型（如 gpt-4o_new）走荣耀云 OpenAI 兼容接口；
    # 旧模型保持历史行为：回退到 MaSS 统一实现。
    if config.is_honor_cloud_new_model(model_name or ""):
        mapped_model = config.get_honor_cloud_model_name(model_name or "")
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        content = []
        for img in _normalize_imgs(imgs):
            image_base64 = _image_to_base64(img, compress_quality=compress_quality)
            if not image_base64:
                continue
            mime_type = _guess_mime_type(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                }
            )
        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": mapped_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        headers = dict(config.HONOR_CLOUD_HEADERS)

        retry = max(1, max_retries)
        for i in range(retry):
            try:
                response = requests.post(
                    url=config.HONOR_CLOUD_API_URL,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=getattr(config, "HONOR_CLOUD_TIMEOUT", 120),
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                print(f"Error: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"请求异常: {e}")
            if i < retry - 1:
                print(f"等待3秒后重试 ({i + 1}/{retry})...")
                time.sleep(3)
        return None

    return call_mass_api(
        query=query,
        imgs=imgs,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        compress_quality=compress_quality,
    )


def call_honor_llm_api(
    query: str,
    imgs: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
    max_tokens: int = 2000,
    max_retries: Optional[int] = None,
    compress_quality: Optional[float] = None,
):
    """调用荣耀 LLM 网关，当前用于 GPT-5.5-SaaS。"""
    if urllib3 is not None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        warnings.filterwarnings(
            "ignore",
            category=urllib3.exceptions.InsecureRequestWarning,
            module=r"urllib3\.connectionpool",
        )

    mapped_model = config.get_model_name(model_name or "gpt55")
    content = []
    if query:
        content.append({"type": "text", "text": query})
    image_list = _normalize_imgs(imgs)
    if str(mapped_model).lower() == "gpt-5.5-saas" and len(image_list) > 50:
        image_list = image_list[-50:]
    for img in image_list:
        image_base64 = _image_to_base64(img, compress_quality=compress_quality)
        if not image_base64:
            continue
        mime_type = _guess_mime_type(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})

    payload = {
        "model": mapped_model,
        "messages": [{"role": "user", "content": content}],
    }
    if str(mapped_model).lower().startswith("gpt-"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.0
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.HONOR_LLM_API_KEY}",
    }
    is_gpt55 = str(mapped_model).lower() == "gpt-5.5-saas"
    default_retries = (
        getattr(config, "HONOR_LLM_GPT55_MAX_RETRIES", 6)
        if is_gpt55
        else getattr(config, "HONOR_LLM_MAX_RETRIES", 3)
    )
    retry = max(1, max_retries or default_retries)
    timeout = (
        getattr(config, "HONOR_LLM_GPT55_TIMEOUT", 300)
        if is_gpt55
        else getattr(config, "HONOR_LLM_TIMEOUT", 180)
    )
    last_error = None

    for i in range(retry):
        try:
            session = requests.Session()
            session.trust_env = False
            with warnings.catch_warnings():
                if urllib3 is not None:
                    warnings.simplefilter(
                        "ignore",
                        urllib3.exceptions.InsecureRequestWarning,
                    )
                response = session.post(
                    config.HONOR_LLM_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    verify=False,
                )
            try:
                data = response.json()
            except Exception:
                data = None
            if response.status_code == 200 and isinstance(data, dict):
                msg = data["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content") or ""
            last_error = f"HTTP {response.status_code}: {data if data is not None else response.text}"
            print(f"Error: {last_error}")
        except Exception as e:
            last_error = str(e)
            print(f"请求异常: {e}")
        if i < retry - 1:
            sleep_s = min(30, 3 * (2 ** i))
            print(f"等待{sleep_s}秒后重试 ({i + 1}/{retry})...")
            time.sleep(sleep_s)
    print(f"达到最大重试次数，停止尝试。last_error={last_error}")
    return None


def call_vllm_api(
    message,
    images=None,
    stream: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    model: Optional[str] = None,
    compress_quality: Optional[float] = None,
):
    model_name = model or config.VLLM_MODEL or config.DASHSCOPE_MODEL or config.DEFAULT_MODEL
    query = message if isinstance(message, str) else str(message)
    return {"choices": [{"message": {"content": call_mass_api(query=query, imgs=images, model_name=model_name, temperature=temperature, max_tokens=max_tokens, compress_quality=compress_quality)}}]}


def call_llm_api(message: str):
    return call_vllm_api(message=message, images=None)


def call_vlm_api(message: str, images):
    return call_vllm_api(message=message, images=images)


def call_ark_api(*, query, imgs=None, model_name=None, compress_quality=None, **kwargs):
    if config.is_root_api_model(model_name or ""):
        return call_root_api(
            query=query,
            imgs=imgs,
            model_name=model_name,
            compress_quality=compress_quality,
        ), None
    from utils.mass_api_wrapper_honor import MassAPIWrapperHonor
    endpoint = config.MODEL_MAP.get(model_name or "", config.ARK_OPENAI_MODEL)
    imgs_list = ([imgs] if isinstance(imgs, str) else list(imgs)) if imgs else None
    return MassAPIWrapperHonor(model=endpoint).chat(
        query=query,
        imgs=imgs_list,
        compress_quality=compress_quality,
    )


def call_vertex_gemini_api(
    query: str,
    imgs: Optional[Union[str, List[str]]] = None,
    model_name: Optional[str] = None,
):
    """
    调用 Vertex Gemini（new 模型名，如 gemini_31_new）。
    返回文本，失败返回 None（与其他 call_* 风格一致）。
    """
    try:
        from google import genai
        from google.genai import types
        from google.oauth2 import service_account
    except Exception:
        print("错误: Vertex Gemini 依赖未安装，请安装: pip install google-genai google-auth")
        return None

    target_model = config.get_model_name(model_name or "gemini_31_new")
    try:
        sa_file = config.GEMINI_VERTEX_SERVICE_ACCOUNT_FILE
        sa_info = getattr(config, "GEMINI_VERTEX_SERVICE_ACCOUNT_INFO", None)
        if sa_file and os.path.isfile(sa_file):
            credentials = service_account.Credentials.from_service_account_file(
                sa_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif sa_info:
            credentials = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            print(f"错误: GEMINI_VERTEX_SERVICE_ACCOUNT_FILE 不存在: {sa_file}")
            print("错误: 且 GEMINI_VERTEX_SERVICE_ACCOUNT_INFO 未配置")
            return None

        client = genai.Client(
            credentials=credentials,
            vertexai=True,
            project=config.GEMINI_VERTEX_PROJECT,
            location=config.GEMINI_VERTEX_LOCATION,
        )
        contents = [query]
        for img in _normalize_imgs(imgs):
            if img.lower().startswith(("http://", "https://")):
                raise ValueError("Vertex 模式暂不支持 URL，请传本地图片路径。")
            with open(img, "rb") as f:
                image_bytes = f.read()
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime_type(img))
            )
        response = client.models.generate_content(model=target_model, contents=contents)
        return getattr(response, "text", None) or str(response)
    except Exception as e:
        print(f"Vertex Gemini 请求失败: {e}")
        return None
