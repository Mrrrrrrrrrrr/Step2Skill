"""
Prompt工具函数
"""
import json
import os
import sys
import hashlib
import io
from typing import Dict, List, Any, Optional, Union
import threading

# 添加项目根目录到路径，以便导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.ask_qwen import (
    call_vllm_api,
    call_llm_api,
    call_vlm_api,
    call_honor_cloud_api,
    call_mass_api,
    call_root_api,
    call_ark_api,
    call_vertex_gemini_api,
    call_honor_llm_api,
)

_cache_dir_lock = threading.Lock()

_current_task_context = threading.local()
_current_model_name = None  # 全局模型名称

# 当前请求/任务使用的 pipeline preset（线程局部），避免用 os.environ 导致 Flask 多线程互相覆盖
_pipeline_preset_tls = threading.local()


class PayloadTooLargeError(Exception):
    """请求体或消息 token 过大，通常因为图片数量/尺寸超限。"""
    pass


def bind_pipeline_preset(preset: Optional[str]) -> None:
    """在单条任务处理入口调用：此后本线程内 get_cache_path 等使用 data/<model>/<preset>/tmp(2)。"""
    if not preset:
        return
    _pipeline_preset_tls.value = preset


def unbind_pipeline_preset() -> None:
    """与 bind_pipeline_preset 成对出现在 finally 中。"""
    if hasattr(_pipeline_preset_tls, "value"):
        delattr(_pipeline_preset_tls, "value")


def _get_pipeline_preset() -> Optional[str]:
    """
    用于 pipeline 缓存目录隔离：data/<model>/<preset>/tmp 与 tmp2。
    优先读本线程 bind 的值（服务并发安全）；否则回退环境变量 ORM_PIPELINE_PRESET（脚本单次进程用法）。
    """
    try:
        v = getattr(_pipeline_preset_tls, "value", None)
        if v:
            return v
        return os.environ.get("ORM_PIPELINE_PRESET") or None
    except Exception:
        return None


get_pipeline_preset = _get_pipeline_preset


def load_json_file(file_path: str, default: Any = None) -> Any:
    """加载JSON文件"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载JSON文件失败 {file_path}: {e}")
            return default if default is not None else {}
    return default if default is not None else {}


def save_json_file(file_path: str, data: Any) -> bool:
    """保存JSON文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存JSON文件失败 {file_path}: {e}")
        return False


def parse_json_response(response_text: str) -> Optional[Dict]:
    """解析LLM返回的JSON响应"""
    try:
        # 尝试直接解析
        return json.loads(response_text)
    except:
        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None


def set_task_context(task_id: Optional[str] = None, step_name: Optional[str] = None):
    """设置当前任务的上下文"""
    _current_task_context.task_id = task_id
    _current_task_context.step_name = step_name

def set_model_name(model_name: Optional[str] = None):
    """设置当前使用的模型名称（全局）"""
    global _current_model_name
    _current_model_name = model_name

def get_model_name() -> Optional[str]:
    """获取当前使用的模型名称"""
    return _current_model_name


def get_task_context() -> tuple:
    """获取当前任务的上下文"""
    task_id = getattr(_current_task_context, 'task_id', None)
    step_name = getattr(_current_task_context, 'step_name', None)
    return task_id, step_name


def get_image_path_from_step(step: Dict) -> str:
    """
    从 step 中解析图片路径：
    1. 优先使用 image_local_path，但会先校验该本地文件是否存在；
    2. 若本地路径不存在或路径不合法，则回退使用 image（可为 URL）。
    """
    local_value = step.get("image_local_path")
    local = local_value.strip() if isinstance(local_value, str) else ""
    if local:
        try:
            if os.path.exists(local):
                return local
        except (TypeError, ValueError, OSError):
            # 本地路径不合法时，回退到 image 字段
            pass
    image_value = step.get("image")
    return image_value.strip() if isinstance(image_value, str) else ""


def _is_payload_too_large_error(exc: Exception) -> bool:
    """判断异常是否表示图片/文本总量超过接口限制。"""
    return _is_payload_too_large_message(str(exc))


def _is_payload_too_large_message(msg: str) -> bool:
    msg = (msg or "").lower()
    return (
        '413' in msg
        or 'too large' in msg
        or 'entity too large' in msg
        or 'payload too large' in msg
        or 'max message tokens' in msg
        or 'total tokens of image and text exceed' in msg
    )


_tee_stdout_lock = threading.Lock()
_tee_stdout_state = threading.local()


def _unwrap_stdout(stream):
    """剥离已泄漏的 _TeeStdout 包装，避免 write 链路过深导致 RecursionError。"""
    while isinstance(stream, _TeeStdout):
        stream = stream._original
    return stream


class _TeeStdout:
    """stdout 包装器：透传所有输出到原始 stdout 的同时缓存内容，用于检测 API 打印的错误信息"""

    def __init__(self, original):
        object.__setattr__(self, "_original", _unwrap_stdout(original))
        object.__setattr__(self, "_buffer", [])

    def write(self, s):
        self._buffer.append(s)
        return self._original.write(s)

    def flush(self):
        return self._original.flush()

    def __getattr__(self, name):
        original = self.__dict__.get("_original")
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)

    def contains_payload_too_large(self) -> bool:
        return _is_payload_too_large_message("".join(self._buffer))


def _install_tee_stdout() -> Optional[_TeeStdout]:
    """
    仅在当前线程最外层 call_llm_with_prompt 时安装 tee。
    嵌套调用复用已有 tee，避免 sys.stdout 层层包装。
    """
    depth = getattr(_tee_stdout_state, "depth", 0)
    if depth > 0:
        _tee_stdout_state.depth = depth + 1
        return getattr(_tee_stdout_state, "tee", None)

    real_stdout = _unwrap_stdout(sys.stdout)
    tee = _TeeStdout(real_stdout)
    with _tee_stdout_lock:
        _tee_stdout_state.real_stdout = real_stdout
        _tee_stdout_state.tee = tee
        sys.stdout = tee
    _tee_stdout_state.depth = 1
    return tee


def _uninstall_tee_stdout() -> None:
    depth = getattr(_tee_stdout_state, "depth", 0)
    if depth <= 0:
        return
    if depth == 1:
        real_stdout = getattr(_tee_stdout_state, "real_stdout", None)
        with _tee_stdout_lock:
            if real_stdout is not None and isinstance(sys.stdout, _TeeStdout):
                sys.stdout = real_stdout
        _tee_stdout_state.tee = None
        _tee_stdout_state.real_stdout = None
    _tee_stdout_state.depth = depth - 1


# 进程启动时若 stdout 已被旧版逻辑层层包装，先还原，避免 print 触发 RecursionError
if isinstance(sys.stdout, _TeeStdout):
    sys.stdout = _unwrap_stdout(sys.stdout)


def call_llm_with_prompt(
    prompt: str, 
    images: Optional[List[str]] = None,
    temperature: float = 0.1, 
    max_tokens: int = 2000,
    task_id: Optional[str] = None,
    step_name: Optional[str] = None,
    compress_quality: Optional[float] = None
) -> Optional[str]:
    """
    调用LLM/VLM API，根据模型名称自动选择API
    
    Args:
        prompt: 文本提示
        images: 图像URL列表（可选，如果提供则使用VLM模式）
        temperature: 温度参数
        max_tokens: 最大token数
        task_id: 任务ID（可选，如果提供则保存prompt和response）
        step_name: 步骤名称（可选，如果提供则保存prompt和response）
    
    Returns:
        str: API响应的文本内容
    
    Raises:
        PayloadTooLargeError: 当 API 返回 HTTP 413 时抛出，调用方可据此降低压缩系数重试
    """
    if task_id is None or step_name is None:
        task_id, step_name = get_task_context()
    
    # 获取当前模型名称
    model_name = get_model_name()
    
    # 判断应该使用哪个API
    # 四个接口：
    # 1. vLLM API（内网）：qwen3-235b, qwen235B 等特定模型
    # 2. DashScope API（阿里云）：qwen3.5-plus（映射为空字符串，在 call_vllm_api 内部识别）
    # 3. MaSS API：doubao-seed-2.0-pro, doubao-seed-1.8, gemini-3-pro, gpt-5.2 等模型
    # 4. 荣耀云 API：gpt-4o, doubao, gemini 等其他模型
    use_vllm = False
    use_mass = False
    use_ark = False
    use_vertex = False
    use_honor_llm = False
    use_root_api = False
    if model_name:
        mapped_name = config.get_model_name(model_name)
        model_lower = model_name.lower()

        # 检查是否使用项目根目录 api.py 统一包装器（seed2.0 等）
        if getattr(config, "is_root_api_model", lambda _name: False)(model_name):
            use_root_api = True
        # 检查是否使用荣耀 LLM 网关（GPT-5.5-SaaS）
        elif config.is_honor_llm_api_model(model_name):
            use_honor_llm = True
        # 检查是否使用 Vertex Gemini API（new）
        elif config.is_vertex_api_model(model_name):
            use_vertex = True
        # 检查是否使用火山方舟 ARK API
        elif config.is_ark_api_model(model_name):
            use_ark = True
        # 检查是否使用 MaSS API
        elif config.is_mass_api_model(model_name):
            use_mass = True
        # 明确使用 vLLM API（内网）的模型列表
        elif model_lower in ['qwen3-235b', 'qwen235b']:
            use_vllm = True
        # 如果映射为空字符串，使用 vLLM API（包括 DashScope，会在 call_vllm_api 内部识别）
        elif mapped_name == '':
            use_vllm = True
        # 其他模型使用荣耀云 API
        else:
            use_vllm = False
            use_mass = False
    else:
        # 如果没有设置模型名称，使用默认的 vLLM API
        use_vllm = True
    
    # 对于 qwen3-235b 模型，设置 max_tokens 为 2048
    if model_name and model_name.lower() in ['qwen3-235b', 'qwen235b']:
        max_tokens = 2048
    
    response_text = None
    full_response = None

    # 监控 API 打印输出以检测请求体/token 超限错误
    # （call_mass_api 等函数在内部 catch 异常、打印错误后返回 None，不抛异常）
    tee = _install_tee_stdout()
    try:
        if use_root_api:
            imgs_for_root_api = None
            if images:
                imgs_for_root_api = images[0] if len(images) == 1 else images
            response_text = call_root_api(
                query=prompt,
                imgs=imgs_for_root_api,
                model_name=model_name,
                max_tokens=max_tokens,
                compress_quality=compress_quality,
            )
            if response_text:
                full_response = {
                    'choices': [{'message': {'content': response_text}}]
                }
        elif use_vertex:
            imgs_for_vertex = None
            if images:
                imgs_for_vertex = images[0] if len(images) == 1 else images
            response_text = call_vertex_gemini_api(
                query=prompt,
                imgs=imgs_for_vertex,
                model_name=model_name,
            )
            if response_text:
                full_response = {
                    'choices': [{'message': {'content': response_text}}]
                }
        elif use_honor_llm:
            imgs_for_honor_llm = None
            if images:
                imgs_for_honor_llm = images[0] if len(images) == 1 else images
            response_text = call_honor_llm_api(
                query=prompt,
                imgs=imgs_for_honor_llm,
                model_name=model_name,
                max_tokens=max_tokens,
                compress_quality=compress_quality,
            )
            if response_text:
                full_response = {
                    'choices': [{'message': {'content': response_text}}]
                }
        elif use_ark:
            # 使用火山方舟 ARK API（MassAPIWrapperHonor）
            imgs_for_ark = None
            if images:
                imgs_for_ark = images[0] if len(images) == 1 else images
            response_text, ark_usage = call_ark_api(
                query=prompt,
                imgs=imgs_for_ark,
                model_name=model_name,
                compress_quality=compress_quality,
            )
            if response_text:
                full_response = {
                    'choices': [{'message': {'content': response_text}}]
                }
                if ark_usage:
                    full_response['usage'] = ark_usage
        elif use_mass:
            # 使用 MaSS API
            imgs_for_mass = None
            if images:
                imgs_for_mass = images[0] if len(images) == 1 else images

            response_text = call_mass_api(
                query=prompt,
                imgs=imgs_for_mass,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                compress_quality=compress_quality
            )
            if response_text:
                full_response = {
                    'choices': [{
                        'message': {
                            'content': response_text
                        }
                    }]
                }
        elif use_vllm:
            # 使用 vLLM API（包括 DashScope API）
            mapped_model_name = config.get_model_name(model_name) if model_name else None
            actual_model = model_name if (model_name and mapped_model_name == '') else mapped_model_name
            response = call_vllm_api(prompt, images=images, temperature=temperature, max_tokens=max_tokens, model=actual_model, compress_quality=compress_quality)
            if response and 'choices' in response and len(response['choices']) > 0:
                response_text = response['choices'][0]['message']['content']
                full_response = response
        else:
            # 使用荣耀云 API
            imgs_for_honor = None
            if images:
                imgs_for_honor = images[0] if len(images) == 1 else images

            response_text = call_honor_cloud_api(
                query=prompt,
                imgs=imgs_for_honor,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                compress_quality=compress_quality
            )
            if response_text:
                full_response = {
                    'choices': [{
                        'message': {
                            'content': response_text
                        }
                    }]
                }
    except Exception as e:
        if _is_payload_too_large_error(e):
            raise PayloadTooLargeError(str(e)) from e
        raise
    finally:
        _uninstall_tee_stdout()

    # API 返回空且输出中包含超限错误 → 判定为请求体过大
    if response_text is None and tee is not None and tee.contains_payload_too_large():
        raise PayloadTooLargeError("请求体或消息 token 过大（从 API 输出中检测到）")
    
    if task_id and step_name and response_text:
        save_prompt_response(task_id, step_name, prompt, images, response_text, full_response)
    
    return response_text


def get_cache_path(task_id: str, step_name: str, base_dir: Optional[str] = None) -> str:
    """获取缓存文件路径（按task_id分文件夹）"""
    if base_dir is None:
        try:
            model_name = get_model_name()
            preset = _get_pipeline_preset()
            if preset:
                base_dir = os.path.join(config.get_model_data_dir(model_name), preset, "tmp")
            else:
                base_dir = config.get_cache_dir(model_name)
        except AttributeError:
            # 如果config模块没有get_cache_dir，使用默认路径
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tmp")
    # 清理文件名中的非法字符
    safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_step_name = step_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    # 按task_id分文件夹
    task_dir = os.path.join(base_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, f"{safe_task_id}_{safe_step_name}.json")


def load_from_cache(task_id: str, step_name: str, base_dir: Optional[str] = None) -> Optional[Any]:
    """从缓存加载数据"""
    cache_path = get_cache_path(task_id, step_name, base_dir)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  加载缓存失败 {step_name}: {e}")
    return None


def save_to_cache(task_id: str, step_name: str, data: Any, base_dir: Optional[str] = None) -> bool:
    """保存数据到缓存（线程安全）"""
    cache_path = get_cache_path(task_id, step_name, base_dir)
    try:
        cache_dir = os.path.dirname(cache_path)
        with _cache_dir_lock:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"  保存缓存失败 {step_name}: {e}")
        return False


def generate_cache_key(data: Any) -> str:
    """生成数据的哈希键"""
    data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()


def get_prompt_response_path(task_id: str, step_name: str, base_dir: Optional[str] = None) -> str:
    """获取 tmp2 中 prompt_response 文件的路径（不创建目录）"""
    if base_dir is None:
        try:
            model_name = get_model_name()
            preset = _get_pipeline_preset()
            if preset:
                base_dir = os.path.join(config.get_model_data_dir(model_name), preset, "tmp2")
            else:
                base_dir = config.get_tmp2_dir(model_name)
        except Exception:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tmp2")
    safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_step_name = step_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    task_dir = os.path.join(base_dir, task_id)
    return os.path.join(task_dir, f"{safe_task_id}_{safe_step_name}_prompt_response.json")


def save_prompt_response(
    task_id: str,
    step_name: str,
    prompt: str,
    images: Optional[List[str]],
    response_text: str,
    full_response: Optional[Dict] = None,
    base_dir: Optional[str] = None
) -> bool:
    """保存prompt和response到tmp2目录（按task_id分文件夹）"""
    if base_dir is None:
        try:
            model_name = get_model_name()
            preset = _get_pipeline_preset()
            if preset:
                base_dir = os.path.join(config.get_model_data_dir(model_name), preset, "tmp2")
            else:
                base_dir = config.get_tmp2_dir(model_name)
        except:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tmp2")
    
    safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_step_name = step_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # 按task_id分文件夹
    task_dir = os.path.join(base_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    file_path = os.path.join(task_dir, f"{safe_task_id}_{safe_step_name}_prompt_response.json")
    
    data = {
        "task_id": task_id,
        "step_name": step_name,
        "prompt": prompt,
        "images": images if images else [],
        "response_text": response_text,
        "full_response": full_response
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存prompt和response失败 {file_path}: {e}")
        return False
