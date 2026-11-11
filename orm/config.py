"""
配置文件
整合所有API接口配置
"""
import os
import base64,json,requests

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 知识库路径
SUPPORTED_KNOWLEDGE_BASES = ("knowledge_base", "knowledge_base_ali", "knowledge_base_ali_original", "knowledge_base_ali_hybrid", "knowledge_base_train")
DEFAULT_KNOWLEDGE_BASE = "knowledge_base"
KNOWLEDGE_BASE_NAME = DEFAULT_KNOWLEDGE_BASE
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, KNOWLEDGE_BASE_NAME)
SLOT_KNOWLEDGE_BASE_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "slot_knowledge_base.json")
STRATEGIES_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "strategies")


def set_knowledge_base(knowledge_base_name: str = DEFAULT_KNOWLEDGE_BASE) -> str:
    """
    切换知识库目录（用于评估规则加载）。

    Args:
        knowledge_base_name: 知识库目录名，仅支持 knowledge_base / knowledge_base_ali / knowledge_base_train

    Returns:
        str: 实际生效的知识库目录名
    """
    global KNOWLEDGE_BASE_NAME, KNOWLEDGE_BASE_DIR, SLOT_KNOWLEDGE_BASE_PATH, STRATEGIES_DIR

    kb_name = (knowledge_base_name or "").strip() or DEFAULT_KNOWLEDGE_BASE
    if kb_name not in SUPPORTED_KNOWLEDGE_BASES:
        raise ValueError(
            f"不支持的知识库: {kb_name}，可选: {', '.join(SUPPORTED_KNOWLEDGE_BASES)}"
        )

    kb_dir = os.path.join(PROJECT_ROOT, kb_name)
    if not os.path.isdir(kb_dir):
        raise ValueError(f"知识库目录不存在: {kb_dir}")

    KNOWLEDGE_BASE_NAME = kb_name
    KNOWLEDGE_BASE_DIR = kb_dir
    SLOT_KNOWLEDGE_BASE_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "slot_knowledge_base.json")
    STRATEGIES_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "strategies")

    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    os.makedirs(STRATEGIES_DIR, exist_ok=True)
    return KNOWLEDGE_BASE_NAME

# ==================== 模型名称映射 ====================
# 将简化的模型名称映射到实际的API模型名称
MODEL_MAP = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o_new': 'gpt-4o',
    'gpt-5.1_new': 'gpt-5.1',
    'gpt-5.2_new': 'gpt-5.2',
    'gpt-5.4_new': 'gpt-5.4',
    'gemini-flash': 'gemini-2.5-flash',
    'gemini_31_new': 'gemini-3.1-pro-preview',
    'gpt-5': 'gpt-5.1-2025-11-13',
    'gemini-3-pro': 'gemini-3-pro-preview',
    'gemini-3-pro-think': 'gemini-3-pro-preview-thinking',
    'doubao': 'doubao-seed-1-6-251015',  # doubao-1.6
    'kimi-k2': 'Kimi-K2-Thinking',
    'qwen3-235b': 'qwen235B',
    'gemini-3-flash-thinking': 'gemini-3-flash-preview-thinking',
    'gpt-5.2':'gpt-5.2',
    'doubao-1.8':'doubao-seed-1-8-251228',
    'qwen3.5-plus': '',  # 使用 DashScope API
    # MaSS API 模型映射
    'doubao-seed-2.0-pro': 'doubao-seed-2.0-pro',
    'doubao-seed-1.8': 'doubao-seed-1.8',
    'doubao-seed-1.6-vision': 'doubao-seed-1.6-vision',
    'gpt-4.1': 'gpt-4.1',
    'claude-sonnet-4.5': 'claude-sonnet-4.5',
    'qwen3-vl-plus': 'qwen3-vl-plus',
    # Short aliases used by main.py --model.
    'seed20': 'DouBao-Seed-2.0-Pro',
    'seed21': 'Doubao-Seed-2.1-Pro-Preview',
    'gpt55': 'GPT-5.5-SaaS',
    # Historical judge alias; routed through root api.py instead of Ark.
    'seed_free': 'DouBao-Seed-2.0-Pro',
}

# ==================== MaSS API 模型映射 ====================
# MaSS API 使用的模型名称映射（extra_body 中的 model_name）
MASS_MODEL_MAP = {
    'doubao-seed-2.0-pro': 'doubao.doubao-seed-2-0-pro-260215-completion',
    'doubao-seed-1.8': 'doubao-seed-1-8-251228-completion',
    'doubao-1.8': 'doubao-seed-1-8-251228-completion',  # doubao-1.8 映射到 doubao-seed-1.8
    'doubao-seed-1.6-vision': 'doubao-seed-1-6-vision-250815',
    'gemini-3-pro': 'gemini-3-pro-preview',
    'gpt-5.2': 'gpt-5.2-2025-12-11',
    'gpt-4.1': 'gpt-4.1-2025-04-14',
    'claude-sonnet-4.5': 'aws.claude-sonnet-4-5-20250929',
    'qwen3.5-plus': 'qwen3.5-plus',
    'qwen3-vl-plus': 'qwen3-vl-plus',
}

# ==================== 使用 MaSS API 的模型列表 ====================
# 这些模型将使用 MaSS API 而不是荣耀云 API
MASS_API_MODELS = [
    'doubao-seed-2.0-pro',
    'doubao-seed-1.8',
    'doubao-1.8',  # doubao-1.8 也使用 MaSS API
    'doubao-seed-1.6-vision',
    'gemini-3-pro',
    'gpt-5.2',
    'gpt-4.1',
    'claude-sonnet-4.5',
    'qwen3.5-plus',
    'qwen3-vl-plus',
]

# ==================== vLLM API 配置 ====================
# vLLM API (用于 Qwen3-VL-235B 等模型)
VLLM_API_URL = "http://10.80.5.27:56789/v1/chat/completions"
VLLM_MODEL = "Qwen3-VL-235B-A22B-Instruct"  # 默认模型
VLLM_TEMPERATURE = 0.1
VLLM_MAX_TOKENS = 4096
VLLM_TOP_P = 0.1
VLLM_REPETITION_PENALTY = 1.05
VLLM_HEADERS = {"Content-Type": "application/json"}

# vLLM API 备用地址（可选）
VLLM_API_URLS = [
    "http://10.80.5.27:56789/v1/chat/completions",
    # "http://10.80.2.212:9902/v1/chat/completions",
    # "http://10.80.1.159:9902/v1/chat/completions",
    # "http://10.80.8.207:9902/v1/chat/completions",
]

# ==================== DashScope API 配置 ====================
# DashScope API (用于 qwen3.5-plus 模型)
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_API_KEY = "sk-d3ec05ee52494029b11261447d556ab5"
DASHSCOPE_MODEL = "qwen3-vl-235b-a22b-instruct"
DASHSCOPE_TEMPERATURE = 0.1
DASHSCOPE_MAX_TOKENS = 4096
DASHSCOPE_TOP_P = 0.1
DASHSCOPE_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
}

# ==================== MaSS API 配置 ====================
# MaSS API (用于 doubao-seed, gemini-3-pro, gpt-5.2 等模型)
MASS_API_BASE_URL = "https://poc-dashscope.aliyuncs.com/compatible-mode/v1"
MASS_API_KEY = "sk-f96c8554864b47beb360e381476d7d74"
MASS_API_MODEL = "pre-ah_agent_server"  # API 的 model 参数
MASS_API_PROXY_URL = None  # 代理地址，如果不需要代理可以设为 None
MASS_API_TIMEOUT = 500  # 请求超时时间（秒）
MASS_API_MAX_RETRIES = 10  # 最大重试次数

# ==================== 火山方舟 OpenAI 兼容 API（MassAPIWrapperHonor）====================
# 实际为 ark.cn-beijing.volces.com，与荣耀云 HONOR_CLOUD 不同；密钥优先读环境变量
ARK_OPENAI_BASE_URL = os.environ.get(
    "ARK_OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
).rstrip("/")
# 环境变量 ARK_OPENAI_API_KEY 优先；未设置、为空或含非 ASCII 占位符时使用下方默认值
_ark_key_env = os.environ.get("ARK_OPENAI_API_KEY", "").strip()
try:
    _ark_key_env.encode("ascii")   # 含非 ASCII（如中文占位符）则视为未设置
except UnicodeEncodeError:
    _ark_key_env = ""
ARK_OPENAI_API_KEY = _ark_key_env or "a09a5bab-3bda-4508-a572-3a3c42b3f8c1"
ARK_OPENAI_MODEL = os.environ.get("ARK_OPENAI_MODEL", "ep-20260319174016-c5fcd")
_ark_proxy = os.environ.get("ARK_OPENAI_PROXY_URL")
if _ark_proxy is None:
    ARK_OPENAI_PROXY_URL = "http://proxysys.his.hihonor.com:8080"
elif _ark_proxy.strip() == "":
    ARK_OPENAI_PROXY_URL = None
else:
    ARK_OPENAI_PROXY_URL = _ark_proxy.strip()
ARK_OPENAI_MAX_COMPLETION_TOKENS = int(os.environ.get("ARK_OPENAI_MAX_COMPLETION_TOKENS", "32000"))

# ==================== 荣耀云 API 配置 ====================
# 荣耀云 API (用于 GPT4oWrapper 类)
HONOR_CLOUD_API_URL = "https://ai.cloud.hihonor.com/ml-athena/cloud-access/aicloud-ml/labs/v1/model-api/openai/v3/chat/completions"
HONOR_CLOUD_HEADERS = {
    "Content-Type": "application/json",
    "ml-source": "multimodal-team",
    "accessKey": "1001201",
    "sign": "NDAwODFDODNGNEQwQkY2N0Y2MkQ3NzhFODczMUExNzgxQThDQThEN0FCMTI1OEVDQjAyRUI2M0E0REM1NzJDRA==",
    "ml-referer": "multimodal-team",
    "ts": "1730893852"
}

# 荣耀云 API 备用 headers（可选）
HONOR_CLOUD_HEADERS_ALT = {
    "Content-Type": "application/json",
    "ml-source": "base-ui-agent-2",
    "accessKey": "1001201",
    "sign": "NDAwODFDODNGNEQwQkY2N0Y2MkQ3NzhFODczMUExNzgxQThDQThEN0FCMTI1OEVDQjAyRUI2M0E0REM1NzJDRA==",
    "ml-referer": "base-ui-agent",
    "ts": "1730893852"
}

# 荣耀云 API 默认参数
HONOR_CLOUD_DEFAULT_MODEL = "gpt-5"  # 默认模型名称（会通过 MODEL_MAP 映射）
HONOR_CLOUD_DEFAULT_TEMPERATURE = 0.1
HONOR_CLOUD_DEFAULT_MAX_TOKENS = 2048
HONOR_CLOUD_TIMEOUT = 500  # 请求超时时间（秒）
HONOR_CLOUD_MAX_RETRIES = 10  # 最大重试次数

# ==================== 荣耀 LLM 网关 API 配置（GPT-5.5-SaaS） ====================
HONOR_LLM_API_URL = os.environ.get(
    "HONOR_LLM_API_URL",
    "https://model-aigw-b.digital.hihonor.com/honor-llm/v1/chat/completions",
)
HONOR_LLM_API_KEY = os.environ.get(
    "HONOR_LLM_API_KEY",
    "b518a7a5dab844d9a8ac3a7e158fa95e",
)
HONOR_LLM_TIMEOUT = int(os.environ.get("HONOR_LLM_TIMEOUT", "180"))
HONOR_LLM_MAX_RETRIES = int(os.environ.get("HONOR_LLM_MAX_RETRIES", "3"))
HONOR_LLM_GPT55_TIMEOUT = int(os.environ.get("HONOR_LLM_GPT55_TIMEOUT", "300"))
HONOR_LLM_GPT55_MAX_RETRIES = int(os.environ.get("HONOR_LLM_GPT55_MAX_RETRIES", "6"))

# ==================== Vertex Gemini API 配置（new） ====================
GEMINI_VERTEX_PROJECT = os.environ.get("GEMINI_VERTEX_PROJECT", "honor-gui-agent-test")
GEMINI_VERTEX_LOCATION = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
GEMINI_VERTEX_SERVICE_ACCOUNT_FILE = os.environ.get(
    "GEMINI_VERTEX_SERVICE_ACCOUNT_FILE",
    os.path.join(PROJECT_ROOT, "credentials", "honor-gui-agent-test-8809420035dd.json"),
)
# 可选：直接在配置中内嵌服务账号信息（避免依赖外部 json 文件路径）
GEMINI_VERTEX_SERVICE_ACCOUNT_INFO = {
    "type": "service_account",
    "project_id": "honor-gui-agent-test",
    "private_key_id": "8809420035dd4d481665316e817dd5117d39585d",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDO/PbdSznaRz0K\nWDDw9cXTl0QHRy1TL6BwBV//urmN5Wre/PFLq0K8ALMKlaRUSB+R2rldS+fM0XGL\ntefZFaHiAdyWzx5hy7S0Jk2SSiZT0NdAePiAYBazIF++1d4R6LR4BTY7PaCNUbnR\nxEVJze7LsykEsvHhd1HTcnbCtzrixqESdHbOTb8+Yvot98ISyDi0p5t5WkgAjXKD\njJCSqClBmjXb19ceUO2SiCJFo9c1WAovzzRAbrNqdk+iqtjQBsDYR+sbGlzlewOR\nw000kE+lIUqUqRDJM2AeN8FlFDQ6ePRy2OtGWmhpNQz2kpboy5V4QT15o85EAcvA\nwKuZpc2DAgMBAAECggEACpYTKs/r9mHPWZCXdt4dueEnfbciYSSd8CzDiKjsUDlh\nZQ81vhjVEcLNZWda32P4QrWXkRHOmgHvlzrR+p8px40wfYFmpzohFKz4Ka5jgG+S\nxWg1JB0F/C+rMJpFQHrmLqnsPfe8bDiM/ZFWuTFwRaMfoBm7gjxf3HsjCiQPPcWI\nUw4MsgVHRkZk1pVaGj0SKsjeQiOxqSviK2lXCe+c3LndUD5IB3m0LhmjohmJ2+lB\nLj6faODtXTzBfByT0uzXwc+5IS2CnVbOzFqxF/CVyt0OogXcegLhjIBDHa9aQXWe\nvJc+vszz8t22zISgAK419O0jCwZwJizmpuJb9AIzIQKBgQDx7qFBdpoixt0vByj1\n+wjQgQ+/9oOqS4kSjoEEpbgLbsQbKQdPvboNcpMVuzn3s+wZDsLIlYt8B0arYJtH\nABN7hN4WQTVJIx9YBtD5VSS2kDt392UwyJNPe5i4WeE6foCJNlSck2cdbWeoTiU6\nEd78uCydGVy3+iGM+tG8A5zeoQKBgQDbBimb3AMysdKrkmG7DdF6KTQOWf5Vir1R\nqkJp9I0ffZ8ZBdC/9xIYskG5uxCUboKHjUnadrWtelGjMuu9ysG22N6Tq/TnJECy\n5pK/NZWZjnl7xMCFV0wsiMUGBKicq+2cAwuRvbRJ/ecoVBnehz9kAIwu0WEheBEj\nnMdL3wntowKBgApycYgwJOvHhpmXR4+HrZPDXvi9HmZmKJ8ihDO9LDtiJxm9JdGt\nsbEO94cCtLgbjuBuIL8kpgdPESApZcslF7JHxIdVfYWQ/LOCZ9e8c39mrwq5Jid7\nIdFW1vtu9ijeeZJzRCgWvWcp6EX+SeE+EFD9av/FrplLNdOuqQvjZAzBAoGBAJ5v\nC8Q513phVYJUxitYKJKUoZpMEQVx8dbUMWyRwU/pQCAKq+Fg6Q9CSvb+UVT9x6Ak\nXSsRugP+OnWyTQMiWxRYNlFeDy7wc3i1/6sfKNNGU6xrO9R4vJpuaQ8caZfDaoLz\nVPgc4QCxEPiENZ5Z+Unud3kaboeWYdd//k5szbK/AoGBAJlBPX5uBJJupf33ofPk\nV51ANsRPLfE8aHN/mZIq2iYykEYEJU33+Q2ptijqOGqfhizWu5Lj2HtGSqQj6eWz\nbX0xG9rCphdIVBfgDt6YwAVAF0Anh+DibZXFXCBFuNnMkki1ZUQ8LZOaF+OK65U7\nUg/I9V/4JWITpGPTEnxt4FSO\n-----END PRIVATE KEY-----\n",
    "client_email": "honor-gui-agent-test@appspot.gserviceaccount.com",
    "client_id": "105022710604083185847",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/honor-gui-agent-test%40appspot.gserviceaccount.com",
    "universe_domain": "googleapis.com",
}

# ==================== 通用 Prompt 配置 ====================
DEFAULT_TEMPERATURE = 0.01
DEFAULT_MAX_TOKENS = 2000

# ==================== 缓存目录 ====================
# 默认模型名称；seed20 通过根目录 api.py 路由到 DouBao-Seed-2.0-Pro。
DEFAULT_MODEL = "seed20"

# 模型相关的数据目录（动态获取）
def get_model_data_dir(model_name: str = None) -> str:
    """
    获取模型相关的数据目录
    
    Args:
        model_name: 模型名称，如果为None则使用默认模型
    
    Returns:
        str: 模型数据目录路径（如 data/qwen3-235b）
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    return os.path.join(PROJECT_ROOT, "data", model_name)

def get_cache_dir(model_name: str = None) -> str:
    """获取缓存目录（tmp）"""
    return os.path.join(get_model_data_dir(model_name), "tmp")

def get_tmp2_dir(model_name: str = None) -> str:
    """获取临时目录2（tmp2）"""
    return os.path.join(get_model_data_dir(model_name), "tmp2")

def get_save_dir(model_name: str = None) -> str:
    """获取保存目录（save）"""
    return os.path.join(get_model_data_dir(model_name), "save")

# 为了向后兼容，保留原有的常量（使用默认模型）
CACHE_DIR = get_cache_dir()
TMP2_DIR = get_tmp2_dir()
SAVE_DIR = get_save_dir()

# ==================== 工具函数 ====================
def get_model_name(model_key: str) -> str:
    """
    根据模型键名获取实际的模型名称
    
    Args:
        model_key: 模型键名（如 'gpt-5', 'doubao' 等）
    
    Returns:
        str: 实际的模型名称，如果映射不存在则返回原键名
    """
    return MODEL_MAP.get(model_key, model_key)

def get_honor_cloud_model_name(model_key: str) -> str:
    """
    获取荣耀云API使用的模型名称（通过映射）
    
    Args:
        model_key: 模型键名
    
    Returns:
        str: 映射后的模型名称
    """
    mapped_name = MODEL_MAP.get(model_key, model_key)
    # 如果映射为空字符串，返回原键名
    return mapped_name if mapped_name else model_key

def get_mass_model_name(model_key: str) -> str:
    """
    获取 MaSS API 使用的模型名称（extra_body 中的 model_name）
    
    Args:
        model_key: 模型键名
    
    Returns:
        str: MaSS API 的 model_name，如果映射不存在则返回原键名
    """
    return MASS_MODEL_MAP.get(model_key, model_key)

def is_mass_api_model(model_key: str) -> bool:
    """
    判断模型是否使用 MaSS API
    
    Args:
        model_key: 模型键名
    
    Returns:
        bool: 如果使用 MaSS API 返回 True，否则返回 False
    """
    return model_key in MASS_API_MODELS

# ==================== 根目录 api.py 模型列表 ====================
ROOT_API_MODELS = [
    'seed20',
    'seed_free',
]

# ==================== 火山方舟 ARK API 模型列表 ====================
ARK_API_MODELS = []

VERTEX_API_MODELS = [
    'gemini_31_new',
]

HONOR_CLOUD_NEW_API_MODELS = [
    'gpt-4o_new',
    'gpt-5.1_new',
    'gpt-5.2_new',
    'gpt-5.4_new',
]

HONOR_LLM_API_MODELS = [
    'seed20',
    'seed21',
    'gpt55',
]


def is_ark_api_model(model_key: str) -> bool:
    """判断模型是否使用火山方舟 OpenAI 兼容 API（MassAPIWrapperHonor）"""
    return model_key in ARK_API_MODELS


def is_root_api_model(model_key: str) -> bool:
    """判断模型是否走项目根目录 api.py 的统一 ApiWrapper。"""
    return model_key in ROOT_API_MODELS


def is_vertex_api_model(model_key: str) -> bool:
    """判断模型是否使用 Vertex Gemini API"""
    return model_key in VERTEX_API_MODELS


def is_honor_cloud_new_model(model_key: str) -> bool:
    """判断模型是否走荣耀云 OpenAI 兼容接口（new 模型名）"""
    return model_key in HONOR_CLOUD_NEW_API_MODELS


def is_honor_llm_api_model(model_key: str) -> bool:
    """判断模型是否走荣耀 LLM 网关（如 GPT-5.5-SaaS）"""
    return model_key in HONOR_LLM_API_MODELS

def vllm_qwen3vl_235b_request(query, image_path='data\gpt-4o-nopic\save\0a1708f843aaj4adba9f4c2157daec04\image_0.jpg'):
    url ="http://10.80.5.27:56789/v1/chat/completions" 
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":f"data:image/jpeg;base64,{base64_img}"}
                },
                {"type": "text",
                "text":  query
                },
            ],
        }
    ]
    headers ={"Content-Type":"application/json"}
    data ={
        "model":"Qwen3-VL-235B-A22B-Instruct",
        "messages":messages,
        "temperature":0.1,
        "top_p":0.1,
        "max_tokens":4096,
        "repetition_penalty":1.05,
    }
    json_payload = json.dumps(data)
    response = requests.post(url,data=json_payload,headers =headers)
    return response.json().get('choices',[])[0].get("message",[]).get("content",[])

# 确保目录存在
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(STRATEGIES_DIR, exist_ok=True)
# 默认模型的目录会在使用时创建，这里只创建基础目录
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
