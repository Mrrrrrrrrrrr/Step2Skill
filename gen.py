import base64
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Optional, Sequence, Union
from openai import AzureOpenAI

# 多组 endpoint 与密钥配置
AZURE_CONFIGS = [
    {
        "endpoint": "*1",
        "api_key": ""
    },
    {
        "endpoint": "*2",
        "api_key": ""
    },
    {
        "endpoint": "*3",
        "api_key": ""
    }
]

API_VERSION = "2025-04-01-preview"
MODEL = "gpt-image-2"

def create_client(config: dict) -> AzureOpenAI:
    """根据配置创建 AzureOpenAI 客户端"""
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=config["endpoint"],
        api_key=config["api_key"],
    )

def generate_single_image(
    client: AzureOpenAI,
    prompt: str,
    output_path: Path,
    input_image_paths: Optional[Sequence[Union[str, Path]]] = None,
    size: str = "1024x2400",
    quality: str = "low"
) -> Path:
    """生成单张图片；传入 input_image_paths 时会把图片作为参考图加入 prompt。"""
    image_paths = _normalize_image_paths(input_image_paths)

    if image_paths:
        result = _edit_image_with_references(
            client=client,
            prompt=prompt,
            image_paths=image_paths,
            size=size,
            quality=quality,
        )
    else:
        result = client.images.generate(
            model=MODEL,
            prompt=prompt,
            size=size,
            n=1,
            output_format="jpeg",
            quality=quality,
        )

    image_base64 = result.data[0].b64_json
    output_path.write_bytes(base64.b64decode(image_base64))
    return output_path


def _normalize_image_paths(
    input_image_paths: Optional[Sequence[Union[str, Path]]]
) -> list[Path]:
    """校验并标准化参考图路径。"""
    if not input_image_paths:
        return []

    image_paths = [Path(image_path) for image_path in input_image_paths]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"参考图不存在: {image_path}")
    return image_paths


def _edit_image_with_references(
    client: AzureOpenAI,
    prompt: str,
    image_paths: Sequence[Path],
    size: str,
    quality: str,
):
    """使用图片编辑接口，把参考图和文字 prompt 一起提交给图像模型。"""
    with ExitStack() as stack:
        image_files = [stack.enter_context(image_path.open("rb")) for image_path in image_paths]
        image_input = image_files[0] if len(image_files) == 1 else image_files
        return client.images.edit(
            model=MODEL,
            image=image_input,
            prompt=prompt,
            size=size,
            n=1,
            output_format="jpeg",
            quality=quality,
        )

def main():
    """示例用法"""
    image_dir = Path(__file__).parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # 1. 单张图片生成（使用第一个配置）
    print("=== 单张图片生成 ===\n")
    client = create_client(AZURE_CONFIGS[0])

    single_start = time.time()
    output_path = image_dir / "generated_image.jpg"
    generate_single_image(
        client,
        prompt="以这个场景为例子，微调一下图像中UI的风格",
        output_path=output_path,
        # 如果要在 prompt 中加入参考图，把本地图片路径放到这里：
        input_image_paths=[Path(__file__).parent / "images" / "reference.png"],
    )
    single_elapsed = time.time() - single_start
    print(f"图像已保存到 {output_path.resolve()}")
    print(f"单条运行时间: {single_elapsed:.2f}s\n")

if __name__ == "__main__":
    main()
