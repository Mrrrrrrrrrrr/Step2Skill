"""
把本地图片上传到 MinIO，并返回上传结果信息。

依赖:
    pip install minio python-dotenv

环境变量(可选，没配则用默认值):
    MINIO_ENDPOINT=http://10.162.200.22:9000
    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=minioadmin@teach
    MINIO_BUCKET_NAME=gui-agent
"""

from __future__ import annotations

import os
import mimetypes
import json
from typing import Any, Dict, Optional

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    raise ImportError("请先安装 minio: pip install minio")

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass  # dotenv 是可选的

# 默认配置（可以被环境变量覆盖）
DEFAULT_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://10.162.200.22:9000")
DEFAULT_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
DEFAULT_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin@teach")
DEFAULT_BUCKET = os.environ.get("MINIO_BUCKET_NAME", "gui-agent")


def _build_minio_client(
    endpoint: str = DEFAULT_ENDPOINT,
    access_key: str = DEFAULT_ACCESS_KEY,
    secret_key: str = DEFAULT_SECRET_KEY,
) -> Minio:
    """构建 MinIO 客户端"""
    # endpoint 不能带协议，Minio 里用 secure 控制 http/https
    endpoint = endpoint.strip()
    secure = True
    if endpoint.startswith("http://"):
        endpoint = endpoint[len("http://") :]
        secure = False
    elif endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]
        secure = True

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def upload_image_to_minio(
    image_path: str,
    bucket_name: str = DEFAULT_BUCKET,
    object_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    上传单张本地图片到 MinIO。

    Args:
        image_path: 本地图片绝对/相对路径
        bucket_name: 目标 bucket 名
        object_name: MinIO 中的对象名(不传则用 images/文件名)
        endpoint: MinIO 端点（可选，覆盖默认值）
        access_key: MinIO 访问密钥（可选，覆盖默认值）
        secret_key: MinIO 密钥（可选，覆盖默认值）

    Returns:
        dict: {
            "ok": bool,
            "bucket": str,
            "object_name": str,
            "url": str,           # 简单拼出的访问 URL
            "error": str|None,
            "etag": str|None,
            "version_id": str|None,
        }
    """
    # 保存原始路径用于 object_name 判断
    original_path = image_path
    
    # 检查文件是否存在（尝试绝对路径和原始路径）
    abs_path = os.path.abspath(image_path) if not os.path.isabs(image_path) else image_path
    if not os.path.exists(abs_path) and not os.path.exists(image_path):
        return {
            "ok": False,
            "bucket": bucket_name,
            "object_name": "",
            "url": "",
            "error": f"file not found: {image_path}",
            "etag": None,
            "version_id": None,
        }
    
    # 使用存在的路径
    actual_path = abs_path if os.path.exists(abs_path) else image_path

    if object_name is None or not str(object_name).strip():
        # 如果原始路径是绝对路径（以 / 开头），去掉开头的 / 作为 object_name
        # 否则使用 images/文件名
        if original_path.startswith('/'):
            object_name = original_path[1:]  # 去掉开头的 /
        else:
            object_name = f"images/{os.path.basename(original_path)}"

    # 使用传入的参数或默认值
    ep = endpoint or DEFAULT_ENDPOINT
    ak = access_key or DEFAULT_ACCESS_KEY
    sk = secret_key or DEFAULT_SECRET_KEY

    try:
        client = _build_minio_client(endpoint=ep, access_key=ak, secret_key=sk)
    except Exception as e:
        return {
            "ok": False,
            "bucket": bucket_name,
            "object_name": "",
            "url": "",
            "error": f"Failed to create MinIO client: {e}",
            "etag": None,
            "version_id": None,
        }

    try:
        # 确保 bucket 存在（添加超时处理）
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
        except Exception as e:
            return {
                "ok": False,
                "bucket": bucket_name,
                "object_name": object_name or "",
                "url": "",
                "error": f"Failed to check/create bucket: {e}",
                "etag": None,
                "version_id": None,
            }

        content_type, _ = mimetypes.guess_type(image_path)

        result = client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=actual_path,
            content_type=content_type or "application/octet-stream",
        )

        # 构造一个简单的 URL（仅供参考，具体要看你网关怎么配置）
        ep_clean = ep.rstrip("/")
        if not ep_clean.startswith("http"):
            ep_clean = f"http://{ep_clean}"
        url = f"{ep_clean}/{bucket_name}/{object_name.lstrip('/')}"

        return {
            "ok": True,
            "bucket": bucket_name,
            "object_name": object_name,
            "url": url,
            "error": None,
            "etag": getattr(result, "etag", None),
            "version_id": getattr(result, "version_id", None),
        }
    except S3Error as e:
        return {
            "ok": False,
            "bucket": bucket_name,
            "object_name": object_name,
            "url": "",
            "error": f"S3Error: {e}",
            "etag": None,
            "version_id": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "bucket": bucket_name,
            "object_name": object_name,
            "url": "",
            "error": str(e),
            "etag": None,
            "version_id": None,
        }


def process_jsonl_images(
    input_jsonl: str,
    output_jsonl: str,
    bucket_name: str = DEFAULT_BUCKET,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    批量处理 JSONL 文件中的图片：上传到 MinIO 并更新 URL。

    Args:
        input_jsonl: 输入 JSONL 文件路径
        output_jsonl: 输出 JSONL 文件路径
        bucket_name: MinIO bucket 名
        endpoint: MinIO 端点（可选）
        access_key: MinIO 访问密钥（可选）
        secret_key: MinIO 密钥（可选）

    Returns:
        dict: 统计信息 {
            "total_tasks": int,
            "total_images": int,
            "success_count": int,
            "failed_count": int,
            "skipped_count": int,  # 已为 URL 或文件不存在
        }
    """
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    stats = {
        "total_tasks": 0,
        "total_images": 0,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
    }

    output_data = []
    failed_images = []

    print(f"正在处理: {input_jsonl}")
    print(f"输出文件: {output_jsonl}")

    # 读取输入 JSONL
    try:
        with open(input_jsonl, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    task_data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行 JSON 解析错误: {e}")
                    continue

                stats["total_tasks"] += 1
                
                # 每处理 10 个任务输出一次进度
                if stats["total_tasks"] % 10 == 0:
                    print(f"已处理 {stats['total_tasks']} 个任务，成功上传 {stats['success_count']} 张图片...")

                traj = task_data.get("traj", [])

                if not traj:
                    output_data.append(task_data)
                    continue

                # 遍历 traj 数组中的每个步骤
                for step_idx, step in enumerate(traj):
                    if "image" not in step:
                        continue

                    image_path = step["image"]
                    stats["total_images"] += 1

                    # 如果已经是 URL，跳过
                    if isinstance(image_path, str) and image_path.startswith("http://"):
                        stats["skipped_count"] += 1
                        continue

                    try:
                        # 检查文件是否存在
                        if not os.path.exists(image_path):
                            print(f"警告：图片文件不存在，跳过: {image_path}")
                            stats["skipped_count"] += 1
                            failed_images.append({"path": image_path, "error": "file not found"})
                            continue

                        # 上传到 MinIO
                        # object_name 使用原路径（去掉开头的 /）
                        result = upload_image_to_minio(
                            image_path=image_path,
                            bucket_name=bucket_name,
                            object_name=None,  # 使用默认逻辑（原路径去掉开头的 /）
                            endpoint=endpoint,
                            access_key=access_key,
                            secret_key=secret_key,
                        )

                        if result["ok"]:
                            # 更新 image 字段为 MinIO URL
                            step["image"] = result["url"]
                            stats["success_count"] += 1
                        else:
                            print(f"警告：上传失败 {image_path}: {result.get('error', 'unknown error')}")
                            stats["failed_count"] += 1
                            failed_images.append({"path": image_path, "error": result.get("error", "unknown")})
                    except Exception as e:
                        print(f"错误：处理图片时出错 {image_path}: {e}")
                        stats["failed_count"] += 1
                        failed_images.append({"path": image_path, "error": str(e)})
                        continue

                output_data.append(task_data)
    except Exception as e:
        print(f"错误：读取或处理文件时出错: {e}")
        raise

    # 写入输出 JSONL
    # 确保输出路径是绝对路径
    if not os.path.isabs(output_jsonl):
        output_jsonl = os.path.abspath(output_jsonl)
    
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在写入输出文件: {output_jsonl}")
    
    try:
        with open(output_jsonl, 'w', encoding='utf-8') as f_out:
            for task_data in output_data:
                f_out.write(json.dumps(task_data, ensure_ascii=False) + '\n')
        
        # 验证文件是否成功写入
        if os.path.exists(output_jsonl):
            file_size = os.path.getsize(output_jsonl)
            print(f"文件写入成功，大小: {file_size} 字节")
        else:
            print(f"警告：文件写入后未找到: {output_jsonl}")
    except Exception as e:
        print(f"错误：写入文件失败: {e}")
        raise

    # 输出统计信息
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"总任务数: {stats['total_tasks']}")
    print(f"总图片数: {stats['total_images']}")
    print(f"成功上传: {stats['success_count']}")
    print(f"上传失败: {stats['failed_count']}")
    print(f"跳过（已为URL或文件不存在）: {stats['skipped_count']}")
    print(f"输出文件: {output_jsonl}")
    if failed_images:
        print(f"\n失败详情（前10条）:")
        for item in failed_images[:10]:
            print(f"  - {item['path']}: {item['error']}")
    print("="*60)

    return stats


def _cli() -> None:
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="批量上传 JSONL 文件中的图片到 MinIO。")
    parser.add_argument(
        "input_jsonl",
        type=str,
        nargs='?',  # 让位置参数变成可选的
        default="data/query_20260212_orm_testset_takeover_subset.jsonl",
        help="输入 JSONL 文件路径 (默认: data/query_20260212_orm_testset_takeover_subset.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSONL 文件路径 (默认: 输入文件名 + _minio.jsonl)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_BUCKET,
        help=f"MinIO bucket 名 (默认: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=f"MinIO 端点 (默认: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        default=None,
        help="MinIO 访问密钥 (默认从环境变量读取)",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        default=None,
        help="MinIO 密钥 (默认从环境变量读取)",
    )
    args = parser.parse_args()

    # 生成输出文件名
    if args.output is None:
        base_name = os.path.splitext(args.input_jsonl)[0]
        output_jsonl = f"{base_name}_minio.jsonl"
    else:
        output_jsonl = args.output

    # 处理 JSONL 文件
    stats = process_jsonl_images(
        input_jsonl=args.input_jsonl,
        output_jsonl=output_jsonl,
        bucket_name=args.bucket,
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
    )


if __name__ == "__main__":
    _cli()
