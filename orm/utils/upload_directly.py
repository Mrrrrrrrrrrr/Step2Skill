"""
JSONL 直传到飞轮（无解析逻辑，全字段透传）。

输入：
- 一个完成的 JSONL：每行是一条完整 task dict（例如 predicted.jsonl）

行为：
- 逐行读取，每条 task 原样作为 payload.task 上传
- 调用新接口：POST /api/save
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_API_URL = "http://10.162.200.22:19003"


def _try_load_dotenv(dotenv_path: str = ".env") -> None:
    """优先加载 .env，让 os.getenv 能读取到配置。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path, override=True)
        return
    except Exception:
        # 没有 python-dotenv 时，手动解析 .env
        pass

    try:
        possible_paths = [
            dotenv_path,
            os.path.join(os.getcwd(), dotenv_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dotenv_path),
        ]
        for path in possible_paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    k, v = raw.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if " # " in v:
                        v = v.split(" # ")[0].strip()
                    if k:
                        os.environ[k] = v
            return
    except Exception:
        pass


_try_load_dotenv(".env")

GUI_AGENT_FLYWHEEL_SYS_API_URL = os.getenv("GUI_AGENT_FLYWHEEL_SYS_API_URL")
GUI_AGENT_FLYWHEEL_SYS_API_KEY = os.getenv("GUI_AGENT_FLYWHEEL_SYS_API_KEY")
GUI_AGENT_SYS_API_KEY = os.getenv("GUI_AGENT_SYS_API_KEY")

def _upload_one_task(
    task: Dict[str, Any],
    api_url: str,
    api_key: Optional[str],
    version_intent: str,
    reason: Optional[str],
) -> Tuple[bool, str, Optional[str]]:
    task_id = task.get("task_id", "")
    if not task_id:
        return False, "", "task_id is missing"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    payload: Dict[str, Any] = {
        "task": task,
        "version_intent": version_intent,
    }
    if reason:
        payload["reason"] = reason

    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/api/save",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if not resp.ok:
            return False, task_id, f"HTTP {resp.status_code}: {resp.text[:300]}"
        return True, task_id, None
    except Exception as e:
        return False, task_id, str(e)


def _read_tasks(input_path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if limit is not None and len(tasks) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_no} 行 JSON 解析失败: {e}")
    return tasks


def _append_tag(tasks: List[Dict[str, Any]], tag: str) -> None:
    if not tag:
        return
    tag_str = str(tag)
    for task_data in tasks:
        tags = task_data.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags] if tags is not None else []
        tags.append(tag_str)
        task_data["tags"] = tags


def main() -> None:
    _try_load_dotenv(".env")
    parser = argparse.ArgumentParser(description="最基础 JSONL 直传到飞轮")
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--api-url", default=GUI_AGENT_FLYWHEEL_SYS_API_URL or DEFAULT_API_URL)
    parser.add_argument("--api-key", default=GUI_AGENT_FLYWHEEL_SYS_API_KEY or GUI_AGENT_SYS_API_KEY)
    parser.add_argument(
        "--version-intent",
        default="auto",
        help="POST /api/save 的 version_intent（create|refresh|auto）",
    )
    parser.add_argument("--reason", default=None, help="POST /api/save 的可选 reason")
    parser.add_argument("--limit", type=int, default=None, help="最多上传条数")
    parser.add_argument("--batch", type=int, default=10, help="并行上传数")
    parser.add_argument("--tag", default="", help="可选：为每条 task 的 tags 追加一个标签")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    if not args.api_key:
        raise ValueError("未读取到 API Key，请在 .env 配置 GUI_AGENT_FLYWHEEL_SYS_API_KEY（或 GUI_AGENT_SYS_API_KEY），或通过 --api-key 传入")

    tasks = _read_tasks(args.input, args.limit)
    _append_tag(tasks, args.tag)
    total = len(tasks)
    success = 0
    failed = 0
    errors: List[Tuple[str, str]] = []

    print(f"正在读取: {args.input}")
    print(f"API 地址: {args.api_url}")
    print(f"version_intent: {args.version_intent}")
    print(f"并行批次: {args.batch}")
    if args.tag:
        print(f"追加 tag: {args.tag}")

    batch_size = max(1, args.batch)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                _upload_one_task, t, args.api_url, args.api_key, args.version_intent, args.reason
            )
            for t in tasks
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            ok, task_id, err = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                errors.append((task_id or "unknown", err or "unknown error"))
            if idx % 10 == 0 or idx == total:
                print(f"已处理 {idx}/{total} 个任务，成功: {success}, 失败: {failed}...")

    print("\n" + "=" * 60)
    print("上传完成！")
    print("=" * 60)
    print(f"总任务数: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    if errors:
        print("\n失败详情（前10条）:")
        for task_id, err in errors[:10]:
            print(f"  - {task_id}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
