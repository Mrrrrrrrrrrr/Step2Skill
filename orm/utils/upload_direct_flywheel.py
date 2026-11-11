"""
直接上传 JSONL 到飞轮 /api/rollout/upload。
输入为 for_upload 格式 JSONL（含 task_id, instruction, traj, evaluation 等），
按参考 payload 结构（含 evaluation、created_at、existing_task_mode）上传，
图片处理与 upload_to_flywheel 一致（本地路径 multipart，URL 不传文件）。

用法:
    python utils/upload_direct_flywheel.py
    python utils/upload_direct_flywheel.py --input data/query_20260212_orm_testset_short_for_upload.jsonl --limit 2
    python utils/upload_direct_flywheel.py --evaluate
    python utils/upload_direct_flywheel.py --batch 10
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

# 项目根加入 path
_project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

DEFAULT_INPUT = os.path.join(_project_root, "data", "query_20260212_orm_testset_short_for_upload.jsonl")
DEFAULT_API_URL = "http://10.162.6.146:19003"


def _try_load_dotenv(dotenv_path: str = ".env") -> None:
    """加载 .env，同一 key 后出现的覆盖前面的。"""
    possible_paths = [
        dotenv_path,
        os.path.join(os.getcwd(), dotenv_path),
        os.path.join(_project_root, dotenv_path),
    ]
    for path in possible_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(path, override=True)
            return
        except Exception:
            pass
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    k, v = raw.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if " # " in v:
                        v = v.split(" # ")[0].strip()
                    if k:
                        os.environ[k] = v
            return
        except Exception:
            continue


_try_load_dotenv(".env")
API_URL = os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_URL", DEFAULT_API_URL)
API_KEY = (
    os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_KEY")
    or os.environ.get("GUI_AGENT_SYS_API_KEY")
)


def upload_one(
    record: Dict[str, Any],
    api_url: str,
    api_key: Optional[str],
    evaluate: bool = False,
) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
    """
    上传单条记录到 /api/rollout/upload。
    traj -> steps（image -> image_0, image_1,...），本地图加入 files，payload 含 evaluation/created_at/existing_task_mode。
    返回 (ok, task_id, result, error)。
    """
    task_id = record.get("task_id", "")
    if not task_id:
        return False, "", None, "task_id is missing"

    traj = record.get("traj", [])
    if not isinstance(traj, list) or not traj:
        return False, task_id, None, "traj is empty or not a list"

    api_url = api_url.strip().rstrip("/")
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    prepared_steps: List[Dict[str, Any]] = []
    files: List[Tuple[str, Tuple[str, Any, str]]] = []
    file_handles: List[Any] = []
    step_index = 0

    try:
        for raw_index, step in enumerate(traj):
            if not isinstance(step, dict):
                continue
            image_path = step.get("image", "") or step.get("image_local_path", "")
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            image_path = image_path.strip()

            if not image_path.startswith(("http://", "https://")):
                if not os.path.exists(image_path):
                    continue
                mime_type, _ = mimetypes.guess_type(image_path)
                handle = open(image_path, "rb")
                file_handles.append(handle)
                image_key = f"image_{step_index}"
                files.append((
                    image_key,
                    (os.path.basename(image_path), handle, mime_type or "application/octet-stream"),
                ))
            else:
                image_key = f"image_{step_index}"

            prepared_step = dict(step)
            prepared_step.setdefault("index", step_index)
            prepared_step["image"] = image_key
            if not prepared_step.get("image_local_path") and image_path:
                prepared_step["image_local_path"] = image_path
            prepared_steps.append(prepared_step)
            step_index += 1

        if not prepared_steps:
            return False, task_id, None, "no valid steps after preparation"

        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        tags = record.get("tags") or []
        if tags and not metadata.get("tags"):
            metadata = dict(metadata)
            metadata["tags"] = tags

        payload = {
            "task_id": task_id,
            "instruction": record.get("instruction", ""),
            "steps": prepared_steps,
            "metadata": metadata,
            "tags": tags,
            "source": record.get("source", "Backflow"),
            "created_at": record.get("created_at"),
            "evaluation": record.get("evaluation"),
            "auto_evaluate": bool(evaluate),
            "existing_task_mode": "overwrite",
        }

        resp = requests.post(
            f"{api_url}/api/rollout/upload",
            data={"payload": json.dumps(payload, ensure_ascii=False)},
            files=files if files else None,
            headers=headers,
            timeout=120,
        )
        for h in file_handles:
            try:
                h.close()
            except Exception:
                pass

        if not resp.ok:
            return False, task_id, None, f"HTTP {resp.status_code}: {resp.text[:500]}"
        result = resp.json()
        return True, task_id, result, None
    except Exception as e:
        for h in file_handles:
            try:
                h.close()
            except Exception:
                pass
        return False, task_id, None, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="直接上传 JSONL 到飞轮 /api/rollout/upload")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="输入 JSONL 路径 (默认: data/query_20260212_orm_testset_short_for_upload.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多上传条数 (默认: 全部)")
    parser.add_argument("--evaluate", action="store_true", help="是否 auto_evaluate")
    parser.add_argument("--batch", type=int, default=10, help="并行上传条数 (默认: 10)")
    parser.add_argument("--retry", type=int, default=10, help="单条上传失败时的重试次数 (默认: 10)")
    args = parser.parse_args()

    api_url = os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_URL", DEFAULT_API_URL)
    api_key = os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_KEY") or os.environ.get("GUI_AGENT_SYS_API_KEY")
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    records: List[Tuple[int, Dict[str, Any]]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if args.limit is not None and len(records) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}")
                continue
            records.append((line_num, record))

    total = len(records)
    success = 0
    failed = 0
    errors: List[Dict[str, str]] = []
    done = 0

    def do_upload(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, bool, str, Optional[Dict], Optional[str]]:
        line_num, record = item
        max_retries = max(1, args.retry)
        last_err: Optional[str] = None
        tid = record.get("task_id", "")
        for attempt in range(max_retries):
            ok, tid, result, err = upload_one(record, api_url, api_key, evaluate=args.evaluate)
            if ok:
                return line_num, True, tid, result, None
            last_err = err
            if attempt < max_retries - 1:
                time.sleep(2)
        return line_num, False, tid, None, last_err

    batch_size = max(1, args.batch)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(do_upload, item): item for item in records}
        for future in as_completed(futures):
            line_num, ok, tid, result, err = future.result()
            done += 1
            if ok:
                success += 1
                if done % 10 == 0 or done <= 3:
                    print(f"  [{done}/{total}] {tid} OK")
            else:
                failed += 1
                errors.append({"task_id": tid, "error": err or "unknown"})
                print(f"  [{done}/{total}] {tid} 失败: {err}")

    print()
    print("=" * 60)
    print("上传完成")
    print("=" * 60)
    print(f"总条数: {total}  成功: {success}  失败: {failed}")
    if errors:
        print("失败详情 (前 10 条):")
        for item in errors[:10]:
            print(f"  - {item['task_id']}: {item['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
