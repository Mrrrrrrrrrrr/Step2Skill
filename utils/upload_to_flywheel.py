"""
批量上传 JSONL 文件中的任务数据到 GUI Agent 数据飞轮系统。

依赖:
    pip install requests python-dotenv

环境变量(可选，写到 .env 文件):
    GUI_AGENT_FLYWHEEL_SYS_API_URL=http://10.162.200.22:19003
    GUI_AGENT_FLYWHEEL_SYS_API_KEY=你的key

用法:
    python utils/upload_to_flywheel.py
    python utils/upload_to_flywheel.py --input data/your_file.jsonl
    python utils/upload_to_flywheel.py --input data/your_file.jsonl --evaluate
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import mimetypes
import glob
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# 项目根加入 path，便于 import config 与 utils
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from utils.calculate_accuracy import load_judgment_from_save_dir

DEFAULT_INPUT_FILE = "data/query_20260212_orm_testset_short.jsonl"


def _try_load_dotenv(dotenv_path: str = ".env") -> None:
    """尝试加载 .env 文件。同一 key 出现多次时，后出现的覆盖前面的（与 .env 中 12-13 行两处 URL 时取后者一致）。"""
    possible_paths = [
        dotenv_path,
        os.path.join(os.getcwd(), dotenv_path),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), dotenv_path),
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


# 模块加载时先读 .env，再从环境变量取默认 API 配置（与 .env 11-21 行对应）
_try_load_dotenv(".env")
DEFAULT_API_URL = os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_URL")
DEFAULT_API_KEY = os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_KEY")


def _get_default_api_url() -> str:
    """获取默认 API URL"""
    return os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_URL", DEFAULT_API_URL)


def _get_default_api_key() -> Optional[str]:
    """获取默认 API Key"""
    return os.environ.get("GUI_AGENT_FLYWHEEL_SYS_API_KEY")


def upload_task_to_flywheel(
    task_data: Dict[str, Any],
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    evaluate: bool = False,
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    上传单个任务到飞轮系统（使用 /api/rollout/upload 端点）。
    参考 data_2_flywheel.py 的 upload_trajectory 函数。

    图片处理逻辑：
    - 本地路径：上传文件，image 字段替换为占位符 image_0, image_1, ...
    - 已有 URL：跳过该步骤的图片上传（不支持直接传 URL）

    Args:
        task_data: 任务数据字典（包含 task_id, instruction, traj 等）
        api_url: 飞轮系统 API 地址
        api_key: API Key
        evaluate: 是否评估
        format: 数据格式（保留兼容性）

    Returns:
        dict: {
            "ok": bool,
            "task_id": str,
            "result": dict|None,
            "error": str|None,
        }
    """
    task_id = task_data.get("task_id", "")
    if not task_id:
        return {
            "ok": False,
            "task_id": "",
            "result": None,
            "error": "task_id is missing",
        }

    api_url = (api_url or _get_default_api_url()).strip().rstrip("/")
    api_key = api_key or _get_default_api_key()

    if not api_url:
        return {
            "ok": False,
            "task_id": task_id,
            "result": None,
            "error": "API URL is not set",
        }

    if not api_key:
        return {
            "ok": False,
            "task_id": task_id,
            "result": None,
            "error": "API Key is not set",
        }

    traj = task_data.get("traj", [])
    if not isinstance(traj, list) or not traj:
        return {
            "ok": False,
            "task_id": task_id,
            "result": None,
            "error": "traj is empty or not a list",
        }

    headers = {
        "Accept": "application/json",
        "X-API-Key": api_key,
    }

    # 将 traj 转换为 steps，处理图片文件上传
    prepared_steps: List[Dict[str, Any]] = []
    files: List[tuple] = []
    file_handles: List[Any] = []

    try:
        for index, step in enumerate(traj):
            if not isinstance(step, dict):
                continue

            image_path = step.get("image_local_path", "") or step.get("image", "")

            if not isinstance(image_path, str) or not image_path.strip():
                continue

            image_path = image_path.strip()

            image_key = f"image_{index}"
            prepared_step = dict(step)
            prepared_step.setdefault("index", index)

            # 如果是本地文件路径，上传文件
            if not image_path.startswith(("http://", "https://")):
                if not os.path.exists(image_path):
                    # 文件不存在，跳过
                    continue

                # 记录 image_local_path
                if not prepared_step.get("image_local_path"):
                    prepared_step["image_local_path"] = image_path

                # 替换 image 为占位符
                prepared_step["image"] = image_key

                mime_type, _ = mimetypes.guess_type(image_path)
                handle = open(image_path, "rb")
                file_handles.append(handle)
                files.append((
                    image_key,
                    (
                        os.path.basename(image_path),
                        handle,
                        mime_type or "application/octet-stream",
                    ),
                ))
            # else: 已有 URL，保留原始 URL（不上传文件）

            prepared_steps.append(prepared_step)

        if not prepared_steps:
            return {
                "ok": False,
                "task_id": task_id,
                "result": None,
                "error": "No valid steps after preparation",
            }

        # 构建 payload
        metadata = task_data.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        tags = task_data.get("tags", [])
        if tags and not metadata.get("tags"):
            metadata = dict(metadata)
            metadata["tags"] = tags

        payload = {
            "instruction": task_data.get("instruction", ""),
            "steps": prepared_steps,
            "metadata": metadata,
            "task_id": task_id,
            "source": task_data.get("source", "Backflow"),
            "auto_evaluate": bool(evaluate),
            "existing_task_mode": "overwrite",
        }

        response = requests.post(
            f"{api_url}/api/rollout/upload",
            data={"payload": json.dumps(payload, ensure_ascii=False)},
            files=files if files else None,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        return {
            "ok": True,
            "task_id": task_id,
            "result": result,
            "error": None,
        }
    except requests.RequestException as e:
        body = ""
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                body = resp.text[:500]
            except Exception:
                pass

        return {
            "ok": False,
            "task_id": task_id,
            "result": None,
            "error": f"{e}; body={body}",
        }
    except Exception as e:
        return {
            "ok": False,
            "task_id": task_id,
            "result": None,
            "error": str(e),
        }
    finally:
        for handle in file_handles:
            try:
                handle.close()
            except Exception:
                pass


def load_takeover_predictions(task_id: str, model_name: str) -> Dict[int, int]:
    """
    从 save 目录加载每步 takeover 预测。
    文件名格式：takeover_judgment_step_{step_index}_{gt}_to_{pred}.json
    返回 step_index -> pred (0/1/-1)，仅包含存在文件的步。
    """
    result: Dict[int, int] = {}
    try:
        save_dir = config.get_save_dir(model_name)
        task_dir = os.path.join(save_dir, task_id)
        if not os.path.exists(task_dir):
            return result
        pattern = os.path.join(task_dir, "takeover_judgment_step_*.json")
        for path in glob.glob(pattern):
            basename = os.path.basename(path)
            if "_to_" not in basename or not basename.endswith(".json"):
                continue
            left, right = basename.replace(".json", "").rsplit("_to_", 1)
            parts = left.split("_")
            if len(parts) < 4:
                continue
            try:
                step_index = int(parts[3])
                pred_str = right
                if pred_str == "-1":
                    pred = -1
                else:
                    pred = int(pred_str)
                result[step_index] = pred
            except (ValueError, IndexError):
                continue
    except Exception:
        pass
    return result


def add_prediction_fields_to_task(task: Dict[str, Any], model_name: str) -> None:
    """
    对 task 原地添加三种预测字段：traj[].value.takeover_prediction、evaluation.orm_prediction。
    仅当预测值为 0/1 时写入对应字段；-1 或缺失则不写。
    """
    task_id = task.get("task_id", "")
    if not task_id:
        return
    evaluation = task.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return
    is_task_completed = evaluation.get("is_task_completed", None)
    is_trajectory_correct = evaluation.get("is_trajectory_correct", None)
    if is_task_completed is not None and is_trajectory_correct is not None:
        ground_truth = 1 if (is_task_completed == 1 and is_trajectory_correct == 1) else 0
    else:
        ground_truth = None

    judgment, _ = load_judgment_from_save_dir(task_id, ground_truth, model_name)
    if judgment is None:
        return

    c_pr = judgment.get("c_pr", -1)
    t_pr = judgment.get("t_pr", -1)

    takeover_map = load_takeover_predictions(task_id, model_name)
    traj = task.get("traj") or []
    for step_idx, step in enumerate(traj):
        if step_idx in takeover_map and takeover_map[step_idx] in (0, 1):
            if "value" not in step:
                step["value"] = {}
            step["value"]["takeover_prediction"] = takeover_map[step_idx] == 1

    orm_prediction: Dict[str, int] = {}
    if c_pr in (0, 1):
        orm_prediction["is_task_completed"] = c_pr
    if t_pr in (0, 1):
        orm_prediction["is_trajectory_correct"] = t_pr
    if orm_prediction:
        if "evaluation" not in task or task["evaluation"] is None:
            task["evaluation"] = {}
        task["evaluation"]["orm_prediction"] = orm_prediction


def batch_upload_jsonl(
    input_jsonl: str,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    evaluate: bool = False,
    format: Optional[str] = None,
    limit: Optional[int] = None,
    batch: int = 10,
) -> Dict[str, Any]:
    """
    批量上传 JSONL 文件中的任务到飞轮系统，支持 batch 个任务并行上传。

    Args:
        input_jsonl: 输入 JSONL 文件路径
        api_url: 飞轮系统 API 地址
        api_key: API Key
        evaluate: 是否评估
        format: 数据格式
        limit: 限制上传的任务数量（None 表示全部）
        batch: 并行上传的批次大小（默认 10）

    Returns:
        dict: 统计信息
    """
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    api_url = api_url or _get_default_api_url()
    api_key = api_key or _get_default_api_key()

    print(f"正在读取: {input_jsonl}")
    print(f"API 地址: {api_url}")
    print(f"评估模式: {evaluate}")
    print(f"并行批次: {batch}")
    if limit:
        print(f"限制数量: {limit}")

    # 先顺序读入所有任务，保持顺序
    tasks: List[Tuple[str, Dict[str, Any]]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(tasks) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                task_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析错误: {e}")
                continue
            task_id = task_data.get("task_id", f"line_{line_num}")
            tasks.append((task_id, task_data))

    stats["total"] = len(tasks)

    def _upload_one(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        tid, tdata = item
        res = upload_task_to_flywheel(
            task_data=tdata,
            api_url=api_url,
            api_key=api_key,
            evaluate=evaluate,
            format=format,
        )
        return (tid, res)

    # 按 batch 并行上传
    for start in range(0, len(tasks), batch):
        chunk = tasks[start : start + batch]
        workers = min(batch, len(chunk))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_upload_one, chunk))
        for task_id, result in results:
            if result["ok"]:
                stats["success"] += 1
                conversion = result.get("result", {}).get("conversion")
                if isinstance(conversion, dict):
                    total_in = conversion.get("total_in", 0)
                    total_out = conversion.get("total_out", 0)
                    skipped = conversion.get("skipped", 0)
                    if skipped > 0:
                        print(f"  任务 {task_id}: total_in={total_in}, total_out={total_out}, skipped={skipped}")
            else:
                stats["failed"] += 1
                error_msg = result.get("error", "unknown error")
                stats["errors"].append({"task_id": task_id, "error": error_msg})
                print(f"  失败 {task_id}: {error_msg}")
        if (start + len(chunk)) % 10 == 0 or start + len(chunk) == len(tasks):
            print(f"已处理 {start + len(chunk)}/{len(tasks)} 个任务，成功: {stats['success']}, 失败: {stats['failed']}...")

    # 输出统计信息
    print("\n" + "="*60)
    print("上传完成！")
    print("="*60)
    print(f"总任务数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")
    if stats["errors"]:
        print(f"\n失败详情（前10条）:")
        for item in stats["errors"][:10]:
            print(f"  - {item['task_id']}: {item['error']}")
    print("="*60)

    return stats


def _cli() -> None:
    """命令行入口"""
    _try_load_dotenv(".env")

    parser = argparse.ArgumentParser(description="批量上传 JSONL 文件中的任务到飞轮系统")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"输入 JSONL 文件路径 (默认: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"飞轮系统 API 地址 (默认: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key (默认从环境变量读取)",
    )
    parser.add_argument(
        "--add-prediction-fields",
        default=True,
        help="是否添加预测字段",
    )    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="是否评估任务",
    )    
    parser.add_argument(
        "--format",
        type=str,
        choices=["trace", "default"],
        default=None,
        help="数据格式 (默认: 自动判断)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制上传的任务数量 (默认: 全部)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="模型名称，用于 evaluate 时从 save 目录读取预测 (默认: gpt-4o)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="并行处理/上传的批次大小 (默认: 10)",
    )
    args = parser.parse_args()

    # 加载 .env 后重新读取环境变量
    api_url = args.api_url or _get_default_api_url()
    api_key = args.api_key or _get_default_api_key()

    if not api_key:
        print("错误: GUI_AGENT_FLYWHEEL_SYS_API_KEY 未设置")
        print("请在 .env 文件中设置，或使用 --api-key 参数")
        return

    input_jsonl = args.input
    batch_size = args.batch

    if args.add_prediction_fields:
        base, ext = os.path.splitext(input_jsonl)
        output_path = base + "_for_upload" + ext
        print(f"预测模式: 先为输入添加预测字段并写入 {output_path}，再上传该文件（并行 batch={batch_size}）")
        if not os.path.exists(input_jsonl):
            print(f"错误: 输入文件不存在: {input_jsonl}")
            return
        # 顺序读入所有任务
        tasks_list: List[Dict[str, Any]] = []
        with open(input_jsonl, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行 JSON 解析错误: {e}")
                    continue
                tasks_list.append(task)
        # 按 batch 并行补充字段（原地修改）
        for start in range(0, len(tasks_list), batch_size):
            chunk = tasks_list[start : start + batch_size]
            workers = min(batch_size, len(chunk))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(lambda t: add_prediction_fields_to_task(t, args.model), chunk))
        with open(output_path, "w", encoding="utf-8") as fout:
            for task in tasks_list:
                fout.write(json.dumps(task, ensure_ascii=False) + "\n")
        print(f"已写入带预测字段的文件: {output_path}")
        input_jsonl = output_path

    batch_upload_jsonl(
        input_jsonl=input_jsonl,
        api_url=api_url,
        api_key=api_key,
        evaluate=args.evaluate,
        format=args.format,
        limit=args.limit,
        batch=batch_size,
    )


if __name__ == "__main__":
    _cli()
