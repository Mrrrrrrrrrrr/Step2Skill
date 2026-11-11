"""
导出原始任务数据（JSONL）

功能：
- 调用后端 `/api/query` 接口
- 支持按 tag / created_at 时间范围筛选
- 将返回的 tasks 原样导出为 JSONL（每行一个 task dict）

环境变量（推荐写到项目根目录的 `.env` 或系统环境变量）：
- GUI_AGENT_FLYWHEEL_SYS_API_URL: 例如 http://10.162.200.22:19003
- GUI_AGENT_FLYWHEEL_SYS_API_KEY: API Key（必填，除非命令行传 --api-key）

提示：
- 仓库内提供了 `env.example` 模板文件：复制为 `.env` 后填入真实 key 即可使用。

示例：
  python utils/export_raw_data.py --tag 2026031618351974 --out data/batch_export_2026031618351974.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import requests


ENV_API_URL_KEY = "GUI_AGENT_FLYWHEEL_SYS_API_URL"
ENV_API_KEY_KEY = "GUI_AGENT_FLYWHEEL_SYS_API_KEY"


def _try_load_dotenv(dotenv_path: str = ".env") -> None:
    """
    尝试加载 .env：
    - 如果安装了 python-dotenv，则使用它
    - 否则做一个最简解析（仅支持 KEY=VALUE，忽略注释/空行）
    """
    if not dotenv_path:
        return
    if not os.path.exists(dotenv_path):
        return

    # 1) python-dotenv（如果可用）
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path)
        return
    except Exception:
        pass

    # 2) fallback：简单解析
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if "=" not in raw:
                    continue
                k, v = raw.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                # 不覆盖已存在的环境变量
                os.environ.setdefault(k, v)
    except Exception:
        # .env 解析失败不应阻塞主流程
        return


def _normalize_tags(tags: Optional[Union[Sequence[str], str]]) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        t = tags.strip()
        return [t] if t else []
    out: List[str] = []
    for t in tags:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            out.append(s)
    return out


def query_api_by_filters(
    *,
    tags: Optional[Union[Sequence[str], str]] = None,
    tags_mode: Literal["and", "or"] = "or",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    instruction: Optional[str] = None,
    instruction_contains: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: int = 30,
    verify_tls: bool = True,
    # 如果服务端支持分页/限制（常见字段：limit/skip），可用它来减少单次返回量
    limit: Optional[int] = None,
    skip: Optional[int] = None,
) -> Dict[str, Any]:
    """
    调用 `/api/query`，返回 tasks 等字段。

    Args:
        tags: tag 列表或单个 tag 字符串；None/空表示不按 tag 过滤
        tags_mode:
            - "or": {"tags": {"$in": tags}}
            - "and": {"tags": {"$all": tags}}
        instruction: 任务名 / 指令，精确匹配（等价于 {"instruction": instruction}）
        instruction_contains: 任务名包含关键字，使用 Mongo 正则（{"instruction": {"$regex": kw, "$options": "i"}}）
        start_time/end_time: created_at 范围（字符串原样下发）
        api_url/api_key: 不传则从环境变量读取
        limit/skip: 可选（服务端若忽略也不影响）
    """
    if api_url is None:
        api_url = os.environ.get(ENV_API_URL_KEY)
    if api_key is None:
        api_key = os.environ.get(ENV_API_KEY_KEY)

    tags_list = _normalize_tags(tags)
    instr_eq = (instruction or "").strip()
    instr_like = (instruction_contains or "").strip()
    start_ok = bool(start_time and str(start_time).strip())
    end_ok = bool(end_time and str(end_time).strip())
    tags_ok = bool(tags_list)
    instr_ok = bool(instr_eq or instr_like)
    if not (tags_ok or start_ok or end_ok or instr_ok):
        raise ValueError("tags / start_time / end_time / instruction 至少需要提供一个")

    base = (api_url or "").strip().rstrip("/")
    if not base:
        raise ValueError(f"{ENV_API_URL_KEY} 为空（可用 --api-url 或环境变量设置）")
    if not api_key or not str(api_key).strip():
        raise ValueError(f"{ENV_API_KEY_KEY} 未设置（可用 --api-key 或环境变量设置）")

    query_and: List[dict] = []

    if start_ok or end_ok:
        created_at: dict = {}
        if start_ok:
            created_at["$gte"] = str(start_time).strip()
        if end_ok:
            created_at["$lt"] = str(end_time).strip()
        query_and.append({"created_at": created_at})

    if tags_ok:
        if tags_mode == "and":
            query_and.append({"tags": {"$all": tags_list}})
        else:
            query_and.append({"tags": {"$in": tags_list}})

    if instr_eq:
        # 精确匹配任务名（instruction 字段）
        query_and.append({"instruction": instr_eq})
    elif instr_like:
        # 模糊匹配（不区分大小写），依赖后端以 Mongo 方式解析
        query_and.append({"instruction": {"$regex": instr_like, "$options": "i"}})

    payload: Dict[str, Any] = {"query": {"$and": query_and}}
    if limit is not None:
        payload["limit"] = int(limit)
    if skip is not None:
        payload["skip"] = int(skip)

    headers = {
        "X-API-Key": str(api_key).strip(),
        "Content-Type": "application/json",
    }
    query_url = base if base.endswith("/api/query") else f"{base}/api/query"

    resp = requests.post(
        query_url,
        headers=headers,
        json=payload,
        timeout=timeout_s,
        verify=verify_tls,
    )
    content_type = (resp.headers.get("Content-Type") or "").lower()
    data: Any
    if "application/json" in content_type:
        try:
            data = resp.json()
        except Exception:
            data = {}
    else:
        data = {}

    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    total_count = data.get("total_count") if isinstance(data, dict) else None

    return {
        "status_code": resp.status_code,
        "data": data,
        "tasks": tasks,
        "total_count": total_count,
        "text": resp.text,
        "query_url": query_url,
        "payload": payload,
    }


def export_tasks_jsonl(tasks: Sequence[Dict[str, Any]], output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    return len(tasks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _try_load_dotenv(".env")

    parser = argparse.ArgumentParser(description="导出原始任务数据（JSONL）")
    parser.add_argument("--tag", type=str, default=None, help="单个 tag（等价于 --tags TAG）")
    parser.add_argument("--tags", type=str, nargs="*", default=None, help="多个 tag（空表示不按 tag 过滤）")
    parser.add_argument("--tags-mode", type=str, choices=["and", "or"], default="or", help="多 tag 匹配方式")
    parser.add_argument("--instruction", type=str, default=None, help="按任务名精确匹配（instruction 字段）")
    parser.add_argument("--instruction-contains", type=str, default=None, help="按任务名包含关键字模糊匹配（不区分大小写）")
    parser.add_argument("--start-time", type=str, default=None, help='created_at 下界（例如 "2026-02-01 00:00:00"）')
    parser.add_argument("--end-time", type=str, default=None, help='created_at 上界（例如 "2026-03-01 00:00:00"）')
    parser.add_argument("--api-url", type=str, default=None, help=f"覆盖环境变量 {ENV_API_URL_KEY}")
    parser.add_argument("--api-key", type=str, default=None, help=f"覆盖环境变量 {ENV_API_KEY_KEY}")
    parser.add_argument("--out", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时（秒）")
    parser.add_argument("--insecure", action="store_true", help="关闭 TLS 校验（仅 https 时有意义）")
    parser.add_argument("--limit", type=int, default=None, help="可选：单次返回限制（服务端若不支持会忽略）")
    parser.add_argument("--skip", type=int, default=None, help="可选：跳过条数（服务端若不支持会忽略）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tags: Optional[Union[List[str], str]]
    if args.tag and (args.tags is None or len(args.tags) == 0):
        tags = args.tag
    elif args.tags is not None:
        tags = args.tags
    else:
        tags = None

    # 同时传 instruction 和 instruction_contains 时，优先使用精确匹配
    instruction = args.instruction
    instruction_contains = None if instruction else args.instruction_contains

    result = query_api_by_filters(
        tags=tags,
        tags_mode=args.tags_mode,
        start_time=args.start_time,
        end_time=args.end_time,
        instruction=instruction,
        instruction_contains=instruction_contains,
        api_url=args.api_url,
        api_key=args.api_key,
        timeout_s=args.timeout,
        verify_tls=not args.insecure,
        limit=args.limit,
        skip=args.skip,
    )

    if result["status_code"] != 200:
        msg = str(result.get("text", ""))[:800]
        raise RuntimeError(f"导出失败: status={result['status_code']}, body={msg}")

    tasks_any = result.get("tasks", [])
    tasks: List[Dict[str, Any]] = [t for t in tasks_any if isinstance(t, dict)]
    exported = export_tasks_jsonl(tasks, args.out)

    total_count = result.get("total_count")
    print(f"status_code={result['status_code']}, tasks={exported}, total_count={total_count}, out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

