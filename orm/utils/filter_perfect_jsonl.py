#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def get_final_result(obj: dict) -> str | None:
    if "final_result" in obj:
        return obj.get("final_result")
    return (
        obj.get("evaluation", {})
        .get("prediction", {})
        .get("vote", {})
        .get("final_result")
    )


def filter_perfect(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if get_final_result(obj) == "perfect":
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    return total, kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Filter JSONL rows where "final_result" == "perfect".'
    )
    parser.add_argument("input", type=Path, help="Input JSONL path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input_stem>_perfect.jsonl)",
    )
    args = parser.parse_args()

    input_path = args.input
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_perfect.jsonl")
    else:
        output_path = args.output

    total, kept = filter_perfect(input_path, output_path)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"total={total}, kept={kept}")


if __name__ == "__main__":
    main()
