"""
Flask 服务：单条 ORM 预测接口。
POST /predict 传入 {"task": <JSONL一行>, "params": {...}}，返回带 orm_prediction（及可选 orm_takeover_prediction）的 JSON。
"""
import logging
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from main import predict_one, configure_knowledge_base, SUPPORTED_KNOWLEDGE_BASES
from utils.input_resolver import normalize_task_for_missing_tags

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "service.log")
DEFAULT_MODEL_NAME = "seed_free"
DEFAULT_PIPELINE_JSON_PATH = os.path.join(
    PROJECT_ROOT, "pipeline", "pipeline_lite_action_validated_trajectory_allimg_validated.json"
)
DEFAULT_KNOWLEDGE_BASE = os.environ.get("ORM_KNOWLEDGE", "knowledge_base").strip() or "knowledge_base"

app = Flask(__name__)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
# 避免请求日志重复到 handler
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _load_default_pipeline_json():
    try:
        with open(DEFAULT_PIPELINE_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        app.logger.warning("failed to load default pipeline json %s: %s", DEFAULT_PIPELINE_JSON_PATH, e)
        return None


DEFAULT_PIPELINE_JSON = _load_default_pipeline_json()


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "service": "orm-predict"})


@app.route("/readyz", methods=["GET"])
def readyz():
    checks = {
        "default_pipeline_loaded": DEFAULT_PIPELINE_JSON is not None,
        "predict_one_callable": callable(predict_one),
    }
    if all(checks.values()):
        return jsonify({"status": "ready", "checks": checks})
    return jsonify({"status": "not_ready", "checks": checks}), 503


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception as e:
        app.logger.warning("invalid json body: %s", e)
        return jsonify({"error": "invalid json", "detail": str(e)}), 400
    if not data:
        return jsonify({"error": "body is empty"}), 400

    task = data.get("task")
    if task is None:
        return jsonify({"error": "missing field: task"}), 400
    if not isinstance(task, dict):
        return jsonify({"error": "field task must be an object"}), 400
    task = normalize_task_for_missing_tags(task)
    params = data.get("params") or {}
    params.pop("save_data", None)
    params.pop("enable_timing", None)
    if "model_name" not in params:
        params["model_name"] = DEFAULT_MODEL_NAME
    if "knowledge" not in params:
        params["knowledge"] = DEFAULT_KNOWLEDGE_BASE
    # lite 模式默认 true；仅在 lite 下未传压缩质量时默认 0.6，非 lite 不压缩
    if "lite_mode" not in params:
        params["lite_mode"] = True
    if params.get("lite_mode") and params.get("compress_quality") is None:
        params["compress_quality"] = 0.6
    # 未显式传 pipeline_json 时，使用服务端默认 pipeline 配置
    if "pipeline_json" not in params and DEFAULT_PIPELINE_JSON is not None:
        params["pipeline_json"] = DEFAULT_PIPELINE_JSON
        if "pipeline_preset" not in params and isinstance(DEFAULT_PIPELINE_JSON, dict):
            default_preset = DEFAULT_PIPELINE_JSON.get("preset")
            if default_preset:
                params["pipeline_preset"] = default_preset

    try:
        result = predict_one(task, **params)
    except Exception as e:
        app.logger.exception("predict_one failed")
        return jsonify({"error": "prediction failed", "detail": str(e)}), 500

    task_id = task.get("task_id", "")
    app.logger.info("predict done task_id=%s", task_id)
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 ORM 预测服务")
    parser.add_argument(
        "--knowledge",
        type=str,
        default=DEFAULT_KNOWLEDGE_BASE,
        choices=list(SUPPORTED_KNOWLEDGE_BASES),
        help="默认评估规则知识库目录（请求未显式传 knowledge 时生效）",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=5000, help="服务监听端口")
    args = parser.parse_args()

    try:
        DEFAULT_KNOWLEDGE_BASE = configure_knowledge_base(args.knowledge, verbose=True)
    except Exception as e:
        raise SystemExit(f"知识库配置失败: {e}")

    app.run(host=args.host, port=args.port)
