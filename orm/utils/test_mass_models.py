"""
测试所有 MaSS API 模型，检查哪些可用
使用 utils/ask_qwen.py 中的测试函数
"""
import sys
import os
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.ask_qwen import quick_test_mass_api_llm, quick_test_mass_api_vlm

print("=" * 80)
print("测试所有 MaSS API 模型")
print("=" * 80)
print()

# 获取所有 MaSS API 模型
mass_models = config.MASS_API_MODELS.copy()
# mass_models = ['gpt-5.2']
print(f"共 {len(mass_models)} 个 MaSS API 模型需要测试:")
for i, model in enumerate(mass_models, 1):
    mass_model_name = config.get_mass_model_name(model)
    print(f"  {i}. {model} -> {mass_model_name}")
print()

# 显示代理配置
print(f"代理配置: {config.MASS_API_PROXY_URL}")
print(f"API Base URL: {config.MASS_API_BASE_URL}")
print()

# 默认测试图片（如果存在）
default_image = r"/var/code/zhaoj/orm/data/gpt-4o-1/save/0a1708f843aaj4adba9f4c2157daec04/image_0.jpg"
test_image = default_image if os.path.exists(default_image) else None

if test_image:
    print(f"使用测试图片: {test_image}")
else:
    print("警告: 未找到测试图片，将只测试 LLM 模式")
print()

# 测试结果
results = {
    'llm': {},  # {model_key: (success, response_preview)}
    'vlm': {}   # {model_key: (success, response_preview)}
}
latency_sec = {
    'llm': {},  # {model_key: seconds}
    'vlm': {}   # {model_key: seconds}
}

# 测试 LLM 模式
print("\n" + "=" * 80)
print("测试 LLM 模式（纯文本）")
print("=" * 80)
for model_key in mass_models:
    print(f"\n测试模型: {model_key}")
    print("-" * 80)
    try:
        t0 = time.perf_counter()
        success = quick_test_mass_api_llm(model_key, "你好，请简单介绍一下你自己")
        elapsed = time.perf_counter() - t0
        latency_sec['llm'][model_key] = elapsed
        if success:
            results['llm'][model_key] = (True, "成功")
        else:
            results['llm'][model_key] = (False, "失败")
    except Exception as e:
        print(f"✗ 异常: {e}")
        results['llm'][model_key] = (False, f"异常: {str(e)[:50]}")
        # 异常也记录耗时（若无意义则留空）
        latency_sec['llm'][model_key] = None

# 测试 VLM 模式（如果有测试图片）
if test_image:
    print("\n" + "=" * 80)
    print("测试 VLM 模式（多模态）")
    print("=" * 80)
    for model_key in mass_models:
        print(f"\n测试模型: {model_key}")
        print("-" * 80)
        try:
            t0 = time.perf_counter()
            success = quick_test_mass_api_vlm(model_key, test_image, "描述这张图片的主要内容")
            elapsed = time.perf_counter() - t0
            latency_sec['vlm'][model_key] = elapsed
            if success:
                results['vlm'][model_key] = (True, "成功")
            else:
                results['vlm'][model_key] = (False, "失败")
        except Exception as e:
            print(f"✗ 异常: {e}")
            results['vlm'][model_key] = (False, f"异常: {str(e)[:50]}")
            latency_sec['vlm'][model_key] = None
else:
    print("\n跳过 VLM 模式测试（未提供测试图片）")

# 汇总结果
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

print("\n【LLM 模式测试结果】")
llm_success = [k for k, v in results['llm'].items() if v[0]]
llm_failed = [k for k, v in results['llm'].items() if not v[0]]

print(f"\n✓ 成功 ({len(llm_success)}/{len(mass_models)}):")
for model in llm_success:
    lat = latency_sec['llm'].get(model)
    lat_str = f"{lat:.2f}s" if isinstance(lat, (int, float)) else "N/A"
    print(f"  - {model}  (时延: {lat_str})")

if llm_failed:
    print(f"\n✗ 失败 ({len(llm_failed)}/{len(mass_models)}):")
    for model in llm_failed:
        status = results['llm'][model][1]
        print(f"  - {model}: {status}")

if test_image:
    print("\n【VLM 模式测试结果】")
    vlm_success = [k for k, v in results['vlm'].items() if v[0]]
    vlm_failed = [k for k, v in results['vlm'].items() if not v[0]]

    print(f"\n✓ 成功 ({len(vlm_success)}/{len(mass_models)}):")
    for model in vlm_success:
        lat = latency_sec['vlm'].get(model)
        lat_str = f"{lat:.2f}s" if isinstance(lat, (int, float)) else "N/A"
        print(f"  - {model}  (时延: {lat_str})")

    if vlm_failed:
        print(f"\n✗ 失败 ({len(vlm_failed)}/{len(mass_models)}):")
        for model in vlm_failed:
            status = results['vlm'][model][1]
            print(f"  - {model}: {status}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
