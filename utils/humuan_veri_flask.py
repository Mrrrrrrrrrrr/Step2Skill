import os
import re
import time
import json
import base64
import mimetypes
import requests
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from postprocess_utils import read_labels, normalize_point, draw_line_and_save, process_boxes, draw_sliding_verification_result
import json
import requests
import base64
from mimetypes import guess_type
import concurrent.futures
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path as osp
from PIL import Image
import shutil
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import re
from PIL import Image, ImageDraw, ImageFont
import importlib.util
import sys
import logging

app = Flask(__name__)

# model_path = "/opt/nas/p/mm/ie_env/wanghui/YOLOv8-main/YOLOv8-main/runs/train3/weights/best.pt"
args = sys.argv
model_path = args[1]
device = args[2]
port = args[3]
log_file = args[4]

import base64
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
file_handler.setFormatter(file_formatter)

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# 添加到 logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# GPT4oWrapper 类保持不变，直接复制你的实现
image_all_save_fold_path = '/opt/nas/p/mm/ie_env/wanghui/人机验证登录/images_all_run_process/'
# 
# waibu_model_name = 'qwen235B'
# waibu_model_name = 'gemini-3-flash-preview-thinking'
waibu_model_name = 'doubao-seed-1-6-251015'

class GPT4oWrapper:
    def  __init__ (self, model_name='gpt-5', temperature=0.1, max_tokens=2048,
        url="https://ai.cloud.hihonor.com/ml-athena/cloud-access/aicloud-ml/labs/v1/model-api/openai/v3/chat/completions",
        headers={"Content-Type":"application/json","ml-source":"multimodal-team",
        "accessKey":"1001201",
        "sign":"NDAwODFDODNGNEQwQkY2N0Y2MkQ3NzhFODczMUExNzgxQThDQThEN0FCMTI1OEVDQjAyRUI2M0E0REM1NzJDRA==",
        "ml-referer":"multimodal-team","ts":"1730893852"}):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = url
        self.headers = headers

        # self.headers = {"Content-Type":"application/json","ml-source":"base-ui-agent-2",
        # "accessKey":"1001201",
        # "sign":"NDAwODFDODNGNEQwQkY2N0Y2MkQ3NzhFODczMUExNzgxQThDQThEN0FCMTI1OEVDQjAyRUI2M0E0REM1NzJDRA==",
        # "ml-referer":"base-ui-agent","ts":"1730893852"}


    def encode_image(self, image_path):
        """将图像文件编码为 Base64 格式"""
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found
        return mime_type, base64_image

    def parse_input(self, query=None, imgs=None, system_prompt=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        if imgs is None:
            messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
            return messages

        if isinstance(imgs, str):
            imgs = [imgs]

        content = []
        for img in imgs:
            mimeType, base64_image = self.encode_image(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mimeType};base64,{base64_image}"}})
        content.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content})
        return messages

    def chat(self, query, imgs, max_entries=10000, system_prompt=None):
        """单次聊天功能"""
        contents = self.parse_input(query, imgs, system_prompt)
        # print(contents)
        data = {
            "model": self.model_name,
            "messages": contents,
            # "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens
        }
        json_payload = json.dumps(data)
        try:
            response = requests.post(url=self.url, data=json_payload, headers=self.headers, timeout=60)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                # raise Exception(f"Error: {response.status_code}, {response.text}")
                # 暂时改成100次请求，避免断开
                if max_entries > 1:
                    print(f"Error: {response.status_code}, {response.text}. 等待3秒后重试...")
                    print(f'Error data: \n {imgs}')
                    time.sleep(3)  # 等待3秒
                    return self.chat(query, imgs, max_entries - 1)  # 重新尝试聊天，减少重试次数
                else:
                    raise Exception(f"Error: {response.status_code}, {response.text}. 达到最大重试次数，停止尝试。")
        except Exception as e:
            # 超时处理，重新发送请求
            if max_entries > 1:
                print(f"报错{e}，等待3秒后重试...")
                time.sleep(3)
                return self.chat(query, imgs, max_entries - 1, system_prompt)
            else:
                raise Exception("请求超时，达到最大重试次数，停止尝试。")

    def batch_chat(self, queries = None, imgs = None):
        # batch processing
        assert len(queries) == len(imgs)
        responses = []
        for i in range(len(queries)):
            responses.append(self.chat(queries[i], imgs[i]))
        return responses
    


def run_detection(image_path, save_label_folder):
    logger.info("开始检测")
    
    results = model(image_path)


    os.makedirs(save_label_folder, exist_ok=True)
    label_path = os.path.join(save_label_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

    with open(label_path, 'w') as f:
        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i].item())
                    x_center, y_center, w, h = r.boxes.xywhn[i].tolist()
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    return label_path

def classify_verification_scene(image_path):
    gpt_wrapper = GPT4oWrapper(model_name='gpt-4o')
    category_map = {
        "1": "拼图验证",
        "2": "滑动验证",
        "3": "点击指定图形验证",
        "4": "输入图片中内容验证",
        "5": "语序依次点击文字验证",
        "6": "滑块带停止条件验证",
        "7": "依次点击图标验证",
        "8": "按照轨迹绘制图形验证",
        "9": "最短连线验证",
        "10": "其他类型验证",
        "11": "不存在人机验证"
    }

    query = """你是一个专业的手机App人机验证场景分类器。你的任务是分析提供的手机App场景描述，并将其准确归类到以下六个预定义类别之一。请严格按照给定的分类标准进行判断。

分类标准：

1.  **拼图验证**: 场景中需要用户滑动一个滑块，将一个缺失的图片部分拖动到正确的位置以完成拼图，从而通过验证。
2.  **滑动验证**: 场景中需要用户将一个滑块从起始位置滑动到指定的目标位置（通常是屏幕的最右侧），以完成验证。
3.  **点击指定图形验证**: 场景中会显示一组图形，并提供一个或多个参照图形，用户需要根据参照图形，点击或选择图中与之对应的图形来完成验证。
4.  **输入图片中内容验证**: 场景中会显示一张包含文本内容的图片（通常是扭曲的字母、数字或组合），用户需要识别图片中的内容并将其输入到文本框中以完成验证。
5.  **语序依次点击文字验证**: 图片中通常为一个成语，需要按照成语的语序顺序依次进行点击完成验证。
6.  **滑块带停止条件验证**: 图中通常为一个滑块，需要刚刚露出完整的指定图形，之后松开。
7.  **依次点击图标验证**: 图中通常有多个图标，需要按照给定对照图标的顺序依次点击指定的图标完成验证。
8.  **按照轨迹绘制图形验证**: 图中为一个轨迹，需要按照轨迹滑动绘制以完成验证。
9.  **最短连线验证**: 图中通常为多个不同颜色的点，需要按照文字要求用最短线连接指定颜色完成验证。
10. **其他类型验证**: 场景中存在人机验证机制，但其类型不属于上述1-9中的任何一种，例如点击按钮完成快速验证等等。
11. **不存在人机验证**: 场景中不存在任何形式的App人机验证信息，即不需要进行人机验证。

输出格式：
请仅输出一个数字，并用`<answer>`标签包裹，不允许输出以下内容之外的其他内容。例如：

*   如果场景属于“拼图验证”，输出：`<answer>1</answer>`
*   如果场景属于“滑动验证”，输出：`<answer>2</answer>`
*   如果场景属于“点击指定图形验证”，输出：`<answer>3</answer>`
*   如果场景属于“输入图片中内容验证”，输出：`<answer>4</answer>`
*   如果场景属于“语序依次点击文字验证”，输出：`<answer>5</answer>`
*   如果场景属于“滑块带停止条件验证”，输出：`<answer>6</answer>`
*   如果场景属于“依次点击图标验证”，输出：`<answer>7</answer>`
*   如果场景属于“按照轨迹绘制图形验证”，输出：`<answer>8</answer>`
*   如果场景属于“最短连线验证”，输出：`<answer>9</answer>`
*   如果场景属于“其他类型验证”，输出：`<answer>10</answer>`
*   如果场景属于“不存在人机验证”，输出：`<answer>11</answer>`
"""

    logger.info("开始调用GPT进行人机验证场景分类...")
    gpt_response = gpt_wrapper.chat(query=query, imgs=image_path)
    # logger.info(f"GPT返回内容: {gpt_response}")
    match = re.search(r"<answer>(\d+)</answer>", gpt_response)

    category_number = "0"
    category_name = "未知类别"
    if match:
        category_number = match.group(1)
        category_name = category_map.get(category_number, "未知类别")
        logger.info(f"识别到的人机验证类别是: {category_name}")
    else:
        logger.info(f"未能从GPT返回中解析出有效的类别数字。")
    return int(category_number), category_name


query_tubiaoshunxudianji = """你是一个智能助手，负责帮助用户完成“依次点击图标验证”任务。验证图片中包含多个分散在不同位置的图标，同时给定一个需要按顺序点击的图标列表。你的任务是：

1. 识别图片中的所有图标，并根据给定的图标顺序，找到这些图标在图片中的位置。
2. 计算每个目标图标的中心点坐标，并将坐标归一化到 0–1000 区间：
   - 坐标原点为图片左上角
   - x 为横坐标，y 为纵坐标
3. 按照给定顺序输出点击指令，格式为：tap(x,y)
4. 输出要求：
   - 只输出点击指令，不包含任何解释或其他内容
   - 所有点击指令必须放在一个 <answer></answer> 标签中
   - 指令顺序必须与图标顺序一致

示例：
若归一化坐标依次为 (119,805)、(496,408)、(685,452)、(874,196)，则输出：

<answer>tap(119,805)tap(496,408)tap(685,452)tap(874,196)</answer>

请根据图片内容完成任务，并输出最终的点击指令。
"""
query_yuxudianji = """你是一个智能助手，负责帮助用户完成“语序依次点击文字验证”的任务。验证图片中会出现一个成语，成语的每个字分散在图片不同位置。你的任务是：

1. 识别图片中的所有文字，并根据成语的正确语序排列这些文字。
2. 计算每个文字的中心点坐标，将坐标归一化到0-1000区间，左上角为坐标原点，横坐标x，纵坐标y。
3. 按照成语的语序顺序，依次输出点击指令，格式为 tap(x,y)，共输出几个字就给出几个点击指令。
4. 输出时，只给出点击指令，不要多余解释，并且将所有点击指令放在一个 <answer></answer> 标签中，注意按照点击的顺序依次给出指令。

示例：

假设图片中成语为“软弱无能”，文字位置分别为：

- “软”在图片左下角，归一化坐标为(119, 805)
- “弱”在图片中间偏上，归一化坐标为(496, 408)
- “无”在图片中间偏右，归一化坐标为(685, 452)
- “能”在图片右上角，归一化坐标为(874, 196)

则输出应为：

<answer>tap(119,805)tap(496,408)tap(685,452)tap(874,196)</answer>

注意！！！！将坐标归一化到0-1000区间！！！！
注意！！！！将坐标归一化到0-1000区间！！！！

请根据图片内容，完成以上步骤，输出正确的点击指令。
"""
query_yaoqiudianji = """你是一个智能助手，负责帮助用户完成“点击指定图形验证”任务。系统会提供一个参照图形，你需要在图片中找到与这个参照图形对应的目标，并执行点击验证操作。

你的任务如下：

1. 识别图片中的所有图标，并识别要求中给定的需要点击的参照图形。
2. 查找图片中与参照图形对应的目标图标，并计算它们的中心点的相对坐标。
3. 将中心点坐标归一化到 0–1000 区间：
   - 坐标原点为图片左上角
   - x 为横坐标，y 为纵坐标
4. 按要求输出点击指令，格式为：tap(x,y)
5. 输出必须遵守以下规则：
   - 只输出点击指令，不包含任何解释、描述或额外文字
   - 所有点击指令必须放在一个 <answer></answer> 标签中

示例：
若归一化坐标为 (119,805)，则输出：

<answer>tap(119,805)</answer>

请根据图片内容完成任务，并输出最终的点击指令。
"""

query_zuiduanlianxian = """你是一名负责执行“最短连线验证”任务的智能助手。任务通常基于一张包含多种颜色点的图片。你需要根据文字说明，从图片中找到指定颜色的两个点，并生成用于连线的点击指令。

你的工作流程如下：

1. 识别说明中要求连接的目标颜色。
2. 在图片中定位该颜色的两个点，并计算它们的中心点相对坐标。
3. 将中心点坐标归一化到 0–1000 区间：
   - 原点位于图片左上角。
   - x 为横坐标，y 为纵坐标。
4. 输出点击指令，格式为：tap(x,y)。
5. 输出规则：
   - 只能输出点击指令，不包含任何解释或额外内容。
   - 所有点击指令必须写在一个 <answer></answer> 标签内。

示例：
若两个点归一化坐标为 (119,805) 和 (256,348)，则输出：
<answer>tap(119,805)tap(256,348)</answer>

注意！！！！将坐标归一化到0-1000区间！！！！
注意！！！！将坐标归一化到0-1000区间！！！！
注意！！！！将坐标归一化到0-1000区间！！！！
注意！！！！将坐标归一化到0-1000区间！！！！
注意！！！！将坐标归一化到0-1000区间！！！！
注意！！！！将坐标归一化到0-1000区间！！！！

请基于图片内容完成任务，并输出最终点击指令。
"""

query_dianjiguanbi = """你是一名专门负责**关闭人机验证界面**的智能助手。根据提供的图片，你需要准确找到用于关闭人机验证的按钮（通常为“X”），并输出点击该按钮的指令。

你的执行流程如下：

1. 在图片中识别并定位关闭人机验证的按钮。
2. 获取该按钮中心点的相对坐标。
3. 将坐标归一化到 0–1000 范围：
   - 原点在图片左上角。
   - x 为横坐标，y 为纵坐标。
4. 将结果输出为点击指令，格式：tap(x,y)。
5. 输出要求：
   - 只能输出点击指令，不得加入任何说明或其他内容。
   - 所有输出必须放入 <answer></answer> 标签中。

示例：
若坐标为 (436,864)，则输出：
<answer>tap(436,864)</answer>

请根据图片内容完成任务，并输出最终点击指令。
"""

query_guanbi_shuxin = """
**角色：** 你是一个图像分析专家，擅长识别界面元素。

**任务：** 我将为您提供一张图片。请仔细分析图片内容，判断其中是否存在“刷新”图标或文字，以及“关闭”图标或文字。

**判断规则及输出格式：**
1.  如果图片中**存在“关闭”图标或文字**，请输出：`<answer>1</answer>`
2.  如果图片中**不存在“关闭”图标或文字，但存在“刷新”图标或文字**，请输出：`<answer>0</answer>`
3.  如果图片中**既不存在“刷新”图标或文字，也不存在“关闭”图标或文字**，请输出：`<answer>2</answer>`

**请注意：** 你的回答必须且只能是上述三种`<answer>X</answer>`格式中的一种，不包含任何额外的文字、解释或标点符号。
"""


def extract_answer_tags(text):
    # 使用正则表达式匹配<answer>和</answer>及其中间内容
    pattern = r'(<answer>.*?</answer>)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]


def process_image_shunxudianji(input_image_path, model_name, query_input, category_name = '其他验证'):
    def extract_tap_list(text):
        """
        从符合格式的文本中提取所有tap指令，返回列表
        """
        inner = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not inner:
            return []
        content = inner.group(1)
        taps = re.findall(r'tap\((\d{1,4}),(\d{1,4})\)', content)
        # 返回坐标元组列表，数字转int
        return [(int(x), int(y)) for x, y in taps]

    def is_valid_format(text):
        """
        判断文本是否严格符合格式：
        <answer>tap(x1,y1)tap(x2,y2)...tap(xn,yn)</answer>
        其中x,y为0-1000的整数
        """
        pattern = r'^<answer>(tap\(\d{1,4},\d{1,4}\))+<\/answer>$'
        return re.match(pattern, text.strip()) is not None

    def draw_points_on_image(image_path, points, save_path):
        """
        在图片上绘制序号标注
        points: [(x1,y1), (x2,y2), ...]，归一化坐标0-1000
        save_path: 保存路径
        """
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # 选择字体，默认字体，或者你可以指定字体路径
        try:
            font = ImageFont.truetype("/opt/nas/p/mm/ie_env/sjg/Font/ARIAL.TTF", 80)
        except:
            font = ImageFont.load_default()

        for idx, (x_norm, y_norm) in enumerate(points, start=1):
            # 归一化坐标转为像素坐标
            x_pix = int(x_norm / 1000 * width)
            y_pix = int(y_norm / 1000 * height)

            # 画圆点
            radius = 20
            draw.ellipse((x_pix - radius, y_pix - radius, x_pix + radius, y_pix + radius), fill='red')

            # 画序号文字，偏移一点避免遮挡圆点
            text = str(idx)
            text_pos = (x_pix + radius + 2, y_pix - radius)
            draw.text(text_pos, text, fill='red', font=font)

        img.save(save_path)
        logger.info(f"带标注的图片已保存到: {save_path}")
    # GPT4 = GPT4oWrapper(model_name=model_name)
    if model_name != 'qwen235B':
        GPT4 = GPT4oWrapper(model_name=model_name)
    

    max_attempts = 50  # 防止死循环
    attempts = 0
    candidate = ""
    while attempts < max_attempts:
        if model_name == 'qwen235B':
            continuation_action = vllm_qwen3vl_235b_request(query=query_input, image_path=input_image_path)
            continuation_action = extract_answer_tags(continuation_action)
            candidate = continuation_action.strip()
        else:
            chat_results = GPT4.chat(query=query_input, imgs=[input_image_path])
            candidate = chat_results.strip()
        if is_valid_format(candidate):
            break
        attempts += 1

    if not candidate or not is_valid_format(candidate):
        logger.info(f"未能生成符合格式的结果")
        return []

    tap_points = extract_tap_list(candidate)
    logger.info(f"提取的坐标点列表： {tap_points}")

    # 生成带标注的图片路径
    save_path = input_image_path.replace(".png", "_marked.png").replace(".jpg", "_marked.jpg")

    draw_points_on_image(input_image_path, tap_points, save_path)
    if os.path.exists(save_path):
        os.makedirs(image_all_save_fold_path + category_name, exist_ok=True)
        shutil.copy(save_path, os.path.join(image_all_save_fold_path, category_name, os.path.basename(save_path)))

    return tap_points


query_shuruneirong = """你是一名智能助手，负责执行“图片文字内容验证”任务。系统会提供一张包含扭曲文字或数字的验证码图片，你需要识别其中的文本并完成验证流程。

你的能力包括：
- text(x: float, y: float, text_input: str) —— 在指定坐标输入文本（坐标为控件中心点，归一化为 0–1000 范围）
- tap(x: float, y: float) —— 点击指定坐标位置（同样归一化为 0–1000 范围）

坐标说明：
- 原点为图片左上角
- x 为横坐标，y 为纵坐标
- 所有坐标需归一化到 0–1000 范围

任务流程如下：

1. 识别验证码图片中的文本，并在输入框位置调用 text 指令完成输入。
2. 输入完成后，在验证按钮位置调用 tap 指令进行确认。
3. 必须严格按照 text 在前、tap 在后的顺序输出。
4. 输出格式要求：
   - 仅输出 text 和 tap 指令，不包含任何解释性文字
   - 所有指令必须包裹在 <answer></answer> 标签内
   - 标签内的指令需直接连续输出，不换行

示例：
若输入框坐标为 (119,805)，验证码为 "YS3V"，验证按钮坐标为 (496,408)，输出应为：

<answer>text(119,805,YS3V)tap(496,408)</answer>

请根据给定图片内容完成任务，并输出最终指令。
"""



def process_image_shuruneirong(input_image_path, model_name, query_input):
    def extract_info(text):
        """
        从 <answer>text(x,y,code)tap(x,y)...</answer> 中提取：
        - text_x, text_y: text 输入框坐标
        - text_code: 识别出的验证码内容
        - taps: 所有 tap 坐标列表
        """
        inner = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not inner:
            return None

        content = inner.group(1)

        # 提取 text(x,y,code)
        text_match = re.search(r'text\((\d{1,4}),(\d{1,4}),([^()]+)\)', content)
        if not text_match:
            return None

        text_x = int(text_match.group(1))
        text_y = int(text_match.group(2))
        text_code = text_match.group(3)

        # 提取所有 tap(x,y)
        taps = re.findall(r'tap\((\d{1,4}),(\d{1,4})\)', content)
        tap_list = [(int(x), int(y)) for x, y in taps]

        return {
            "text_coord": (text_x, text_y),
            "text_code": text_code,
            "tap_list": tap_list
        }

    def is_valid_format(text):
        """
        判断文本是否严格符合格式：
        <answer>text(x,y,code)tap(x,y)</answer>
        其中 x, y 为 0–1000 的整数，code 为任意不含括号的字符串
        """
        pattern = r'^<answer>text\(\d{1,4},\d{1,4},[^()]+\)tap\(\d{1,4},\d{1,4}\)</answer>$'
        return re.match(pattern, text.strip()) is not None

    def draw_points_with_info(image_path, info_dict, save_path):
        """
        在图片上绘制 text_coord 和 tap_list 的标注信息:
        - text_coord 画一个点，并标注 text_code
        - tap_list 画多个点，并按顺序标注 1,2,3,...
        """
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # 字体
        try:
            font = ImageFont.truetype("/opt/nas/p/mm/ie_env/sjg/Font/ARIAL.TTF", 80)
        except:
            font = ImageFont.load_default()

        # 画 text_coord
        text_x_norm, text_y_norm = info_dict["text_coord"]
        text_code = info_dict["text_code"]

        text_x = int(text_x_norm / 1000 * width)
        text_y = int(text_y_norm / 1000 * height)

        radius = 25
        draw.ellipse((text_x - radius, text_y - radius, text_x + radius, text_y + radius), fill='blue')
        draw.text((text_x + radius + 5, text_y - radius), text_code, fill="blue", font=font)

        # 画 tap_list
        for idx, (x_norm, y_norm) in enumerate(info_dict["tap_list"], start=1):
            x_pix = int(x_norm / 1000 * width)
            y_pix = int(y_norm / 1000 * height)

            draw.ellipse((x_pix - radius, y_pix - radius, x_pix + radius, y_pix + radius), fill='red')
            draw.text((x_pix + radius + 5, y_pix - radius), str(idx), fill="red", font=font)

        img.save(save_path)
        logger.info(f"带标注的图片已保存到: {save_path}")

    if model_name != 'qwen235B':
        GPT4 = GPT4oWrapper(model_name=model_name)
    

    max_attempts = 50  # 防止死循环
    attempts = 0
    candidate = ""
    while attempts < max_attempts:
        if model_name == 'qwen235B':
            logger.info(f'正在请求qwen输出')
            continuation_action = vllm_qwen3vl_235b_request(query=query_input, image_path=input_image_path)
            logger.info(f'完整输出：{continuation_action}')
            continuation_action = extract_answer_tags(continuation_action)
            logger.info(f'处理后的输出：{continuation_action}')
            candidate = continuation_action.strip()
        else:
            chat_results = GPT4.chat(query=query_input, imgs=[input_image_path])
            candidate = chat_results.strip()
        if is_valid_format(candidate):
            break
        attempts += 1

    if not candidate or not is_valid_format(candidate):
        logger.info(f"未能生成符合格式的结果")
        return []

    result_map = extract_info(candidate)
    logger.info(f"输入验证码内容坐标：{result_map['text_coord']}")
    logger.info(f"输入验证码内容：{result_map['text_code']}")
    logger.info(f"点击验证坐标：{result_map['tap_list']}")

    # 生成带标注的图片路径
    save_path = input_image_path.replace(".png", "_marked.png").replace(".jpg", "_marked.jpg")

    draw_points_with_info(input_image_path, result_map, save_path)

    return result_map


def run(image_path):
    start_time = time.time()
    category_number, category_name = classify_verification_scene(image_path)
    # logger.info(f"类别编号: {category_number}, 类别名称: {category_name}")
    if os.path.exists(image_path):
        os.makedirs(image_all_save_fold_path + category_name, exist_ok=True)
        shutil.copy(image_path, os.path.join(image_all_save_fold_path, category_name, os.path.basename(image_path)))
    logger.info(f"原始图片已保存 {os.path.join(image_all_save_fold_path, category_name, os.path.basename(image_path))}")

    action_result = ''
    action_result_abs = ''
    response = {}
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    if category_number == 1:  # 拼图验证

        save_label_folder = "/opt/nas/p/mm/ie_env/wanghui/人机验证登录/labels/"
        # logger.info("开始检测图片，生成标签文件...")
        label_path = run_detection(image_path, save_label_folder)
        # logger.info(f"标签文件保存到: {label_path}")

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        boxes_label0, centers_label1 = read_labels(label_path, img_w, img_h)

        if len(centers_label1) != 1 or len(boxes_label0) < 2:
            logger.info(f"检测到的拼图或滑块数量不符合要求，跳过动作生成。")
            gpt_wrapper = GPT4oWrapper(model_name = waibu_model_name)
            gpt_response = gpt_wrapper.chat(query = query_guanbi_shuxin, imgs=image_path)
            if gpt_response == '<answer>0</answer>':
                return {
                "type": "query",
                "detail": "点击刷新"
            }
            else:
                return {
                    "type": "query",
                    "detail": "点击关闭"
                }

        result = process_boxes(boxes_label0)
        if result is None:
            logger.info(f"不满足条件，无法计算距离")
            label1_center = centers_label1[0] if len(centers_label1) > 0 else None
            save_dir = "/opt/nas/p/mm/ie_env/wanghui/人机验证登录/img"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            draw_line_and_save(img, (0,0), (0,0), label1_center, None, save_path)
        else:
            distance, y_line, pt1, pt2 = result
            # logger.info(f"两点水平距离: {distance:.2f} 像素，直线y坐标: {y_line:.2f}")
            label1_center = centers_label1[0]
            label1_moved = (label1_center[0] + distance, label1_center[1])

            pt1_norm = normalize_point(pt1, img_w, img_h)
            pt2_norm = normalize_point(pt2, img_w, img_h)
            label1_center_norm = normalize_point(label1_center, img_w, img_h)
            label1_moved_norm = normalize_point(label1_moved, img_w, img_h)

            save_dir = "/opt/nas/p/mm/ie_env/wanghui/人机验证登录/img/"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            save_path = save_path.replace(".png", "_marked.png").replace(".jpg", "_marked.jpg")
            # logger.info(f"处理完成，结果保存到 {save_path}")
            draw_line_and_save(img, pt1, pt2, label1_center, label1_moved, save_path)

            x1, y1 = map(lambda v: int(round(v)), label1_center_norm)
            x2, y2 = map(lambda v: int(round(v)), label1_moved_norm)
            logger.info(f"drag({x1},{y1},{x2},{y2})")
            action_result = f"drag({x1},{y1},{x2},{y2})"
            action_result_abs = f"drag({x1/1000*img_w},{y1/1000*img_h},{x2/1000*img_w},{y2/1000*img_h})"

        logger.info(f"处理完成，结果保存到 {save_path}")
        end_time = time.time()
        logger.info(f"滑动拼图全部流程所需时间 {end_time - start_time}")

        response = {
            "type": "action",
            "detail": [
                {
                    "action": {
                        "type":"drag",
                        "detail": {"start_x": x1/1000*img_w,"start_y": y1/1000*img_h,"end_x": x2/1000*img_w,"end_y": y2/1000*img_h,"duration_s": 2,"mock":True,"end_hold_duration": 1.0}
                    }
                }
            ]
        }


        if os.path.exists(save_path):
            os.makedirs(image_all_save_fold_path + category_name, exist_ok=True)
            shutil.copy(save_path, os.path.join(image_all_save_fold_path, category_name, os.path.basename(save_path)))
    

    elif category_number == 2:  # 滑动验证
        save_label_folder = "/opt/nas/p/mm/ie_env/wanghui/人机验证登录/labels/"
        # logger.info(f"开始检测图片，生成标签文件...")
        label_path = run_detection(image_path, save_label_folder)
        # logger.info(f"标签文件保存到: {label_path}")

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        boxes_label0, centers_label1 = read_labels(label_path, img_w, img_h)
        logger.info(f"检测到标签0数量: {len(boxes_label0)}，标签1数量: {len(centers_label1)}")
        if len(centers_label1) == 0:
            logger.info(f"未检测到标签1，无法进行滑动验证处理。")
            return {
                "type": "query",
                "detail": "点击关闭"
            }

        label1_center = centers_label1[0]

        box_w_norm, box_h_norm = None, None
        x_center_norm, y_center_norm = None, None
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                if cls == 1:
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    box_w_norm = float(parts[3])
                    box_h_norm = float(parts[4])
                    break
        if None in [box_w_norm, box_h_norm, x_center_norm, y_center_norm]:
            logger.info(f"未能获取标签1的中心和宽高，使用默认值。")
            box_w_norm, box_h_norm = 0.1, 0.1
            x_center_norm, y_center_norm = 0.5, 0.5

        target_x_norm = 0.995 - box_w_norm / 2
        target_x_norm = max(0, min(1, target_x_norm))
        target_x_norm_1000 = target_x_norm * 1000

        save_dir = "/opt/nas/p/mm/ie_env/wanghui/人机验证登录/img/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        save_path = save_path.replace(".png", "_marked.png").replace(".jpg", "_marked.jpg")
        logger.info(f"处理完成，结果保存到 {save_path}")

        draw_sliding_verification_result(img, label1_center, box_w_norm, box_h_norm, target_x_norm_1000, img_w, img_h, save_path)

        label1_center_norm = (x_center_norm * 1000, y_center_norm * 1000)

        x1, y1 = map(lambda v: round(v, 2), label1_center_norm)
        x2, y2 = round(target_x_norm_1000, 2), round(y_center_norm * 1000, 2)
        logger.info(f"drag({x1},{y1},{x2},{y2})")
        action_result = f"drag({x1},{y1},{x2},{y2})"
        action_result_abs = f"drag({x1/1000*img_w:.2f},{y1/1000*img_h:.2f},{x2/1000*img_w:.2f},{y2/1000*img_h:.2f})"

        logger.info(f"滑动验证处理完成，结果保存到 {save_path}")
        end_time = time.time()
        logger.info(f"滑动验证全部流程所需时间 {end_time - start_time}")

        response = {
            "type": "action",
            "detail": [
                {
                    "action": {
                        "type":"drag",
                        "detail": {"start_x": x1/1000*img_w,"start_y": y1/1000*img_h,"end_x": x2/1000*img_w,"end_y": y2/1000*img_h,"duration_s": 2, "mock":True,"end_hold_duration": 1.0}
                    }
                }
            ]
        }

        if os.path.exists(save_path):
            os.makedirs(image_all_save_fold_path + category_name, exist_ok=True)
            shutil.copy(save_path, os.path.join(image_all_save_fold_path, category_name, os.path.basename(save_path)))

    elif category_number == 3:  #按照语序依次点击
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        points = process_image_shunxudianji(image_path, waibu_model_name,  query_yaoqiudianji, category_name)
        response = {
            "type": "action",
            "detail": []
        }

        for x, y in points:
            action_item = {
                "action": {
                    "type":"touch",
                    "detail": {
                        "point": [x/1000*img_w, y/1000*img_h]
                    }
                }
            }
            response["detail"].append(action_item)


    elif category_number == 4:  #输入内容验证
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        result_map = process_image_shuruneirong(image_path, 'qwen235B',  query_shuruneirong)
        
        response = {
            "type": "action",
            "detail": []
        }
        text_x, text_y = result_map["text_coord"]
        text_ppp = result_map["text_code"]

        action_item1 = {
            "action": {
                "type":"type",
                "detail": {
                    "point": [text_x/1000*img_w, text_y/1000*img_h],
                    "text": text_ppp,
                }
            }
        }
        response["detail"].append(action_item1)


        x, y = result_map["tap_list"][0]
        action_item2 = {
            "action": {
                "type":"touch",
                "detail": {
                    "point": [x/1000*img_w, y/1000*img_h]
                }
            }
        }
        response["detail"].append(action_item2)



    elif category_number == 5:  #按照语序依次点击
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]

        points = process_image_shunxudianji(image_path, waibu_model_name, query_yuxudianji, category_name)
        response = {
            "type": "action",
            "detail": []
        }

        for x, y in points:
            action_item = {
                "action": {
                    "type":"touch",
                    "detail": {
                        "point": [x/1000*img_w, y/1000*img_h]
                    }
                }
            }
            response["detail"].append(action_item)
    
    elif  category_number == 7: #如果场景属于“依次点击图标验证”，输出：
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        points = process_image_shunxudianji(image_path, waibu_model_name, query_tubiaoshunxudianji, category_name)
        response = {
            "type": "action",
            "detail": []
        }

        for x, y in points:
            action_item = {
                "action": {
                    "type":"touch",
                    "detail": {
                        "point": [x/1000*img_w, y/1000*img_h]
                    }
                }
            }
            response["detail"].append(action_item)


    # elif category_number == 9:  #如果场景属于“最短连线验证”，输出：`<answer>9</answer>`
    #     img = cv2.imread(image_path)
    #     img_h, img_w = img.shape[:2]
    #     num_retry_max = 3
    #     num_retry = 0
    #     response = {}
    #     while num_retry<num_retry_max:
    #         points = process_image_shunxudianji(image_path, waibu_model_name, query_zuiduanlianxian, category_name)
    #         if len(points) == 2:
    #             x1 = points[0][0]
    #             y1 = points[0][1]
    #             x2 = points[1][0]
    #             y2 = points[1][1]

    #             response = {
    #                 "type": "action",
    #                 "detail": [
    #                     {
    #                         "action": {
    #                             "type":"drag",
    #                             "detail": {"start_x": x1/1000*img_w,"start_y": y1/1000*img_h,"end_x": x2/1000*img_w,"end_y": y2/1000*img_h,"duration_s": 2, "mock":True,"end_hold_duration": 1.0}
    #                         }
    #                     }
    #                 ]
    #             }
    #             break

    #         num_retry +=1
    #     if response == {}:
    #         points = process_image_shunxudianji(image_path, waibu_model_name,  query_dianjiguanbi, category_name)
    #         if len(points)!=1 or points == [(0, 0)]:
    #             response = {
    #                 "type": "processing_failure",
    #                 "detail": []
    #             }
    #         else:
    #             response = {
    #                 "type": "action",
    #                 "detail": []
    #             }
    #             for x, y in points:
    #                 action_item = {
    #                     "action": {
    #                         "type":"touch",
    #                         "detail": {
    #                             "point": [x/1000*img_w, y/1000*img_h]
    #                         }
    #                     }
    #                 }
    #                 response["detail"].append(action_item)


    elif category_number == 11:
        logger.info(f"场景是{category_name}")
        response = {
            "type": "no_human_verification",
            "detail": []
        }
        

    else:
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        logger.info(f"场景是{category_name}，不需要调用检测模型，跳过后续流程。")
        logger.info(f"返回点击关闭指令")
        points = process_image_shunxudianji(image_path, waibu_model_name,  query_dianjiguanbi, category_name)
        if points == [(0, 0)] or len(points) != 1:
            response = {
                "type": "processing_failure",
                "detail": []
            }
        else:
            response = {
                "type": "action",
                "detail": []
            }
            for x, y in points:
                action_item = {
                    "action": {
                        "type":"touch",
                        "detail": {
                            "point": [x/1000*img_w, y/1000*img_h]
                        }
                    }
                }
                response["detail"].append(action_item)
    return response

def vllm_qwen3vl_235b_request(query, image_path):
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


os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
model = YOLO(model_path)


def save_base64_image_with_millisecond(base64_str, filename):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    image_data = base64.b64decode(base64_str)

    # 获取当前时间，包含毫秒
    save_dir = '/opt/nas/p/mm/ie_env/wanghui/人机验证登录/human_verification/tmp_image'
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, 'wb') as f:
        f.write(image_data)

    # logger.info(f"图片已保存到 {save_path}")
    return save_path


@app.route('/human_verification/classify_and_action', methods=['POST'])
def classify_and_action():
    """
    接口接收JSON格式数据，包含图片路径image_path字段。
    返回分类结果和动作指令。
    """
    data = request.get_json()
    if not data or 'image_path' not in data:
        return jsonify({"error": "缺少参数 image_path"}), 400

    image_path = data['image_path']
    t = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
    millisecond = int((t - int(t)) * 1000)

    filename = f"{timestamp}_{millisecond}.png"
    save_path = save_base64_image_with_millisecond(image_path, filename)
    
    try:
        response = run(save_path)
        return jsonify(response)
    except Exception as e:
        logger.info(f"error:{e}")
        return jsonify(f"error:{e}"), 500


if __name__ == '__main__':
    logger.info("Start flask server...")
    # 监听所有IP，端口可根据需要修改
    app.run(host='0.0.0.0', port=port)
