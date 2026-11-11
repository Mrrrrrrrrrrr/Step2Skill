"""
Badcase可视化图片生成模块
当检测到badcase时，自动生成包含关键帧、索引、原始action和修正后action的可视化图片
"""
import os
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

# 添加项目根目录到路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# 配置常量
TEXT_PADDING = 200  # 文本预留高度（像素，上下都预留）
INSTRUCTION_FONT_SIZE = 80  # instruction文本字号
SUBIMAGE_FONT_SIZE = 48  # 子图文本字号
PLACEHOLDER_SIZE = (200, 150)  # 占位符图片尺寸（宽x高）

# 缓存已加载的字体，避免重复加载
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}
_FONT_PATH_CACHE: Optional[str] = None


def find_chinese_font_path() -> Optional[str]:
    """
    查找系统中可用的中文字体路径
    
    Returns:
        Optional[str]: 字体文件路径，如果未找到则返回None
    """
    global _FONT_PATH_CACHE
    
    # 如果已缓存，直接返回
    if _FONT_PATH_CACHE is not None:
        return _FONT_PATH_CACHE if _FONT_PATH_CACHE != "" else None
    
    # 按优先级排列的字体路径列表
    font_paths = [
        # Linux 常见中文字体路径
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        # Ubuntu/Debian
        "/usr/share/fonts/truetype/fonts-cjk/SimSun.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 备选，部分支持
        # CentOS/RHEL
        "/usr/share/fonts/chinese/TrueType/uming.ttf",
        "/usr/share/fonts/cjkuni-uming/uming.ttc",
        # Windows
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simkai.ttf",    # 楷体
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            # 验证字体是否可以正常加载
            try:
                test_font = ImageFont.truetype(font_path, 20, index=0)
                # 尝试渲染中文字符验证
                test_img = Image.new('RGB', (100, 50), color='white')
                test_draw = ImageDraw.Draw(test_img)
                test_draw.text((0, 0), "测试中文", font=test_font, fill='black')
                
                _FONT_PATH_CACHE = font_path
                print(f"  [字体] 成功加载中文字体: {font_path}")
                return font_path
            except Exception as e:
                print(f"  [字体] 尝试加载失败 {font_path}: {e}")
                continue
    
    # 尝试使用 fc-list 命令查找字体（Linux）
    try:
        import subprocess
        result = subprocess.run(
            ['fc-list', ':lang=zh', '-f', '%{file}\n'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            font_files = result.stdout.strip().split('\n')
            for font_file in font_files:
                font_file = font_file.strip()
                if font_file and os.path.exists(font_file):
                    try:
                        test_font = ImageFont.truetype(font_file, 20, index=0)
                        _FONT_PATH_CACHE = font_file
                        print(f"  [字体] 通过fc-list找到中文字体: {font_file}")
                        return font_file
                    except Exception:
                        continue
    except Exception:
        pass
    
    print("  [字体] 警告: 未找到中文字体，将使用默认字体（可能显示乱码）")
    _FONT_PATH_CACHE = ""  # 标记为已搜索但未找到
    return None


def get_chinese_font(size: int) -> ImageFont.FreeTypeFont:
    """
    获取支持中文的字体
    
    Args:
        size: 字体大小
    
    Returns:
        ImageFont.FreeTypeFont: 字体对象
    """
    # 检查缓存
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    
    font_path = find_chinese_font_path()
    
    if font_path:
        try:
            # 对于 .ttc 文件，需要指定 index 参数
            font = ImageFont.truetype(font_path, size, index=0)
            _FONT_CACHE[size] = font
            return font
        except Exception as e:
            print(f"  [字体] 加载字体失败 (size={size}): {e}")
    
    # 回退方案：使用 Pillow 内置的默认字体
    # 注意：Pillow 10.0+ 的 load_default 支持 size 参数
    try:
        # Pillow >= 10.1.0
        font = ImageFont.load_default(size=size)
        _FONT_CACHE[size] = font
        return font
    except TypeError:
        # Pillow < 10.1.0，load_default 不支持 size 参数
        font = ImageFont.load_default()
        _FONT_CACHE[size] = font
        return font


def clean_action_text(action_text: str) -> str:
    """
    清理action文本，去除特殊标记
    
    Args:
        action_text: 原始action文本
    
    Returns:
        str: 清理后的文本
    """
    if not action_text:
        return ""
    # 去除 <|begin_of_box|> 和 <|end_of_box|> 标记
    text = action_text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return text.strip()


def truncate_text_to_fit(text: str, max_width: int, draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont) -> str:
    """
    根据字体和最大宽度截断文本，确保不换行
    
    Args:
        text: 原始文本
        max_width: 最大宽度（像素）
        draw: ImageDraw对象
        font: 字体对象
    
    Returns:
        str: 截断后的文本（如果超出，添加"..."）
    """
    if not text:
        return ""
    
    # 获取文本宽度
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    
    if text_width <= max_width:
        return text
    
    # 需要截断，逐步减少字符直到加上"..."后能放下
    ellipsis = "..."
    ellipsis_bbox = draw.textbbox((0, 0), ellipsis, font=font)
    ellipsis_width = ellipsis_bbox[2] - ellipsis_bbox[0]
    max_text_width = max_width - ellipsis_width
    
    # 二分查找合适的截断位置
    left, right = 0, len(text)
    while left < right:
        mid = (left + right + 1) // 2
        truncated = text[:mid]
        bbox = draw.textbbox((0, 0), truncated, font=font)
        truncated_width = bbox[2] - bbox[0]
        if truncated_width <= max_text_width:
            left = mid
        else:
            right = mid - 1
    
    return text[:left] + ellipsis


def draw_text_wrapped(
    draw: ImageDraw.Draw,
    text: str,
    x: int,
    y: int,
    max_width: int,
    font: ImageFont.FreeTypeFont,
    max_lines: int = 2,
    fill: str = 'black'
) -> int:
    """
    绘制自动换行的文本，最多显示 max_lines 行
    
    Args:
        draw: ImageDraw对象
        text: 要绘制的文本
        x: 起始x坐标
        y: 起始y坐标
        max_width: 最大宽度（像素）
        font: 字体对象
        max_lines: 最多显示的行数（默认2）
        fill: 文本颜色（默认黑色）
    
    Returns:
        int: 实际绘制的总高度（像素）
    """
    if not text:
        return 0
    
    # 获取单行高度
    bbox = draw.textbbox((0, 0), "测试", font=font)
    line_height = bbox[3] - bbox[1]
    
    # 尝试将文本分成多行
    lines = []
    current_line = ""
    
    # 按字符分割（中文字符每个字符独立，英文单词尽量保持完整）
    i = 0
    while i < len(text):
        char = text[i]
        test_line = current_line + char
        
        # 检查当前行宽度
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        
        if test_width <= max_width:
            current_line = test_line
            i += 1
        else:
            # 当前行已满，保存并开始新行
            if current_line:
                lines.append(current_line)
                current_line = char
            else:
                # 单个字符就超宽，强制添加
                lines.append(char)
                current_line = ""
            i += 1
            
            # 如果已达到最大行数，截断剩余文本
            if len(lines) >= max_lines:
                # 在最后一行添加省略号
                if current_line:
                    # 尝试在最后一行末尾添加省略号
                    ellipsis = "..."
                    while current_line and len(current_line) > 0:
                        test_with_ellipsis = current_line + ellipsis
                        bbox = draw.textbbox((0, 0), test_with_ellipsis, font=font)
                        if bbox[2] - bbox[0] <= max_width:
                            lines.append(test_with_ellipsis)
                            break
                        current_line = current_line[:-1]
                    else:
                        # 如果连省略号都放不下，直接添加
                        lines.append(ellipsis)
                break
    
    # 添加最后一行（如果还有剩余）
    if current_line and len(lines) < max_lines:
        lines.append(current_line)
    
    # 绘制每一行
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=fill, font=font)
        current_y += line_height
    
    # 返回总高度
    return len(lines) * line_height


def create_placeholder_image(width: int, height: int) -> Image.Image:
    """
    创建灰色占位符图片（当原图无法加载时使用）
    
    Args:
        width: 图片宽度
        height: 图片高度
    
    Returns:
        Image.Image: PIL Image对象
    """
    img = Image.new('RGB', (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    # 在占位符上绘制文本（使用中文字体）
    font = get_chinese_font(20)
    text = "图片\n未找到"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=(100, 100, 100), font=font)
    return img


def load_keyframe_images(original_task_data: Dict, keyframe_indices: List[int]) -> List[Image.Image]:
    """
    根据关键帧索引从original_task_data中获取对应的图片并加载
    
    Args:
        original_task_data: 原始任务数据
        keyframe_indices: 关键帧索引列表
    
    Returns:
        List[Image.Image]: 图片对象列表
    """
    images = []
    traj = original_task_data.get("traj", [])
    base_height = None
    
    # 第一遍：确定基准高度
    temp_images = []
    for idx in keyframe_indices:
        if idx < len(traj):
            image_path = traj[idx].get("image", "")
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert('RGB')
                    if base_height is None:
                        base_height = img.height
                    temp_images.append((img, True))  # (图片, 是否成功加载)
                except Exception as e:
                    print(f"  [图片] 加载失败 {image_path}: {e}")
                    temp_images.append((None, False))
            else:
                temp_images.append((None, False))
        else:
            temp_images.append((None, False))
    
    # 如果没有成功加载任何图片，使用默认高度
    if base_height is None:
        base_height = 400
    
    # 第二遍：统一高度并处理占位符
    for img, loaded in temp_images:
        if loaded and img:
            # 计算缩放比例
            scale = base_height / img.height
            new_width = int(img.width * scale)
            # 调整图片大小
            resized_img = img.resize((new_width, base_height), Image.Resampling.LANCZOS)
            images.append(resized_img)
        else:
            # 使用占位符，宽度按比例计算
            placeholder_width = int(PLACEHOLDER_SIZE[0] * (base_height / PLACEHOLDER_SIZE[1]))
            placeholder = create_placeholder_image(placeholder_width, base_height)
            images.append(placeholder)
    
    return images


def create_subimage_with_text(
    image: Image.Image,
    index: int,
    original_action: str,
    corrected_action: str,
    base_height: int
) -> Image.Image:
    """
    为单个关键帧创建带文本的中间图
    
    Args:
        image: 关键帧图片
        index: 关键帧索引
        original_action: 原始action描述
        corrected_action: 修正后action描述
        base_height: 基准高度
    
    Returns:
        Image.Image: 带文本的中间图
    """
    # 清理action文本
    original_action = clean_action_text(original_action)
    corrected_action = clean_action_text(corrected_action)
    
    # 创建白色画布
    canvas_height = base_height + TEXT_PADDING
    canvas_width = image.width
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    
    # 将图片粘贴到画布顶部
    canvas.paste(image, (0, 0))
    
    # 创建绘图对象
    draw = ImageDraw.Draw(canvas)
    
    # 使用支持中文的字体
    font = get_chinese_font(SUBIMAGE_FONT_SIZE)
    
    # 计算文本位置（上下都预留空间）
    text_y = base_height + 20  # 从图片下方20像素开始
    line_height = (TEXT_PADDING - 40) // 3  # 三行文本，每行间距
    
    # 第一行：索引
    index_text = f"索引: {index}"
    text_x = 5
    draw.text((text_x, text_y), index_text, fill='black', font=font)
    
    # 第二行：原始action
    original_text = f"原始: {original_action}"
    max_text_width = canvas_width - 10
    truncated_original = truncate_text_to_fit(original_text, max_text_width, draw, font)
    draw.text((text_x, text_y + line_height), truncated_original, fill='black', font=font)
    
    # 第三行：修正后action
    corrected_text = f"修正: {corrected_action}"
    truncated_corrected = truncate_text_to_fit(corrected_text, max_text_width, draw, font)
    draw.text((text_x, text_y + line_height * 2), truncated_corrected, fill='black', font=font)
    
    return canvas


def load_task_data(task_id: str, model_name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    加载任务数据
    
    Args:
        task_id: 任务ID
        model_name: 模型名称
    
    Returns:
        Tuple[Optional[Dict], Optional[Dict]]: (original_task_data, keyframes_data) 或 (None, None)
    """
    try:
        save_dir = config.get_save_dir(model_name)
        tmp_dir = config.get_cache_dir(model_name)
        
        # 加载original_task_data.json
        original_task_file = os.path.join(save_dir, task_id, "original_task_data.json")
        if not os.path.exists(original_task_file):
            return None, None
        
        with open(original_task_file, 'r', encoding='utf-8') as f:
            original_task_data = json.load(f)
        
        # 加载keyframes_full.json
        keyframes_file = os.path.join(tmp_dir, task_id, f"{task_id}_keyframes_full.json")
        if not os.path.exists(keyframes_file):
            return None, None
        
        with open(keyframes_file, 'r', encoding='utf-8') as f:
            keyframes_data = json.load(f)
        
        return original_task_data, keyframes_data
    
    except Exception as e:
        print(f"  警告: 加载任务数据失败 {task_id}: {e}")
        return None, None


def create_badcase_image(
    task_id: str,
    model_name: str,
    ground_truth: Optional[int],
    predicted: int,
    generate_json: bool = True,
    completion_gt: Optional[int] = None,
    completion_pr: Optional[int] = None,
    completion_reason: Optional[str] = None,
    trajectory_gt: Optional[int] = None,
    trajectory_pr: Optional[int] = None,
    trajectory_reason: Optional[str] = None
) -> Optional[str]:
    """
    创建badcase可视化图片
    
    Args:
        task_id: 任务ID
        model_name: 模型名称
        ground_truth: 真实标签（0或1），不能为None
        predicted: 预测值（0或1）
        generate_json: 是否生成JSON文件（默认True，新格式badcase应设为False）
        completion_gt: 完成度判断的ground_truth（可选）
        completion_pr: 完成度判断的predicted（可选）
        completion_reason: 完成度判断的reason（可选）
        trajectory_gt: 轨迹正确性判断的ground_truth（可选）
        trajectory_pr: 轨迹正确性判断的predicted（可选）
        trajectory_reason: 轨迹正确性判断的reason（可选）
    
    Returns:
        Optional[str]: 生成的图片路径，如果失败则返回None
    
    Raises:
        ValueError: 如果 ground_truth 为 None
    """
    # 检查 ground_truth 不能为 None
    if ground_truth is None:
        raise ValueError(f"ground_truth 不能为 None，task_id: {task_id}")
    
    try:
        # 加载数据
        original_task_data, keyframes_data = load_task_data(task_id, model_name)
        if not original_task_data or not keyframes_data:
            return None
        
        # 获取instruction
        instruction = original_task_data.get("instruction", "")
        
        # 获取关键帧信息
        keyframe_indices = keyframes_data.get("keyframe_indices", [])
        keyframe_actions = keyframes_data.get("keyframe_actions", [])
        
        if not keyframe_indices or not keyframe_actions:
            return None
        
        # 加载关键帧图片
        images = load_keyframe_images(original_task_data, keyframe_indices)
        if not images:
            return None
        
        # 确定基准高度（第一张图片的高度）
        base_height = images[0].height if images else 400
        
        # 创建每个关键帧的中间图
        subimages = []
        traj = original_task_data.get("traj", [])
        
        for i, idx in enumerate(keyframe_indices):
            # 获取原始action
            original_action = ""
            if idx < len(traj):
                original_action = traj[idx].get("value", {}).get("action", "")
            
            # 获取修正后action
            corrected_action = ""
            if i < len(keyframe_actions):
                corrected_action = keyframe_actions[i]
            
            # 创建中间图
            subimage = create_subimage_with_text(
                images[i],
                idx,
                original_action,
                corrected_action,
                base_height
            )
            subimages.append(subimage)
        
        # 横向拼接所有中间图
        total_width = sum(img.width for img in subimages)
        subimage_height = subimages[0].height if subimages else (base_height + TEXT_PADDING)
        combined_image = Image.new('RGB', (total_width, subimage_height), color='white')
        
        x_offset = 0
        for subimage in subimages:
            combined_image.paste(subimage, (x_offset, 0))
            x_offset += subimage.width
        
        # 添加instruction文本到顶部（上下都预留空间）
        # 计算自适应字体大小（根据图片宽度）
        # 后两行字体：宽度960px时24号，宽度1920px时48号，在合理区间自适应
        detail_font_size = max(24, min(48, total_width // 40))
        
        # 使用支持中文的字体
        instruction_font = get_chinese_font(INSTRUCTION_FONT_SIZE)
        detail_font = get_chinese_font(detail_font_size)
        
        # 创建临时图像来计算文本高度（需要先有draw对象）
        temp_image = Image.new('RGB', (100, 100), color='white')
        temp_draw = ImageDraw.Draw(temp_image)
        
        # 计算instruction行高度
        instruction_bbox = temp_draw.textbbox((0, 0), "测试", font=instruction_font)
        instruction_line_height = instruction_bbox[3] - instruction_bbox[1]
        
        # 计算detail行高度（每行最多2个子行）
        detail_bbox = temp_draw.textbbox((0, 0), "测试", font=detail_font)
        detail_line_height = detail_bbox[3] - detail_bbox[1]
        detail_max_height = detail_line_height * 2  # 最多2行
        
        # 计算总高度
        padding = 20
        instruction_height = instruction_line_height + padding * 2
        if completion_gt is not None and completion_pr is not None:
            instruction_height += detail_max_height + 10  # 10像素行间距
        if trajectory_gt is not None and trajectory_pr is not None:
            instruction_height += detail_max_height + 10  # 10像素行间距
        
        final_image = Image.new('RGB', (total_width, subimage_height + instruction_height), color='white')
        final_image.paste(combined_image, (0, instruction_height))
        
        # 绘制文本
        draw = ImageDraw.Draw(final_image)
        
        # 计算文本区域
        max_text_width = total_width - 20  # 左右各留10像素边距
        current_y = padding
        
        # 第一行：instruction（保持大字体，居中）
        instruction_text = f"instruction: {instruction}"
        truncated_instruction = truncate_text_to_fit(instruction_text, max_text_width, draw, instruction_font)
        bbox = draw.textbbox((0, 0), truncated_instruction, font=instruction_font)
        text_width = bbox[2] - bbox[0]
        text_x = (total_width - text_width) // 2
        draw.text((text_x, current_y), truncated_instruction, fill='black', font=instruction_font)
        current_y += instruction_line_height + 10  # 10像素行间距
        
        # 第二行：完成度（如果存在，使用小字体，左对齐，最多2行）
        if completion_gt is not None and completion_pr is not None:
            completion_reason_str = completion_reason or ""
            completion_text = f"final_state：gt_to_pr({completion_gt}_to_{completion_pr}) {completion_reason_str}"
            text_x = 10  # 左对齐，左边距10像素
            drawn_height = draw_text_wrapped(
                draw, completion_text, text_x, current_y,
                max_text_width, detail_font, max_lines=2, fill='black'
            )
            current_y += drawn_height + 10  # 10像素行间距
        
        # 第三行：轨迹正确性（如果存在，使用小字体，左对齐，最多2行）
        if trajectory_gt is not None and trajectory_pr is not None:
            trajectory_reason_str = trajectory_reason or ""
            trajectory_text = f"trajectory_correct：gt_to_pr({trajectory_gt}_to_{trajectory_pr}) {trajectory_reason_str}"
            text_x = 10  # 左对齐，左边距10像素
            draw_text_wrapped(
                draw, trajectory_text, text_x, current_y,
                max_text_width, detail_font, max_lines=2, fill='black'
            )
        
        # 保存图片
        bdcs_dir = os.path.join(config.get_model_data_dir(model_name), "bdcs")
        os.makedirs(bdcs_dir, exist_ok=True)
        
        # 图片命名格式：{task_id}_keyframe_judgment_{ground_truth}_to_{predicted}.jpg
        # 与 judgment 文件命名格式一致
        output_filename = f"{task_id}_keyframe_judgment_{ground_truth}_to_{predicted}.jpg"
        output_path = os.path.join(bdcs_dir, output_filename)
        
        final_image.save(output_path, 'JPEG', quality=95)
        
        # 生成对应的JSON文件（如果启用）
        if generate_json:
            try:
                save_dir = config.get_save_dir(model_name)
                tmp_dir = config.get_cache_dir(model_name)
                
                # 构建文件路径
                keyframe_judgment_file = os.path.join(save_dir, task_id, f"keyframe_judgment_{ground_truth}_to_{predicted}.json")
                category_file = os.path.join(tmp_dir, task_id, f"{task_id}_category.json")
                
                # 读取keyframe_judgment文件
                keyframe_data = {}
                if os.path.exists(keyframe_judgment_file):
                    with open(keyframe_judgment_file, 'r', encoding='utf-8') as f:
                        keyframe_data = json.load(f)
                
                # 读取category.json文件
                category_value = None
                if os.path.exists(category_file):
                    with open(category_file, 'r', encoding='utf-8') as f:
                        category_value = json.load(f)
                
                # 合并数据：category字段在最前面
                merged_data = {"category": category_value}
                merged_data.update(keyframe_data)
                
                # 保存JSON文件到bdcs目录
                json_filename = f"{task_id}.json"
                json_output_path = os.path.join(bdcs_dir, json_filename)
                
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            except Exception as json_error:
                print(f"  警告: 生成badcase JSON文件失败 {task_id}: {json_error}")
                # JSON生成失败不影响图片生成，继续执行
        
        return output_path
    
    except Exception as e:
        print(f"  警告: 生成badcase图片失败 {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# 模块加载时预热字体缓存
def _warmup_font_cache():
    """预热字体缓存，提前检测字体问题"""
    print("  [字体] 正在初始化字体...")
    find_chinese_font_path()
    # 预加载常用字号
    for size in [INSTRUCTION_FONT_SIZE, SUBIMAGE_FONT_SIZE, 20]:
        get_chinese_font(size)
    print("  [字体] 字体初始化完成")


# 取消自动预热，改为首次使用时加载
# _warmup_font_cache()
