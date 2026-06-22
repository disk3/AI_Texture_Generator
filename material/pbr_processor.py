import numpy as np
from PIL import Image


# =============================================================================
# 材质物理属性映射表
# =============================================================================

# Roughness 基础范围 (0=镜面光滑, 255=完全粗糙)
# 由 material_category 决定
ROUGHNESS_BASE = {
    'Met': (30, 120),   # 金属：通常较光滑
    'Glas': (20, 80),    # 玻璃：非常光滑
    'Cer': (40, 140),   # 陶瓷：中等偏光滑
    'Plas': (50, 160),   # 塑料：中等
    'Conc': (150, 240),   # 混凝土：粗糙
    'Cem': (140, 230),   # 水泥砂浆：粗糙
    'Asph': (160, 250),   # 沥青：很粗糙
    'Stn': (100, 220),   # 石材：中等偏粗糙
    'Wd': (80, 200),   # 木材：中等
    'Veg': (120, 240),   # 植被：粗糙
    'Comp': (60, 180),   # 复合材料：中等
    'Coat': (40, 140),   # 涂料涂层：中等偏光滑
    'Rub': (80, 180),   # 橡胶：中等
    'Fab': (100, 220),   # 织物：中等偏粗糙
    'Soil': (180, 255),   # 土壤：非常粗糙
    'Insul': (120, 220),   # 保温材料：粗糙
    'Seal': (60, 160),   # 密封剂：中等
    'Gyp': (80, 180),   # 石膏：中等
    'StnPaint': (60, 160),  # 石材漆：中等
}

# 由 surface_finish 进行微调 (偏移量)
FINISH_ROUGHNESS_OFFSET = {
    'Pol': -40,   # 抛光 → 更光滑
    'Gloss': -30,   # 光泽 → 更光滑
    'Smo': -20,   # 光滑 → 稍光滑
    'Brsh': +10,   # 拉丝 → 稍粗糙
    'Mat': +20,   # 哑光 → 稍粗糙
    'Rgh': +40,   # 粗糙 → 更粗糙
    'Tex': +30,   # 纹理 → 更粗糙
    'Sbl': +25,   # 喷砂 → 更粗糙
}

# Metallic 基础值 (0=非金属, 255=金属)
METALLIC_BASE = {
    'Met':    220,   # 金属类 → 高金属度
    'Glas':   10,    # 玻璃 → 非金属
    'Cer':    15,    # 陶瓷 → 非金属
    'Plas':   10,    # 塑料 → 非金属
    'Conc':   5,     # 混凝土 → 非金属
    'Cem':    5,     # 水泥 → 非金属
    'Asph':   5,     # 沥青 → 非金属
    'Stn':    10,    # 石材 → 非金属
    'Wd':     5,     # 木材 → 非金属
    'Veg':    5,     # 植被 → 非金属
    'Comp':   30,    # 复合材料 → 可能有金属成分
    'Coat':   10,    # 涂料 → 非金属
    'Rub':    5,     # 橡胶 → 非金属
    'Fab':    5,     # 织物 → 非金属
    'Soil':   5,     # 土壤 → 非金属
    'Insul':  5,     # 保温 → 非金属
    'Seal':   5,     # 密封剂 → 非金属
    'Gyp':    5,     # 石膏 → 非金属
    'StnPaint': 10,  # 石材漆 → 非金属
}


def _get_roughness_range(category: str, finish: str) -> tuple:
    """根据材质大类和表面处理返回 roughness 的 (min, max) 范围。"""
    base_min, base_max = ROUGHNESS_BASE.get(category, (60, 180))
    offset = FINISH_ROUGHNESS_OFFSET.get(finish, 0)
    return (
        max(0, base_min + offset),
        min(255, base_max + offset),
    )


def generate_normal_map(diffuse: Image.Image, strength: float = 0.8) -> Image.Image:
    """参考 NormalMap-Online 算法生成法线贴图（OpenGL 切线空间格式）。

    核心算法（对应 NormalMap-Online 默认 Sobel 模式）：
    1. 对灰度图直接做 3x3 Sobel 计算 dx, dy（不预处理模糊，保留细节）
    2. dz = (1 + 2^7) / strength = 129 / strength
    3. normal = normalize([-dx, -dy, dz])
    4. 编码：R=(X+1)/2, G=(Y+1)/2, B=Z（保持归一化值）
    """
    try:
        import cv2
        gray = np.array(diffuse.convert('L'), dtype=np.float32)
        h, w = gray.shape

        # Sobel 计算梯度
        # cv2.Sobel 不支持 BORDER_WRAP，手动 wrap padding 保证无缝纹理边缘正确
        padded = np.pad(gray, ((1, 1), (1, 1)), mode='wrap')
        dx = cv2.Sobel(padded, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
        dy = cv2.Sobel(padded, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]

        # 参考 NormalMap-Online: dz = (1 + 2^level) / strength, level=7
        dz = 129.0 / max(strength, 0.01)

        # 构建法线向量并归一化
        normal = np.zeros((h, w, 3), dtype=np.float32)
        normal[:, :, 0] = -dx  # R = X（OpenGL 切线空间）
        normal[:, :, 1] = -dy  # G = Y
        normal[:, :, 2] = dz   # B = Z

        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-8)

        # OpenGL 编码：XY 映射到 [0,1]，Z 保持 [0,1]
        normal[:, :, 0] = normal[:, :, 0] * 0.5 + 0.5
        normal[:, :, 1] = normal[:, :, 1] * 0.5 + 0.5
        # Z 分量已在 [0,1]（normalize 后 Z 始终为正）

        normal = np.clip(normal * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(normal)

    except ImportError:
        # fallback: 简化版 PIL 实现
        gray = np.array(diffuse.convert('L'), dtype=np.float32)
        h, w = gray.shape

        dx = np.zeros_like(gray)
        dy = np.zeros_like(gray)
        dx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        dy[:-1, :] = gray[1:, :] - gray[:-1, :]

        dz = 129.0 / max(strength, 0.01)

        normal = np.zeros((h, w, 3), dtype=np.float32)
        normal[:, :, 0] = -dx
        normal[:, :, 1] = -dy
        normal[:, :, 2] = dz

        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-8)

        normal[:, :, 0] = normal[:, :, 0] * 0.5 + 0.5
        normal[:, :, 1] = normal[:, :, 1] * 0.5 + 0.5

        normal = np.clip(normal * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(normal)


def generate_roughness_map(
    diffuse: Image.Image,
    category: str = "",
    finish: str = "",
    contrast: float = 1.0,
) -> Image.Image:
    """从 diffuse 生成 roughness 贴图，根据材质物理属性计算基础范围。

    逻辑：
    1. 反转亮度（暗部/裂缝通常更粗糙）
    2. 根据 material_category 和 surface_finish 确定 roughness 的物理范围
    3. 将反转后的亮度映射到该物理范围内
    """
    gray = np.array(diffuse.convert('L'), dtype=np.float32)

    # 反转亮度：暗部 → 更粗糙（高 Roughness）
    inverted = 255.0 - gray

    # 获取物理范围
    r_min, r_max = _get_roughness_range(category, finish)

    # 将 [0,255] 的反转亮度映射到 [r_min, r_max]
    rough = inverted * (r_max - r_min) / 255.0 + r_min
    if contrast != 1.0:
        midpoint = (r_min + r_max) * 0.5
        rough = (rough - midpoint) * contrast + midpoint
    rough = np.clip(rough, r_min, r_max)

    return Image.fromarray(rough.astype(np.uint8), mode='L').convert('RGB')


def generate_metallic_map(
    diffuse: Image.Image,
    category: str = "",
    finish: str = "",
    threshold: int = 220,
) -> Image.Image:
    """从 diffuse 生成 metallic 贴图，根据材质物理属性决定金属度。

    逻辑：
    1. 金属类材质（Met）：高亮且低饱和度区域标记为金属
    2. 非金属类材质：几乎不标记金属，除非极端高亮（如釉面反光）
    3. 由 material_category 决定 base metallic 水平
    """
    arr = np.array(diffuse.convert('RGB'), dtype=np.float32)
    max_c = np.max(arr, axis=2)
    min_c = np.min(arr, axis=2)
    brightness = max_c
    saturation = (max_c - min_c) / (max_c + 1e-8)

    metallic = np.zeros_like(brightness)

    base = METALLIC_BASE.get(category, 10)
    threshold = int(np.clip(threshold, 0, 255))

    if base >= 200:
        # 金属类：高亮且低饱和度区域标记为金属
        metallic[(brightness > threshold) & (saturation < 0.2)] = 255
        # 中等亮度也可能有金属成分（如氧化/污渍区域）
        mask = (brightness > max(40, threshold * 0.5)) & (saturation < 0.3)
        metallic[mask] = np.maximum(metallic[mask], brightness[mask] * 0.6)
    else:
        # 非金属：只有极端高亮接近纯灰度才可能（如釉面反光）
        metallic[(brightness > max(245, threshold)) & (saturation < 0.03)] = 180

    metallic = np.clip(metallic, 0, 255)
    return Image.fromarray(metallic.astype(np.uint8), mode='L').convert('RGB')


def generate_packed_map(
    roughness: Image.Image,
    metallic: Image.Image,
    pack_config: dict,
    ao: Image.Image = None,
    height: Image.Image = None,
) -> Image.Image:
    """按 pack_config 将贴图打包到 RGBA 通道，生成 Channel Packed 贴图。

    pack_config 格式: {'r'|'g'|'b'|'a': 'AO'|'ROUGHNESS'|'METALLIC'|'HEIGHT'|'NONE', ...}

    游戏引擎常见约定：
    - R = Ambient Occlusion
    - G = Roughness
    - B = Metallic
    - A = Height / Opacity
    但本函数不固定顺序，完全按 pack_config 配置。

    Args:
        roughness: 算法提取的 roughness 贴图
        metallic: 算法提取的 metallic 贴图
        pack_config: 通道分配配置
        ao: 可选的 AO 贴图；未提供时 AO 通道填白 255
        height: 可选的 height 贴图；未提供时 HEIGHT 通道填白 255

    Returns:
        RGBA 图像，可直接作为 Channel Packed 贴图使用
    """
    r_gray = np.array(roughness.convert('L'), dtype=np.uint8)
    m_gray = np.array(metallic.convert('L'), dtype=np.uint8)

    h, w = r_gray.shape
    packed = np.zeros((h, w, 4), dtype=np.uint8)

    image_map = {
        'ROUGHNESS': r_gray,
        'METALLIC': m_gray,
    }
    if ao is not None:
        image_map['AO'] = np.array(ao.convert('L'), dtype=np.uint8)
    if height is not None:
        image_map['HEIGHT'] = np.array(height.convert('L'), dtype=np.uint8)

    for ch_name, ch_idx in [('r', 0), ('g', 1), ('b', 2), ('a', 3)]:
        content = pack_config.get(ch_name, 'NONE')
        if content == 'NONE':
            # R 默认白（兼容 AO 惯例），GBA 默认黑
            packed[:, :, ch_idx] = 255 if ch_name == 'r' else 0
        elif content in image_map:
            packed[:, :, ch_idx] = image_map[content]
        elif content == 'AO' and ao is None:
            # AO 未提供时填白（游戏引擎默认 AO = 1.0）
            packed[:, :, ch_idx] = 255
        elif content == 'HEIGHT' and height is None:
            # HEIGHT 未提供时填白
            packed[:, :, ch_idx] = 255

    return Image.fromarray(packed, mode='RGBA')
