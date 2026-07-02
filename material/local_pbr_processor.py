# =============================================================================
# Local PBR Fallback — 不依赖 ComfyUI/CHORD 的 CPU 算法
# =============================================================================
# 本文件提供 AI_Texture_Generator 在本地 ComfyUI 不可用时生成法线与无缝贴图的
# 兜底算法。实现参考了常见的图像处理原理（Sobel 梯度、循环混合、接缝缝合），
# 代码为独立编写，不直接复制任何第三方源码。
# =============================================================================

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image


# =============================================================================
# 图像格式转换
# =============================================================================

def _image_to_float_rgba(src: Image.Image) -> np.ndarray:
    """任意 PIL 图像 → float32 RGBA (H,W,4)。"""
    if src.mode == "RGBA":
        arr = np.array(src, dtype=np.float32) / 255.0
    elif src.mode in ("L", "LA"):
        if src.mode == "LA":
            la = np.array(src, dtype=np.float32) / 255.0
            gray = la[..., 0]
            alpha = la[..., 1]
        else:
            gray = np.array(src, dtype=np.float32) / 255.0
            alpha = np.ones_like(gray)
        arr = np.stack([gray, gray, gray, alpha], axis=-1)
    elif src.mode == "P":
        return _image_to_float_rgba(src.convert("RGBA"))
    else:
        rgb = np.array(src.convert("RGB"), dtype=np.float32) / 255.0
        alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
        arr = np.concatenate([rgb, alpha], axis=-1)
    return arr.astype(np.float32)


def _float_rgba_to_image(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    """float32 RGBA → PIL Image。"""
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    if mode == "RGBA":
        return Image.fromarray(u8, "RGBA")
    if mode == "L":
        return Image.fromarray(u8[..., 0], "L")
    return Image.fromarray(u8[..., :3], "RGB")


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """RGBA float32 → 灰度。"""
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] >= 3:
        return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    return arr[..., 0]


# =============================================================================
# 缩放
# =============================================================================

def _resize_float_array(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """双线性缩放 float 数组。"""
    h, w = arr.shape[:2]
    if h == new_h and w == new_w:
        return arr.copy()
    c = arr.shape[2] if arr.ndim == 3 else 1
    if c == 1:
        pil_img = Image.fromarray((arr[..., 0] * 255).astype(np.uint8), "L")
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        out = np.array(pil_img, dtype=np.float32) / 255.0
        return out[..., np.newaxis]
    mode = "RGBA" if c == 4 else "RGB"
    pil_img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    out = np.array(pil_img, dtype=np.float32) / 255.0
    return out


# =============================================================================
# 盒式模糊（用于法线 detail 控制）
# =============================================================================

def _box_blur_channel(channel: np.ndarray, radius: int, edge_mode: str = "edge") -> np.ndarray:
    """单通道积分图盒式模糊，保持平均亮度。"""
    if radius <= 0:
        return channel.copy()
    mean_before = np.mean(channel) or 1e-8

    if edge_mode == "wrap":
        padded = np.pad(channel, radius, mode="wrap")
    elif edge_mode == "constant":
        fill = 1.0 if mean_before > 0.9 else mean_before
        padded = np.pad(channel, radius, mode="constant", constant_values=fill)
    else:
        padded = np.pad(channel, radius, mode="edge")

    row_sum = np.cumsum(padded, axis=1)
    row_window = row_sum[:, 2 * radius:] - row_sum[:, :-2 * radius]
    col_sum = np.cumsum(row_window, axis=0)
    col_window = col_sum[2 * radius:, :] - col_sum[:-2 * radius, :]

    area = (2 * radius + 1) ** 2
    blurred = col_window / area
    mean_after = np.mean(blurred) or 1e-8
    blurred = blurred * (mean_before / mean_after)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def _box_blur(arr: np.ndarray, percent: float = 1.0, edge_mode: str = "edge") -> np.ndarray:
    """多通道盒式模糊，percent 基于对角线比例。"""
    if percent <= 0:
        return arr.copy()
    flat = arr.ndim == 2
    img = arr[..., np.newaxis].astype(np.float32) if flat else arr.copy().astype(np.float32)
    h, w, c = img.shape
    diag = math.hypot(w, h)
    radius = int(round(diag / 4 * percent / 100))
    radius = max(0, min(radius, int(diag / 4)))
    if radius == 0:
        return arr.copy()
    out = np.zeros_like(img)
    for i in range(c):
        out[..., i] = _box_blur_channel(img[..., i], radius, edge_mode)
    return out[..., 0] if flat else out


# =============================================================================
# 法线生成
# =============================================================================

def _compute_normal_from_height(
    height: np.ndarray,
    strength: float = 2.0,
    invert: bool = False,
) -> np.ndarray:
    """灰度高度图 → RGBA 法线贴图（OpenGL 切线空间）。

    Args:
        invert: 反转法线 R/X 方向（与 Image Editor Master 的“反转绿色通道”行为一致）。
    """
    gray = np.clip(height.astype(np.float32), 0.0, 1.0)
    h, w = gray.shape

    # Sobel 算子
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    padded = np.pad(gray, 1, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))
    dx = np.sum(window * kx, axis=(2, 3))
    dy = np.sum(window * ky, axis=(2, 3))

    nx = -dx * strength
    ny = -dy * strength
    if invert:
        nx = -nx
    nz = np.ones_like(gray)

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    length[length == 0] = 1.0
    nx /= length
    ny /= length
    nz /= length

    out = np.empty((*gray.shape, 4), dtype=np.float32)
    out[..., 0] = (nx + 1.0) * 0.5
    out[..., 1] = (ny + 1.0) * 0.5
    out[..., 2] = (nz + 1.0) * 0.5
    out[..., 3] = 1.0
    return np.clip(out, 0.0, 1.0)


def generate_normal_from_diffuse(
    diffuse: Image.Image,
    strength: float = 1.5,
    detail: float = 0.4,
    invert: bool = False,
) -> Image.Image:
    """从漫反射贴图生成法线贴图。

    Args:
        strength: 凹凸强度。默认 1.5 更接近 Image Editor Master GPU 输出。
        detail: 0~1，越高保留越多高频细节；0 只保留大形。
        invert: 反转法线 R/X 方向（与 Image Editor Master 的“反转绿色通道”行为一致）。
    """
    rgba = _image_to_float_rgba(diffuse)

    if detail <= 0.0:
        normal = _compute_normal_from_height(_to_grayscale(rgba), strength, invert)
    else:
        full = _compute_normal_from_height(_to_grayscale(rgba), strength, invert)
        gray = _to_grayscale(rgba)
        smooth_gray = _box_blur(gray, 1.5, "edge")
        smooth_rgba = np.stack([smooth_gray, smooth_gray, smooth_gray, rgba[..., 3]], axis=-1)
        base = _compute_normal_from_height(_to_grayscale(smooth_rgba), strength, invert)
        result = np.zeros_like(full)
        result[..., :3] = np.clip(base[..., :3] + (full[..., :3] - base[..., :3]) * detail, 0.0, 1.0)
        result[..., 3] = full[..., 3]
        normal = result

    return _float_rgba_to_image(normal, mode="RGB")


# =============================================================================
# 基础无缝化：循环偏移 + 边缘线性混合
# =============================================================================

def _roll_image(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """循环平移图像。"""
    h, w = arr.shape[:2]
    dx = dx % w
    dy = dy % h
    if dx:
        arr = np.concatenate([arr[:, -dx:, :], arr[:, :-dx, :]], axis=1)
    if dy:
        arr = np.concatenate([arr[-dy:, :, :], arr[:-dy, :, :]], axis=0)
    return arr


def _basic_seamless(arr: np.ndarray, blend_ratio: float = 0.125) -> np.ndarray:
    """基础无缝化，输出尺寸不变。"""
    h, w = arr.shape[:2]
    short = min(w, h)
    band = max(2, min(int(round(short * blend_ratio)), short // 2))
    half = band // 2

    def make_mask(direction: str) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        if half <= 0:
            return mask
        a1 = 1.0 - np.arange(half) / half
        a2 = np.arange(half) / half
        if direction == "h":
            mask[:, :half] = a1[np.newaxis, :]
            mask[:, w - half:w] = a2[np.newaxis, :]
        else:
            mask[:half, :] = a1[:, np.newaxis]
            mask[h - half:h, :] = a2[:, np.newaxis]
        return mask[..., np.newaxis]

    shifted_h = _roll_image(arr.copy(), -w // 2, 0)
    mask_h = make_mask("h")
    blended_h = arr * (1.0 - mask_h) + shifted_h * mask_h

    shifted_v = _roll_image(blended_h.copy(), 0, -h // 2)
    mask_v = make_mask("v")
    blended_v = blended_h * (1.0 - mask_v) + shifted_v * mask_v

    return _force_seamless_edges(np.clip(blended_v, 0.0, 1.0), band=1)


# =============================================================================
# 高级无缝化：接缝缝合 + 双频融合
# =============================================================================

def _match_local_color(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """将 source 的颜色统计量对齐到 target。"""
    mu_s = np.mean(source, axis=(0, 1), keepdims=True)
    std_s = np.std(source, axis=(0, 1), keepdims=True)
    mu_t = np.mean(target, axis=(0, 1), keepdims=True)
    std_t = np.std(target, axis=(0, 1), keepdims=True)
    std_s = np.where(std_s == 0, 1e-5, std_s)
    matched = (source - mu_s) * (std_t / std_s) + mu_t
    return np.clip(matched, 0.0, 1.0)


def _gradient_penalty(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """计算两区域梯度结构差异。"""
    g_l = np.mean(left, axis=2)
    g_r = np.mean(right, axis=2)
    gy_l, gx_l = np.gradient(g_l)
    gy_r, gx_r = np.gradient(g_r)
    return (gx_r - gx_l) ** 2 + (gy_r - gy_l) ** 2


def _iterative_blur(arr: np.ndarray, iterations: int = 5) -> np.ndarray:
    """多次小半径盒式模糊，近似高斯模糊。"""
    pad = iterations * 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
    two = np.float32(2.0)
    four = np.float32(4.0)
    for _ in range(iterations):
        padded = (np.roll(padded, 1, axis=1) + two * padded + np.roll(padded, -1, axis=1)) / four
        padded = (np.roll(padded, 1, axis=0) + two * padded + np.roll(padded, -1, axis=0)) / four
    return padded[pad:-pad, pad:-pad, :]


def _find_vertical_seam(error: np.ndarray) -> np.ndarray:
    """动态规划找垂直方向最优缝合线。"""
    h, w = error.shape
    cost = np.zeros_like(error, dtype=np.float32)
    path = np.zeros_like(error, dtype=np.int8)
    cost[0] = error[0]

    for y in range(1, h):
        left = np.roll(cost[y - 1], 1)
        left[0] = np.inf
        right = np.roll(cost[y - 1], -1)
        right[-1] = np.inf
        up = cost[y - 1]
        best = np.minimum(np.minimum(left, up), right)
        cost[y] = error[y] + best
        path[y] = np.where(best == left, -1, np.where(best == up, 0, 1))

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    for y in range(h - 2, -1, -1):
        seam[y] = seam[y + 1] + path[y + 1, seam[y + 1]]
    return seam


def _two_band_blend(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """双频融合：低频软过渡，高频硬保留。"""
    left_low = _iterative_blur(left, 5)
    right_low = _iterative_blur(right, 5)
    left_high = left.astype(np.float32) - left_low
    right_high = right.astype(np.float32) - right_low

    high = right_high * mask + left_high * (1.0 - mask)
    soft_mask = _iterative_blur(mask, 8)
    low = right_low * soft_mask + left_low * (1.0 - soft_mask)

    return np.clip(low + high, 0.0, 1.0)


def _remove_macro_gradient(arr: np.ndarray, overlap_ratio: float) -> np.ndarray:
    """拟合并去除左右/上下边缘的线性光照梯度。"""
    h, w, c = arr.shape
    ow = int(w * overlap_ratio)
    oh = int(h * overlap_ratio)
    eps = 1e-5

    left_mean = np.mean(arr[:, :ow], axis=(0, 1))
    right_mean = np.mean(arr[:, -ow:], axis=(0, 1))
    target_x = (left_mean + right_mean) * 0.5
    x_profile = np.linspace(0, 1, w, dtype=np.float32).reshape(1, w, 1)
    x_light = left_mean * (1 - x_profile) + right_mean * x_profile
    arr = arr * (target_x / (x_light + eps))

    top_mean = np.mean(arr[:oh, :], axis=(0, 1))
    bottom_mean = np.mean(arr[-oh:, :], axis=(0, 1))
    target_y = (top_mean + bottom_mean) * 0.5
    y_profile = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
    y_light = top_mean * (1 - y_profile) + bottom_mean * y_profile
    arr = arr * (target_y / (y_light + eps))

    return np.clip(arr, 0.0, 1.0)


def _stitch_horizontal_inplace(img: np.ndarray, overlap_w: int, beta: float) -> np.ndarray:
    """水平方向接缝缝合（保持图像尺寸不变）。"""
    h, w, c = img.shape
    if overlap_w <= 0 or overlap_w >= w // 2:
        return img.copy()
    right = img[:, -overlap_w:, :].astype(np.float32)
    left = img[:, :overlap_w, :].astype(np.float32)

    right_matched = _match_local_color(right, left)
    diff = right_matched - left
    rgb_error = np.sum(diff * diff, axis=2)
    grad_error = _gradient_penalty(left, right_matched)
    error = rgb_error + beta * grad_error

    seam = _find_vertical_seam(error)
    hard_mask = np.zeros((h, overlap_w, 1), dtype=np.float32)
    for y in range(h):
        hard_mask[y, :seam[y]] = 1.0

    blended = _two_band_blend(left, right_matched, hard_mask)
    result = img.copy().astype(np.float32)
    # 让左右边缘都等于融合后的结果，保证左右连续
    result[:, :overlap_w, :] = blended
    result[:, -overlap_w:, :] = blended
    return result


def _stitch_vertical_inplace(img: np.ndarray, overlap_h: int, beta: float) -> np.ndarray:
    """垂直方向接缝缝合（保持图像尺寸不变，转置复用水平逻辑）。"""
    transposed = np.transpose(img, (1, 0, 2))
    stitched = _stitch_horizontal_inplace(transposed, overlap_h, beta)
    return np.transpose(stitched, (1, 0, 2))


def _force_seamless_edges(arr: np.ndarray, band: int = 2) -> np.ndarray:
    """强制让左右边缘、上下边缘完全镜像对称，消除残余接缝。

    只对最边缘的 band 个像素做平均，band 很小（默认 2）时对主体内容影响极小。
    """
    if band <= 0:
        return arr
    h, w = arr.shape[:2]
    band = min(band, w // 2, h // 2)
    result = arr.copy().astype(np.float32)

    # 左右边缘对称
    avg_h = (result[:, :band, :] + result[:, -band:, :][:, ::-1, :]) * 0.5
    result[:, :band, :] = avg_h
    result[:, -band:, :] = avg_h[:, ::-1, :]

    # 上下边缘对称
    avg_v = (result[:band, :, :] + result[-band:, :, :][::-1, :, :]) * 0.5
    result[:band, :, :] = avg_v
    result[-band:, :, :] = avg_v[::-1, :, :]

    return np.clip(result, 0.0, 1.0)


def _advanced_seamless(arr: np.ndarray, overlap_ratio: float = 0.2, beta: float = 5.0) -> np.ndarray:
    """高级无缝化：循环偏移 + 双频融合 + 边缘对称修正。

    思路：把原图与它自身在水平/垂直方向循环偏移半个周期的版本进行多频段融合，
    使结果在四个边缘都周期连续，同时尽量保留高频细节。
    """
    h, w = arr.shape[:2]
    band_w = max(2, min(int(w * overlap_ratio), w // 2))
    band_h = max(2, min(int(h * overlap_ratio), h // 2))

    # 水平方向：原图 vs 水平偏移 w//2
    shifted_h = _roll_image(arr.copy(), -w // 2, 0)
    mask_h = np.zeros((h, w, 1), dtype=np.float32)
    ramp_w = np.linspace(0.0, 1.0, band_w, dtype=np.float32).reshape(1, band_w, 1)
    mask_h[:, :band_w] = ramp_w
    mask_h[:, -band_w:] = ramp_w[:, ::-1, :]
    horizontal = _two_band_blend(arr, shifted_h, mask_h)

    # 垂直方向：在水平结果基础上，与垂直偏移 h//2 的版本融合
    shifted_v = _roll_image(horizontal, 0, -h // 2)
    mask_v = np.zeros((h, w, 1), dtype=np.float32)
    ramp_h = np.linspace(0.0, 1.0, band_h, dtype=np.float32).reshape(band_h, 1, 1)
    mask_v[:band_h, :] = ramp_h
    mask_v[-band_h:, :] = ramp_h[::-1, :, :]
    result = _two_band_blend(horizontal, shifted_v, mask_v)

    # 最后强制边缘完全对称，消除任何残余接缝
    return _force_seamless_edges(result, band=min(2, band_w, band_h))


# =============================================================================
# 高度图生成：从法线贴图反推（FFT 求解 Poisson 方程）
# =============================================================================

def generate_height_from_normal(
    normal: Image.Image,
    flip_green: bool = False,
) -> Image.Image:
    """从法线贴图反推高度图。

    算法思路：
    1. 将法线从 [0,1] 解码到 [-1,1]。
    2. 恢复梯度场 grad_x = -nx / nz, grad_y = -ny / nz。
    3. 计算散度并做 FFT 频域求解 Poisson 方程，得到高度。
    4. 归一化到 [0,1]。
    """
    rgba = _image_to_float_rgba(normal)
    rgb = rgba[..., :3]

    nx = rgb[..., 0] * 2.0 - 1.0
    ny = rgb[..., 1] * 2.0 - 1.0
    nz = np.maximum(rgb[..., 2] * 2.0 - 1.0, 1e-6)

    if flip_green:
        ny = -ny

    grad_x = -nx / nz
    grad_y = -ny / nz
    grad_x -= np.mean(grad_x)
    grad_y -= np.mean(grad_y)

    h, w = grad_x.shape

    # 周期性中心差分计算散度
    div_x = (np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1)) * 0.5
    div_y = (np.roll(grad_y, -1, axis=0) - np.roll(grad_y, 1, axis=0)) * 0.5
    divergence = div_x + div_y

    # FFT 频域求解 Poisson 方程
    kx = np.fft.fftfreq(w, d=1.0).astype(np.float32) * (2.0 * np.pi)
    ky = np.fft.fftfreq(h, d=1.0).astype(np.float32) * (2.0 * np.pi)
    k2 = kx[np.newaxis, :] ** 2 + ky[:, np.newaxis] ** 2
    k2[0, 0] = 1.0

    div_fft = np.fft.fft2(divergence)
    height_fft = -div_fft / k2
    height_fft[0, 0] = 0.0

    height = np.real(np.fft.ifft2(height_fft))
    h_min, h_max = height.min(), height.max()
    height = (height - h_min) / (h_max - h_min + 1e-6)

    height_rgba = np.stack([height, height, height, np.ones_like(height)], axis=-1)
    return _float_rgba_to_image(height_rgba, mode="RGB")


# =============================================================================
# 公共入口
# =============================================================================

def make_seamless_tile_iem(
    image: Image.Image,
    method: str = "BASIC",
    blend_ratio: float = 0.125,
    overlap: float = 0.2,
    beta: float = 5.0,
) -> Image.Image:
    """贴图无缝化本地算法入口。

    Args:
        method: "BASIC" 或 "ADVANCED"。
        blend_ratio: BASIC 混合带宽度占短边比例。
        overlap: ADVANCED 重叠区比例。
        beta: ADVANCED 边缘结构保护强度。
    """
    if method not in {"BASIC", "ADVANCED"}:
        raise ValueError(f"method must be BASIC or ADVANCED, got {method}")

    keep_alpha = image.mode in ("RGBA", "LA", "P")
    arr = _image_to_float_rgba(image)

    if method == "BASIC":
        result = _basic_seamless(arr, blend_ratio)
    else:
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        seamless_rgb = _advanced_seamless(rgb, overlap, beta)
        # alpha 用基础算法保持边缘一致
        alpha_rgba = np.concatenate([alpha, alpha, alpha, alpha], axis=-1)
        seamless_alpha = _basic_seamless(alpha_rgba, blend_ratio)[..., 3:4]
        result = np.concatenate([seamless_rgb, seamless_alpha], axis=-1)

    return _float_rgba_to_image(result, mode="RGBA" if keep_alpha else "RGB")
