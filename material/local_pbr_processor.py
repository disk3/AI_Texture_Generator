# =============================================================================
# Local PBR Fallback — 不依赖 ComfyUI/CHORD 的 CPU 算法
# =============================================================================
# 提供法线/高度图生成与无缝贴图入口。
# 无缝化核心已迁移到 seamless_processor.py，本文件仅保留兼容入口与工具函数。
# =============================================================================

from __future__ import annotations

import math

import numpy as np

from ..utils.image_utils import resize_numpy_image


# =============================================================================
# 图像格式转换
# =============================================================================

def _image_to_float_rgba(src: np.ndarray) -> np.ndarray:
    """任意 numpy 图像（H,W,C uint8） → float32 RGBA (H,W,4)。"""
    src = np.asarray(src)
    if src.ndim == 2:
        gray = src.astype(np.float32) / 255.0
        alpha = np.ones_like(gray)
        arr = np.stack([gray, gray, gray, alpha], axis=-1)
    elif src.shape[-1] == 1:
        gray = src[..., 0].astype(np.float32) / 255.0
        alpha = np.ones_like(gray)
        arr = np.stack([gray, gray, gray, alpha], axis=-1)
    elif src.shape[-1] == 2:
        gray = src[..., 0].astype(np.float32) / 255.0
        alpha = src[..., 1].astype(np.float32) / 255.0
        arr = np.stack([gray, gray, gray, alpha], axis=-1)
    elif src.shape[-1] == 3:
        rgb = src.astype(np.float32) / 255.0
        alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
        arr = np.concatenate([rgb, alpha], axis=-1)
    elif src.shape[-1] == 4:
        arr = src.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported image shape: {src.shape}")
    return arr.astype(np.float32)


def _float_rgba_to_image(arr: np.ndarray, mode: str = "RGB") -> np.ndarray:
    """float32 RGBA → numpy uint8 图像。"""
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    if mode == "RGBA":
        return u8
    if mode == "L":
        return u8[..., 0]
    return u8[..., :3]


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
        u8 = (np.clip(arr[..., 0], 0.0, 1.0) * 255).astype(np.uint8)
        resized = resize_numpy_image(u8, new_w, new_h)
        return resized.astype(np.float32)[..., np.newaxis] / 255.0

    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    resized = resize_numpy_image(u8, new_w, new_h)
    return resized.astype(np.float32) / 255.0


# =============================================================================
# 盒式模糊（用于法线 detail 控制）
# =============================================================================

def _box_blur_channel(channel: np.ndarray, radius: int, edge_mode: str = "edge") -> np.ndarray:
    """单通道积分图盒式模糊，保持平均亮度。"""
    if radius <= 0:
        return channel.copy()
    mean_before = max(float(np.mean(channel)), 1e-8)

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
    mean_after = max(float(np.mean(blurred)), 1e-8)
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
    edge_mode: str = "wrap",
) -> np.ndarray:
    """灰度高度图 → RGBA 法线贴图（OpenGL 切线空间）。"""
    gray = np.clip(height.astype(np.float32), 0.0, 1.0)
    h, w = gray.shape

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    padded = np.pad(gray, 1, mode=edge_mode)
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
    diffuse: np.ndarray,
    strength: float = 1.5,
    detail: float = 0.4,
    invert: bool = False,
) -> np.ndarray:
    """从漫反射贴图生成法线贴图（numpy uint8 RGB/RGBA）。"""
    rgba = _image_to_float_rgba(diffuse)
    gray = _to_grayscale(rgba)

    if detail > 0:
        base = _box_blur(gray, percent=detail * 100, edge_mode="wrap")
        detail_layer = gray - base
        height = base + detail_layer
    else:
        height = gray

    normal_rgba = _compute_normal_from_height(height, strength=strength, invert=invert, edge_mode="wrap")
    return _float_rgba_to_image(normal_rgba, mode="RGB")


def renormalize_normal_map(normal: np.ndarray) -> np.ndarray:
    """对法线贴图重新归一化（uint8 RGB/RGBA）。"""
    rgba = _image_to_float_rgba(normal)
    vectors = rgba[..., :3] * 2.0 - 1.0
    vectors[..., 2] = np.abs(vectors[..., 2])
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors = vectors / np.maximum(length, 1e-8)
    rgba[..., :3] = vectors * 0.5 + 0.5
    return _float_rgba_to_image(rgba, mode="RGB")


# =============================================================================
# QA：接缝指标
# =============================================================================

def compute_seam_metrics(image: np.ndarray, map_type: str = "", band: int = 4) -> dict:
    """返回贴图无缝化后的数值接缝指标。"""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr_f = arr.astype(np.float32)[..., np.newaxis]
    else:
        arr_f = arr[..., :3].astype(np.float32)

    h, w = arr_f.shape[:2]
    band = max(1, min(int(band), h // 2, w // 2))
    if band <= 0:
        return {"ok": True, "mean_delta": 0.0, "max_delta": 0.0}

    if map_type == "normal":
        arr_n = arr_f / 255.0 * 2.0 - 1.0
        arr_n = arr_n / np.maximum(np.linalg.norm(arr_n, axis=2, keepdims=True), 1e-8)
        lr_dot = np.sum(arr_n[:, 0, :] * arr_n[:, -1, :], axis=1)
        tb_dot = np.sum(arr_n[0, :, :] * arr_n[-1, :, :], axis=1)
        delta = np.concatenate([(1.0 - lr_dot).reshape(-1), (1.0 - tb_dot).reshape(-1)])
        mean_delta = float(np.mean(delta))
        max_delta = float(np.max(delta))
        return {
            "ok": mean_delta <= 0.03 and max_delta <= 0.25,
            "mean_delta": mean_delta,
            "max_delta": max_delta,
        }

    lr = np.abs(arr_f[:, 0, :] - arr_f[:, -1, :])
    tb = np.abs(arr_f[0, :, :] - arr_f[-1, :, :])
    delta = np.concatenate([lr.reshape(-1), tb.reshape(-1)])
    mean_delta = float(np.mean(delta))
    max_delta = float(np.max(delta))
    return {
        "ok": mean_delta <= 6.0 and max_delta <= 48.0,
        "mean_delta": mean_delta,
        "max_delta": max_delta,
    }


# =============================================================================
# 高度图生成：从法线贴图反推（FFT 求解 Poisson 方程）
# =============================================================================

def generate_height_from_normal(
    normal: np.ndarray,
    flip_green: bool = False,
) -> np.ndarray:
    """从法线贴图反推高度图（numpy uint8 RGB/RGBA）。"""
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

    div_x = (np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1)) * 0.5
    div_y = (np.roll(grad_y, -1, axis=0) - np.roll(grad_y, 1, axis=0)) * 0.5
    divergence = div_x + div_y

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
