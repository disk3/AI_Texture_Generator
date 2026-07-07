# =============================================================================
# Seamless Texture Processor
# =============================================================================
# 参考 add-on-image-editor-tools-v0.3.8 的 seamless 管线实现，
# 提供两种无缝化策略：
#   - BASIC：中心偏移 + 边缘线性渐变混合，保持尺寸，适合法线等方向性贴图。
#   - ADVANCED：宏观梯度补偿 + Seam Carving 最优边界 + 双频融合，
#              输出尺寸会缩小，最后再插值回原尺寸，适合 diffuse / roughness 等。
# 仅依赖 numpy。
# =============================================================================

from __future__ import annotations

import math

import numpy as np


# =============================================================================
# 图像格式转换
# =============================================================================

def _to_float_rgba(src: np.ndarray) -> tuple[np.ndarray, dict]:
    """任意 numpy 图像 → float32 RGBA (H,W,4)，返回 (rgba, shape_info)。"""
    src = np.asarray(src)
    info = {
        "ndim": src.ndim,
        "shape": src.shape,
        "dtype": src.dtype,
        "has_alpha": src.ndim == 3 and src.shape[-1] == 4,
    }

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

    info["h"], info["w"] = arr.shape[:2]
    return np.clip(arr, 0.0, 1.0).astype(np.float32), info


def _from_float_rgba(arr: np.ndarray, info: dict) -> np.ndarray:
    """float32 RGBA → 原始格式 uint8。"""
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)

    if info["ndim"] == 2:
        return u8[..., 0]
    if info["shape"][-1] == 1:
        return u8[..., 0:1]
    if info["shape"][-1] == 2:
        return u8[..., [0, 3]]
    if info["shape"][-1] == 3:
        return u8[..., :3]
    return u8[..., :4]


# =============================================================================
# 纯 NumPy 图像工具
# =============================================================================

def _np_offset(img: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    """循环偏移（wrap-around），支持任意通道数。"""
    h, w = img.shape[:2]
    x_offset = x_offset % w
    y_offset = y_offset % h
    if x_offset != 0:
        img = np.concatenate([img[:, -x_offset:, :], img[:, :-x_offset, :]], axis=1)
    if y_offset != 0:
        img = np.concatenate([img[-y_offset:, :, :], img[:-y_offset, :, :]], axis=0)
    return img


def _np_resize_img(img: np.ndarray, target_w: int, target_h: int, wrap: bool = False) -> np.ndarray:
    """双线性 resize，支持任意通道数。wrap=True 时使用周期边界保持平铺性。"""
    img = img.astype(np.float32, copy=False)
    h_ori, w_ori = img.shape[:2]
    target_h = int(round(target_h))
    target_w = int(round(target_w))
    if target_h == h_ori and target_w == w_ori:
        return img.copy()

    y_target = np.linspace(0, h_ori - 1, target_h, dtype=np.float32)
    x_target = np.linspace(0, w_ori - 1, target_w, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_target, y_target)

    x0 = np.floor(x_grid).astype(np.int32)
    y0 = np.floor(y_grid).astype(np.int32)
    if wrap:
        x1 = (x0 + 1) % w_ori
        y1 = (y0 + 1) % h_ori
    else:
        x1 = np.minimum(x0 + 1, w_ori - 1)
        y1 = np.minimum(y0 + 1, h_ori - 1)

    dx = (x_grid - x0).astype(np.float32)
    dy = (y_grid - y0).astype(np.float32)
    dw = dx[..., np.newaxis]
    dh = dy[..., np.newaxis]

    val00 = img[y0, x0, :]
    val01 = img[y0, x1, :]
    val10 = img[y1, x0, :]
    val11 = img[y1, x1, :]

    val_x0 = val00 * (1 - dw) + val01 * dw
    val_x1 = val10 * (1 - dw) + val11 * dw
    resized = val_x0 * (1 - dh) + val_x1 * dh
    return np.clip(resized, 0.0, 1.0)


# =============================================================================
# BASIC：线性边缘混合（保持尺寸）
# =============================================================================

def np_make_seamless_tile_basic(img: np.ndarray, blend_ratio: float = 0.125) -> np.ndarray:
    """中心偏移 + 边缘线性渐变混合，输入输出 [0,1] float，任意通道。"""
    img = img.astype(np.float32, copy=False)
    h, w = img.shape[:2]
    short_side = min(w, h)
    blend_width = max(2, min(int(round(short_side * blend_ratio)), short_side // 2))
    bw = blend_width // 2

    def blend(base: np.ndarray, offset: np.ndarray, direction: str) -> np.ndarray:
        if bw <= 0:
            return base
        mask = np.zeros((h, w), dtype=np.float32)
        alpha_1 = 1.0 - np.arange(bw, dtype=np.float32) / bw
        alpha_2 = np.arange(bw, dtype=np.float32) / bw
        if direction == "horizontal":
            mask[:, :bw] = alpha_1[np.newaxis, :]
            mask[:, w - bw:w] = alpha_2[np.newaxis, :]
        else:
            mask[:bw, :] = alpha_1[:, np.newaxis]
            mask[h - bw:h, :] = alpha_2[:, np.newaxis]
        mask = mask[..., np.newaxis]
        return base * (1.0 - mask) + offset * mask

    h_offset = _np_offset(img.copy(), -w // 2, 0)
    img_b = blend(img, h_offset, "horizontal")
    v_offset = _np_offset(img_b.copy(), 0, -h // 2)
    img_c = blend(img_b, v_offset, "vertical")
    return np.clip(img_c, 0.0, 1.0)


# =============================================================================
# ADVANCED：宏观梯度补偿 + Seam Carving + 双频融合
# =============================================================================

def _np_fast_blur_padded(img: np.ndarray, iterations: int = 5) -> np.ndarray:
    """多次 roll 近似盒式/高斯模糊。"""
    pad_size = iterations * 2
    padded = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="edge")
    res = padded.astype(np.float32)
    two = np.float32(2.0)
    four = np.float32(4.0)
    for _ in range(iterations):
        res = (np.roll(res, 1, axis=1) + two * res + np.roll(res, -1, axis=1)) / four
        res = (np.roll(res, 1, axis=0) + two * res + np.roll(res, -1, axis=0)) / four
    return res[pad_size:-pad_size, pad_size:-pad_size, :]


def _np_local_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """把 source 的均值/方差对齐到 target。"""
    mu_s = np.mean(source, axis=(0, 1), keepdims=True).astype(np.float32)
    std_s = np.std(source, axis=(0, 1), keepdims=True).astype(np.float32)
    mu_t = np.mean(target, axis=(0, 1), keepdims=True).astype(np.float32)
    std_t = np.std(target, axis=(0, 1), keepdims=True).astype(np.float32)
    std_s = np.where(std_s == 0, 1e-5, std_s)
    matched = (source - mu_s) * (std_t / std_s) + mu_t
    return np.clip(matched, 0.0, 1.0)


def _np_compute_gradient_penalty(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """左右重叠区梯度差异（SSD of Gradients）。"""
    L_gray = np.mean(L, axis=2).astype(np.float32)
    R_gray = np.mean(R, axis=2).astype(np.float32)
    grad_L_y, grad_L_x = np.gradient(L_gray)
    grad_R_y, grad_R_x = np.gradient(R_gray)
    return (grad_R_x - grad_L_x) ** 2 + (grad_R_y - grad_L_y) ** 2


def _np_find_vertical_seam(error_matrix: np.ndarray) -> np.ndarray:
    """动态规划找垂直最优缝合线。"""
    h, w = error_matrix.shape
    cost = np.zeros_like(error_matrix, dtype=np.float32)
    paths = np.zeros_like(error_matrix, dtype=np.int8)
    cost[0, :] = error_matrix[0, :]

    for i in range(1, h):
        prev = cost[i - 1]
        left = np.roll(prev, 1)
        left[0] = np.inf
        right = np.roll(prev, -1)
        right[-1] = np.inf
        up = prev
        mins = np.minimum(np.minimum(left, up), right)
        cost[i] = error_matrix[i] + mins
        paths[i] = np.where(mins == left, -1, np.where(mins == up, 0, 1))

    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + paths[i + 1, seam[i + 1]]
    return seam


def _np_two_band_blend(L: np.ndarray, R: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """低频软混合 + 高频硬混合。"""
    L_low = _np_fast_blur_padded(L, iterations=5)
    R_low = _np_fast_blur_padded(R, iterations=5)
    L_high = L.astype(np.float32) - L_low
    R_high = R.astype(np.float32) - R_low

    high_blended = R_high * seam_mask + L_high * (1.0 - seam_mask)
    soft_mask = _np_fast_blur_padded(seam_mask, iterations=8)
    low_blended = R_low * soft_mask + L_low * (1.0 - soft_mask)

    return np.clip(low_blended + high_blended, 0.0, 1.0)


def _np_flatten_macro_gradient(img: np.ndarray, overlap_ratio: float = 0.2) -> np.ndarray:
    """用左右/上下边缘均值拟合线性光照并补偿，消除宏观色偏。"""
    img_float = img.astype(np.float32)
    H, W, C = img_float.shape
    overlap_w = max(1, int(W * overlap_ratio))
    overlap_h = max(1, int(H * overlap_ratio))
    eps = np.float32(1e-5)

    mean_L = np.mean(img_float[:, :overlap_w], axis=(0, 1))
    mean_R = np.mean(img_float[:, -overlap_w:], axis=(0, 1))
    target_X = (mean_L + mean_R) * 0.5
    profile_X = np.linspace(0, 1, W, dtype=np.float32).reshape(1, W, 1)
    lighting_X = mean_L * (1 - profile_X) + mean_R * profile_X
    img_float = img_float * (target_X / (lighting_X + eps))

    mean_T = np.mean(img_float[:overlap_h], axis=(0, 1))
    mean_B = np.mean(img_float[-overlap_h:], axis=(0, 1))
    target_Y = (mean_T + mean_B) * 0.5
    profile_Y = np.linspace(0, 1, H, dtype=np.float32).reshape(H, 1, 1)
    lighting_Y = mean_T * (1 - profile_Y) + mean_B * profile_Y
    img_float = img_float * (target_Y / (lighting_Y + eps))

    return np.clip(img_float, 0.0, 1.0)


def _np_quilt_horizontal(
    img: np.ndarray,
    overlap_w: int,
    alpha: float = 1.0,
    beta: float = 5.0,
) -> np.ndarray:
    """水平方向 Seam Carving + 双频融合。"""
    h, w, c = img.shape
    R = img[:, -overlap_w:, :].astype(np.float32)
    L = img[:, :overlap_w, :].astype(np.float32)

    R_matched = _np_local_color_transfer(R, L)
    diff = R_matched - L
    ssd_rgb = np.sum(diff ** 2, axis=2)
    ssd_grad = _np_compute_gradient_penalty(L, R_matched)
    error_matrix = np.float32(alpha) * ssd_rgb + np.float32(beta) * ssd_grad

    seam = _np_find_vertical_seam(error_matrix)
    hard_mask = np.zeros((h, overlap_w, 1), dtype=np.float32)
    for i in range(h):
        hard_mask[i, :seam[i]] = 1.0

    stitched = _np_two_band_blend(L, R_matched, hard_mask)
    middle = img[:, overlap_w:-overlap_w, :]
    return np.concatenate([middle, stitched], axis=1)


def _np_quilt_vertical(
    img: np.ndarray,
    overlap_h: int,
    alpha: float = 1.0,
    beta: float = 5.0,
) -> np.ndarray:
    """垂直方向 Seam Carving，转置后复用水平逻辑。"""
    transposed = np.transpose(img, (1, 0, 2))
    stitched = _np_quilt_horizontal(transposed, overlap_h, alpha, beta)
    return np.transpose(stitched, (1, 0, 2))


def np_make_texture_seamless_advanced(
    img: np.ndarray,
    overlap_ratio: float = 0.2,
    beta: float = 5.0,
) -> np.ndarray:
    """高级无缝：宏观梯度补偿 → 水平缝合 → 垂直缝合。

    输出尺寸会缩小 (H - 2*overlap_h, W - 2*overlap_w)。
    """
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    overlap_w = max(1, int(w * overlap_ratio))
    overlap_h = max(1, int(h * overlap_ratio))

    # 避免 overlap 过大导致中间区域为零
    overlap_w = min(overlap_w, w // 3)
    overlap_h = min(overlap_h, h // 3)

    img_corrected = _np_flatten_macro_gradient(img, overlap_ratio)
    img_h = _np_quilt_horizontal(img_corrected, overlap_w, alpha=1.0, beta=beta)
    final_img = _np_quilt_vertical(img_h, overlap_h, alpha=1.0, beta=beta)
    return np.clip(final_img, 0.0, 1.0)


# =============================================================================
# 周期类纹理：PPS + 晶格对齐（保留砖/瓦/地砖等规则图案尺寸）
# =============================================================================

def _period_from_profile(profile: np.ndarray) -> tuple[float, float]:
    """从一维 ACF 剖面（中心已去除）找第一个显著峰。"""
    if len(profile) < 3:
        return 0.0, 0.0
    center_val = profile[0]
    if center_val <= 0:
        return 0.0, 0.0
    profile = profile / center_val
    for i in range(1, len(profile) - 1):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > 0.05:
            local = profile[max(0, i - 2):min(len(profile), i + 3)]
            prominence = float(profile[i] - np.mean(local))
            return float(i + 1), prominence
    return 0.0, 0.0


def _detect_periodicity(gray: np.ndarray) -> tuple[float, float, float]:
    """返回 (period_y, period_x, score)。score 越大越像规则周期纹理。"""
    gray = gray - gray.mean()
    H, W = gray.shape
    F = np.fft.fft2(gray)
    acf = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    cy, cx = H // 2, W // 2

    px, sx = _period_from_profile(acf[cy, cx + 1:])
    py, sy = _period_from_profile(acf[cy + 1:, cx])

    if px <= 0 or py <= 0:
        return 0.0, 0.0, 0.0

    # 正方形图像且两个方向周期接近时，强制统一，避免地砖被拉成长方形
    if H == W and abs(px - py) <= 2:
        avg = (px + py) * 0.5
        px = py = avg

    score = min((sx + sy) * 4.0, 1.0)
    return py, px, score


def _lattice_resize(H: int, W: int, py: float, px: float) -> tuple[int, int]:
    """把图像缩放到周期整数倍，避免砖块被切半，并尽量保持原图宽高比。"""
    py_i = max(1, int(round(py)))
    px_i = max(1, int(round(px)))

    # 正方形图像：使用两个周期的最小公倍数，确保输出仍是正方形
    if H == W:
        lcm = (py_i * px_i) // math.gcd(py_i, px_i)
        n = max(lcm, round(H / lcm) * lcm)
        return int(n), int(n)

    nH = max(py_i, round(H / py_i) * py_i)
    nW = max(px_i, round(W / px_i) * px_i)
    return int(nH), int(nW)


def _periodic_smooth_decomposition(u: np.ndarray) -> np.ndarray:
    """Moisan Periodic-plus-smooth 分解，让对向边界完全一致。"""
    H, W = u.shape
    v = np.zeros_like(u)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] += u[0, :] - u[-1, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] += u[:, 0] - u[:, -1]

    fv = np.fft.fft2(v)
    cy = np.cos(2.0 * np.pi * np.arange(H) / H)[:, None]
    cx = np.cos(2.0 * np.pi * np.arange(W) / W)[None, :]
    denom = 2.0 * cy + 2.0 * cx - 4.0
    denom[0, 0] = 1.0
    fs = fv / denom
    fs[0, 0] = 0.0
    p = u - np.real(np.fft.ifft2(fs))
    return p


def np_make_periodic_seamless(img: np.ndarray) -> np.ndarray | None:
    """针对规则周期纹理（砖/瓦/地砖/编织）的无缝化。

    检测周期 → 晶格对齐 → PPS → 恢复原尺寸。非周期纹理返回 None，便于回退。
    """
    H, W = img.shape[:2]
    if H < 64 or W < 64:
        return None

    gray = np.mean(img, axis=2) if img.ndim == 3 else img
    py, px, score = _detect_periodicity(gray)
    if score < 0.25 or py <= 0 or px <= 0:
        return None

    nH, nW = _lattice_resize(H, W, py, px)
    resized = _np_resize_img(img, nW, nH, wrap=True)

    out = np.empty_like(resized)
    for c in range(resized.shape[2]):
        out[..., c] = _periodic_smooth_decomposition(resized[..., c])

    out = _np_resize_img(out, W, H, wrap=True)
    return _lock_edges(out)


def np_make_pps_seamless(img: np.ndarray) -> np.ndarray:
    """Moisan PPS 无缝化：不裁剪、不融合偏移副本、不拉伸周期。

    只通过频域平滑让对向边界一致，因此不会在原图边缘产生花纹重叠或错位。
    适合“参考图 + 空提示词”这种需要最大限度保留原图内容的场景。
    """
    out = img.astype(np.float32).copy()
    for c in range(out.shape[2]):
        out[..., c] = _periodic_smooth_decomposition(out[..., c])
    return _lock_edges(out)


def np_make_preserve_seamless(img: np.ndarray) -> np.ndarray:
    """尽量保留原图内容的无缝化。

    规则周期纹理走晶格对齐；非周期纹理直接做 PPS，不裁剪、不融合偏移副本，
    最大程度保留原图细节。适用于“参考图 + 空提示词”的本地离线 PBR 烘焙。
    """
    periodic = np_make_periodic_seamless(img)
    if periodic is not None:
        return periodic

    return np_make_pps_seamless(img)


# =============================================================================
# 公共入口
# =============================================================================

def _lock_edges(img: np.ndarray) -> np.ndarray:
    """让对向边界像素完全一致，消除最后的数值残余。"""
    img[0, :] = img[-1, :] = (img[0, :] + img[-1, :]) * 0.5
    img[:, 0] = img[:, -1] = (img[:, 0] + img[:, -1]) * 0.5
    return img


def make_seamless_tile_local(
    image: np.ndarray,
    method: str = "SMART",
    blend_ratio: float = 0.125,
    overlap: float = 0.20,
    beta: float = 5.0,
    structure_radius: float = 0.02,  # 保留兼容，未使用
    levels: int = 5,                 # 保留兼容，未使用
    is_normal: bool = False,
) -> np.ndarray:
    """贴图无缝化本地入口（输入输出 numpy uint8 RGB/RGBA/灰度/单通道）。

    Args:
        method: "BASIC"、"SMART"、"ADVANCED"、"PERIODIC"、"PRESERVE" 或 "PPS"。
                SMART 会自动检测：规则周期纹理走 PERIODIC，否则走 ADVANCED。
                PRESERVE 尽量保留原图，周期纹理会晶格对齐。
                PPS 只通过频域平滑让边界一致，不偏移、不重叠原图内容，适合参考图本地烘焙。
        blend_ratio: BASIC 模式下边缘混合带占短边比例。
        overlap: ADVANCED 模式下重叠区比例。
        beta: ADVANCED 模式下梯度惩罚权重（砖石等强纹理可设 8~10，平滑纹理 2~3）。
        is_normal: 是否按法线贴图处理（使用 BASIC + 向量解码/归一化）。
    """
    rgba, info = _to_float_rgba(image)
    h0, w0 = info["h"], info["w"]

    if is_normal:
        # 法线在向量域做 BASIC 偏移混合，最后归一化
        vectors = rgba[..., :3] * 2.0 - 1.0
        blended = np_make_seamless_tile_basic(vectors, blend_ratio)
        length = np.linalg.norm(blended, axis=-1, keepdims=True)
        vectors = blended / np.maximum(length, 1e-8)
        rgba[..., :3] = vectors * 0.5 + 0.5
    else:
        if method == "BASIC":
            rgba = np_make_seamless_tile_basic(rgba, blend_ratio)
        elif method == "PPS":
            rgba = np_make_pps_seamless(rgba)
        elif method == "PERIODIC":
            periodic = np_make_periodic_seamless(rgba)
            rgba = periodic if periodic is not None else np_make_texture_seamless_advanced(rgba, overlap, beta)
        elif method == "PRESERVE":
            rgba = np_make_preserve_seamless(rgba)
        elif method == "SMART":
            periodic = np_make_periodic_seamless(rgba)
            if periodic is not None:
                rgba = periodic
            else:
                rgba = np_make_texture_seamless_advanced(rgba, overlap, beta)
                if rgba.shape[:2] != (h0, w0):
                    rgba = _np_resize_img(rgba, w0, h0, wrap=True)
        else:
            # ADVANCED
            rgba = np_make_texture_seamless_advanced(rgba, overlap, beta)
            if rgba.shape[:2] != (h0, w0):
                rgba = _np_resize_img(rgba, w0, h0, wrap=True)

    rgba = _lock_edges(rgba)
    return _from_float_rgba(rgba, info)


def process_single_image(
    image: np.ndarray,
    channel: str = "diffuse",
    manual_class=None,
) -> tuple[np.ndarray, None]:
    """兼容旧调用点的单图入口，固定返回 (image, None)。"""
    out = make_seamless_tile_local(
        image,
        method="SMART" if channel != "normal" else "BASIC",
        is_normal=(channel == "normal"),
    )
    return out, None


# 兼容旧导入名
TextureClass = None
