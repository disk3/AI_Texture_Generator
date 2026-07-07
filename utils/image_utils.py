import os

import numpy as np


def blender_image_to_numpy(img) -> np.ndarray:
    """Blender bpy.types.Image → numpy ndarray (H, W, 4) uint8 RGBA。"""
    width, height = img.size
    pixels = np.array(img.pixels[:], dtype=np.float32)
    if pixels.size != width * height * 4:
        # 某些格式可能不是 4 通道，先补齐或截断到 4 通道
        pixels = np.resize(pixels, width * height * 4)
    arr = pixels.reshape((height, width, 4))
    # Blender pixels 原点在左下角，numpy 图像通常按左上角处理，做垂直翻转
    arr = np.flipud(arr)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def numpy_to_blender_image(name: str, arr: np.ndarray) -> "bpy.types.Image":
    """numpy ndarray (H, W, C) uint8 → Blender bpy.types.Image。"""
    import bpy

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.ones_like(arr) * 255], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.ones((*arr.shape[:2], 1), dtype=np.uint8) * 255
        arr = np.concatenate([arr, alpha], axis=-1)
    elif arr.shape[-1] != 4:
        raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

    height, width = arr.shape[:2]
    # numpy 原点在左上角，Blender pixels 原点在左下角，需垂直翻转
    flipped = np.flipud(arr)
    flat = (flipped.astype(np.float32) / 255.0).reshape(-1)

    blender_img = bpy.data.images.new(name=name, width=width, height=height)
    blender_img.pixels.foreach_set(flat.tolist())
    blender_img.update()
    return blender_img


def save_numpy_image(arr: np.ndarray, path: str, file_format: str = "PNG") -> None:
    """使用 Blender API 保存 numpy 图像到磁盘，不依赖 Pillow。"""
    import bpy

    name = os.path.splitext(os.path.basename(path))[0]
    blender_img = numpy_to_blender_image(name, arr)
    blender_img.filepath_raw = path.replace("\\", "/")
    blender_img.file_format = file_format
    blender_img.save()
    # 保存后从 bpy.data.images 移除临时 image，避免污染
    bpy.data.images.remove(blender_img)


def resize_numpy_image(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """numpy 图像双线性缩放（支持 2D 灰度或 HWC 彩色，uint8/float32）。"""
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]

    h, w, c = arr.shape
    if h == new_h and w == new_w:
        return arr.copy()

    src_dtype = arr.dtype
    arr_f = arr.astype(np.float32)

    # 坐标映射：目标像素中心反向映射到源图像
    x_src = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    y_src = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = x_src - x0
    wy = y_src - y0

    # 取四个角
    Ia = arr_f[y0[:, None], x0[None, :]]
    Ib = arr_f[y1[:, None], x0[None, :]]
    Ic = arr_f[y0[:, None], x1[None, :]]
    Id = arr_f[y1[:, None], x1[None, :]]

    wa = (1 - wx[None, :]) * (1 - wy[:, None])
    wb = (1 - wx[None, :]) * wy[:, None]
    wc = wx[None, :] * (1 - wy[:, None])
    wd = wx[None, :] * wy[:, None]

    resized = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

    if np.issubdtype(src_dtype, np.integer):
        resized = np.clip(resized, 0, 255).astype(src_dtype)
    else:
        resized = np.clip(resized, 0.0, 1.0).astype(src_dtype)

    if c == 1:
        return resized[..., 0]
    return resized


def resize_numpy_image_keep_aspect(
    arr: np.ndarray,
    target_w: int,
    target_h: int,
    mode: str = "cover",
) -> np.ndarray:
    """等比缩放 numpy 图像，避免拉伸变形。

    mode="cover"：缩放至填满目标画布，居中裁剪溢出部分（适合纹理，无黑边）。
    mode="contain"：缩放至完全装入目标画布，居中填充边缘（适合需保留全部内容）。

    输入：2D 灰度或 HWC 彩色（uint8/float32）。
    输出：target_h x target_w，保持原图宽高比。
    """
    flat = arr.ndim == 2
    if flat:
        arr = arr[..., np.newaxis]

    h, w, c = arr.shape
    if h == target_h and w == target_w:
        out = arr.copy()
    else:
        scale = (
            max(target_w / w, target_h / h)
            if mode == "cover"
            else min(target_w / w, target_h / h)
        )
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = resize_numpy_image(arr, new_w, new_h)

        if mode == "cover":
            # 居中裁剪
            start_y = (new_h - target_h) // 2
            start_x = (new_w - target_w) // 2
            out = resized[start_y:start_y + target_h, start_x:start_x + target_w]
        else:
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            out = np.pad(
                resized,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

    if flat:
        return out[..., 0]
    return out


def pil_to_blender_pixels(pil_img) -> list:
    """PIL Image → Blender pixels 扁平列表（需要 Pillow）。"""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("pil_to_blender_pixels 需要 Pillow，请安装 Pillow") from e

    if not isinstance(pil_img, Image.Image):
        raise TypeError("pil_img must be a PIL Image")
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    pixels = list(pil_img.getdata())
    return [c / 255.0 for px in pixels for c in px]


def blender_pixels_to_pil(blender_img) -> "Image.Image":
    """Blender Image → PIL Image（需要 Pillow）。"""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("blender_pixels_to_pil 需要 Pillow，请安装 Pillow") from e

    width = blender_img.size[0]
    height = blender_img.size[1]
    pixels = np.array(blender_img.pixels[:]).reshape((height, width, 4))
    pixels = (pixels * 255).astype(np.uint8)
    return Image.fromarray(pixels, 'RGBA')


def normalize_depth(z_pass: np.ndarray, near: float, far: float) -> np.ndarray:
    valid = np.clip(z_pass, near, far)
    normalized = (valid - near) / (far - near)
    normalized = 1.0 - normalized
    return (normalized * 65535).astype(np.uint16)


def _guess_image_extension(data: bytes) -> str:
    """根据文件头猜测图像格式扩展名。"""
    if data.startswith(b'\x89PNG'):
        return '.png'
    if data.startswith(b'\xff\xd8'):
        return '.jpg'
    if data.startswith(b'RIFF') or data.startswith(b'WEBP'):
        return '.webp'
    if data.startswith(b'BM'):
        return '.bmp'
    if data.startswith(b'GIF'):
        return '.gif'
    return '.png'


def load_image_bytes_to_numpy(data: bytes) -> np.ndarray:
    """从图像文件 bytes 加载为 numpy RGBA 数组。

    Pillow 可用时直接在线程内解码；无 Pillow 时回到 Blender 主线程
    使用 bpy 解码，避免 worker 线程直接访问 Blender 数据 API。
    """
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(data)) as pil_img:
            return np.array(pil_img.convert('RGBA'), dtype=np.uint8)
    except ImportError:
        pass

    import tempfile

    ext = _guess_image_extension(data)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        from .async_bridge import run_on_main_thread

        def _decode_on_main_thread():
            import bpy

            blender_img = bpy.data.images.load(temp_path, check_existing=False)
            try:
                return blender_image_to_numpy(blender_img)
            finally:
                bpy.data.images.remove(blender_img)

        return run_on_main_thread(_decode_on_main_thread, timeout=60.0)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def encode_numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    """把 numpy 图像编码为 PNG bytes，优先使用 Pillow，无 Pillow 时用 Blender API。"""
    try:
        from PIL import Image
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, 'L')
        elif arr.shape[-1] == 4:
            pil_img = Image.fromarray(arr, 'RGBA')
        elif arr.shape[-1] == 3:
            pil_img = Image.fromarray(arr, 'RGB')
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return buf.getvalue()
    except ImportError:
        pass

    # Pillow-free fallback. Blender encoding must happen on the main thread.
    import io
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name

    try:
        from .async_bridge import run_on_main_thread

        def _encode_on_main_thread():
            import bpy

            blender_img = numpy_to_blender_image('_temp_encode', arr)
            try:
                blender_img.filepath_raw = temp_path.replace('\\', '/')
                blender_img.file_format = 'PNG'
                blender_img.save()
            finally:
                bpy.data.images.remove(blender_img)

        run_on_main_thread(_encode_on_main_thread, timeout=60.0)
        with open(temp_path, 'rb') as f:
            return f.read()
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
