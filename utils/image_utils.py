import numpy as np
from PIL import Image


def pil_to_blender_pixels(pil_img: Image.Image) -> list:
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    pixels = list(pil_img.getdata())
    return [c / 255.0 for px in pixels for c in px]


def blender_pixels_to_pil(blender_img) -> Image.Image:
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
