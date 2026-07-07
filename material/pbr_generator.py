import bpy
from typing import Dict, Any

import numpy as np

from .uv_extractor import UVExtractor
from .shader_builder import ShaderBuilder
from ..sd_backend.abstract_client import GenerationConfig
from ..utils.image_utils import numpy_to_blender_image


class PBRGenerator:
    def __init__(self, client):
        self.client = client

    def generate_for_object(self, obj: bpy.types.Object, props) -> bpy.types.Material:
        if obj.type != 'MESH':
            raise RuntimeError("Selected object is not a mesh")

        extractor = UVExtractor(resolution=int(props.width))
        validation = extractor.validate_uv(obj)
        if not validation.is_valid:
            raise RuntimeError(
                f"UV validation failed: {'; '.join(validation.error_messages)}"
            )

        uv_layout = extractor.render_uv_layout(obj, image_size=int(props.width))

        images: Dict[str, bpy.types.Image] = {}

        # Diffuse
        diffuse_config = self._build_config(
            props, uv_layout,
            "seamless texture, highly detailed, diffuse albedo, matte flat color, "
            "no highlights, no reflections, no specular, no shadows"
        )
        diffuse_result = self.client.img2img(diffuse_config)
        images['diffuse'] = self._np_to_blender_image(np.array(diffuse_result.images[0]), f"{obj.name}_diffuse")

        # Normal
        normal_config = self._build_config(props, uv_layout, "normal map, seamless texture")
        normal_result = self.client.img2img(normal_config)
        images['normal'] = self._np_to_blender_image(np.array(normal_result.images[0]), f"{obj.name}_normal")

        # Roughness
        rough_config = self._build_config(props, uv_layout, "roughness map, grayscale, seamless texture")
        rough_result = self.client.img2img(rough_config)
        images['roughness'] = self._np_to_blender_image(np.array(rough_result.images[0]), f"{obj.name}_roughness")

        # Metallic
        metal_config = self._build_config(props, uv_layout, "metallic map, grayscale, seamless texture")
        metal_result = self.client.img2img(metal_config)
        images['metallic'] = self._np_to_blender_image(np.array(metal_result.images[0]), f"{obj.name}_metallic")

        mat = ShaderBuilder.build_principled_bsdf(f"Mat_{obj.name}_PBR", images)

        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

        return mat

    def _build_config(self, props, init_image: Any, prompt_suffix: str) -> GenerationConfig:
        # 安全读取属性：CTProperties 中不存在 steps/cfg_scale/sampler/denoising_strength，
        # 使用合理的默认值兼容旧代码路径
        steps = getattr(props, "steps", 25)
        cfg_scale = getattr(props, "cfg_scale", 7.0)
        sampler = getattr(props, "sampler", "DPM++ 2M Karras")
        denoising_strength = getattr(props, "denoising_strength", 0.75)
        return GenerationConfig(
            prompt=f"{props.prompt}, {prompt_suffix}" if props.prompt else prompt_suffix,
            negative_prompt=props.negative_prompt,
            width=int(props.width),
            height=int(props.height),
            steps=steps,
            cfg_scale=cfg_scale,
            seed=props.seed,
            sampler=sampler,
            batch_size=1,
            init_image=init_image,
            denoising_strength=denoising_strength,
        )

    def _np_to_blender_image(self, arr: np.ndarray, name: str) -> bpy.types.Image:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.ones_like(arr) * 255], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.ones((*arr.shape[:2], 1), dtype=np.uint8) * 255
            arr = np.concatenate([arr, alpha], axis=-1)
        elif arr.shape[-1] != 4:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        blender_img = numpy_to_blender_image(name, arr)
        # 明确指定为 sRGB，避免 3D 视口因线性/非线性解释错误而发灰
        blender_img.colorspace_settings.name = 'sRGB'
        return blender_img
