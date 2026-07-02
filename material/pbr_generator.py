import bpy
from typing import Dict
from PIL import Image

from .uv_extractor import UVExtractor
from .shader_builder import ShaderBuilder
from ..sd_backend.abstract_client import GenerationConfig


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
        diffuse_config = self._build_config(props, uv_layout, "seamless texture, highly detailed, diffuse albedo")
        diffuse_result = self.client.img2img(diffuse_config)
        images['diffuse'] = self._pil_to_blender_image(diffuse_result.images[0], f"{obj.name}_diffuse")

        # Normal
        normal_config = self._build_config(props, uv_layout, "normal map, seamless texture")
        normal_result = self.client.img2img(normal_config)
        images['normal'] = self._pil_to_blender_image(normal_result.images[0], f"{obj.name}_normal")

        # Roughness
        rough_config = self._build_config(props, uv_layout, "roughness map, grayscale, seamless texture")
        rough_result = self.client.img2img(rough_config)
        images['roughness'] = self._pil_to_blender_image(rough_result.images[0], f"{obj.name}_roughness")

        # Metallic
        metal_config = self._build_config(props, uv_layout, "metallic map, grayscale, seamless texture")
        metal_result = self.client.img2img(metal_config)
        images['metallic'] = self._pil_to_blender_image(metal_result.images[0], f"{obj.name}_metallic")

        mat = ShaderBuilder.build_principled_bsdf(f"Mat_{obj.name}_PBR", images)

        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

        return mat

    def _build_config(self, props, init_image: Image.Image, prompt_suffix: str) -> GenerationConfig:
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

    def _pil_to_blender_image(self, pil_img: Image.Image, name: str) -> bpy.types.Image:
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        blender_img = bpy.data.images.new(name, width=pil_img.size[0], height=pil_img.size[1])
        resized = pil_img.resize((blender_img.size[0], blender_img.size[1]))
        # Blender pixels 原点在左下角，PIL 原点在左上角，需垂直翻转才能一致
        resized = resized.transpose(Image.FLIP_TOP_BOTTOM)
        pixels = list(resized.getdata())
        blender_img.pixels = [c / 255.0 for px in pixels for c in px]
        # 明确指定为 sRGB，避免 3D 视口因线性/非线性解释错误而发灰
        blender_img.colorspace_settings.name = 'sRGB'
        blender_img.update()
        return blender_img
