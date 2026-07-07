from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def with_reference_mode_hint(prompt: str, config: "GenerationConfig") -> str:
    """Prepend a reference-mode hint when an init_image is provided.

    Shared by GPTImageClient and NanobananaClient — avoids duplication.
    """
    if config.init_image is None:
        return prompt
    # 若 prompt 已由 build_texture_prompt 生成参考图前缀，避免重复追加
    if "参考图" in prompt and "以参考图" in prompt:
        return prompt
    hint = (
        "参考图是强视觉参考。请尽量保留参考图的纹理图案、颜色比例和表面质感，"
        "同时生成可平铺的 PBR 材质贴图。"
        "如果文字提示和参考图的材质类型、颜色或纹理冲突，必须以参考图为准。"
        "不要按文字提示改成另一种 Material Config 材质。"
        "漫反射 albedo，平光照明，无阴影，无高光，无反光，无镜面反射。"
    )
    if hint in prompt:
        return prompt
    return f"{hint}\n\n{prompt}"


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "DPM++ 2M Karras"
    batch_size: int = 1
    init_image: Optional[Any] = None
    denoising_strength: float = 0.75
    reference_tile_strength: float = 0.5
    use_chord_enhanced: bool = True
    # 本地 ComfyUI 工作流模型覆盖：{节点 ID: {输入字段: 值}}
    model_overrides: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class GenerationResult:
    images: List[Any]
    seed: int
    info: Dict
    metadata: Dict
    pbr_maps: Dict[str, Any] = field(default_factory=dict)  # e.g. {"basecolor": img, "normal": img, ...}


class AbstractSDClient(ABC):
    @abstractmethod
    def check_health(self) -> bool:
        pass

    @abstractmethod
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        pass
