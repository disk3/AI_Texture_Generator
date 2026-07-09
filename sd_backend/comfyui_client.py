import json
import os
import time
import uuid
import random
import tempfile
from typing import Dict, Any, Tuple
import io
import base64

from .abstract_client import AbstractSDClient, GenerationConfig, GenerationResult
from ..utils.async_bridge import thread_safe_callback
from ..utils.logger import get_logger

log = get_logger(__name__)


def _parse_version(version_str: str):
    """把 'v0.24.0' / '0.20.1' 解析为可比较的整数元组；解析失败返回 None。"""
    if not version_str:
        return None
    parts = version_str.lstrip("v").split(".")
    nums = []
    for part in parts[:3]:
        try:
            nums.append(int(part))
        except ValueError:
            break
    return tuple(nums) if nums else None


def _requests_module():
    try:
        import requests
        return requests
    except ImportError:
        from ..utils import simple_requests
        return simple_requests


class ComfyUIClient(AbstractSDClient):

    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: int = 600,
                 zimage_workflow_path: str = ""):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self.timeout = timeout
        self._progress_cb = None
        self._zimage_workflow_path = zimage_workflow_path

    def check_health(self, auto_launch_path: str = "") -> bool:
        """检查 ComfyUI 是否在线，若不在线且提供了安装路径则自动启动。

        同时缓存 /system_stats，用于后续核心版本判断。
        """
        # 优先尝试复用 ComfyUI 便携版中的 Pillow/requests/websocket
        if auto_launch_path:
            try:
                from .comfyui_env_resolver import inject_comfyui_packages
                inject_comfyui_packages(auto_launch_path)
            except Exception as e:
                log.debug("Could not inject ComfyUI packages: %s", e)

        requests = _requests_module()
        try:
            # 启动初期 /system_stats 可能响应较慢，使用更宽容的 10 秒超时
            r = requests.get(f"{self.base_url}/system_stats", timeout=(3, 10))
            if r.status_code == 200:
                try:
                    self._system_stats_cache = r.json()
                except Exception:
                    pass
                return True
        except requests.exceptions.ReadTimeout:
            log.debug("ComfyUI at %s connected but response read timed out", self.base_url)
        except Exception:
            log.debug("ComfyUI not responding at %s (may not be running)", self.base_url)

        # 未响应且提供了安装路径 → 尝试自动启动
        if auto_launch_path and os.path.isdir(auto_launch_path):
            from .comfyui_launcher import launch_comfyui, wait_for_comfyui

            log.info("ComfyUI not running, launching from %s ...", auto_launch_path)
            if launch_comfyui(auto_launch_path):
                log.info("Waiting for ComfyUI to start ...")
                if wait_for_comfyui(self.base_url, timeout=120, comfyui_path=auto_launch_path):
                    log.info("ComfyUI is ready.")
                    # 启动成功后再次拉取 system_stats 以获取核心版本
                    try:
                        r = requests.get(f"{self.base_url}/system_stats", timeout=(3, 10))
                        if r.status_code == 200:
                            try:
                                self._system_stats_cache = r.json()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return True
                else:
                    log.warning("ComfyUI failed to start within 120s.")
            else:
                log.warning("Failed to launch ComfyUI.")
        return False

    def check_workflow_nodes(self, family_id: str = "zimage") -> list:
        """检查指定模型族的完整 workflow 所需节点在 ComfyUI 中是否都存在。

        返回缺失节点列表，每个元素为 (node_id, class_type, hint)。
        若无法获取 /object_info 则返回空列表（由后续执行时再做校验）。
        """
        try:
            from .abstract_client import GenerationConfig
            config = GenerationConfig(
                prompt="",
                width=512,
                height=512,
                model_family=family_id,
                model_selections={},
            )
            workflow, _, _ = self._build_local_workflow(config)
        except Exception as e:
            log.warning("Could not build workflow for node check: %s", e)
            return []

        try:
            self._validate_workflow_nodes(workflow)
        except RuntimeError as exc:
            message = str(exc)
            import re
            missing = []
            for line in message.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.search(r"节点\s+([^:]+):\s+(\S+)\s*->\s*(.*)", line)
                if m:
                    missing.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
                elif "->" in line:
                    parts = line.split("->")
                    missing.append(("", parts[0].strip().strip(":"), parts[1].strip()))
                else:
                    missing.append(("", line, ""))
            return missing
        return []

    def set_progress_callback(self, callback):
        self._progress_cb = callback

    def interrupt(self) -> bool:
        """Best-effort cancellation for the currently running ComfyUI prompt."""
        requests = _requests_module()
        try:
            resp = requests.post(f"{self.base_url}/interrupt", timeout=5)
            return resp.status_code < 400
        except Exception:
            log.debug("ComfyUI interrupt failed")
            return False

    def execute_workflow_json(self, config: GenerationConfig, workflow_path: str = "") -> GenerationResult:
        """Local ComfyUI 路径：按模型族动态组装生图子图 + 后处理子图。

        - <=1024：生图 -> CHORD（无放大）
        - >1024： 生图 -> CHORD@1024 -> SeedVR2（先 CHORD 再放大）
        """
        target_w = int(config.width)
        target_h = int(config.height)
        use_upscaler = target_w > 1024 or target_h > 1024

        # 兼容旧的外部自定义 workflow 路径（仍按 Z-Image 硬编码节点处理）
        legacy_path = workflow_path or (
            self._zimage_workflow_path
            if os.path.isfile(getattr(self, "_zimage_workflow_path", "") or "")
            else ""
        )
        if legacy_path and os.path.isfile(legacy_path):
            workflow = self._load_legacy_zimage_workflow(legacy_path, config, target_w, target_h)
            actual_seed = workflow.get("7", {}).get("inputs", {}).get("seed", -1)
        else:
            workflow, actual_seed, _ = self._build_local_workflow(config)

        result = self._execute_workflow(
            workflow,
            output_mode="by_prefix",
            timeout=self._workflow_timeout(target_w, target_h),
        )
        result.seed = actual_seed
        return result

    def _build_local_workflow(self, config: GenerationConfig) -> Tuple[Dict[str, Any], int, bool]:
        """根据 model_family 选择模板，合并生图/后处理子图并注入用户参数。"""
        from . import workflow_specs as specs

        family_id = getattr(config, "model_family", "zimage") or "zimage"
        family = dict(specs.get_family_spec(family_id))
        family["_id"] = family_id
        target_w = int(config.width)
        target_h = int(config.height)
        use_upscaler = target_w > 1024 or target_h > 1024
        chord_first = use_upscaler

        gen_wf = specs.load_workflow_template(family["gen_template"])
        post_template_name = family["post_hires_template"] if use_upscaler else family["post_lowres_template"]
        post_wf = specs.load_workflow_template(post_template_name)

        vae_decode_spec = family["nodes"]["vae_decode"]
        post_input_spec = family["nodes"]["post_input"]
        workflow, post_id_map = specs.merge_workflow_subgraphs(
            gen_wf,
            post_wf,
            gen_output_node_id=vae_decode_spec["id"],
            post_input_node_id=post_input_spec["id"],
            post_input_field=post_input_spec.get("field", "image"),
        )

        resolver = lambda logical_name: specs.resolve_node_id(family, logical_name, post_id_map)

        print(
            f"[AI Texture] Local ComfyUI: family={family_id}, "
            f"templates={family['gen_template']}+{post_template_name} ({target_w}x{target_h})"
        )

        actual_seed = self._apply_family_overrides(workflow, family, config, resolver)
        self._apply_model_selections(workflow, family, config.model_selections, resolver)

        # 后处理入口尺寸
        post_input_id = resolver("post_input")
        if chord_first:
            workflow[post_input_id]["inputs"]["target_width"] = 1024
            workflow[post_input_id]["inputs"]["target_height"] = 1024
        else:
            workflow[post_input_id]["inputs"]["target_width"] = target_w
            workflow[post_input_id]["inputs"]["target_height"] = target_h

        if use_upscaler:
            self._set_seedvr2_resolution_by_spec(workflow, family, resolver, target_w, target_h)

        if config.init_image is not None:
            self._apply_reference_image(workflow, family, resolver, config)

        # 保留对旧 model_overrides 的兼容（仅当节点 ID 存在时生效）
        for node_id, params in getattr(config, "model_overrides", {}).items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(params)
                log.debug("Applied legacy model override node %s: %s", node_id, params)

        # 预校验模型文件是否存在于 ComfyUI 对应目录，避免提交后才报 400
        self._validate_model_files(workflow, family, getattr(config, "model_selections", {}))

        return workflow, actual_seed, chord_first

    def _apply_family_overrides(
        self,
        workflow: Dict[str, Any],
        family: Dict[str, Any],
        config: GenerationConfig,
        resolver,
    ) -> int:
        """注入提示词、latent 尺寸、采样参数等族相关默认值。"""
        return self._apply_zimage_overrides(workflow, family, config, resolver)

    def _apply_zimage_overrides(
        self,
        workflow: Dict[str, Any],
        family: Dict[str, Any],
        config: GenerationConfig,
        resolver,
    ) -> int:
        """Z-Image / Lumina2 族的默认值注入。"""
        defaults = family.get("defaults", {})
        seed = config.seed if config.seed >= 0 else random.randint(0, 2147483647)

        positive_id = resolver("positive")
        workflow[positive_id]["inputs"]["text"] = self._tileable_prompt(config.prompt)

        latent_id = resolver("latent")
        workflow[latent_id]["inputs"]["width"] = defaults.get("width", 1024)
        workflow[latent_id]["inputs"]["height"] = defaults.get("height", 1024)

        sampler_id = resolver("sampler")
        sampler_inputs = workflow[sampler_id]["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["cfg"] = defaults.get("cfg", 1.0)
        sampler_inputs["steps"] = defaults.get("steps", 9)
        sampler_inputs["sampler_name"] = defaults.get("sampler_name", "euler")
        sampler_inputs["scheduler"] = defaults.get("scheduler", "simple")
        sampler_inputs["denoise"] = defaults.get("denoise", 1.0)

        log.debug("Node %s prompt length: %d", positive_id, len(workflow[positive_id]["inputs"]["text"]))
        return seed

    def _apply_model_selections(
        self,
        workflow: Dict[str, Any],
        family: Dict[str, Any],
        selections: Dict[str, str],
        resolver,
    ) -> None:
        """注入用户在该模型族下选择的具体模型文件。"""
        if not selections:
            return

        mapping = {
            "main_model": ("main_model", "field"),
            "vae": ("vae", "field"),
            "text_encoder": ("text_encoder", "field"),
            "chord_model": ("chord_model", "field"),
        }
        for selection_key, filename in selections.items():
            if not filename or filename in ("DEFAULT", "__EMPTY__", "NONE"):
                continue
            logical_name, field_attr = mapping.get(selection_key, (None, None))
            if logical_name is None or logical_name not in family["nodes"]:
                continue
            node_spec = family["nodes"][logical_name]
            node_id = resolver(logical_name)
            field_name = node_spec.get(field_attr)
            if field_name:
                workflow[node_id]["inputs"][field_name] = filename
                log.debug(
                    "Applied selection %s -> node %s.%s = %s",
                    selection_key, node_id, field_name, filename,
                )

    def _set_seedvr2_resolution_by_spec(
        self,
        workflow: Dict[str, Any],
        family: Dict[str, Any],
        resolver,
        target_w: int,
        target_h: int,
    ) -> None:
        """按族规范设置 SeedVR2 放大目标分辨率。"""
        resolution = max(target_w, target_h)
        for logical_name in ("seedvr2_basecolor", "seedvr2_normal"):
            if logical_name not in family["nodes"]:
                continue
            node_id = resolver(logical_name)
            if workflow[node_id].get("class_type") == "SeedVR2VideoUpscaler":
                workflow[node_id]["inputs"]["resolution"] = resolution

    def _apply_reference_image(
        self,
        workflow: Dict[str, Any],
        family: Dict[str, Any],
        resolver,
        config: GenerationConfig,
    ) -> None:
        """把参考图编码后接入该族 sampler 的 latent_image。"""
        uploaded_name = self._upload_image(config.init_image, "ref")
        denoise = getattr(config, "denoising_strength", 0.65)

        vae_node_id = resolver("vae")
        sampler_node_id = resolver("sampler")

        workflow["100"] = {
            "inputs": {"image": uploaded_name},
            "class_type": "LoadImage",
        }
        workflow["101"] = {
            "inputs": {
                "pixels": ["100", 0],
                "vae": [vae_node_id, 0],
            },
            "class_type": "VAEEncode",
        }
        if sampler_node_id in workflow:
            workflow[sampler_node_id]["inputs"]["latent_image"] = ["101", 0]
            if "denoise" in workflow[sampler_node_id]["inputs"]:
                workflow[sampler_node_id]["inputs"]["denoise"] = denoise

        # CHORD 参考图分支（如果后处理子图包含 ETN_LoadImageBase64 节点）
        for node_id, node_def in workflow.items():
            if node_id in ("100", "101"):
                continue
            if node_def.get("class_type") == "ETN_LoadImageBase64":
                import numpy as np
                from ..utils.image_utils import encode_numpy_to_png_bytes
                init_img = config.init_image
                if hasattr(init_img, 'shape'):
                    arr = np.asarray(init_img)
                    ref_bytes = encode_numpy_to_png_bytes(arr)
                elif hasattr(init_img, 'save'):
                    ref_buffer = io.BytesIO()
                    init_img.save(ref_buffer, format="PNG")
                    ref_bytes = ref_buffer.getvalue()
                else:
                    raise TypeError("CHORD 参考图必须是 PIL Image 或 numpy 数组")
                ref_b64 = base64.b64encode(ref_bytes).decode()
                workflow[node_id]["inputs"]["image"] = ref_b64
                break

    def _load_legacy_zimage_workflow(
        self,
        path: str,
        config: GenerationConfig,
        target_w: int,
        target_h: int,
    ) -> Dict[str, Any]:
        """加载旧版完整 workflow JSON（仅用于外部自定义路径兼容）。"""
        with open(path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        print(f"[AI Texture] Local ComfyUI (legacy): running workflow '{os.path.basename(path)}' ({target_w}x{target_h})")

        overrides = self._build_zimage_overrides(config)
        for node_id, params in overrides.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(params)

        for node_id, params in getattr(config, "model_overrides", {}).items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(params)
                log.debug("Applied model override node %s: %s", node_id, params)

        chord_first = target_w > 1024 or target_h > 1024
        if "11" in workflow:
            if chord_first:
                workflow["11"]["inputs"]["target_width"] = 1024
                workflow["11"]["inputs"]["target_height"] = 1024
            else:
                workflow["11"]["inputs"]["target_width"] = target_w
                workflow["11"]["inputs"]["target_height"] = target_h

        if chord_first:
            self._set_seedvr2_resolution(workflow, target_w, target_h)

        if config.init_image is not None:
            uploaded_name = self._upload_image(config.init_image, "ref")
            denoise = getattr(config, "denoising_strength", 0.65)
            workflow["100"] = {
                "inputs": {"image": uploaded_name},
                "class_type": "LoadImage",
            }
            workflow["101"] = {
                "inputs": {
                    "pixels": ["100", 0],
                    "vae": ["8", 0],
                },
                "class_type": "VAEEncode",
            }
            if "7" in workflow:
                workflow["7"]["inputs"]["latent_image"] = ["101", 0]
                workflow["7"]["inputs"]["denoise"] = denoise

            if "58" in workflow and workflow["58"].get("class_type") == "ETN_LoadImageBase64":
                import numpy as np
                from ..utils.image_utils import encode_numpy_to_png_bytes
                init_img = config.init_image
                if hasattr(init_img, 'shape'):
                    arr = np.asarray(init_img)
                    ref_bytes = encode_numpy_to_png_bytes(arr)
                elif hasattr(init_img, 'save'):
                    ref_buffer = io.BytesIO()
                    init_img.save(ref_buffer, format="PNG")
                    ref_bytes = ref_buffer.getvalue()
                else:
                    raise TypeError("CHORD 参考图必须是 PIL Image 或 numpy 数组")
                ref_b64 = base64.b64encode(ref_bytes).decode()
                workflow["58"]["inputs"]["image"] = ref_b64

        return workflow

    def execute_chord_workflow(
        self,
        diffuse_image: Any,
        width: int = 2048,
        height: int = 2048,
        model_name: str = "",
    ) -> GenerationResult:
        """上传 diffuse 贴图并执行 CHORD-only 工作流，返回高质量 PBR maps。

        Args:
            diffuse_image: API 生成的 diffuse 贴图 (PIL Image 或 numpy 数组)
            width: 输出贴图宽度（覆盖 workflow 中 Node 11 的 target_width）
            height: 输出贴图高度（覆盖 workflow 中 Node 11 的 target_height）
            model_name: 可选。覆盖 CHORD 主模型文件名

        Returns:
            GenerationResult with pbr_maps containing basecolor/normal/roughness/metalness/height
        """
        # 1. 上传 diffuse 到 ComfyUI
        uploaded_name = self._upload_image(diffuse_image, "diffuse")

        # 2. 加载 CHORD-only 工作流
        addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(addon_dir, "workflows", "chord_only_api.json")

        with open(path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        print(f"[AI Texture] CHORD-only: running workflow '{os.path.basename(path)}' ({width}x{height})")

        # 3. 替换 LoadImage 节点为上传的图片
        workflow["100"]["inputs"]["image"] = uploaded_name

        # 4. 覆盖输出尺寸（避免 workflow 里硬编码 2048）
        if "11" in workflow:
            workflow["11"]["inputs"]["target_width"] = width
            workflow["11"]["inputs"]["target_height"] = height

        # 5. 应用用户选择的 CHORD 模型
        if model_name and "12" in workflow:
            workflow["12"]["inputs"]["ckpt_name"] = model_name
            log.debug("CHORD-only workflow model override: %s", model_name)

        # 6. 执行工作流
        return self._execute_workflow(workflow, output_mode="by_prefix")

    def _set_seedvr2_resolution(self, workflow: dict, target_w: int, target_h: int) -> None:
        """把 workflow 中所有 SeedVR2VideoUpscaler 节点的分辨率统一设为目标长边。"""
        resolution = max(target_w, target_h)
        for node_id in ("111", "121"):
            if node_id in workflow and workflow[node_id].get("class_type") == "SeedVR2VideoUpscaler":
                workflow[node_id]["inputs"]["resolution"] = resolution

    def _workflow_timeout(self, target_w: int, target_h: int) -> int:
        """高分辨率 SeedVR2 流程需要更多时间，按目标尺寸动态计算超时。"""
        max_target = max(target_w, target_h)
        if max_target <= 1024:
            return self.timeout
        # 每 1024 像素额外 10 分钟，2048=20min，4096=40min
        extra_minutes = (max_target // 1024) * 10
        return max(self.timeout, extra_minutes * 60)

    def _build_zimage_overrides(self, config: GenerationConfig) -> Dict[str, Dict]:
        """Map GenerationConfig to Zimage workflow node overrides."""
        seed = config.seed if config.seed >= 0 else random.randint(0, 2147483647)
        prompt = self._tileable_prompt(config.prompt)
        return {
            "4": {"text": prompt},
            # Z-Image Turbo 固定 1024² 生成（最佳精度/显存平衡点）
            # 最终输出尺寸由 _import_texture_results 做 resize
            "6": {"width": 1024, "height": 1024},
            "7": {"seed": seed},
        }

    @staticmethod
    def _tileable_prompt(prompt: str) -> str:
        """Add workflow-specific tileability constraints without overwriting user intent."""
        prompt = (prompt or "").strip()
        hint = (
            "四向无缝可平铺，左右边缘和上下边缘纹理连续，"
            "不要边框、不要中心主体、不要透视畸变，"
            "平光照明，无阴影，无高光，无反光，无镜面反射"
        )
        if not prompt:
            return hint
        if any(key in prompt for key in ("四向无缝", "四向平铺", "边缘连续", "tileable")):
            return prompt
        return f"{prompt}。{hint}"

    @staticmethod
    def _map_prefix_to_map_type(prefix: str) -> str:
        """Map SaveImage filename_prefix to standard PBR map type key."""
        text = os.path.basename(str(prefix or "")).strip().lower()
        if not text:
            return ""
        text = text.replace("-", "_").replace(" ", "_")
        mapping = {
            "basecolor": "basecolor",
            "base_color": "basecolor",
            "albedo": "basecolor",
            "diffuse": "diffuse",
            "color": "diffuse",
            "texture_image": "diffuse",
            "normal": "normal",
            "roughness": "roughness",
            "rough": "roughness",
            "height": "height",
            "displacement": "height",
            "metalness": "metalness",
            "metallic": "metalness",
            "metal": "metalness",
            "放大": "diffuse",
        }
        if text in mapping:
            return mapping[text]
        for key, map_type in mapping.items():
            if text.startswith(f"{key}_") or text.startswith(f"{key}.") or f"_{key}_" in text:
                return map_type
        return ""

    def _upload_image(self, image: Any, name: str = "input") -> str:
        requests = _requests_module()
        import numpy as np
        from ..utils.image_utils import encode_numpy_to_png_bytes

        if hasattr(image, 'shape'):
            arr = np.asarray(image)
            img_bytes = encode_numpy_to_png_bytes(arr)
        elif hasattr(image, 'save'):
            # PIL Image fallback
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
        else:
            raise TypeError("image must be PIL Image or numpy array")

        files = {"image": (f"{name}.png", img_bytes, "image/png")}
        data = {"type": "input", "subfolder": ""}
        resp = requests.post(f"{self.base_url}/upload/image", files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json().get("name", f"{name}.png")

    def _fetch_object_info(self) -> Dict[str, Any]:
        """拉取 ComfyUI /object_info，缓存一次避免重复请求。"""
        if hasattr(self, "_object_info_cache"):
            return self._object_info_cache
        requests = _requests_module()
        try:
            resp = requests.get(f"{self.base_url}/object_info", timeout=10)
            resp.raise_for_status()
            self._object_info_cache = resp.json()
        except Exception as e:
            log.debug("Could not fetch ComfyUI object_info: %s", e)
            self._object_info_cache = {}
        return self._object_info_cache

    def get_comfyui_core_version(self):
        """返回 ComfyUI 核心版本元组，例如 (0, 24, 0)；获取失败返回 None。"""
        if hasattr(self, "_core_version_tuple"):
            return self._core_version_tuple
        stats = getattr(self, "_system_stats_cache", None)
        if stats is None:
            stats = self._fetch_system_stats()
        version_str = (stats or {}).get("system", {}).get("comfyui_version", "")
        self._core_version_tuple = _parse_version(version_str)
        return self._core_version_tuple

    def _fetch_system_stats(self) -> Dict[str, Any]:
        """拉取 ComfyUI /system_stats，缓存一次避免重复请求。"""
        if hasattr(self, "_system_stats_cache"):
            return self._system_stats_cache
        requests = _requests_module()
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=10)
            resp.raise_for_status()
            self._system_stats_cache = resp.json()
        except Exception as e:
            log.debug("Could not fetch ComfyUI system_stats: %s", e)
            self._system_stats_cache = {}
        return self._system_stats_cache

    def _validate_workflow_nodes(self, workflow: dict) -> None:
        """在提交前检查 workflow 中所有 class_type 是否在 ComfyUI 中可用。"""
        object_info = self._fetch_object_info()
        if not object_info:
            return

        missing = []
        for node_id, node_def in workflow.items():
            class_type = node_def.get("class_type")
            if class_type and class_type not in object_info:
                missing.append((node_id, class_type))

        if not missing:
            return

        # 将缺失节点映射到安装建议
        custom_hints = {
            "ChordLoadModel": "ComfyUI-Chord (https://github.com/ubisoft/ComfyUI-Chord)",
            "ChordMaterialEstimation": "ComfyUI-Chord",
            "ChordNormalToHeight": "ComfyUI-Chord",
            "ResizeAndPadImage": "ComfyUI-Chord",
            "SeedVR2LoadVAEModel": "ComfyUI-SeedVR2_VideoUpscaler (https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)",
            "SeedVR2LoadDiTModel": "ComfyUI-SeedVR2_VideoUpscaler",
            "SeedVR2VideoUpscaler": "ComfyUI-SeedVR2_VideoUpscaler",
        }
        core_version = self.get_comfyui_core_version()

        def _hint_for(class_type: str) -> str:
            if class_type in custom_hints:
                return custom_hints[class_type]
            req = self._BUILT_IN_NODE_VERSION_REQUIREMENTS.get(class_type)
            if not req:
                return "请检查 ComfyUI 版本或安装对应 custom nodes"
            required_str, feature = req
            required_tuple = _parse_version(required_str)
            if core_version is None:
                return f"升级 ComfyUI 核心到 v{required_str}+（{feature} 节点已内置）"
            actual_str = ".".join(str(x) for x in core_version)
            if required_tuple and core_version < required_tuple:
                return (
                    f"当前 ComfyUI 核心版本 {actual_str} 不包含此节点，"
                    f"请将核心升级到 v{required_str}+（{feature}）"
                    "；注意 ComfyUI Desktop 的应用版本号不等于核心版本"
                )
            return (
                f"当前核心版本 {actual_str} 已应包含此节点，但 /object_info 未返回，"
                "可能节点未加载或存在 custom nodes 冲突，请重启 ComfyUI 或检查节点冲突"
            )

        details = []
        for node_id, class_type in missing:
            hint = _hint_for(class_type)
            details.append(f"  节点 {node_id}: {class_type} -> {hint}")
        raise RuntimeError(
            "ComfyUI 缺少以下工作流必需的节点，请安装/升级后再试：\n" + "\n".join(details)
        )

    def _validate_model_files(
        self,
        workflow: dict,
        family: Dict[str, Any],
        selections: Dict[str, str],
    ) -> None:
        """检查用户选择的模型文件是否真实存在于 ComfyUI 对应目录中。

        只校验当前模型族实际会用到的 loader：UNETLoader / CLIPLoader / VAELoader。
        """
        object_info = self._fetch_object_info()
        if not object_info:
            return

        logical_map = {
            "main_model": "main_model",
            "secondary_model": "secondary_model",
            "vae": "vae",
            "text_encoder": "text_encoder",
            "text_encoder_1": "text_encoder",
            "text_encoder_2": "text_encoder",
        }
        errors = []
        for selection_key, filename in selections.items():
            if not filename or filename in ("DEFAULT", "__EMPTY__", "NONE"):
                continue
            logical_name = logical_map.get(selection_key)
            if not logical_name or logical_name not in family.get("nodes", {}):
                continue
            node_spec = family["nodes"][logical_name]
            class_type = node_spec.get("class")
            field = node_spec.get("field")
            if not class_type or not field:
                continue
            info = object_info.get(class_type, {})
            inputs = info.get("input", {})
            required = inputs.get("required", {})
            field_config = required.get(field, [[]])
            options = field_config[0] if isinstance(field_config, list) and field_config else []
            if filename not in options:
                errors.append(f"  {selection_key}: {filename}（{class_type}.{field}）")

        if errors:
            raise RuntimeError(
                "以下模型文件在本地 ComfyUI 中未找到，请确认已下载并放在正确目录：\n"
                + "\n".join(errors)
            )

    def _execute_workflow(self, workflow: dict, output_mode: str = "flat", require_images: bool = True, timeout: int = None) -> GenerationResult:
        requests = _requests_module()
        client_id = str(uuid.uuid4())
        ws = None
        workflow_timeout = timeout or self.timeout

        try:
            self._validate_workflow_nodes(workflow)

            try:
                import websocket
                ws = websocket.create_connection(f"{self.ws_url}?clientId={client_id}", timeout=workflow_timeout)
            except ImportError:
                log.debug("websocket-client not installed; falling back to HTTP polling")
                thread_safe_callback({
                    "status": "warning",
                    "message": "缺少 websocket-client，ComfyUI 实时进度不可用。请在插件偏好设置中点击「一键修复环境依赖（COMFYUI）」，然后重启 Blender。",
                })
            except Exception as e:
                log.debug("WebSocket connection failed, falling back to HTTP polling: %s", e)

            prompt_data = {"prompt": workflow, "client_id": client_id}

            # 调试用：始终保存最近一次提交的实际 workflow JSON，便于与 ComfyUI 直接运行对比
            last_workflow_path = os.path.join(tempfile.gettempdir(), "ai_texture_last_workflow.json")
            try:
                with open(last_workflow_path, "w", encoding="utf-8") as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            resp = requests.post(f"{self.base_url}/prompt", json=prompt_data, timeout=30)
            if resp.status_code >= 400:
                # 记录完整错误响应与当前 workflow，方便定位 400 Bad Request 根因
                detail = resp.text[:2048]
                log.error("ComfyUI /prompt %s: %s", resp.status_code, detail)
                dump_path = os.path.join(tempfile.gettempdir(), f"ai_texture_workflow_{client_id}.json")
                try:
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(workflow, f, ensure_ascii=False, indent=2)
                except Exception:
                    dump_path = "<failed to dump>"
                raise RuntimeError(
                    f"ComfyUI 提交失败 ({resp.status_code}): {detail}\n"
                    f"workflow 已转储: {dump_path}"
                )
            prompt_id = resp.json()["prompt_id"]
            log.debug("ComfyUI prompt_id: %s", prompt_id)

            if self._progress_cb:
                self._progress_cb(0.2, "工作流已提交到 ComfyUI...")

            history = None
            if ws is not None:
                while True:
                    msg = ws.recv()
                    if isinstance(msg, str):
                        msg_data = json.loads(msg)
                        msg_type = msg_data.get("type")
                        if msg_type == "progress":
                            value = msg_data.get("data", {}).get("value", 0)
                            max_val = msg_data.get("data", {}).get("max", 1)
                            progress = 0.2 + (value / max_val) * 0.6
                            if self._progress_cb:
                                self._progress_cb(progress, f"Sampling step {value}/{max_val}")
                        elif msg_type == "executing":
                            if msg_data.get("data", {}).get("node") is None:
                                break
                    elif isinstance(msg, bytes):
                        pass
            else:
                deadline = time.time() + workflow_timeout
                last_report = 0.0
                while time.time() < deadline:
                    history_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
                    if history_resp.status_code == 200:
                        candidate = history_resp.json()
                        entry = candidate.get(prompt_id, {})
                        if entry.get("outputs") or entry.get("status", {}).get("status_str") == "error":
                            history = candidate
                            break
                    now = time.time()
                    if self._progress_cb and now - last_report > 5:
                        elapsed = max(0.0, workflow_timeout - (deadline - now))
                        progress = 0.2 + min(elapsed / max(workflow_timeout, 1), 1.0) * 0.65
                        self._progress_cb(progress, "ComfyUI 正在执行工作流...")
                        last_report = now
                    time.sleep(1)
                if history is None:
                    raise TimeoutError(f"ComfyUI 工作流轮询超时（{workflow_timeout} 秒）")

            if self._progress_cb:
                self._progress_cb(0.9, "正在获取结果图像...")

            if history is None:
                history_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
                history_resp.raise_for_status()
                history = history_resp.json()
            outputs = history.get(prompt_id, {}).get("outputs", {})
            log.debug("History outputs nodes: %s", list(outputs.keys()))

            images = []
            pbr_maps = {}
            output_debug = []
            for node_id, node_output in outputs.items():
                # Determine prefix from the workflow definition
                prefix = ""
                if node_id in workflow:
                    node_def = workflow[node_id]
                    if node_def.get("class_type") == "SaveImage":
                        prefix = node_def.get("inputs", {}).get("filename_prefix", "")

                for img_info in node_output.get("images", []):
                    filename = img_info.get('filename', '')
                    output_debug.append({
                        "node": node_id,
                        "prefix": prefix,
                        "filename": filename,
                    })
                    from urllib.parse import urlencode
                    view_query = urlencode({
                        "filename": filename,
                        "subfolder": img_info.get('subfolder', ''),
                        "type": img_info.get('type', 'output'),
                    })
                    img_resp = requests.get(
                        f"{self.base_url}/view?{view_query}",
                        timeout=30,
                    )
                    img_resp.raise_for_status()
                    from ..utils.image_utils import load_image_bytes_to_numpy
                    np_img = load_image_bytes_to_numpy(img_resp.content)
                    images.append(np_img)

                    if output_mode == "by_prefix":
                        map_type = self._map_prefix_to_map_type(prefix)
                        if not map_type:
                            map_type = self._map_prefix_to_map_type(filename)
                        if map_type:
                            pbr_maps[map_type] = np_img

            log.debug("pbr_maps keys: %s", list(pbr_maps.keys()))
            log.debug("Total images fetched: %d", len(images))

            if not images and require_images:
                # 尝试从 history 中提取具体错误信息
                status_info = history.get(prompt_id, {}).get("status", {})
                error_msgs = []
                if status_info.get("status_str") == "error":
                    for msg in status_info.get("messages", []):
                        if isinstance(msg, list) and len(msg) >= 2:
                            error_msgs.append(str(msg[1]))
                if error_msgs:
                    detail = "; ".join(error_msgs[:3])
                    raise RuntimeError(f"ComfyUI 执行失败: {detail}")
                raise RuntimeError("ComfyUI 未返回任何图像，请检查 checkpoint 和 ControlNet 模型是否存在于 ComfyUI 中。")

            if self._progress_cb:
                self._progress_cb(1.0, "完成")

            return GenerationResult(
                images=images,
                seed=-1,
                info={},
                metadata={
                    "backend": "comfyui",
                    "timestamp": time.time(),
                    "prompt_id": prompt_id,
                    "outputs": output_debug,
                },
                pbr_maps=pbr_maps,
            )
        finally:
            if ws is not None:
                ws.close()


def register():
    pass


def unregister():
    pass
