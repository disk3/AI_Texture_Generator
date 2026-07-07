import threading
import bpy
import os
import time
import numpy as np

from ..utils.async_bridge import run_on_main_thread, thread_safe_callback
from ..utils.logger import get_logger
from ..utils.image_utils import (
    blender_image_to_numpy,
    numpy_to_blender_image,
    resize_numpy_image,
    resize_numpy_image_keep_aspect,
    save_numpy_image,
)
from ..sd_backend.connection_pool import ConnectionPool
from ..sd_backend.abstract_client import GenerationConfig
from ..ui import preview_manager

from ..material.uv_extractor import UVExtractor
from ..material.shader_builder import ShaderBuilder
from ..material.pbr_processor import generate_normal_map, generate_roughness_map, generate_metallic_map
from ..material.local_pbr_processor import (
    compute_seam_metrics,
    generate_height_from_normal,
    generate_normal_from_diffuse,
    make_seamless_tile_local,
    renormalize_normal_map,
)
from ..properties import _build_preservation_prompt
from ..preferences import get_selected_api_provider_snapshot, resolve_asset_output_path
from ..sd_backend import comfyui_installer
from ..sd_backend.comfyui_client import ComfyUIClient
from ..sd_backend.comfyui_env_resolver import inject_comfyui_packages

log = get_logger(__name__)


def _sanitize_filename(name: str, max_len: int = 64) -> str:
    """将字符串转换为安全的文件/文件夹名称。

    移除 Windows / 多数文件系统不支持的字符，并截断长度。
    """
    import re
    # Windows 非法字符
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    # 控制字符
    name = re.sub(r'[\x00-\x1f]', '', name)
    # 首尾空格/句点
    name = name.strip(' .')
    # 避免 Windows 保留名
    reserved = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
    }
    if name.upper() in reserved:
        name = f"_{name}"
    if not name:
        name = "unnamed"
    if len(name) > max_len:
        name = name[:max_len].rsplit('_', 1)[0]
    return name


class GenerationOrchestrator:
    def __init__(self):
        self._pool = ConnectionPool()
        self._thread = None
        self._cancel_event = None
        self._texture_result = {}
        self._generation_id = 0
        self._active_client = None

    @staticmethod
    def _check_client_health(client, backend_type: str, auto_launch_path: str = "") -> bool:
        if backend_type == 'COMFYUI':
            return client.check_health(auto_launch_path=auto_launch_path)
        return client.check_health()

    @staticmethod
    def _is_comfyui_url_reachable(url: str) -> bool:
        """快速探测 ComfyUI URL 是否可达。"""
        if not url:
            return False
        try:
            try:
                import requests
            except ImportError:
                from ..utils import simple_requests as requests
            r = requests.get(url.rstrip('/') + '/system_stats', timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def start_generation(self, context, cancel_event: threading.Event):
        self._cancel_event = cancel_event
        self._generation_id += 1
        generation_id = self._generation_id

        addon_name = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_name].preferences
        scene = context.scene
        props = context.scene.ai_concept_props
        texture_generator = getattr(props, "texture_generator", "LOCAL_COMFYUI")

        # 选择 Local ComfyUI 但未安装时，弹出安装确认对话框。
        # 例外 1：参考图模式 + 无用户 prompt + 启用本地 PBR 时，可直接处理原图，无需 ComfyUI。
        # 例外 2：用户已填写可访问的 ComfyUI URL（桌面版/手动启动）时也跳过安装提示。
        if texture_generator == 'LOCAL_COMFYUI':
            install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()
            installed = comfyui_installer.is_comfyui_installed(install_path)
            url_reachable = self._is_comfyui_url_reachable(prefs.comfyui_url)
            if not installed and not url_reachable:
                has_reference = getattr(props, 'reference_image', None) is not None
                has_prompt = bool(props.prompt.strip())
                use_local_pbr = getattr(props, "use_local_pbr", False)
                if not (has_reference and not has_prompt and use_local_pbr):
                    props.is_generating = False
                    bpy.ops.ai_concept.install_comfyui('INVOKE_DEFAULT')
                    return

        # ---- 主线程中安全执行所有 bpy 操作 ----
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            thread_safe_callback({
                "status": "error",
                "message": "请选择一个网格对象用于贴图生成",
            })
            props.is_generating = False
            return

        thread_safe_callback({
            "status": "progress",
            "progress": 0.05,
            "message": "正在验证 UV...",
        })

        extractor = UVExtractor(resolution=int(props.width))
        validation = extractor.validate_uv(obj)

        has_uv_layer = obj.data.uv_layers.active is not None
        if not has_uv_layer:
            thread_safe_callback({
                "status": "error",
                "message": "未找到 UV 层，请先展开 UV。",
            })
            props.is_generating = False
            return

        if validation.error_messages:
            thread_safe_callback({
                "status": "progress",
                "progress": 0.05,
                "message": f"UV 警告: {'; '.join(validation.error_messages)}",
            })

        thread_safe_callback({
            "status": "progress",
            "progress": 0.1,
            "message": "正在渲染 UV 布局...",
        })

        try:
            uv_layout = extractor.render_uv_layout(obj, image_size=int(props.width))
        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": f"UV 布局渲染失败: {e}",
            })
            props.is_generating = False
            return

        # Prompt 由 Material Config 在 UI 中预填充；用户可手动编辑。
        prompt = props.prompt.strip()
        has_reference = getattr(props, 'reference_image', None) is not None
        has_user_prompt = bool(prompt)

        # 有参考图时允许 prompt 为空（表示直接按原图生成）；无参考图时必须填写 prompt
        if not prompt and not has_reference:
            thread_safe_callback({
                "status": "error",
                "message": "Prompt 为空，请先输入提示词或调整 Material Config 自动生成",
            })
            props.is_generating = False
            return

        if has_reference and has_user_prompt:
            # 参考图 + 用户写了 prompt：追加保留前缀，以参考图为主
            preservation = _build_preservation_prompt(props)
            if preservation and not prompt.startswith(preservation):
                prompt = f"{preservation}\n\n{prompt}"

        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=props.negative_prompt,
            width=int(props.width),
            height=int(props.height),
            seed=props.seed,
            sampler="DPM++ 2M Karras",
            batch_size=props.batch_size,
            reference_tile_strength=getattr(props, "reference_tile_strength", 0.5),
            use_chord_enhanced=True,
        )

        reference_numpy = None

        # 参考图处理：转换为 numpy 数组，本地 PBR 路径直接使用；
        # ComfyUI/API 路径在 worker 中再转成 PIL Image。
        if props.reference_image is not None:
            ref_img = props.reference_image
            width, height = ref_img.size
            if width <= 0 or height <= 0:
                thread_safe_callback({
                    "status": "error",
                    "message": f"参考图尺寸无效 ({width}x{height})，请重新选择一张有效的图片。",
                })
                props.is_generating = False
                return
            if not ref_img.pixels or len(ref_img.pixels) < width * height * 4:
                thread_safe_callback({
                    "status": "error",
                    "message": "参考图像素数据未加载或为空，请检查图片是否已加载。",
                })
                props.is_generating = False
                return
            try:
                ref_np = blender_image_to_numpy(ref_img)
            except Exception as e:
                thread_safe_callback({
                    "status": "error",
                    "message": f"参考图转换失败: {e}",
                })
                props.is_generating = False
                return
            target_w = int(props.width)
            target_h = int(props.height)
            if target_w <= 0 or target_h <= 0:
                thread_safe_callback({
                    "status": "error",
                    "message": f"目标尺寸无效 ({target_w}x{target_h})，请在面板中选择有效的宽度/高度。",
                })
                props.is_generating = False
                return
            if has_user_prompt:
                # img2img / 有提示词：按目标尺寸拉伸，便于后端直接处理
                if ref_np.shape[:2] != (target_h, target_w):
                    ref_np = resize_numpy_image(ref_np, target_w, target_h)
            else:
                # 空提示词：先 fit 到目标尺寸，再做无缝化（参考 v1.0.0 的处理顺序）
                if ref_np.shape[:2] != (target_h, target_w):
                    ref_np = resize_numpy_image_keep_aspect(ref_np, target_w, target_h, mode="cover")
            reference_numpy = ref_np
            config.denoising_strength = props.reference_denoise

        # 解析 asset_output_path 为绝对路径
        asset_output_path = resolve_asset_output_path(prefs)
        if not asset_output_path:
            thread_safe_callback({
                "status": "error",
                "message": "请先在偏好设置 > 输出中设置资源输出目录",
            })
            props.is_generating = False
            return

        api_provider = get_selected_api_provider_snapshot(
            context,
            texture_generator,
            getattr(props, "api_image_model", "DEFAULT"),
            "DEFAULT",
        )

        if api_provider:
            selected_backend = 'NANOBANANA' if api_provider.get("protocol") == 'GEMINI' else 'GPT_IMAGE'
            selected_url = api_provider.get("base_url", "api")
        else:
            selected_backend = 'COMFYUI'
            selected_url = prefs.comfyui_url

        # 构建轻量数据包传给后台线程
        data = {
            "mode": 'TEXTURE',
            "config": config,
            "backend_type": selected_backend,
            "url": selected_url,
            "asset_output_path": asset_output_path,
            "comfyui_url": prefs.comfyui_url,
            "comfyui_path": prefs.comfyui_path,
            "api_provider": api_provider,
            "uv_layout": uv_layout,
            "scene_name": scene.name,
            "active_object_name": obj.name,
            "generation_id": generation_id,
            "texture_category": props.texture_category,
            "texture_finish": props.texture_finish,
            "texture_generator": texture_generator,
            "has_user_prompt": has_user_prompt,
            "output_basecolor": props.output_basecolor,
            "output_normal": props.output_normal,
            "output_roughness": props.output_roughness,
            "output_metalness": props.output_metalness,
            "output_height": props.output_height,
        }

        # 将用户本地 PBR 开关传递给后台线程
        data["use_local_pbr"] = getattr(props, "use_local_pbr", False)
        data["normal_strength"] = getattr(props, "normal_strength", 1.5)
        data["normal_detail"] = getattr(props, "normal_detail", 0.4)
        data["normal_invert"] = getattr(props, "normal_invert", False)
        data["local_comfyui_model"] = getattr(props, "local_comfyui_model", "DEFAULT")
        if reference_numpy is not None:
            data["reference_numpy"] = reference_numpy

        self._thread = threading.Thread(
            target=self._generate_worker,
            args=(data,),
            daemon=True,
        )
        self._thread.start()

    def stop_generation(self):
        if self._cancel_event:
            self._cancel_event.set()
        client = self._active_client
        if client and hasattr(client, "interrupt"):
            try:
                client.interrupt()
            except Exception:
                log.debug("Backend interrupt failed")

    def _is_cancelled(self, data: dict) -> bool:
        if self._cancel_event and self._cancel_event.is_set():
            return True
        return data.get("generation_id") != self._generation_id

    def _generate_worker(self, data):
        try:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return

            backend_type = data["backend_type"]
            url = data["url"]
            base_config = data["config"]

            # 参考图模式 + 无用户 prompt：直接本地处理参考图，
            # 无需初始化任何后端客户端，也避免在 LOCAL_COMFYUI 模式下自动启动 ComfyUI。
            is_reference_mode = (
                data.get("reference_numpy") is not None
                and not data.get("has_user_prompt", False)
            )
            if is_reference_mode:
                thread_safe_callback({
                    "status": "progress",
                    "progress": 0.1,
                    "message": "正在本地处理参考图...",
                })
                self._process_reference(data, use_chord=False)
                return

            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return

            thread_safe_callback({
                "status": "progress",
                "progress": 0.05,
                "message": "正在初始化后端客户端...",
            })

            texture_generator = data.get("texture_generator", "LOCAL_COMFYUI")

            if texture_generator == 'LOCAL_COMFYUI':
                url = data.get("comfyui_url", url)
                comfyui_path = data.get("comfyui_path", "") or comfyui_installer.get_default_install_path()
                auto_launch_path = comfyui_path if comfyui_path and os.path.isdir(comfyui_path) else ""

                # 尝试复用 ComfyUI 便携版中的 Pillow/requests/websocket
                if comfyui_path:
                    try:
                        inject_comfyui_packages(comfyui_path)
                    except Exception as e:
                        log.debug("ComfyUI package injection failed: %s", e)

                client = self._pool.get_client('COMFYUI', url)
                self._active_client = client
                if not self._check_client_health(client, 'COMFYUI', auto_launch_path=auto_launch_path):
                    thread_safe_callback({
                        "status": "error",
                        "message": f"无法连接到 ComfyUI ({url})。请手动启动 ComfyUI 或在偏好设置中配置正确的安装路径以自动启动。",
                    })
                    return
                self._generate_texture_worker(data, client)
                return

            elif backend_type in {'GPT_IMAGE', 'NANOBANANA'}:
                api_provider = data.get("api_provider", {})
                if backend_type == 'GPT_IMAGE':
                    client = self._pool.get_client(
                        'GPT_IMAGE', 'api',
                        api_key=api_provider.get("api_key", ""),
                        model=api_provider.get("image_model", "gpt-image-2"),
                        base_url=api_provider.get("base_url", ""),
                    )
                else:
                    client = self._pool.get_client(
                        'NANOBANANA', 'api',
                        api_key=api_provider.get("api_key", ""),
                        base_url=api_provider.get("base_url", ""),
                        model=api_provider.get("image_model", "gemini-2.5-flash-image"),
                    )
                self._active_client = client
                if not client.check_health():
                    thread_safe_callback({
                        "status": "error",
                        "message": f"API 客户端 {api_provider.get('name', backend_type)} 未就绪，请检查偏好设置中的 API Key。",
                    })
                    return
                self._generate_texture_api_worker(data, client)
                return

            else:
                thread_safe_callback({
                    "status": "error",
                    "message": f"未知后端: {backend_type}",
                })
                return

        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": str(e),
            })
        finally:
            self._active_client = None

    @staticmethod
    def _make_seamless_tile(image: np.ndarray, method: str = "SMART") -> np.ndarray:
        """用本地算法把图片处理成可平铺的无缝贴图。"""
        return make_seamless_tile_local(
            image,
            method=method,
            overlap=0.12,
            structure_radius=0.02,
            levels=5,
        )

    @staticmethod
    def _postprocess_comfyui_seams(textures: dict) -> dict:
        """对 ComfyUI/CHORD 返回的完整贴图集做本地无缝修复。

        对每个 PBR 通道分别调用本地无缝化算法；normal 使用向量域归一化。
        """
        supported = {'diffuse', 'normal', 'roughness', 'metallic', 'height', 'ao'}
        processed = {}
        for map_type, img in textures.items():
            if map_type not in supported or img is None:
                processed[map_type] = img
                continue
            out = make_seamless_tile_local(
                img,
                method="SMART",
                overlap=0.12,
                is_normal=(map_type == "normal"),
            )
            if map_type == 'normal' and out.shape[-1] == 3:
                out = renormalize_normal_map(out)
            processed[map_type] = out
        return processed

    def _process_reference(self, data, use_chord: bool = True):
        """有参考图且 prompt 为空时：直接处理参考图为无缝 diffuse，再提取 PBR。

        处理顺序参考 v1.0.0：先 fit 到目标尺寸，再做无缝化，避免在无缝化后再裁剪
        导致图案错位。
        """
        diffuse = data.get("reference_numpy")
        if diffuse is None:
            raise RuntimeError("参考图模式需要 reference_numpy")

        base_config = data["config"]
        target_w = int(base_config.width)
        target_h = int(base_config.height)

        thread_safe_callback({
            "status": "progress",
            "progress": 0.2,
            "message": "正在将参考图适配到目标尺寸...",
        })
        # 先 fit 到目标尺寸（等比裁剪，与 v1.0.0 的 ImageOps.fit 行为一致）
        if diffuse.shape[:2] != (target_h, target_w):
            diffuse = resize_numpy_image_keep_aspect(diffuse, target_w, target_h, mode="cover")

        thread_safe_callback({
            "status": "progress",
            "progress": 0.25,
            "message": "正在对参考图做四方连续处理...",
        })
        # 使用 PPS 无缝化：不偏移、不重叠原图内容，避免带花纹图像边缘出现叠影
        diffuse = self._make_seamless_tile(diffuse, method="PPS")

        self._extract_pbr(diffuse, data, use_chord=False, already_seamless=True)

    @staticmethod
    def _is_comfyui_installed_for_data(data: dict) -> bool:
        """根据 data 中的 comfyui_path 判断本地 ComfyUI 是否已安装。"""
        path = data.get("comfyui_path", "") or comfyui_installer.get_default_install_path()
        return comfyui_installer.is_comfyui_installed(path)

    def _should_use_chord(self, data: dict, requested_use_chord: bool) -> bool:
        """决定是否真正调用 CHORD。

        以下情况跳过 CHORD：
        1. 用户手动开启"不用 ComfyUI，本地生成 PBR"；
        2. 本地 ComfyUI 未安装（修复 API 路径仍尝试 CHORD 的问题）。
        """
        if data.get("use_local_pbr", False):
            return False
        if not self._is_comfyui_installed_for_data(data):
            return False
        return requested_use_chord

    def _extract_pbr(self, diffuse: np.ndarray, data, use_chord: bool = True, existing_textures: dict = None, already_seamless: bool = False):
        """从 diffuse（numpy uint8 RGB/RGBA）提取 PBR：优先使用 CHORD，不可用时使用本地算法 fallback。

        Args:
            existing_textures: 调用方已有的贴图（如 CHORD 返回的部分贴图），会优先保留。
            already_seamless: 为 True 时跳过本地模式下的二次无缝化（参考图模式已处理）。
        """
        target_w = data["config"].width
        target_h = data["config"].height
        if diffuse.shape[:2] != (target_h, target_w):
            diffuse = resize_numpy_image(diffuse, target_w, target_h)

        if self._is_cancelled(data):
            thread_safe_callback({"status": "cancelled"})
            return

        using_local_pbr = data.get("use_local_pbr", False) or not self._is_comfyui_installed_for_data(data)
        if using_local_pbr and not already_seamless:
            diffuse = self._make_seamless_tile(diffuse)

        textures = existing_textures.copy() if existing_textures else {}
        textures['diffuse'] = diffuse

        effective_use_chord = self._should_use_chord(data, use_chord)

        if effective_use_chord:
            chord_success = False
            try:
                comfyui_path = data.get("comfyui_path", "") or comfyui_installer.get_default_install_path()
                auto_launch_path = comfyui_path if comfyui_path and os.path.isdir(comfyui_path) else ""
                url = data.get("comfyui_url", "http://127.0.0.1:8188")

                # 尝试复用 ComfyUI 便携版中的 Pillow/requests/websocket
                if comfyui_path:
                    try:
                        inject_comfyui_packages(comfyui_path)
                    except Exception as e:
                        log.debug("ComfyUI package injection failed: %s", e)

                cclient = ComfyUIClient(base_url=url)
                if cclient.check_health(auto_launch_path=auto_launch_path):
                    thread_safe_callback({
                        "status": "progress",
                        "progress": 0.4,
                        "message": "正在运行 CHORD 材质估算...",
                    })
                    # CHORD 工作流现在支持 numpy 数组作为输入
                    chord_result = cclient.execute_chord_workflow(
                        diffuse,
                        width=target_w,
                        height=target_h,
                    )
                    pbr_maps = chord_result.pbr_maps
                    if pbr_maps:
                        for map_type, img in pbr_maps.items():
                            np_img = img if isinstance(img, np.ndarray) else np.array(img)
                            if np_img.shape[:2] != (target_h, target_w):
                                np_img = resize_numpy_image(np_img, target_w, target_h)
                            pbr_maps[map_type] = np_img
                        if "basecolor" in pbr_maps:
                            textures['diffuse'] = pbr_maps["basecolor"]
                        if "normal" in pbr_maps:
                            textures['normal'] = pbr_maps["normal"]
                        if "roughness" in pbr_maps:
                            textures['roughness'] = pbr_maps["roughness"]
                        if "height" in pbr_maps:
                            textures['height'] = pbr_maps["height"]
                        if "metalness" in pbr_maps:
                            textures['metallic'] = pbr_maps["metalness"]
                        chord_success = True
                        thread_safe_callback({
                            "status": "progress",
                            "progress": 0.7,
                            "message": f"CHORD 返回了 {len(pbr_maps)} 张贴图",
                        })
            except Exception as e:
                log.debug("CHORD failed: %s", e)

            if not chord_success:
                thread_safe_callback({
                    "status": "progress",
                    "progress": 0.4,
                    "message": "CHORD 失败，回退到本地算法提取...",
                })
        else:
            thread_safe_callback({
                "status": "progress",
                "progress": 0.4,
                "message": "ComfyUI 未安装或已启用本地 PBR，使用本地算法提取...",
            })

        # 算法填充未提取到的贴图
        if 'normal' not in textures:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return
            thread_safe_callback({
                "status": "progress",
                "progress": 0.55,
                "message": "正在提取法线贴图...",
            })
            if using_local_pbr:
                textures['normal'] = generate_normal_from_diffuse(
                    diffuse,
                    strength=data.get("normal_strength", 1.5),
                    detail=data.get("normal_detail", 0.4),
                    invert=data.get("normal_invert", False),
                )
            else:
                textures['normal'] = generate_normal_map(diffuse, strength=0.8)
        if 'normal' in textures:
            textures['normal'] = renormalize_normal_map(textures['normal'])

        if 'roughness' not in textures:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return
            thread_safe_callback({
                "status": "progress",
                "progress": 0.7,
                "message": "正在提取粗糙度贴图...",
            })
            textures['roughness'] = generate_roughness_map(
                diffuse,
                category=data.get("texture_category", ""),
                finish=data.get("texture_finish", ""),
                contrast=1.0,
            )
        if 'metallic' not in textures:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return
            thread_safe_callback({
                "status": "progress",
                "progress": 0.85,
                "message": "正在提取金属度贴图...",
            })
            textures['metallic'] = generate_metallic_map(
                diffuse,
                category=data.get("texture_category", ""),
                finish=data.get("texture_finish", ""),
                threshold=220,
            )

        # 本地 PBR 模式下，从生成的法线反推高度图
        if using_local_pbr and 'height' not in textures and 'normal' in textures:
            thread_safe_callback({
                "status": "progress",
                "progress": 0.88,
                "message": "正在从法线生成高度图...",
            })
            textures['height'] = generate_height_from_normal(textures['normal'], flip_green=False)

        if using_local_pbr:
            failed = []
            for map_type, img in textures.items():
                metrics = compute_seam_metrics(img, map_type=map_type)
                if not metrics.get("ok", True):
                    failed.append(f"{map_type}: mean={metrics.get('mean_delta', 0):.1f}, max={metrics.get('max_delta', 0):.1f}")
            if failed:
                thread_safe_callback({
                    "status": "progress",
                    "progress": 0.89,
                    "message": "无缝检测提示: " + "; ".join(failed[:3]),
                })
        elif not already_seamless:
            # LOCAL_COMFYUI / API 非本地 PBR 模式：CHORD 完成后统一在 Blender 里做无缝
            textures = self._postprocess_comfyui_seams(textures)

        textures_for_import = {k: v.copy() for k, v in textures.items()}
        thread_safe_callback({
            "status": "progress",
            "progress": 0.9,
            "message": "正在应用 PBR 材质...",
        })
        run_on_main_thread(lambda: self._import_texture_results(data, textures_for_import), timeout=300.0)

    def _generate_texture_worker(self, data, client):
        """后台线程生成 PBR 贴图：ComfyUI Zimage + CHORD 工作流提取 PBR。"""
        try:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return

            base_config = data["config"]
            use_chord = True  # 进入此函数说明 ComfyUI 健康检查已通过

            # 有参考图 + prompt 为空：直接本地处理参考图
            if data.get("reference_numpy") is not None and not data.get("has_user_prompt", False):
                self._process_reference(data, use_chord=use_chord)
                return

            # 有参考图 + 有 prompt：直接传 numpy 数组给后端，由后端决定编码方式
            if data.get("reference_numpy") is not None:
                base_config.init_image = data["reference_numpy"][..., :3]

            def _comfy_progress_cb(progress: float, message: str):
                # ComfyUI 客户端进度范围约 0.2~0.9，映射到整体 0.1~0.85
                overall = 0.1 + min(max(progress, 0.0), 1.0) * 0.75
                thread_safe_callback({
                    "status": "progress",
                    "progress": overall,
                    "message": message or "正在 ComfyUI 中运行 Zimage+CHORD 工作流...",
                })

            client.set_progress_callback(_comfy_progress_cb)

            thread_safe_callback({
                "status": "progress",
                "progress": 0.1,
                "message": "正在 ComfyUI 中运行 Zimage+CHORD 工作流...",
            })

            # 注入用户选择的本地 ComfyUI 主模型（Z-Image Turbo / UNETLoader node 1）
            selected_model = data.get("local_comfyui_model", "DEFAULT")
            if selected_model and selected_model not in ("DEFAULT", "__EMPTY__"):
                base_config.model_overrides["1"] = {"unet_name": selected_model}
                log.debug("Override main UNET model to %s", selected_model)

            result = client.execute_workflow_json(base_config)
            client.set_progress_callback(None)
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return

            pbr_maps = result.pbr_maps
            log.debug("result.pbr_maps keys: %s", list(pbr_maps.keys()))

            if not pbr_maps:
                outputs = result.metadata.get("outputs", []) if result.metadata else []
                log.debug("No named PBR maps. ComfyUI outputs: %s", outputs)
                if result.images:
                    pbr_maps = {"diffuse": result.images[0]}
                    thread_safe_callback({
                        "status": "progress",
                        "progress": 0.45,
                        "message": "无法识别 CHORD 贴图命名，将使用返回图像并提取 PBR 贴图。",
                    })
                else:
                    raise RuntimeError(
                        "ComfyUI workflow returned no usable images. "
                        "Open the latest ComfyUI history entry and check whether SaveImage nodes for basecolor/normal/roughness/metalness/height executed."
                    )

            def _pil_to_np(img):
                """将 ComfyUI 返回的 PIL Image 转为 numpy uint8。"""
                if isinstance(img, np.ndarray):
                    return img
                return np.array(img)

            diffuse = None
            textures = {}
            if "basecolor" in pbr_maps:
                textures['diffuse'] = _pil_to_np(pbr_maps["basecolor"])
            elif "diffuse" in pbr_maps:
                textures['diffuse'] = _pil_to_np(pbr_maps["diffuse"])
            if "normal" in pbr_maps:
                textures['normal'] = _pil_to_np(pbr_maps["normal"])
            if "roughness" in pbr_maps:
                textures['roughness'] = _pil_to_np(pbr_maps["roughness"])
            if "height" in pbr_maps:
                textures['height'] = _pil_to_np(pbr_maps["height"])
            if "metalness" in pbr_maps:
                textures['metallic'] = _pil_to_np(pbr_maps["metalness"])

            if 'diffuse' not in textures and result.images:
                textures['diffuse'] = _pil_to_np(result.images[0])

            diffuse = textures.get('diffuse')
            if diffuse is None:
                raise RuntimeError("ComfyUI 工作流未返回 diffuse 贴图。")

            chord_maps = [k for k in ['diffuse', 'normal', 'roughness', 'height', 'metallic'] if k in textures]
            thread_safe_callback({
                "status": "progress",
                "progress": 0.5,
                "message": f"贴图就绪: {', '.join(chord_maps)}",
            })

            # 如果 CHORD 没返回完整 PBR，用算法补齐
            if 'normal' not in textures or 'roughness' not in textures or 'metallic' not in textures:
                self._extract_pbr(diffuse, data, use_chord=False, existing_textures=textures)
            else:
                # CHORD 完成后统一在 Blender 里做无缝修复
                textures = self._postprocess_comfyui_seams(textures)
                textures_for_import = {k: v.copy() for k, v in textures.items()}
                thread_safe_callback({
                    "status": "progress",
                    "progress": 0.9,
                    "message": "正在应用 PBR 材质...",
                })
                run_on_main_thread(lambda: self._import_texture_results(data, textures_for_import), timeout=300.0)

        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": str(e),
            })

    def _generate_texture_api_worker(self, data, client):
        """使用 GPT Image-2 / Nanobanana API 生成 diffuse，可选 CHORD 增强。"""
        try:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return

            base_config = data["config"]

            # 有参考图 + prompt 为空：直接本地处理参考图
            if data.get("reference_numpy") is not None and not data.get("has_user_prompt", False):
                self._process_reference(data, use_chord=self._is_comfyui_installed_for_data(data))
                return

            # 有参考图 + 有 prompt：直接传 numpy 数组给 API 客户端
            if data.get("reference_numpy") is not None:
                base_config.init_image = data["reference_numpy"][..., :3]

            thread_safe_callback({
                "status": "progress",
                "progress": 0.1,
                "message": f"正在通过 {data.get('texture_generator', 'API')} 生成贴图...",
            })

            # API 生成：txt2img 或 img2img
            if base_config.init_image is not None:
                result = client.img2img(base_config)
            else:
                result = client.txt2img(base_config)

            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return

            if not result.images:
                raise RuntimeError("API 未返回任何图像。")

            for idx, diffuse in enumerate(result.images):
                if self._is_cancelled(data):
                    thread_safe_callback({"status": "cancelled"})
                    return
                item_data = dict(data)
                item_data["batch_index"] = idx
                item_data["batch_total"] = len(result.images)
                if not isinstance(diffuse, np.ndarray):
                    diffuse = np.array(diffuse)
                self._extract_pbr(diffuse, item_data, use_chord=self._is_comfyui_installed_for_data(data))

        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": str(e),
            })

    def _import_texture_results(self, data, textures=None):
        """在主线程中创建 Blender Image 和 PBR 材质。"""
        try:
            if self._is_cancelled(data):
                thread_safe_callback({"status": "cancelled"})
                return None

            scene_name = data.get("scene_name", "")
            scene = bpy.data.scenes.get(scene_name) if scene_name else bpy.context.scene
            if scene is None:
                thread_safe_callback({
                    "status": "error",
                    "message": f"生成开始时的场景已不存在: {scene_name}",
                })
                return None
            props = scene.ai_concept_props
            obj = bpy.data.objects.get(data.get("active_object_name", ""))
            if not obj or obj.type != 'MESH':
                thread_safe_callback({
                    "status": "error",
                    "message": "找不到目标网格对象",
                })
                return

            textures = textures or self._texture_result
            if not textures:
                thread_safe_callback({
                    "status": "error",
                    "message": "未生成任何贴图",
                })
                return

            # 根据 Maps 面板选择过滤输出贴图
            output_config = {
                'diffuse': data.get("output_basecolor", True),
                'normal': data.get("output_normal", True),
                'roughness': data.get("output_roughness", True),
                'metallic': data.get("output_metalness", True),
                'height': data.get("output_height", True),
            }
            if not any(output_config.values()):
                output_config = {k: True for k in output_config}

            filtered_textures = {}
            for map_type, np_img in textures.items():
                if map_type == 'packed':
                    # packed 只有在需要 roughness 或 metallic 通道时才输出
                    if output_config.get('roughness', True) or output_config.get('metallic', True):
                        filtered_textures[map_type] = np_img
                elif output_config.get(map_type, True):
                    filtered_textures[map_type] = np_img

            if not filtered_textures and 'diffuse' in textures:
                filtered_textures['diffuse'] = textures['diffuse']

            textures = filtered_textures

            # 将贴图缩放到用户选择的输出尺寸
            target_w = int(data["config"].width)
            target_h = int(data["config"].height)
            for k in textures:
                if textures[k].shape[:2] != (target_h, target_w):
                    textures[k] = resize_numpy_image(textures[k], target_w, target_h)

            output_dir = data.get("asset_output_path", "")
            if not output_dir or not os.path.isdir(output_dir):
                thread_safe_callback({
                    "status": "error",
                    "message": "资源输出目录无效，请在偏好设置 > 输出中设置有效的目录",
                })
                return

            images = {}
            batch_id = time.strftime("%Y%m%d_%H%M%S")
            if data.get("batch_total", 1) > 1:
                batch_id = f"{batch_id}_b{int(data.get('batch_index', 0)) + 1:02d}"
            safe_obj_name = _sanitize_filename(obj.name)

            for map_type, np_img in textures.items():
                img_name = f"{safe_obj_name}_{map_type}_{batch_id}"

                # 统一转换为 RGBA uint8
                if np_img.ndim == 2:
                    np_img = np.stack([np_img, np_img, np_img, np.ones_like(np_img) * 255], axis=-1)
                elif np_img.shape[-1] == 3:
                    alpha = np.ones((*np_img.shape[:2], 1), dtype=np.uint8) * 255
                    np_img = np.concatenate([np_img, alpha], axis=-1)
                elif np_img.shape[-1] != 4:
                    raise ValueError(f"Unsupported texture shape for {map_type}: {np_img.shape}")

                blender_img = numpy_to_blender_image(img_name, np_img)
                blender_img.pack()

                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{img_name}.png")
                blender_img.filepath_raw = save_path.replace('\\', '/')
                blender_img.file_format = 'PNG'
                try:
                    blender_img.save()
                except Exception as e:
                    log.warning("Blender image.save() failed (%s), falling back to PIL", e)
                    from PIL import Image
                    Image.fromarray(np_img, 'RGBA').save(save_path)

                images[map_type] = blender_img

            mat = ShaderBuilder.build_principled_bsdf(
                f"Mat_{safe_obj_name}_PBR",
                images,
                output_config=output_config,
            )

            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat

            # 记录到结果列表
            item = props.results.add()
            item.name = f"PBR {safe_obj_name} ({batch_id})"
            item.result_type = 'pbr'
            item.timestamp = batch_id

            for map_type, blender_img in images.items():
                map_item = item.pbr_maps.add()
                map_item.map_type = map_type
                map_item.image_name = blender_img.name

                save_path = os.path.join(output_dir, f"{blender_img.name}.png")
                preview_manager.load_preview(blender_img.name, save_path)

            props.active_result_index = len(props.results) - 1

            thread_safe_callback({
                "status": "progress",
                "progress": 1.0,
                "message": f"完成 — PBR 材质已应用到 {obj.name}",
            })
            batch_total = int(data.get("batch_total", 1))
            batch_index = int(data.get("batch_index", 0))
            if batch_index >= batch_total - 1:
                thread_safe_callback({"status": "done"})

        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": str(e),
            })
        return None

    def _run_texture_generation(self, context, client):
        # Kept for compatibility; texture generation logic is now in worker
        pass


def register():
    pass


def unregister():
    pass
