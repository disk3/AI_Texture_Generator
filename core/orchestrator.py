import threading
import bpy
import os
import time
import numpy as np
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter

from ..utils.async_bridge import thread_safe_callback
from ..utils.logger import get_logger
from ..sd_backend.connection_pool import ConnectionPool
from ..sd_backend.abstract_client import GenerationConfig
from ..ui import preview_manager

from ..material.uv_extractor import UVExtractor
from ..material.shader_builder import ShaderBuilder
from ..material.pbr_processor import generate_normal_map, generate_roughness_map, generate_metallic_map
from ..properties import _build_preservation_prompt
from ..preferences import get_selected_api_provider_snapshot, resolve_asset_output_path
from ..sd_backend import comfyui_installer

log = get_logger(__name__)


class GenerationOrchestrator:
    def __init__(self):
        self._pool = ConnectionPool()
        self._thread = None
        self._cancel_event = None
        self._texture_result = {}

    @staticmethod
    def _check_client_health(client, backend_type: str, auto_launch_path: str = "") -> bool:
        if backend_type == 'COMFYUI':
            return client.check_health(auto_launch_path=auto_launch_path)
        return client.check_health()

    def start_generation(self, context, cancel_event: threading.Event):
        self._cancel_event = cancel_event

        addon_name = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_name].preferences
        props = context.scene.ai_concept_props
        texture_generator = getattr(props, "texture_generator", "LOCAL_COMFYUI")

        # 选择 Local ComfyUI 但未安装时，弹出安装确认对话框
        if texture_generator == 'LOCAL_COMFYUI':
            install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()
            if not comfyui_installer.is_comfyui_installed(install_path):
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

        # 参考图处理
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
            pixels = np.array(ref_img.pixels[:], dtype=np.float32)
            if pixels.size != width * height * 4:
                thread_safe_callback({
                    "status": "error",
                    "message": "参考图像素数据长度不匹配，请重新选择图片。",
                })
                props.is_generating = False
                return
            pixels = pixels.reshape((height, width, 4))
            pixels = (pixels * 255).astype(np.uint8)
            pil_ref = Image.fromarray(pixels, 'RGBA').convert('RGB')
            target_w = int(props.width)
            target_h = int(props.height)
            if target_w <= 0 or target_h <= 0:
                thread_safe_callback({
                    "status": "error",
                    "message": f"目标尺寸无效 ({target_w}x{target_h})，请在面板中选择有效的宽度/高度。",
                })
                props.is_generating = False
                return
            if pil_ref.size != (target_w, target_h):
                pil_ref = ImageOps.fit(pil_ref, (target_w, target_h), method=Image.LANCZOS)
            config.init_image = pil_ref
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
            "comfyui_controlnet_tile": prefs.comfyui_controlnet_tile_model,
            "comfyui_path": prefs.comfyui_path,
            "api_provider": api_provider,
            "uv_layout": uv_layout,
            "active_object_name": obj.name,
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

        self._thread = threading.Thread(
            target=self._generate_worker,
            args=(data,),
            daemon=True,
        )
        self._thread.start()

    def stop_generation(self):
        if self._cancel_event:
            self._cancel_event.set()

    def _generate_worker(self, data):
        try:
            backend_type = data["backend_type"]
            url = data["url"]

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
                client = self._pool.get_client(
                    'COMFYUI', url,
                    controlnet_tile=data.get("comfyui_controlnet_tile", ""),
                )
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

    @staticmethod
    def _make_seamless_tile(image: Image.Image) -> Image.Image:
        """用 offset + 羽化 mask 把图片处理成可平铺的无缝贴图。"""
        img = image.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return img

        # 50% 平移，让接缝移到中心
        offset = ImageChops.offset(img, w // 2, h // 2)

        # 中心羽化 mask：中心 255，边缘 0，过渡区用高斯模糊
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        feather = max(min(w, h) // 8, 8)
        draw.rectangle([feather, feather, w - feather, h - feather], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather // 2))

        # 在原始图和 offset 图之间按 mask 融合
        blended = Image.composite(img, offset, mask)
        return blended

    def _process_reference(self, data, use_chord: bool = True):
        """有参考图且 prompt 为空时：直接处理参考图为无缝 diffuse，再提取 PBR。

        use_chord 默认 True（CHORD），CHORD 失败时由 _extract_pbr fallback 到算法。
        """
        base_config = data["config"]
        diffuse = base_config.init_image.copy() if base_config.init_image else None
        if diffuse is None:
            raise RuntimeError("参考图模式需要 init_image")

        target_w = int(base_config.width)
        target_h = int(base_config.height)
        if diffuse.size != (target_w, target_h):
            diffuse = diffuse.resize((target_w, target_h), Image.LANCZOS)

        thread_safe_callback({
            "status": "progress",
            "progress": 0.2,
            "message": "正在处理参考图（缩放 + 无缝平铺）...",
        })
        diffuse = self._make_seamless_tile(diffuse)

        self._extract_pbr(diffuse, data, use_chord=use_chord)

    def _extract_pbr(self, diffuse: Image.Image, data, use_chord: bool = True):
        """从 diffuse 提取 PBR：默认始终使用 CHORD，CHORD 失败时算法提取。"""
        target_w = data["config"].width
        target_h = data["config"].height
        if diffuse.size != (target_w, target_h):
            diffuse = diffuse.resize((target_w, target_h), Image.LANCZOS)

        textures = {'diffuse': diffuse}

        if use_chord:
            chord_success = False
            try:
                comfyui_path = data.get("comfyui_path", "") or comfyui_installer.get_default_install_path()
                auto_launch_path = comfyui_path if comfyui_path and os.path.isdir(comfyui_path) else ""
                url = data.get("comfyui_url", "http://127.0.0.1:8188")
                from ..sd_backend.comfyui_client import ComfyUIClient
                cclient = ComfyUIClient(base_url=url)
                if cclient.check_health(auto_launch_path=auto_launch_path):
                    thread_safe_callback({
                        "status": "progress",
                        "progress": 0.4,
                        "message": "正在运行 CHORD 材质估算...",
                    })
                    chord_result = cclient.execute_chord_workflow(
                        diffuse,
                        width=target_w,
                        height=target_h,
                    )
                    pbr_maps = chord_result.pbr_maps
                    if pbr_maps:
                        for map_type, img in pbr_maps.items():
                            if img.size != (target_w, target_h):
                                pbr_maps[map_type] = img.resize((target_w, target_h), Image.LANCZOS)
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
                    "message": "CHORD 失败，回退到算法提取...",
                })

        # 算法填充未提取到的贴图
        if 'normal' not in textures:
            thread_safe_callback({
                "status": "progress",
                "progress": 0.55,
                "message": "正在提取法线贴图...",
            })
            textures['normal'] = generate_normal_map(diffuse, strength=0.8)
        if 'roughness' not in textures:
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

        self._texture_result = textures
        thread_safe_callback({
            "status": "progress",
            "progress": 0.9,
            "message": "正在应用 PBR 材质...",
        })
        bpy.app.timers.register(
            lambda: self._import_texture_results(data),
            first_interval=0.1,
        )

    def _generate_texture_worker(self, data, client):
        """后台线程生成 PBR 贴图：ComfyUI Zimage + CHORD 工作流提取 PBR。"""
        try:
            base_config = data["config"]
            use_chord = True  # 始终使用 CHORD

            # 有参考图 + prompt 为空：直接处理参考图
            if base_config.init_image is not None and not data.get("has_user_prompt", False):
                self._process_reference(data, use_chord=use_chord)
                return

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

            result = client.execute_workflow_json(base_config)
            client.set_progress_callback(None)
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

            diffuse = None
            textures = {}
            if "basecolor" in pbr_maps:
                textures['diffuse'] = pbr_maps["basecolor"]
            elif "diffuse" in pbr_maps:
                textures['diffuse'] = pbr_maps["diffuse"]
            if "normal" in pbr_maps:
                textures['normal'] = pbr_maps["normal"]
            if "roughness" in pbr_maps:
                textures['roughness'] = pbr_maps["roughness"]
            if "height" in pbr_maps:
                textures['height'] = pbr_maps["height"]
            if "metalness" in pbr_maps:
                textures['metallic'] = pbr_maps["metalness"]

            if 'diffuse' not in textures and result.images:
                textures['diffuse'] = result.images[0]

            diffuse = textures.get('diffuse')
            if not diffuse:
                raise RuntimeError("ComfyUI 工作流未返回 diffuse 贴图。")

            chord_maps = [k for k in ['diffuse', 'normal', 'roughness', 'height', 'metallic'] if k in textures]
            thread_safe_callback({
                "status": "progress",
                "progress": 0.5,
                "message": f"贴图就绪: {', '.join(chord_maps)}",
            })

            # 如果 CHORD 没返回完整 PBR，用算法补齐
            if 'normal' not in textures or 'roughness' not in textures or 'metallic' not in textures:
                self._extract_pbr(diffuse, data, use_chord=False)
            else:
                self._texture_result = textures
                thread_safe_callback({
                    "status": "progress",
                    "progress": 0.9,
                    "message": "正在应用 PBR 材质...",
                })
                bpy.app.timers.register(
                    lambda: self._import_texture_results(data),
                    first_interval=0.1,
                )

        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": str(e),
            })

    def _generate_texture_api_worker(self, data, client):
        """使用 GPT Image-2 / Nanobanana API 生成 diffuse，可选 CHORD 增强。"""
        try:
            base_config = data["config"]

            # 有参考图 + prompt 为空：直接处理参考图
            if base_config.init_image is not None and not data.get("has_user_prompt", False):
                self._process_reference(data, use_chord=True)
                return

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

            if not result.images:
                raise RuntimeError("API 未返回任何图像。")

            diffuse = result.images[0]
            self._extract_pbr(diffuse, data, use_chord=True)

        except Exception as e:
            thread_safe_callback({
                "status": "error",
                "message": str(e),
            })

    def _import_texture_results(self, data):
        """在主线程中创建 Blender Image 和 PBR 材质。"""
        try:
            props = bpy.context.scene.ai_concept_props
            obj = bpy.data.objects.get(data.get("active_object_name", ""))
            if not obj or obj.type != 'MESH':
                thread_safe_callback({
                    "status": "error",
                    "message": "找不到目标网格对象",
                })
                return

            textures = self._texture_result
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
            for map_type, pil_img in textures.items():
                if map_type == 'packed':
                    # packed 只有在需要 roughness 或 metallic 通道时才输出
                    if output_config.get('roughness', True) or output_config.get('metallic', True):
                        filtered_textures[map_type] = pil_img
                elif output_config.get(map_type, True):
                    filtered_textures[map_type] = pil_img

            if not filtered_textures and 'diffuse' in textures:
                filtered_textures['diffuse'] = textures['diffuse']

            textures = filtered_textures

            output_dir = data.get("asset_output_path", "")
            images = {}
            batch_id = time.strftime("%Y%m%d_%H%M%S")

            for map_type, pil_img in textures.items():
                img_name = f"{obj.name}_{map_type}_{batch_id}"
                blender_img = bpy.data.images.new(
                    name=img_name,
                    width=pil_img.size[0],
                    height=pil_img.size[1],
                )

                if pil_img.mode != 'RGBA':
                    pil_img = pil_img.convert('RGBA')
                resized = pil_img.resize((blender_img.size[0], blender_img.size[1]))
                pixels = list(resized.getdata())
                flat = [float(c) / 255.0 for px in pixels for c in px]
                blender_img.pixels.foreach_set(flat)
                blender_img.update()
                blender_img.pack()

                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{img_name}.png")
                pil_img.save(save_path)
                blender_img.filepath_raw = save_path.replace('\\', '/')
                blender_img.file_format = 'PNG'

                images[map_type] = blender_img

            mat = ShaderBuilder.build_principled_bsdf(
                f"Mat_{obj.name}_PBR",
                images,
                output_config=output_config,
            )

            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat

            # 记录到结果列表
            item = props.results.add()
            item.name = f"PBR {obj.name} ({batch_id})"
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
