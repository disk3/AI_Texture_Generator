import bpy
import threading
import os
import webbrowser
import ctypes
import subprocess

from .utils.async_bridge import get_orchestrator
from .ui import preview_manager
from .sd_backend import comfyui_installer
from .utils.logger import get_logger
from . import preferences as pref_utils

log = get_logger(__name__)


def _open_file_explorer(folder: str):
    """Cross-platform file explorer: open `folder` in the OS file manager."""
    folder = os.path.normpath(folder)
    if os.name == "nt":
        os.startfile(folder)
    elif os.name == "posix":
        try:
            subprocess.run(["open", folder], check=False)
        except Exception:
            try:
                subprocess.run(["xdg-open", folder], check=False)
            except Exception:
                log.warning("Could not open file explorer for %s", folder)
    else:
        log.warning("Unsupported OS: cannot open file explorer")


def _enum_process_windows():
    """枚举属于当前进程的所有顶层窗口句柄（仅 Windows）。"""
    if os.name != "nt":
        return []
    from ctypes import wintypes

    current_pid = os.getpid()
    handles = []

    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    GetWindowThreadProcessId = ctypes.windll.user32.GetWindowThreadProcessId

    def callback(hwnd, _):
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == current_pid:
            handles.append(hwnd)
        return True

    try:
        EnumWindows(EnumWindowsProc(callback), 0)
    except Exception:
        log.debug("EnumWindows failed (expected on non-Windows)")
    return handles


def _set_window_pos(hwnd, x, y, w, h):
    """使用 Windows API 调整窗口位置和大小。"""
    if os.name != "nt":
        return
    try:
        SWP_NOZORDER = 0x0004
        ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, SWP_NOZORDER)
    except Exception:
        log.debug("SetWindowPos failed (expected on non-Windows)")


class AI_OT_GenerateTexture(bpy.types.Operator):
    bl_idname = "ai_concept.generate"
    bl_label = "生成 PBR"
    bl_options = {'REGISTER', 'UNDO'}

    _thread = None
    _cancel_event = None

    @classmethod
    def poll(cls, context):
        return not context.scene.ai_concept_props.is_generating

    def execute(self, context):
        props = context.scene.ai_concept_props

        if props.is_generating:
            self.report({'WARNING'}, "正在生成中")
            return {'CANCELLED'}

        props.is_generating = True
        props.progress = 0.0
        props.status_message = "准备生成..."

        self._cancel_event = threading.Event()
        orch = get_orchestrator()
        orch.start_generation(context, self._cancel_event)

        return {'FINISHED'}

    def cancel(self, context):
        if self._cancel_event:
            self._cancel_event.set()
        context.scene.ai_concept_props.is_generating = False
        context.scene.ai_concept_props.status_message = "已取消"


class AI_OT_StopGeneration(bpy.types.Operator):
    bl_idname = "ai_concept.stop"
    bl_label = "停止"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        props = context.scene.ai_concept_props
        return props.is_generating

    def execute(self, context):
        props = context.scene.ai_concept_props
        props.is_generating = False
        props.status_message = "已取消"
        orch = get_orchestrator()
        orch.stop_generation()
        return {'FINISHED'}


class AI_OT_LoadReferenceImage(bpy.types.Operator):
    bl_idname = "ai_concept.load_reference_image"
    bl_label = "上传参考图"
    bl_description = "选择一张本地图片作为材质参考图"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp",
        options={'HIDDEN'}
    )

    def execute(self, context):
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "未选择有效文件")
            return {'CANCELLED'}

        # 加载图片到 Blender
        img_name = os.path.basename(self.filepath)
        # 避免命名冲突
        if img_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[img_name])

        image = bpy.data.images.load(self.filepath, check_existing=False)
        image.name = img_name
        image.update()

        # 使用和生成结果相同的 preview collection 机制加载预览，保证放大清晰度一致
        preview_manager.remove_preview(img_name)
        preview_manager.load_preview(img_name, self.filepath)

        # 设置到属性
        props = context.scene.ai_concept_props
        props.reference_image = image
        self.report({'INFO'}, f"已加载参考图: {img_name}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class AI_OT_PasteReferenceImage(bpy.types.Operator):
    bl_idname = "ai_concept.paste_reference_image"
    bl_label = "粘贴参考图"
    bl_description = "从系统剪贴板粘贴图像作为参考图 (Ctrl+V)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            from PIL import Image, ImageGrab
            pil_img = ImageGrab.grabclipboard()
            if pil_img is None or not isinstance(pil_img, Image.Image):
                self.report({'ERROR'}, "剪贴板中没有图像")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"无法读取剪贴板: {e}")
            return {'CANCELLED'}

        import time
        import tempfile

        # 颜色模式与扩展名
        if pil_img.mode in ('RGBA', 'P'):
            pil_img = pil_img.convert('RGBA')
            ext = '.png'
        else:
            pil_img = pil_img.convert('RGB')
            ext = '.jpg'

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_name = f"Ref_Pasted_{timestamp}"

        # 保存到系统临时目录用于预览（图片本身 pack 进 .blend，不会永久保留在磁盘）
        temp_dir = os.path.join(tempfile.gettempdir(), "BlenderAI")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{img_name}{ext}")
        pil_img.save(temp_path)

        # 加载到 Blender 并 pack（数据存入 .blend，不依赖外部文件）
        if img_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[img_name])

        image = bpy.data.images.load(temp_path, check_existing=False)
        image.name = img_name
        image.pack()
        image.update()

        # 注册预览（需要磁盘文件；预览读取后即可删除临时文件）
        preview_manager.remove_preview(img_name)
        preview_manager.load_preview(img_name, temp_path)

        props = context.scene.ai_concept_props
        props.reference_image = image
        self.report({'INFO'}, f"已从剪贴板粘贴参考图 ({pil_img.size[0]}x{pil_img.size[1]})")
        return {'FINISHED'}


class AI_OT_OptimizePrompt(bpy.types.Operator):
    """使用 LLM 将当前提示词优化为更详细的图像生成描述。"""
    bl_idname = "ai_concept.optimize_prompt"
    bl_label = "AI 优化提示词"
    bl_description = "使用 LLM 将当前提示词优化为更详细的图像生成描述"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ai_concept_props
        addon_name = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_name].preferences

        pref_utils.ensure_default_api_providers(prefs)
        provider_value = props.texture_generator
        provider_snapshot = pref_utils.get_selected_api_provider_snapshot(
            context,
            provider_value,
            getattr(props, "api_image_model", "DEFAULT"),
            "DEFAULT",
        )

        # 若当前选中的后端不是 API provider，或其 API Key 为空，
        # 自动回退到任意一个已配置 API Key 的 provider。
        if not provider_snapshot or not provider_snapshot.get("api_key"):
            provider_snapshot = pref_utils.find_any_configured_api_provider(prefs, context)

        if not provider_snapshot or not provider_snapshot.get("api_key"):
            provider_name = provider_snapshot.get("name") if provider_snapshot else "未选中"
            self.report(
                {'ERROR'},
                f"未找到可用 API Key (当前 provider: {provider_name})。"
                f"请确认 Preferences > AI Texture to PBR 中对应 provider 的 API Key 已填写并保存"
            )
            return {'CANCELLED'}

        api_key = provider_snapshot["api_key"]
        base_url = (provider_snapshot.get("base_url") or "").rstrip("/")
        model = provider_snapshot.get("text_model") or provider_snapshot.get("image_model")
        protocol = provider_snapshot.get("protocol", "OPENAI_COMPATIBLE")

        original = props.prompt.strip()
        if not original:
            self.report({'WARNING'}, "提示词为空，请输入描述后再优化")
            return {'CANCELLED'}

        system_prompt = (
            "你是一位专业的 3A 游戏与影视级 PBR 材质纹理提示词工程师。"
            "请将用户的简单描述扩展成一段高质量、细节极其丰富的材质纹理生成提示词。\n\n"
            "要求：\n"
            "1. 保留用户原意，不要改成完全不同的材质。\n"
            "2. 必须加入以下维度的具体细节描述：\n"
            "   - 材质类型与宏观特征（如混凝土、金属、石材、木材、织物、土壤、涂层、陶瓷等）；\n"
            "   - 整体色调、饱和度、冷暖倾向；\n"
            "   - 表面老化与风化程度（全新、轻微磨损、中度风化、严重腐蚀、陈旧古旧等）；\n"
            "   - 主要纹理图案（裂纹、划痕、凹坑、凸起、鳞片、纤维、编织、颗粒、层理、孔洞等）；\n"
            "   - 纹理密度、方向、规律性或随机性分布；\n"
            "   - 磨损区域、边缘破损、磕碰痕迹、使用痕迹；\n"
            "   - 污渍、油渍、水渍、锈迹、霉斑、氧化物、沉积物、尘埃覆盖；\n"
            "   - 表面粗糙度与触感（光滑、磨砂、颗粒感、绒毛感、油腻感、干涩感等）；\n"
            "   - 反光特性（哑光、半哑光、微光泽、镜面、各向异性、金属高光等）；\n"
            "   - PBR 技术规范：无缝平铺（seamless tiling）、UV 就绪、正投影俯视、平光照明、无阴影、无高光、无透视变形。\n"
            "3. 使用中文输出，输出一段连续自然的描述文本，不要加标题、列表、引号或 Markdown。\n"
            "4. 直接返回优化后的提示词，不要添加任何解释或额外内容。"
        )

        import requests
        try:
            if protocol == 'GEMINI':
                # SECURITY: Gemini API 要求通过 URL query param 传递 API Key。
                # Key 可能被中间代理 / CDN / 服务端日志记录。
                # 建议使用专属的低权限 Key，勿与高价值服务共用同一 Key。
                gemini_base = base_url if base_url.endswith("/v1beta") else f"{base_url}/v1beta"
                resp = requests.post(
                    f"{gemini_base}/models/{model}:generateContent?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [{"text": f"{system_prompt}\n\n用户描述：{original}"}]
                        }],
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates or not isinstance(candidates[0], dict):
                    raise ValueError("Gemini 返回空 candidates")
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                optimized = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            else:
                api_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
                resp = requests.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": original},
                        ],
                        "temperature": 0.7,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    raise ValueError("API 返回空 choices")
                message = choices[0].get("message") or {}
                optimized = message.get("content", "").strip()
            props.prompt = optimized
            self.report({'INFO'}, "提示词已优化")
        except requests.exceptions.ReadTimeout:
            self.report({'ERROR'}, f"优化失败: {provider_snapshot.get('name', 'API')} 响应超时。/models 测试通过只代表模型列表可访问；请检查 Default Text Model 是否支持 chat/completions，或稍后重试。")
        except requests.exceptions.HTTPError as e:
            self.report({'ERROR'}, f"API 错误: {e}")
        except Exception as e:
            self.report({'ERROR'}, f"优化失败: {e}")

        return {'FINISHED'}


class AI_OT_OpenImageFolder(bpy.types.Operator):
    """在文件浏览器中打开贴图所在文件夹。"""
    bl_idname = "ai_concept.open_image_folder"
    bl_label = "打开文件夹"
    bl_options = {'REGISTER', 'INTERNAL'}

    image_name: bpy.props.StringProperty()

    def execute(self, context):
        img = bpy.data.images.get(self.image_name)
        if not img:
            self.report({'WARNING'}, f"找不到图像 '{self.image_name}'")
            return {'CANCELLED'}

        filepath = img.filepath_raw
        if not filepath:
            self.report({'WARNING'}, "图像无文件路径")
            return {'CANCELLED'}

        # filepath_raw 已经是绝对路径（_import_texture_results 中设置）
        folder = os.path.dirname(filepath)
        if not os.path.isdir(folder):
            # fallback：处理相对路径或格式问题
            folder = os.path.dirname(bpy.path.abspath(filepath))
        if not os.path.isdir(folder):
            self.report({'WARNING'}, f"找不到文件夹: {folder}")
            return {'CANCELLED'}

        _open_file_explorer(folder)
        return {'FINISHED'}


class AI_OT_EnlargePreview(bpy.types.Operator):
    """打开一个独立的 Image Editor 窗口显示原图。"""
    bl_idname = "ai_concept.enlarge_preview"
    bl_label = "放大预览"
    bl_description = "点击在 Image Editor 中查看原图"
    bl_options = {'REGISTER', 'INTERNAL'}

    image_name: bpy.props.StringProperty()

    def execute(self, context):
        img = bpy.data.images.get(self.image_name)
        if not img:
            self.report({'ERROR'}, "找不到图像")
            return {'CANCELLED'}

        # 记录当前已有窗口指针，用于找出新窗口
        existing = {w.as_pointer() for w in context.window_manager.windows}

        # Windows 下记录创建前的进程窗口句柄，用于后续强制调整尺寸
        before_hwnds = set(_enum_process_windows())

        try:
            bpy.ops.wm.window_new()
        except Exception as e:
            self.report({'ERROR'}, f"无法创建预览窗口: {e}")
            return {'CANCELLED'}

        # 找到新创建的窗口
        new_window = None
        for w in context.window_manager.windows:
            if w.as_pointer() not in existing:
                new_window = w
                break

        if not new_window or not new_window.screen.areas:
            self.report({'ERROR'}, "无法定位新窗口")
            return {'CANCELLED'}

        # 计算预览窗口的目标尺寸和居中位置
        main_win = context.window
        target_w = 2400
        target_h = 1500
        pos_x = main_win.x + max(0, (main_win.width - target_w) // 2)
        pos_y = main_win.y + max(0, (main_win.height - target_h) // 2)

        # 限制预览窗口尺寸，避免默认窗口过大
        try:
            new_window.width = target_w
            new_window.height = target_h
            new_window.x = pos_x
            new_window.y = pos_y
        except Exception:
            log.debug("Failed to set new window geometry via Blender API")

        # Windows 下若 Blender API 未生效，使用 Win32 API 兜底
        if os.name == "nt":
            try:
                after_hwnds = set(_enum_process_windows())
                new_hwnds = list(after_hwnds - before_hwnds)
                if new_hwnds:
                    _set_window_pos(new_hwnds[0], pos_x, pos_y, target_w, target_h)
            except Exception:
                log.debug("Win32 SetWindowPos fallback failed")

        # 把新窗口的第一个区域设为 Image Editor 并加载图像
        area = new_window.screen.areas[0]
        area.type = 'IMAGE_EDITOR'
        image_space = None
        for space in area.spaces:
            if space.type == 'IMAGE_EDITOR':
                space.image = img
                # 隐藏工具栏和侧边栏，最大化图像显示区域
                space.show_region_toolbar = False
                space.show_region_ui = False
                image_space = space
                break

        # 让图像自适应窗口大小
        if image_space:
            try:
                region = None
                for r in area.regions:
                    if r.type == 'WINDOW':
                        region = r
                        break
                if region:
                    with context.temp_override(
                        window=new_window,
                        screen=new_window.screen,
                        area=area,
                        region=region,
                    ):
                        bpy.ops.image.view_all()
            except Exception:
                log.debug("Failed to fit image to preview window")

        return {'FINISHED'}


class AI_OT_ChannelPackRebuild(bpy.types.Operator):
    """基于当前结果的 Roughness + Metallic 贴图生成 Channel Packed 贴图并重建材质。"""
    bl_idname = "ai_concept.channel_pack_rebuild"
    bl_label = "重建通道打包材质"
    bl_description = "基于当前结果的粗糙度/金属度贴图生成通道打包贴图并重建材质"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import numpy as np
        from PIL import Image

        props = context.scene.ai_concept_props
        addon_name = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_name].preferences
        obj = context.active_object

        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "请选择一个网格对象")
            return {'CANCELLED'}

        # 获取当前结果
        if not props.results or props.active_result_index < 0 or props.active_result_index >= len(props.results):
            self.report({'ERROR'}, "没有可用的生成结果")
            return {'CANCELLED'}

        result = props.results[props.active_result_index]
        if result.result_type != 'pbr':
            self.report({'ERROR'}, "当前结果不是 PBR 贴图")
            return {'CANCELLED'}

        # 收集贴图名称
        map_names = {}
        for m in result.pbr_maps:
            map_names[m.map_type] = m.image_name

        for req in ('roughness', 'metallic'):
            if req not in map_names:
                self.report({'ERROR'}, f"当前结果缺少 {req} 贴图")
                return {'CANCELLED'}

        roughness_img = bpy.data.images.get(map_names['roughness'])
        metallic_img = bpy.data.images.get(map_names['metallic'])
        if not roughness_img or not metallic_img:
            self.report({'ERROR'}, "找不到贴图数据，可能已被删除")
            return {'CANCELLED'}

        # 可选 height 贴图
        height_img = None
        if 'height' in map_names:
            height_img = bpy.data.images.get(map_names['height'])

        # Blender Image -> PIL
        def _to_pil(img):
            w, h = img.size
            arr = np.array(img.pixels[:]).reshape((h, w, 4))
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr, 'RGBA').convert('RGB')

        roughness_pil = _to_pil(roughness_img)
        metallic_pil = _to_pil(metallic_img)
        height_pil = _to_pil(height_img) if height_img else None

        # 生成 packed 贴图
        from .material.pbr_processor import generate_packed_map
        pack_config = {
            'r': props.pack_channel_r,
            'g': props.pack_channel_g,
            'b': props.pack_channel_b,
            'a': props.pack_channel_a,
        }
        packed_pil = generate_packed_map(roughness_pil, metallic_pil, pack_config, ao=None, height=height_pil)

        # 输出目录：与已有贴图同目录，解析失败时用偏好设置路径
        output_dir = os.path.dirname(bpy.path.abspath(roughness_img.filepath_raw))
        if not os.path.isdir(output_dir):
            output_dir = pref_utils.resolve_asset_output_path(prefs)
            if not output_dir:
                self.report({'ERROR'}, "找不到输出目录，请先在偏好设置中设置资源输出目录")
                return {'CANCELLED'}
            os.makedirs(output_dir, exist_ok=True)

        batch_id = result.timestamp
        packed_name = f"{obj.name}_packed_{batch_id}"

        # 创建 Blender Image
        packed_blender = bpy.data.images.new(
            name=packed_name,
            width=packed_pil.size[0],
            height=packed_pil.size[1],
        )

        rgba = packed_pil.convert('RGBA')
        pixels = list(rgba.getdata())
        flat = [float(c) / 255.0 for px in pixels for c in px]
        packed_blender.pixels.foreach_set(flat)
        packed_blender.update()
        packed_blender.pack()

        save_path = os.path.join(output_dir, f"{packed_name}.png")
        packed_pil.save(save_path)
        packed_blender.filepath_raw = save_path.replace('\\', '/')
        packed_blender.file_format = 'PNG'
        packed_blender.colorspace_settings.name = 'Non-Color'

        # 构建 images 字典重建材质
        images = {}
        for key in ('diffuse', 'normal'):
            if key in map_names:
                img = bpy.data.images.get(map_names[key])
                if img:
                    images[key] = img
        images['packed'] = packed_blender

        from .material.shader_builder import ShaderBuilder
        mat = ShaderBuilder.build_principled_bsdf(
            f"Mat_{obj.name}_PBR_Packed",
            images,
            pack_config=pack_config,
        )

        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

        # 记录 packed 贴图到结果中
        has_packed = any(m.map_type == 'packed' for m in result.pbr_maps)
        if not has_packed:
            map_item = result.pbr_maps.add()
            map_item.map_type = 'packed'
            map_item.image_name = packed_blender.name

        # 被 pack 进通道的原始贴图不再单独显示在预览中
        content_to_map_type = {
            'ROUGHNESS': 'roughness',
            'METALLIC': 'metallic',
            'AO': 'ao',
            'HEIGHT': 'height',
        }
        packed_map_types = {
            content_to_map_type[content]
            for content in pack_config.values()
            if content in content_to_map_type
        }
        # 从后往前移除，避免索引变化
        for i in range(len(result.pbr_maps) - 1, -1, -1):
            if result.pbr_maps[i].map_type in packed_map_types:
                result.pbr_maps.remove(i)

        from .ui import preview_manager
        preview_manager.load_preview(packed_blender.name, save_path)

        self.report({'INFO'}, "通道打包材质已重建")
        return {'FINISHED'}


class AI_OT_ClearResults(bpy.types.Operator):
    bl_idname = "ai_concept.clear_results"
    bl_label = "清空结果"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.ai_concept_props
        for item in props.results:
            for map_item in item.pbr_maps:
                if map_item.image_name in bpy.data.images:
                    bpy.data.images.remove(bpy.data.images[map_item.image_name])
        props.results.clear()
        preview_manager.clear_previews()
        props.active_result_index = -1
        return {'FINISHED'}


class AI_OT_PrevResult(bpy.types.Operator):
    bl_idname = "ai_concept.prev_result"
    bl_label = "上一个"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        props = context.scene.ai_concept_props
        if props.active_result_index > 0:
            props.active_result_index -= 1
        return {'FINISHED'}


class AI_OT_NextResult(bpy.types.Operator):
    bl_idname = "ai_concept.next_result"
    bl_label = "下一个"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        props = context.scene.ai_concept_props
        if props.active_result_index < len(props.results) - 1:
            props.active_result_index += 1
        return {'FINISHED'}


class AI_OT_ClearReferenceImage(bpy.types.Operator):
    bl_idname = "ai_concept.clear_reference_image"
    bl_label = "清除参考图"
    bl_description = "移除当前参考图"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ai_concept_props
        if props.reference_image:
            preview_manager.remove_preview(props.reference_image.name)
            if props.reference_image.name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[props.reference_image.name])
        props.reference_image = None
        return {'FINISHED'}


class AI_OT_CaptionReferenceImage(bpy.types.Operator):
    """使用视觉模型分析参考图并反推中文材质纹理提示词。"""
    bl_idname = "ai_concept.caption_reference_image"
    bl_label = "反推提示词"
    bl_description = "根据参考图生成中文材质纹理描述提示词"
    bl_options = {'REGISTER', 'UNDO'}

    SYSTEM_PROMPT = (
        "你是一位专业的 3A 级 PBR 材质纹理分析师。\n"
        "请观察图片并输出一个用于复现该材质纹理的 JSON 对象。\n"
        "不要标题，不要解释，不要 Markdown，只输出合法 JSON。\n\n"
        "JSON 字段：\n"
        "- subject: 材质主体名称（如大理石纹理表面、锈蚀金属板等）\n"
        "- materials: 材质类型与表面质感（如天然大理石、抛光表面、光滑质感）\n"
        "- colors: 颜色描述（如深棕色为主基调，米白色纹理脉络）\n"
        "- key_details: 关键纹理细节（如裂纹、磨损、阴影过渡、反光特征等）\n\n"
        "所有字段值使用中文，信息尽量完整具体。"
    )

    USER_PROMPT = "请分析这张图片的材质纹理特征，按上述字段输出 JSON。"

    PBR_SUFFIX = "无缝拼接，照片扫描超写实，平光照明，无阴影，无高光，正投影材质视角，UV就绪"

    def _get_provider_snapshot(self, context, prefs):
        pref_utils.ensure_default_api_providers(prefs)

        props = context.scene.ai_concept_props
        provider_value = props.texture_generator
        snapshot = None

        if pref_utils.is_api_provider_value(provider_value):
            snapshot = pref_utils.get_selected_api_provider_snapshot(
                context,
                provider_value,
                getattr(props, "api_image_model", "DEFAULT"),
                "DEFAULT",
            )

        if not snapshot or not snapshot.get("api_key"):
            snapshot = pref_utils.find_any_configured_api_provider(prefs, context)

        return snapshot

    def _image_to_base64(self, blender_image):
        import numpy as np
        from PIL import Image
        import io
        import base64

        width, height = blender_image.size
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸无效")
        pixels = blender_image.pixels
        if pixels is None or len(pixels) == 0:
            raise ValueError("无法读取图片像素数据（图片可能未加载或已被删除）")

        np_pixels = np.array(pixels[:], dtype=np.float32)
        np_pixels = np_pixels.reshape((height, width, 4))
        np_pixels = (np_pixels * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_pixels, 'RGBA').convert('RGB')

        max_size = 1024
        if max(pil_img.size) > max_size:
            pil_img.thumbnail((max_size, max_size), Image.LANCZOS)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _parse_caption_json(self, text):
        """从模型返回文本中解析 JSON 并融合为自然语言。"""
        import json
        text = (text or "").strip()
        if not text:
            return ""

        # 尝试提取 ```json ... ``` 代码块
        import re
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        # 尝试提取 {} 包裹的 JSON
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            text = json_match.group(0)

        try:
            data = json.loads(text)
        except Exception:
            # 解析失败则直接返回原文本
            return text

        if not isinstance(data, dict):
            return text

        parts = []
        for key in ("subject", "materials", "colors", "key_details"):
            value = data.get(key)
            if value and isinstance(value, str):
                value = value.strip()
                if value and value not in parts:
                    parts.append(value)

        if not parts:
            return text

        return "，".join(parts)

    def execute(self, context):
        props = context.scene.ai_concept_props
        addon_name = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_name].preferences

        if not props.reference_image:
            self.report({'ERROR'}, "请先加载参考图")
            return {'CANCELLED'}

        provider_snapshot = self._get_provider_snapshot(context, prefs)
        if not provider_snapshot or not provider_snapshot.get("api_key"):
            self.report({'ERROR'}, "请先在 Preferences > AI Texture to PBR 中配置至少一个 API Provider")
            return {'CANCELLED'}

        api_key = provider_snapshot["api_key"]
        base_url = (provider_snapshot.get("base_url") or "").rstrip("/")
        model = provider_snapshot.get("vision_model") or provider_snapshot.get("text_model") or provider_snapshot.get("image_model")
        protocol = provider_snapshot.get("protocol", "OPENAI_COMPATIBLE")

        if not model:
            self.report({'ERROR'}, "未找到可用的视觉模型，请先在 Preferences 中设置 Default Vision Model")
            return {'CANCELLED'}

        try:
            b64 = self._image_to_base64(props.reference_image)
        except Exception as e:
            self.report({'ERROR'}, f"参考图转换失败: {e}")
            return {'CANCELLED'}

        import requests
        try:
            if protocol == 'GEMINI':
                # SECURITY: Gemini API Key 通过 URL query param 传递，见上文说明。
                gemini_base = base_url if base_url.endswith("/v1beta") else f"{base_url}/v1beta"
                resp = requests.post(
                    f"{gemini_base}/models/{model}:generateContent?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [
                                {"text": self.SYSTEM_PROMPT},
                                {"inline_data": {"mime_type": "image/png", "data": b64}},
                                {"text": self.USER_PROMPT},
                            ]
                        }],
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates or not isinstance(candidates[0], dict):
                    raise ValueError("Gemini 返回空 candidates")
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                raw_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            else:
                api_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
                data_url = f"data:image/png;base64,{b64}"
                resp = requests.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                    {"type": "text", "text": self.USER_PROMPT},
                                ],
                            },
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1200,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    raise ValueError("API 返回空 choices")
                message = choices[0].get("message") or {}
                raw_text = message.get("content", "").strip()

            caption = self._parse_caption_json(raw_text)
            if not caption:
                self.report({'ERROR'}, "模型返回空提示词")
                return {'CANCELLED'}

            # 追加 PBR 技术后缀
            full_prompt = f"{caption}。{self.PBR_SUFFIX}"
            props.prompt = full_prompt
            self.report({'INFO'}, "参考图提示词已生成")
        except requests.exceptions.ReadTimeout:
            self.report({'ERROR'}, "反推超时，请稍后重试")
            return {'CANCELLED'}
        except requests.exceptions.HTTPError as e:
            self.report({'ERROR'}, f"API 错误: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"反推失败: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class AI_OT_EditPromptInTextEditor(bpy.types.Operator):
    """在 3D View 右侧拆分出一个文本编辑器编辑提示词，接受后自动恢复。"""
    bl_idname = "ai_concept.edit_prompt_in_text_editor"
    bl_label = "编辑提示词"
    bl_description = "在 3D View 右侧打开文本编辑器编辑提示词 (Ctrl+Enter)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # 只在 3D View 的 N-panel（UI region）中生效，避免在 viewport 主体里误触发
        if context.area.type != 'VIEW_3D':
            return False
        if context.region.type != 'UI':
            return False
        # 避免在已拆出的文本编辑器里重复触发；若记录的区域已被手动关闭则清理状态
        props = context.scene.ai_concept_props
        if props.prompt_editor_new_area_ptr:
            for area in context.screen.areas:
                if str(area.as_pointer()) == props.prompt_editor_new_area_ptr:
                    return False
            props.prompt_editor_new_area_ptr = ""
        return True

    def execute(self, context):
        props = context.scene.ai_concept_props
        text_name = "CT_Prompt_Edit"

        # 获取或创建临时文本块
        if text_name in bpy.data.texts:
            text = bpy.data.texts[text_name]
        else:
            text = bpy.data.texts.new(text_name)

        # 将当前 prompt 写入文本块
        text.from_string(props.prompt)

        # 记录拆分前所有 area 的指针
        existing_pointers = {str(a.as_pointer()) for a in context.screen.areas}

        # 保存原区域信息
        props.prompt_editor_previous_area_type = context.area.type
        props.prompt_editor_area_ptr = str(context.area.as_pointer())

        # 拆分当前 area，新区域出现在右侧（VERTICAL），占右侧约 35% 宽度
        try:
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.65)
        except Exception as e:
            log.warning("Failed to split area: %s", e)
            # fallback: 直接切换当前区域
            context.area.type = 'TEXT_EDITOR'
            for space in context.area.spaces:
                if space.type == 'TEXT_EDITOR':
                    space.text = text
                    space.show_line_numbers = True
                    space.show_word_wrap = True
                    break
            return {'FINISHED'}

        # 找到新创建的 area
        new_area = None
        for area in context.screen.areas:
            ptr = str(area.as_pointer())
            if ptr not in existing_pointers:
                new_area = area
                break

        if new_area is None:
            # 没找到新区域，fallback
            context.area.type = 'TEXT_EDITOR'
            new_area = context.area

        # 将新区域设为文本编辑器并加载文本
        new_area.type = 'TEXT_EDITOR'
        for space in new_area.spaces:
            if space.type == 'TEXT_EDITOR':
                space.text = text
                space.show_line_numbers = True
                space.show_word_wrap = True
                break

        # 保存新区域指针，用于 Accept 时关闭
        if new_area:
            props.prompt_editor_new_area_ptr = str(new_area.as_pointer())

        return {'FINISHED'}


class AI_OT_ApplyPromptFromTextEditor(bpy.types.Operator):
    """将文本编辑器中的内容应用回提示词，并关闭拆出的文本编辑器区域。"""
    bl_idname = "ai_concept.apply_prompt_from_text_editor"
    bl_label = "应用提示词"
    bl_description = "将文本内容写回提示词并关闭文本编辑器区域"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # 只有在 Prompt 编辑器实际存在且当前焦点在文本编辑器时才触发
        if context.area.type != 'TEXT_EDITOR':
            return False
        props = context.scene.ai_concept_props
        if not props.prompt_editor_new_area_ptr:
            return False
        for area in context.screen.areas:
            if str(area.as_pointer()) == props.prompt_editor_new_area_ptr:
                return True
        return False

    def execute(self, context):
        props = context.scene.ai_concept_props
        text_name = "CT_Prompt_Edit"

        if text_name in bpy.data.texts:
            text = bpy.data.texts[text_name]
            props.prompt = text.as_string()

        # 找到拆出的文本编辑器区域并关闭
        new_area = None
        ptr = props.prompt_editor_new_area_ptr
        if ptr:
            for area in context.screen.areas:
                if str(area.as_pointer()) == ptr:
                    new_area = area
                    break

        if new_area:
            try:
                # 使用 temp_override 关闭指定 area
                region = None
                for r in new_area.regions:
                    if r.type == 'WINDOW':
                        region = r
                        break
                if region:
                    with context.temp_override(area=new_area, region=region):
                        bpy.ops.screen.area_close()
            except Exception as e:
                log.warning("Failed to close prompt editor area: %s", e)
                # fallback: 改回 VIEW_3D
                try:
                    new_area.type = props.prompt_editor_previous_area_type or 'VIEW_3D'
                except Exception:
                    log.debug("Failed to restore area type during prompt editor close")

        # 清理临时文本块（可选：保留以方便撤销）
        # if text_name in bpy.data.texts:
        #     bpy.data.texts.remove(bpy.data.texts[text_name])

        props.prompt_editor_area_ptr = ""
        props.prompt_editor_previous_area_type = ""
        props.prompt_editor_new_area_ptr = ""

        return {'FINISHED'}


class AI_PT_PromptEditorPanel(bpy.types.Panel):
    """在文本编辑器侧栏显示的应用面板。"""
    bl_label = "提示词编辑"
    bl_idname = "AI_PT_prompt_editor_panel"
    bl_space_type = 'TEXT_EDITOR'
    bl_region_type = 'UI'
    bl_category = "提示词"

    def draw(self, context):
        layout = self.layout
        text_name = "CT_Prompt_Edit"
        if text_name in bpy.data.texts and context.space_data.text == bpy.data.texts[text_name]:
            row = layout.row()
            row.alignment = 'CENTER'
            row.scale_y = 1.5
            row.operator("ai_concept.apply_prompt_from_text_editor", text="接受", icon='CHECKMARK')
            layout.label(text="Ctrl+Enter 也可接受")
        else:
            layout.label(text="未在编辑提示词")


def _prompt_editor_menu_func(self, context):
    """在 Text Editor 的 Text 菜单中追加 Accept 项。"""
    text_name = "CT_Prompt_Edit"
    if text_name in bpy.data.texts and context.space_data.text == bpy.data.texts[text_name]:
        self.layout.separator()
        self.layout.operator("ai_concept.apply_prompt_from_text_editor", text="接受提示词", icon='CHECKMARK')


class AI_MT_ReferenceImageMenu(bpy.types.Menu):
    bl_label = "参考图"
    bl_idname = "AI_MT_reference_image_menu"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ai_concept_props
        layout.operator("ai_concept.load_reference_image", text="选择文件...", icon='FILEBROWSER')
        layout.operator("ai_concept.paste_reference_image", text="从剪贴板粘贴", icon='PASTEDOWN')
        if props.reference_image:
            layout.separator()
            layout.operator("ai_concept.clear_reference_image", text="清除", icon='X')


# =============================================================================
# ComfyUI 安装 / 模型下载
# =============================================================================

_install_state = {
    "thread": None,
    "status": "",
    "running": False,
}


def _get_prefs(context):
    addon_pkg = __package__.split('.')[0]
    return context.preferences.addons[addon_pkg].preferences


def _get_install_path(prefs):
    return prefs.comfyui_path or comfyui_installer.get_default_install_path()


def _poll_install():
    """轮询后台安装线程并更新 UI 状态。"""
    state = _install_state
    if state["thread"] is None:
        return None
    if state["thread"].is_alive():
        try:
            if bpy.context and bpy.context.scene:
                bpy.context.scene.ai_concept_props.status_message = f"正在安装 ComfyUI: {state['status']}"
                log.debug("ComfyUI install: %s", state['status'])
        except Exception:
            log.debug("Could not update install status (context unavailable)")
        return 0.5
    state["running"] = False
    try:
        if bpy.context and bpy.context.scene:
            bpy.context.scene.ai_concept_props.status_message = f"安装完成: {state['status']}"
            log.debug("ComfyUI install finished: %s", state['status'])
    except Exception:
        log.debug("Could not update install finished status (context unavailable)")
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    state["thread"] = None
    return None


class AI_OT_InstallComfyUI(bpy.types.Operator):
    bl_idname = "ai_concept.install_comfyui"
    bl_label = "安装 ComfyUI"
    bl_description = "下载并安装 ComfyUI（约 15 GB）+ 模型（约 10 GB），共需约 25 GB 磁盘空间，NVIDIA GPU 推荐"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=480)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="即将下载并安装 ComfyUI 及必需模型。", icon='INFO')
        col.separator()
        col.label(text=f"⚠ 预计需要约 25 GB 可用磁盘空间", icon='ERROR')
        col.label(text="⚠ 需要 NVIDIA GPU (4 GB+ 显存) 才能正常使用", icon='ERROR')
        col.label(text="⚠ 下载可能需要 15~60 分钟（取决于网络）", icon='ERROR')
        col.separator()
        col.label(text="仅需执行一次，安装完成后即可使用本地 ComfyUI。")
        col.separator()
        col.label(text="确认继续？")

    def execute(self, context):
        if os.name != "nt":
            self.report({'ERROR'}, "ComfyUI 自动安装仅支持 Windows。请在 macOS/Linux 上手动安装 ComfyUI 并在偏好设置中设置路径。")
            return {'CANCELLED'}

        state = _install_state
        if state["running"]:
            self.report({'WARNING'}, "安装已在进行中")
            return {'CANCELLED'}

        prefs = _get_prefs(context)
        target = _get_install_path(prefs)

        state["running"] = True
        state["status"] = "准备中..."

        def progress_cb(phase, progress, message):
            state["status"] = message
            log.debug("Install [%s]: %s", phase, message)

        use_mirror = getattr(prefs, "use_china_mirror", True)

        def worker():
            try:
                comfyui_installer.install_comfyui(target, progress_cb, use_mirror=use_mirror)
            except Exception as e:
                state["status"] = f"Error: {e}"
            finally:
                state["running"] = False

        state["thread"] = threading.Thread(target=worker, daemon=True)
        state["thread"].start()

        bpy.app.timers.register(_poll_install, first_interval=0.5)
        self.report({'INFO'}, "ComfyUI 后台安装已启动")
        return {'FINISHED'}


_download_state = {
    "thread": None,
    "status": "",
}


def _poll_download():
    state = _download_state
    if state["thread"] is None:
        return None
    if state["thread"].is_alive():
        try:
            if bpy.context and bpy.context.scene:
                bpy.context.scene.ai_concept_props.status_message = f"正在下载模型: {state['status']}"
                log.debug("Model download: %s", state['status'])
        except Exception:
            log.debug("Could not update download status (context unavailable)")
        return 0.5
    try:
        if bpy.context and bpy.context.scene:
            bpy.context.scene.ai_concept_props.status_message = f"下载完成: {state['status']}"
            log.debug("Model download finished: %s", state['status'])
    except Exception:
        log.debug("Could not update download finished status (context unavailable)")
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    state["thread"] = None
    return None


class AI_OT_DownloadModel(bpy.types.Operator):
    bl_idname = "ai_concept.download_model"
    bl_label = "下载模型"
    bl_description = "下载选中的模型到 ComfyUI models 目录"
    bl_options = {'REGISTER'}

    model_id: bpy.props.StringProperty()

    def execute(self, context):
        state = _download_state
        if state["thread"] and state["thread"].is_alive():
            self.report({'WARNING'}, "已有下载进行中")
            return {'CANCELLED'}

        prefs = _get_prefs(context)
        comfyui_path = _get_install_path(prefs)

        if not os.path.isdir(comfyui_path):
            self.report({'ERROR'}, "ComfyUI 路径不存在")
            return {'CANCELLED'}

        model_id = self.model_id

        hf_token = prefs.huggingface_token
        use_mirror = getattr(prefs, "use_china_mirror", True)

        def progress_cb(phase, progress, message):
            state["status"] = message
            log.debug("Download [%s]: %s", phase, message)

        def worker():
            try:
                comfyui_installer.download_model(comfyui_path, model_id, progress_cb, hf_token=hf_token, use_mirror=use_mirror)
            except Exception as e:
                state["status"] = f"Error: {e}"

        state["thread"] = threading.Thread(target=worker, daemon=True)
        state["thread"].start()

        bpy.app.timers.register(_poll_download, first_interval=0.5)
        self.report({'INFO'}, f"开始下载模型: {model_id}")
        return {'FINISHED'}


class AI_OT_OpenModelDownloadPage(bpy.types.Operator):
    bl_idname = "ai_concept.open_model_download_page"
    bl_label = "打开模型下载页"
    bl_description = "在浏览器中打开模型下载页面"
    bl_options = {'REGISTER'}

    model_id: bpy.props.StringProperty()

    def execute(self, context):
        model = next((m for m in comfyui_installer.get_model_registry() if m["id"] == self.model_id), None)
        if not model:
            self.report({'ERROR'}, "未知模型")
            return {'CANCELLED'}

        prefs = _get_prefs(context)
        use_mirror = getattr(prefs, "use_china_mirror", True)

        page = model.get("page") or ""
        if not use_mirror:
            # 关闭镜像时尝试打开官方 HuggingFace 页面
            official_url = ""
            for u in [model.get("url", "")] + list(model.get("mirrors", [])):
                if "huggingface.co" in u:
                    official_url = u
                    break
            if official_url:
                # 从 resolve URL 推导文件树页面
                import re
                page = re.sub(r"/resolve/main/[^/]+$", "/tree/main", official_url)
                if "/tree/main" not in page:
                    page = official_url.rsplit("/", 1)[0] + "/tree/main"

        if not page:
            page = model.get("url")
        if not page:
            self.report({'ERROR'}, "无可用的下载页面")
            return {'CANCELLED'}
        webbrowser.open(page)
        return {'FINISHED'}


addon_keymaps = []

classes = [
    AI_OT_GenerateTexture,
    AI_OT_StopGeneration,
    AI_OT_LoadReferenceImage,
    AI_OT_PasteReferenceImage,
    AI_OT_ClearReferenceImage,
    AI_OT_CaptionReferenceImage,
    AI_OT_EditPromptInTextEditor,
    AI_OT_ApplyPromptFromTextEditor,
    AI_PT_PromptEditorPanel,
    AI_MT_ReferenceImageMenu,
    AI_OT_ClearResults,
    AI_OT_PrevResult,
    AI_OT_NextResult,
    AI_OT_OpenImageFolder,
    AI_OT_EnlargePreview,
    AI_OT_OptimizePrompt,
    AI_OT_ChannelPackRebuild,
    AI_OT_InstallComfyUI,
    AI_OT_DownloadModel,
    AI_OT_OpenModelDownloadPage,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            if "already registered" in str(e):
                log.debug("%s already registered, skipping.", cls.__name__)
                continue
            raise

    # 注册菜单追加
    bpy.types.TEXT_MT_text.append(_prompt_editor_menu_func)

    # 注册快捷键
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Window')
        kmi = km.keymap_items.new(
            idname="ai_concept.paste_reference_image",
            type='V',
            value='PRESS',
            ctrl=True,
            shift=False,
            alt=False,
        )
        addon_keymaps.append((km, kmi))

        # Ctrl+Enter（含小键盘 Enter）：打开 Prompt 文本编辑器编辑模式
        # 通过 Window 全局注册，但 operator poll 限制只在 3D View 的 N-panel 中生效
        km_edit = kc.keymaps.new(name='Window')
        kmi_edit = km_edit.keymap_items.new(
            idname="ai_concept.edit_prompt_in_text_editor",
            type='RET',
            value='PRESS',
            ctrl=True,
        )
        addon_keymaps.append((km_edit, kmi_edit))

        km_edit_pad = kc.keymaps.new(name='Window')
        kmi_edit_pad = km_edit_pad.keymap_items.new(
            idname="ai_concept.edit_prompt_in_text_editor",
            type='NUMPAD_ENTER',
            value='PRESS',
            ctrl=True,
        )
        addon_keymaps.append((km_edit_pad, kmi_edit_pad))

        # Ctrl+Enter（含小键盘 Enter）在文本编辑器中触发 Accept
        km_accept = kc.keymaps.new(name='Text', space_type='TEXT_EDITOR')
        kmi_accept = km_accept.keymap_items.new(
            idname="ai_concept.apply_prompt_from_text_editor",
            type='RET',
            value='PRESS',
            ctrl=True,
        )
        addon_keymaps.append((km_accept, kmi_accept))

        km_accept_pad = kc.keymaps.new(name='Text', space_type='TEXT_EDITOR')
        kmi_accept_pad = km_accept_pad.keymap_items.new(
            idname="ai_concept.apply_prompt_from_text_editor",
            type='NUMPAD_ENTER',
            value='PRESS',
            ctrl=True,
        )
        addon_keymaps.append((km_accept_pad, kmi_accept_pad))


def unregister():
    bpy.types.TEXT_MT_text.remove(_prompt_editor_menu_func)

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
