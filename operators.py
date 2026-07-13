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
from .properties import build_texture_prompt
from .utils.image_utils import (
    blender_image_to_numpy,
    numpy_to_blender_image,
    save_numpy_image,
)

log = get_logger(__name__)


def _open_file_explorer(folder: str, select_file: str = ""):
    """Cross-platform file explorer: open `folder` and optionally select a file."""
    folder = os.path.normpath(folder)
    select_file = os.path.normpath(select_file) if select_file else ""
    if os.name == "nt":
        if select_file and os.path.isfile(select_file):
            # /select 会打开文件夹并高亮指定文件
            subprocess.run(["explorer", f"/select,{select_file}"], check=False)
        else:
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


def _unique_image_name(prefix: str, basename: str) -> str:
    import time

    stem, ext = os.path.splitext(os.path.basename(basename))
    stem = stem or "Image"
    candidate = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{stem}{ext}"
    if candidate not in bpy.data.images:
        return candidate
    idx = 1
    while f"{candidate}_{idx:02d}" in bpy.data.images:
        idx += 1
    return f"{candidate}_{idx:02d}"


def _requests_module():
    try:
        import requests
        return requests
    except ImportError:
        from .utils import simple_requests
        return simple_requests


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

        # 加载图片到 Blender，使用插件私有名称，避免删除用户已有同名图像
        img_name = _unique_image_name("AIRef", self.filepath)

        image = bpy.data.images.load(self.filepath, check_existing=False)
        image.name = img_name
        image["ai_concept_owned"] = True
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


def _grab_clipboard_with_pil():
    """使用 Pillow 读取剪贴板图像或图片文件路径，返回 (numpy RGBA array, width, height) 或 None。"""
    import numpy as np
    from PIL import Image, ImageGrab
    result = ImageGrab.grabclipboard()
    if result is None:
        return None

    if isinstance(result, Image.Image):
        pil_img = result
    elif isinstance(result, list) and result:
        # 剪贴板中是文件路径列表，加载第一个图片文件
        for path in result:
            if os.path.isfile(path):
                try:
                    with Image.open(path) as pil_img:
                        if pil_img.mode != 'RGBA':
                            pil_img = pil_img.convert('RGBA')
                        return np.array(pil_img), pil_img.size[0], pil_img.size[1]
                except Exception as e:
                    log.warning("Cannot open clipboard file %s: %s", path, e)
                    continue
        return None
    else:
        return None


def _grab_clipboard_windows_ctypes():
    """Windows 无 Pillow 时使用 ctypes 读取剪贴板图像。

    支持格式（按优先级）：image/png、PNG、CF_HDROP、CF_DIBV5、CF_DIB。
    返回 (numpy RGBA array, width, height) 或 None。
    失败时通过 log.warning 输出诊断信息。
    """
    import ctypes
    import struct
    import tempfile

    from .utils.image_utils import blender_image_to_numpy

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.restype = ctypes.c_bool
    user32.RegisterClipboardFormatW.argtypes = [ctypes.c_wchar_p]
    user32.RegisterClipboardFormatW.restype = ctypes.c_uint
    user32.EnumClipboardFormats.argtypes = [ctypes.c_uint]
    user32.EnumClipboardFormats.restype = ctypes.c_uint
    user32.GetClipboardFormatNameW.argtypes = [ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_int]
    user32.GetClipboardFormatNameW.restype = ctypes.c_int
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool
    kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.restype = ctypes.c_size_t

    CF_DIB = 8
    CF_DIBV5 = 17
    CF_BITMAP = 2
    CF_HDROP = 15

    def _last_error():
        return ctypes.windll.kernel32.GetLastError()

    def _read_hglobal(hglobal):
        if not hglobal:
            return None
        ptr = kernel32.GlobalLock(hglobal)
        if not ptr:
            log.warning("GlobalLock failed, err=%s", _last_error())
            return None
        try:
            size = kernel32.GlobalSize(hglobal)
            return ctypes.string_at(ptr, size)
        finally:
            kernel32.GlobalUnlock(hglobal)

    def _load_temp_image(temp_path):
        try:
            blender_img = bpy.data.images.load(temp_path, check_existing=False)
            width, height = blender_img.size
            arr = blender_image_to_numpy(blender_img)
            bpy.data.images.remove(blender_img)
            log.warning("Loaded clipboard image via Blender: %sx%s", int(width), int(height))
            return arr, int(width), int(height)
        except Exception as e:
            log.warning("Load temp image failed for %s: %s", temp_path, e)
            return None

    if not user32.OpenClipboard(None):
        log.warning("OpenClipboard failed, err=%s", _last_error())
        raise RuntimeError("无法打开剪贴板")

    temp_path = None
    try:
        # 1) 优先尝试 image/png / PNG 格式（QQ/微信/截图工具常用）
        for fmt_name in ("image/png", "PNG"):
            fmt_id = user32.RegisterClipboardFormatW(fmt_name)
            if not fmt_id:
                continue
            data = _read_hglobal(user32.GetClipboardData(fmt_id))
            if data:
                log.warning("%s format data size=%s, header=%s", fmt_name, len(data), data[:8])
                if data[:4] == b'\x89PNG':
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        f.write(data)
                        temp_path = f.name
                    result = _load_temp_image(temp_path)
                    if result:
                        return result
                else:
                    log.warning("%s format data has invalid header", fmt_name)

        # 2) 尝试 CF_HDROP（剪贴板中是图片文件路径）
        hdrop = user32.GetClipboardData(CF_HDROP)
        if hdrop:
            log.warning("Using CF_HDROP clipboard format")
            data = _read_hglobal(hdrop)
            if data and len(data) >= 20:
                # DROPFILES 结构，pFiles 是文件列表偏移
                p_files = struct.unpack_from('<I', data, 0)[0]
                f_wide = struct.unpack_from('<I', data, 16)[0]
                paths_bytes = data[p_files:]
                if f_wide:
                    paths = paths_bytes.decode('utf-16-le').split('\x00')
                else:
                    paths = paths_bytes.decode('mbcs').split('\x00')
                # 过滤空字符串和最后一个双 null 后的空项
                paths = [p for p in paths if p]
                log.warning("CF_HDROP paths: %s", paths[:5])
                for path in paths:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'):
                        try:
                            result = _load_temp_image(path)
                            if result:
                                return result
                        except Exception as e:
                            log.warning("Failed to load dropped image %s: %s", path, e)

        # 3) 尝试 CF_DIBV5 / CF_DIB
        hglobal = user32.GetClipboardData(CF_DIBV5)
        if hglobal:
            log.warning("Using CF_DIBV5 clipboard format")
        else:
            hglobal = user32.GetClipboardData(CF_DIB)
            if hglobal:
                log.warning("Using CF_DIB clipboard format")

        if hglobal:
            raw_bytes = _read_hglobal(hglobal)
            if raw_bytes and len(raw_bytes) >= 4:
                log.warning("DIB data size=%s", len(raw_bytes))
                header_size = struct.unpack_from('<I', raw_bytes, 0)[0]
                log.warning("DIB header size=%s", header_size)
                if header_size in (12, 40, 52, 56, 64, 108, 124) and len(raw_bytes) >= header_size:
                    width = struct.unpack_from('<i', raw_bytes, 4)[0]
                    height = struct.unpack_from('<i', raw_bytes, 8)[0]
                    bit_count = struct.unpack_from('<H', raw_bytes, 14)[0]
                    compression = struct.unpack_from('<I', raw_bytes, 16)[0]
                    clr_used = struct.unpack_from('<I', raw_bytes, 32)[0]
                    log.warning("DIB width=%s height=%s bit_count=%s compression=%s", width, height, bit_count, compression)

                    if compression in (0, 3, 6):
                        if height < 0:
                            height = -height
                        color_table_size = 0
                        if bit_count <= 8:
                            color_count = clr_used if clr_used else (1 << bit_count)
                            color_table_size = color_count * 4

                        pixel_offset = 14 + header_size + color_table_size
                        file_header = struct.pack(
                            '<2sIHHI', b'BM', 14 + len(raw_bytes), 0, 0, pixel_offset,
                        )
                        with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as f:
                            f.write(file_header + raw_bytes)
                            temp_path = f.name
                        result = _load_temp_image(temp_path)
                        if result:
                            return result
                    else:
                        log.warning("Compressed DIB not supported: compression=%s", compression)

        # 3) 调试：记录可用的剪贴板格式
        fmt = 0
        available = []
        format_names = {}
        while True:
            fmt = user32.EnumClipboardFormats(fmt)
            if fmt == 0:
                break
            available.append(fmt)
            buf = ctypes.create_unicode_buffer(256)
            if user32.GetClipboardFormatNameW(fmt, buf, 256):
                format_names[fmt] = buf.value
            elif fmt == CF_BITMAP:
                format_names[fmt] = "CF_BITMAP"
            elif fmt == CF_DIB:
                format_names[fmt] = "CF_DIB"
            elif fmt == CF_DIBV5:
                format_names[fmt] = "CF_DIBV5"
        log.warning("Available clipboard formats: %s", available)
        log.warning("Clipboard format names: %s", format_names)
        return None
    finally:
        user32.CloseClipboard()
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


class AI_OT_PasteReferenceImage(bpy.types.Operator):
    bl_idname = "ai_concept.paste_reference_image"
    bl_label = "粘贴参考图"
    bl_description = "从系统剪贴板粘贴图像作为参考图 (Ctrl+V)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import numpy as np
        import time
        import tempfile

        # 1) 优先使用 Pillow（跨平台、支持更多格式）
        clipboard_result = None
        pil_error = ""
        try:
            clipboard_result = _grab_clipboard_with_pil()
        except ImportError:
            pil_error = "Pillow not installed"
        except Exception as e:
            pil_error = str(e)
            log.warning("PIL clipboard failed: %s", e)

        # 2) 无 Pillow 时，Windows 用 ctypes 读剪贴板
        ctypes_error = ""
        if clipboard_result is None and os.name == 'nt':
            try:
                clipboard_result = _grab_clipboard_windows_ctypes()
            except Exception as e:
                ctypes_error = str(e)
                log.warning("Windows ctypes clipboard failed: %s", e)

        if clipboard_result is None:
            msg = "无法读取剪贴板图像。"
            if pil_error:
                msg += f" Pillow: {pil_error}."
            if ctypes_error:
                msg += f" ctypes: {ctypes_error}."
            self.report(
                {'ERROR'},
                msg + " 打开「Window → Toggle System Console」查看详细格式日志。"
            )
            return {'CANCELLED'}

        arr, width, height = clipboard_result

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_name = f"Ref_Pasted_{timestamp}"

        # 保存到系统临时目录用于预览
        temp_dir = os.path.join(tempfile.gettempdir(), "BlenderAI")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{img_name}.png")

        try:
            save_numpy_image(arr, temp_path, file_format='PNG')
        except Exception as e:
            # 无 Pillow 且 Blender API 保存失败时，尝试用 Pillow
            try:
                from PIL import Image
                Image.fromarray(arr, 'RGBA').save(temp_path)
            except ImportError:
                self.report({'ERROR'}, f"保存剪贴板图像失败: {e}")
                return {'CANCELLED'}

        # 加载到 Blender 并 pack
        if img_name in bpy.data.images:
            img_name = _unique_image_name("Ref_Pasted", img_name)

        image = bpy.data.images.load(temp_path, check_existing=False)
        image.name = img_name
        image["ai_concept_owned"] = True
        image.pack()
        image.update()

        # 注册预览
        preview_manager.remove_preview(img_name)
        preview_manager.load_preview(img_name, temp_path)

        props = context.scene.ai_concept_props
        props.reference_image = image
        self.report({'INFO'}, f"已从剪贴板粘贴参考图 ({width}x{height})")
        return {'FINISHED'}


class AI_OT_OptimizePrompt(bpy.types.Operator):
    """使用 LLM 将当前提示词优化为更详细的图像生成描述。"""
    bl_idname = "ai_concept.optimize_prompt"
    bl_label = "AI 优化提示词"
    bl_description = "使用 LLM 将当前提示词优化为更详细的图像生成描述"
    bl_options = {'REGISTER', 'UNDO'}

    def _local_fallback_optimize(self, props):
        """没有可用 API 时，使用材质配置规则扩展提示词。"""
        expanded = build_texture_prompt(props)
        props.prompt = expanded

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
            # 无可用 API：回退到本地基于 Material Config 的规则扩展
            try:
                self._local_fallback_optimize(props)
                self.report(
                    {'INFO'},
                    "未配置 API，已使用本地规则根据材质配置扩展提示词"
                )
            except Exception as e:
                self.report(
                    {'ERROR'},
                    f"未找到可用 API Key，且本地规则扩展失败: {e}"
                )
                return {'CANCELLED'}
            return {'FINISHED'}

        original = props.prompt.strip()
        if not original:
            self.report({'WARNING'}, "提示词为空，请输入描述后再优化")
            return {'CANCELLED'}

        requests = _requests_module()
        try:
            api_key = provider_snapshot["api_key"]
            base_url = (provider_snapshot.get("base_url") or "").rstrip("/")
            model = provider_snapshot.get("text_model") or provider_snapshot.get("image_model")
            protocol = provider_snapshot.get("protocol", "OPENAI_COMPATIBLE")

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

            if protocol == 'GEMINI':
                # SECURITY: Gemini API Key 要求通过 URL query param 传递 API Key。
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
            return {'FINISHED'}
        except requests.exceptions.ReadTimeout:
            self.report(
                {'ERROR'},
                f"优化失败: {provider_snapshot.get('name', 'API')} 响应超时。/models 测试通过只代表模型列表可访问；请检查 Default Text Model 是否支持 chat/completions，或稍后重试。"
            )
            return {'CANCELLED'}
        except requests.exceptions.HTTPError as e:
            self.report({'ERROR'}, f"API 错误: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"优化失败: {e}")
            return {'CANCELLED'}

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

        _open_file_explorer(folder, filepath)
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

        # Blender Image -> numpy RGB
        def _to_numpy_rgb(img):
            arr = blender_image_to_numpy(img)
            return arr[..., :3]

        roughness_np = _to_numpy_rgb(roughness_img)
        metallic_np = _to_numpy_rgb(metallic_img)
        height_np = _to_numpy_rgb(height_img) if height_img else None

        # 生成 packed 贴图
        from .material.pbr_processor import generate_packed_map
        pack_config = {
            'r': props.pack_channel_r,
            'g': props.pack_channel_g,
            'b': props.pack_channel_b,
            'a': props.pack_channel_a,
        }
        packed_np = generate_packed_map(roughness_np, metallic_np, pack_config, ao=None, height=height_np)

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
        packed_blender = numpy_to_blender_image(packed_name, packed_np)
        packed_blender.pack()

        save_path = os.path.join(output_dir, f"{packed_name}.png")
        packed_blender.filepath_raw = save_path.replace('\\', '/')
        packed_blender.file_format = 'PNG'
        packed_blender.colorspace_settings.name = 'Non-Color'
        try:
            packed_blender.save()
        except Exception as e:
            from PIL import Image
            Image.fromarray(packed_np, 'RGBA').save(save_path)

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

        # 记录 packed 贴图到结果中；已有 packed 时更新引用
        packed_item = next((m for m in result.pbr_maps if m.map_type == 'packed'), None)
        if packed_item is None:
            map_item = result.pbr_maps.add()
            map_item.map_type = 'packed'
            map_item.image_name = packed_blender.name
        else:
            packed_item.image_name = packed_blender.name

        from .ui import preview_manager
        preview_manager.remove_preview(packed_blender.name)
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


class AI_OT_RemoveResult(bpy.types.Operator):
    bl_idname = "ai_concept.remove_result"
    bl_label = "删除此结果"
    bl_description = "删除该条结果及其关联的 Blender 图像"
    bl_options = {'REGISTER'}

    index: bpy.props.IntProperty(name="Index", default=-1)

    def execute(self, context):
        props = context.scene.ai_concept_props
        idx = self.index
        if idx < 0 or idx >= len(props.results):
            return {'CANCELLED'}

        item = props.results[idx]
        for map_item in item.pbr_maps:
            preview_manager.remove_preview(map_item.image_name)
            if map_item.image_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[map_item.image_name])
        props.results.remove(idx)

        # 修正当前选中索引
        if props.active_result_index >= len(props.results):
            props.active_result_index = len(props.results) - 1

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
            owned = bool(props.reference_image.get("ai_concept_owned", False))
            if owned and props.reference_image.name in bpy.data.images:
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

    PBR_SUFFIX = (
        "无缝拼接，照片扫描超写实，平光照明，无阴影，无高光，无反光，"
        "无镜面反射，正投影材质视角，UV就绪"
    )

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
        import base64
        from .utils.image_utils import blender_image_to_numpy, resize_numpy_image, encode_numpy_to_png_bytes

        width, height = blender_image.size
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸无效")
        pixels = blender_image.pixels
        if pixels is None or len(pixels) == 0:
            raise ValueError("无法读取图片像素数据（图片可能未加载或已被删除）")

        np_pixels = blender_image_to_numpy(blender_image)[..., :3]  # RGB

        max_size = 1024
        if max(width, height) > max_size:
            if width > height:
                new_w = max_size
                new_h = int(height * max_size / width)
            else:
                new_h = max_size
                new_w = int(width * max_size / height)
            np_pixels = resize_numpy_image(np_pixels, new_w, new_h)

        img_bytes = encode_numpy_to_png_bytes(np_pixels)
        return base64.b64encode(img_bytes).decode()

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

        if not isinstance(text, str):
            return text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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
            self.report(
                {'ERROR'},
                "反推提示词需要视觉模型 API。请在 Preferences 中配置 API Provider，"
                "或本地运行 Ollama（OpenAI 兼容端点 http://localhost:11434/v1）后添加为 Provider"
            )
            return {'CANCELLED'}

        model = provider_snapshot.get("vision_model") or provider_snapshot.get("text_model") or provider_snapshot.get("image_model")
        if not model:
            self.report({'ERROR'}, "未找到可用的视觉模型，请先在 Preferences 中设置 Default Vision Model")
            return {'CANCELLED'}

        try:
            b64 = self._image_to_base64(props.reference_image)
        except Exception as e:
            self.report({'ERROR'}, f"参考图转换失败: {e}")
            return {'CANCELLED'}

        requests = _requests_module()
        try:
            api_key = provider_snapshot["api_key"]
            base_url = (provider_snapshot.get("base_url") or "").rstrip("/")
            protocol = provider_snapshot.get("protocol", "OPENAI_COMPATIBLE")

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

            props.prompt = f"{caption}。{self.PBR_SUFFIX}"
            self.report({'INFO'}, "参考图提示词已生成")
            return {'FINISHED'}
        except requests.exceptions.ReadTimeout:
            self.report({'ERROR'}, "反推超时，请稍后重试")
            return {'CANCELLED'}
        except requests.exceptions.HTTPError as e:
            self.report({'ERROR'}, f"API 错误: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"反推失败: {e}")
            return {'CANCELLED'}

class AI_OT_EditPromptInTextEditor(bpy.types.Operator):
    """在 3D View 右侧拆分出一个文本编辑器编辑提示词，接受后自动恢复。"""
    bl_idname = "ai_concept.edit_prompt_in_text_editor"
    bl_label = "编辑提示词"
    bl_description = "在 3D View 右侧打开文本编辑器编辑提示词 (Ctrl+Enter)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # 只在 3D View 中生效（无论焦点在 viewport 还是 N-panel）
        if context.area is None or context.area.type != 'VIEW_3D':
            return False
        return True

    def _find_existing_editor_area(self, context):
        props = context.scene.ai_concept_props
        ptr = props.prompt_editor_new_area_ptr
        if not ptr:
            return None
        for area in context.screen.areas:
            if str(area.as_pointer()) == ptr:
                return area
        # 记录的区域已被关闭，清理状态
        props.prompt_editor_new_area_ptr = ""
        return None

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

        # 如果编辑器已经存在，仅刷新内容并聚焦，不再拆分
        existing_area = self._find_existing_editor_area(context)
        if existing_area is not None:
            existing_area.type = 'TEXT_EDITOR'
            for space in existing_area.spaces:
                if space.type == 'TEXT_EDITOR':
                    space.text = text
                    space.show_line_numbers = True
                    space.show_word_wrap = True
                    break
            return {'FINISHED'}

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
        if context.area is None or context.area.type != 'TEXT_EDITOR':
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
# ComfyUI 获取 / 模型下载页
# =============================================================================

def _get_prefs(context):
    addon_pkg = __package__.split('.')[0]
    return context.preferences.addons[addon_pkg].preferences


def _get_install_path(prefs):
    return prefs.comfyui_path or comfyui_installer.get_default_install_path()


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

        page = model.get("page") or model.get("url", "")
        if not page:
            self.report({'ERROR'}, "该模型没有下载链接")
            return {'CANCELLED'}

        import webbrowser
        webbrowser.open(page)
        return {'FINISHED'}


# Operator registration
classes = [
    AI_OT_GenerateTexture,
    AI_OT_StopGeneration,
    AI_OT_LoadReferenceImage,
    AI_OT_PasteReferenceImage,
    AI_OT_OptimizePrompt,
    AI_OT_OpenImageFolder,
    AI_OT_EnlargePreview,
    AI_OT_ChannelPackRebuild,
    AI_OT_ClearResults,
    AI_OT_RemoveResult,
    AI_OT_PrevResult,
    AI_OT_NextResult,
    AI_OT_ClearReferenceImage,
    AI_OT_CaptionReferenceImage,
    AI_OT_EditPromptInTextEditor,
    AI_OT_ApplyPromptFromTextEditor,
    AI_PT_PromptEditorPanel,
    AI_MT_ReferenceImageMenu,
    AI_OT_OpenModelDownloadPage,
]

_addon_keymaps = []


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            if "already registered" in str(e):
                log.debug("%s already registered, skipping.", cls.__name__)
                continue
            raise

    # 注册快捷键（使用 Window 全局键位，operator poll 负责限制触发范围）
    try:
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.addon
        if kc:
            # 参考图面板 Ctrl+V 粘贴剪贴板图像
            km = kc.keymaps.new(name='Window')
            kmi = km.keymap_items.new(
                "ai_concept.paste_reference_image",
                type='V',
                value='PRESS',
                ctrl=True,
            )
            _addon_keymaps.append((km, kmi))

            # Ctrl+Enter（含小键盘 Enter）：在 3D View N-panel 打开 Prompt 文本编辑器
            km_edit = kc.keymaps.new(name='Window')
            kmi_edit = km_edit.keymap_items.new(
                "ai_concept.edit_prompt_in_text_editor",
                type='RET',
                value='PRESS',
                ctrl=True,
            )
            _addon_keymaps.append((km_edit, kmi_edit))

            km_edit_pad = kc.keymaps.new(name='Window')
            kmi_edit_pad = km_edit_pad.keymap_items.new(
                "ai_concept.edit_prompt_in_text_editor",
                type='NUMPAD_ENTER',
                value='PRESS',
                ctrl=True,
            )
            _addon_keymaps.append((km_edit_pad, kmi_edit_pad))

            # Ctrl+Enter（含小键盘 Enter）在文本编辑器中触发“接受提示词”
            km_accept = kc.keymaps.new(name='Text', space_type='TEXT_EDITOR')
            kmi_accept = km_accept.keymap_items.new(
                "ai_concept.apply_prompt_from_text_editor",
                type='RET',
                value='PRESS',
                ctrl=True,
            )
            _addon_keymaps.append((km_accept, kmi_accept))

            km_accept_pad = kc.keymaps.new(name='Text', space_type='TEXT_EDITOR')
            kmi_accept_pad = km_accept_pad.keymap_items.new(
                "ai_concept.apply_prompt_from_text_editor",
                type='NUMPAD_ENTER',
                value='PRESS',
                ctrl=True,
            )
            _addon_keymaps.append((km_accept_pad, kmi_accept_pad))
    except Exception as e:
        log.debug("Could not register keymaps: %s", e)

    # 在 Text Editor 的 Text 菜单追加“接受提示词”项
    try:
        bpy.types.TEXT_MT_text.append(_prompt_editor_menu_func)
    except Exception as e:
        log.debug("Could not register prompt editor menu: %s", e)


def unregister():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()

    try:
        bpy.types.TEXT_MT_text.remove(_prompt_editor_menu_func)
    except Exception:
        pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
