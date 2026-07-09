import bpy
import time
import os
import sys
import importlib

from .sd_backend import comfyui_installer
from .ui import icons as icon_manager
from .utils.logger import get_logger

log = get_logger(__name__)

API_PROVIDER_PREFIX = "API:"
MAX_PANEL_IMAGE_MODELS = 8

DEPENDENCY_PROFILES = {
    'LOCAL_PBR': [
        ("numpy", "numpy", "数值计算（本地 PBR）"),
    ],
    'API': [
        ("PIL", "Pillow", "图像解码增强"),
        ("requests", "requests", "API 通信增强"),
    ],
    'COMFYUI': [
        ("PIL", "Pillow", "图像解码增强"),
        ("requests", "requests", "ComfyUI HTTP 通信增强"),
        ("websocket", "websocket-client", "ComfyUI 实时进度增强"),
    ],
    'INSTALLER': [
        ("py7zr", "py7zr", "7z 解压（可选）"),
    ],
}


def resolve_asset_output_path(prefs) -> str:
    """解析插件的资源输出路径为绝对路径。

    返回空字符串表示用户未设置输出目录。
    """
    asset_output_path = getattr(prefs, "asset_output_path", "")
    if asset_output_path.startswith("//"):
        return bpy.path.abspath(asset_output_path)
    if not asset_output_path:
        return ""
    return os.path.abspath(asset_output_path)


def get_default_asset_output_path() -> str:
    """首次安装时的默认资源输出目录：系统 Pictures 下的 AI_Texture_Generator 子目录。"""
    return os.path.join(os.path.expanduser("~"), "Pictures", "AI_Texture_Generator")


def ensure_default_asset_output_path(prefs) -> None:
    """如果用户尚未设置资源输出目录，则填入系统 Pictures 默认路径并自动创建。"""
    if not getattr(prefs, "asset_output_path", ""):
        prefs.asset_output_path = get_default_asset_output_path()
    default_path = get_default_asset_output_path()
    if prefs.asset_output_path == default_path:
        try:
            os.makedirs(default_path, exist_ok=True)
        except Exception:
            pass


class AIAPIModelItem(bpy.types.PropertyGroup):
    model_id: bpy.props.StringProperty(name="模型 ID")
    label: bpy.props.StringProperty(name="标签")
    model_kind: bpy.props.EnumProperty(
        name="类型",
        items=[
            ('IMAGE', "图像", ""),
            ('TEXT', "文本", ""),
            ('VIDEO', "视频", ""),
            ('OTHER', "其他", ""),
        ],
        default='OTHER',
    )


class AIComfyUIModelItem(bpy.types.PropertyGroup):
    """本地 ComfyUI 中扫描到的模型文件缓存项。"""
    model_id: bpy.props.StringProperty(name="模型 ID")
    label: bpy.props.StringProperty(name="标签")
    model_kind: bpy.props.StringProperty(name="类型")
    model_family: bpy.props.StringProperty(name="模型族")


def _classify_local_model(filename: str, model_kind: str) -> str:
    """根据内置注册表、文件名和文件夹判断模型属于哪个族。

    返回空字符串表示未知/未分类。"""
    lower = filename.lower()

    # 1. 优先匹配内置下载注册表（包括 alt_filenames）
    for entry in comfyui_installer.get_model_registry():
        if entry.get("filename") == filename:
            return entry.get("family", "")
        for alt in entry.get("alt_filenames", []):
            if alt == filename:
                return entry.get("family", "")

    # 2. 文件夹强提示
    if model_kind == "vae":
        return "vae"
    if model_kind in ("clip", "text_encoders"):
        return "text_encoder"

    # 3. 文件名关键字启发式（仅对生图主模型类生效）
    if any(k in lower for k in ("flux2", "flux.2", "flux_2")):
        return "flux2"
    if any(k in lower for k in ("flux1", "flux.1", "flux_1")):
        return "flux1"
    if any(k in lower for k in ("z_image", "zimage", "z-image", "lumina", "aura")):
        return "zimage"
    if "sdxl" in lower:
        return "sdxl"

    return ""


class AIAPIProviderItem(bpy.types.PropertyGroup):
    provider_id: bpy.props.StringProperty(name="Provider ID")
    name: bpy.props.StringProperty(name="名称", default="自定义 API")
    protocol: bpy.props.EnumProperty(
        name="协议",
        items=[
            ('OPENAI_COMPATIBLE', "OpenAI 兼容", ""),
            ('GEMINI', "Gemini", ""),
        ],
        default='OPENAI_COMPATIBLE',
    )
    api_key: bpy.props.StringProperty(name="API Key", subtype='PASSWORD')
    base_url: bpy.props.StringProperty(name="Base URL", default="https://api.openai.com/v1")
    default_image_model: bpy.props.StringProperty(name="默认图像模型", default="gpt-image-2")
    default_text_model: bpy.props.StringProperty(name="默认文本模型", default="gpt-5.5")
    default_vision_model: bpy.props.StringProperty(name="默认视觉模型", default="")
    models: bpy.props.CollectionProperty(type=AIAPIModelItem)
    active_model_index: bpy.props.IntProperty(name="Active Model", default=0)


def _new_provider_id(prefix: str = "custom") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def _provider_value(provider_id: str) -> str:
    return f"{API_PROVIDER_PREFIX}{provider_id}"


def is_api_provider_value(value: str) -> bool:
    return bool(value and value.startswith(API_PROVIDER_PREFIX))


def provider_id_from_value(value: str) -> str:
    return value[len(API_PROVIDER_PREFIX):] if is_api_provider_value(value) else ""


def find_api_provider(prefs, provider_id: str):
    for provider in prefs.api_providers:
        if provider.provider_id == provider_id:
            return provider
    return None


def selected_model_or_default(selected: str, default: str) -> str:
    if selected and selected not in {'DEFAULT', 'NONE'}:
        return selected
    return default


def dependency_missing_for_profile(profile: str) -> list:
    deps = []
    if profile == 'ALL':
        seen = set()
        for items in DEPENDENCY_PROFILES.values():
            for dep in items:
                if dep[0] not in seen:
                    deps.append(dep)
                    seen.add(dep[0])
    else:
        deps = DEPENDENCY_PROFILES.get(profile, [])

    missing = []
    for import_name, pip_name, desc in deps:
        try:
            __import__(import_name)
        except ImportError:
            missing.append((import_name, pip_name, desc))
    return missing


def _requests_module():
    try:
        import requests
        return requests
    except ImportError:
        from .utils import simple_requests
        return simple_requests


def ensure_default_api_providers(prefs):
    """只预置一个空的 ModelScope provider 作为示例，用户可自行新增/编辑。"""
    existing = {provider.provider_id for provider in prefs.api_providers}

    if "modelscope" not in existing:
        provider = prefs.api_providers.add()
        provider.provider_id = "modelscope"
        provider.name = "ModelScope"
        provider.protocol = 'OPENAI_COMPATIBLE'
        provider.api_key = ""
        provider.base_url = "https://api-inference.modelscope.cn/v1"
        provider.default_image_model = "Tongyi-MAI/Z-Image-Turbo"
        provider.default_text_model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        provider.default_vision_model = "Qwen/Qwen2-VL-72B-Instruct"


def _addon_prefs_from_context(context):
    if not context:
        return None
    addon_pkg = __package__.split('.')[0]
    addon = context.preferences.addons.get(addon_pkg)
    if not addon:
        return None
    prefs = addon.preferences
    ensure_default_api_providers(prefs)
    return prefs


def _api_provider_enum_items(context):
    prefs = _addon_prefs_from_context(context)
    if not prefs:
        return [(_provider_value("openai"), "OpenAI", "")]
    items = []
    for provider in prefs.api_providers:
        if not provider.provider_id:
            provider.provider_id = _new_provider_id("api")
        items.append((_provider_value(provider.provider_id), provider.name or "Custom API", provider.base_url))
    return items or [(_provider_value("modelscope"), "ModelScope", "")]


def get_texture_provider_items(self, context):
    items = []
    # 仅在检测到本地 ComfyUI 已安装时才在面板中显示该后端
    try:
        addon_pkg = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_pkg].preferences
        install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()
        if comfyui_installer.is_comfyui_installed(install_path):
            items.append(('LOCAL_COMFYUI', "Local ComfyUI", "Use local ZImage+CHORD workflow"))
    except Exception:
        pass
    return items + _api_provider_enum_items(context)


def _selected_api_provider_from_props(props, context):
    provider_value = props.texture_generator
    if not is_api_provider_value(provider_value):
        return None
    prefs = _addon_prefs_from_context(context)
    if not prefs:
        return None
    return find_api_provider(prefs, provider_id_from_value(provider_value))


def _model_enum_items(provider, model_kind: str):
    if not provider:
        return [('DEFAULT', "Use provider default", "")]
    default_model = provider.default_text_model if model_kind == 'TEXT' else provider.default_image_model
    items = [('DEFAULT', f"Default: {default_model or 'Not set'}", "")]
    wanted = {'TEXT'} if model_kind == 'TEXT' else {'IMAGE'}
    models = [model for model in provider.models if model.model_kind in wanted and model.model_id]
    if model_kind == 'IMAGE':
        core_order = [
            provider.default_image_model,
            "gpt-image-2",
            "gpt-image-1",
            "dall-e-3",
            "Tongyi-MAI/Z-Image-Turbo",
            "Qwen/Qwen-Image-2512",
            "Qwen/Qwen-Image-Edit-2511",
            "black-forest-labs/FLUX.2-klein-9B",
            "gemini-2.5-flash-image",
            "gemini-2.0-flash-exp-image-generation",
        ]

        def rank(model):
            mid = model.model_id
            try:
                return core_order.index(mid)
            except ValueError:
                return len(core_order)

        models = sorted(models, key=lambda model: (rank(model), model.model_id.lower()))[:MAX_PANEL_IMAGE_MODELS]

    for model in models:
        model_id = model.model_id
        items.append((model_id, model.label or model_id, model.model_kind.title()))
    return items


def get_api_image_model_items(self, context):
    return _model_enum_items(_selected_api_provider_from_props(self, context), 'IMAGE')


def get_api_text_model_items(self, context):
    return _model_enum_items(_selected_api_provider_from_props(self, context), 'TEXT')


def find_any_configured_api_provider(prefs, context) -> dict | None:
    """Return a snapshot of the first API provider with a valid API key, or None.

    Shared helper used by operators that need an API provider but don't have
    one explicitly selected.
    """
    if not prefs.api_providers:
        return None
    for provider in prefs.api_providers:
        if not provider.api_key:
            continue
        candidate = get_selected_api_provider_snapshot(
            context,
            f"{API_PROVIDER_PREFIX}{provider.provider_id}",
            "DEFAULT",
            "DEFAULT",
        )
        if candidate and candidate.get("api_key"):
            return candidate
    return None


def get_selected_api_provider_snapshot(context, provider_value: str, image_model_selection: str = "DEFAULT", text_model_selection: str = "DEFAULT") -> dict:
    prefs = _addon_prefs_from_context(context)
    provider = find_api_provider(prefs, provider_id_from_value(provider_value)) if prefs and is_api_provider_value(provider_value) else None
    if not provider:
        return {}
    return {
        "provider_id": provider.provider_id,
        "name": provider.name,
        "protocol": provider.protocol,
        "api_key": provider.api_key,
        "base_url": provider.base_url,
        "image_model": selected_model_or_default(image_model_selection, provider.default_image_model),
        "text_model": selected_model_or_default(text_model_selection, provider.default_text_model),
        "vision_model": provider.default_vision_model,
    }


class AI_OT_TestProviderConnection(bpy.types.Operator):
    bl_idname = "ai_concept.test_provider_connection"
    bl_label = "测试连接"
    bl_description = "测试本地 ComfyUI 是否安装正确，或远程后端是否可连接"
    bl_options = {'REGISTER', 'INTERNAL'}

    provider: bpy.props.StringProperty(default='COMFYUI')

    def execute(self, context):
        requests = _requests_module()

        addon_pkg = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_pkg].preferences

        try:
            if self.provider == 'COMFYUI':
                url = prefs.comfyui_url.rstrip("/")
                install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()
                ctype = comfyui_installer.get_comfyui_type(install_path)

                from .sd_backend.comfyui_client import ComfyUIClient
                from .sd_backend import workflow_specs as _wf_specs
                client = ComfyUIClient(base_url=url)

                # 本地已安装：尝试连接；桌面版不自动启动，需要用户先打开 Desktop 应用
                if ctype:
                    if ctype == "desktop":
                        self.report({'INFO'}, f"正在测试 ComfyUI Desktop 连接 ({url})...")
                        if not client.check_health():
                            self.report(
                                {'ERROR'},
                                f"ComfyUI Desktop 未在 {url} 响应。请先启动 ComfyUI Desktop，"
                                f"并确认偏好设置中的 URL 与其监听地址一致。",
                            )
                            return {'CANCELLED'}
                    else:
                        self.report({'INFO'}, f"正在测试 ComfyUI {ctype} 连接，必要时将自动启动...")
                        if not client.check_health(auto_launch_path=install_path):
                            self.report({'ERROR'}, f"ComfyUI {ctype} 启动或连接失败，请检查安装与日志: {install_path}")
                            return {'CANCELLED'}
                else:
                    # 本地未安装时，探测 URL 是否已有运行实例
                    try:
                        resp = requests.get(f"{url}/system_stats", timeout=(3, 10))
                        resp.raise_for_status()
                    except requests.exceptions.ReadTimeout:
                        self.report({'WARNING'}, f"{url} 已连接但响应读取超时，ComfyUI 可能正在初始化，请稍后再试")
                        return {'CANCELLED'}
                    except Exception as e:
                        self.report({'ERROR'}, f"未检测到 ComfyUI 安装，且 {url} 无法连接: {e}")
                        return {'CANCELLED'}

                # 连接可达后，先按模型族所需的最低核心版本做快速判断
                props = getattr(context.scene, "ai_concept_props", None)
                family_id = getattr(props, "local_comfyui_family", "zimage") if props else "zimage"
                core_version = client.get_comfyui_core_version()
                min_version_str = _wf_specs.get_family_min_version(family_id)
                if core_version and min_version_str:
                    min_version = tuple(int(x) for x in min_version_str.split(".") if x.isdigit())
                    if len(min_version) >= 2 and core_version < min_version:
                        actual_str = ".".join(str(x) for x in core_version)
                        self.report(
                            {'ERROR'},
                            f"当前 ComfyUI 核心版本为 {actual_str}，但模型族 '{family_id}' "
                            f"需要核心版本 {min_version_str}+。\n"
                            "请先升级 ComfyUI 核心；若使用 ComfyUI Desktop，"
                            "请在应用内更新核心版本（应用版本号不等于核心版本）。",
                        )
                        return {'CANCELLED'}

                # 校验当前模型族 workflow 所需节点完整性
                missing = client.check_workflow_nodes(family_id=family_id)
                if missing:
                    lines = [f"{cls_type} ({hint})" if hint else cls_type for _nid, cls_type, hint in missing]
                    self.report(
                        {'ERROR'},
                        f"ComfyUI 缺少当前模型族 ({family_id}) 所需节点:\n" + "\n".join(lines[:8]),
                    )
                    return {'CANCELLED'}

                ver_str = ".".join(str(x) for x in core_version) if core_version else "未知"
                self.report({'INFO'}, f"ComfyUI 连接正常 (核心版本 {ver_str}) 且节点完整 ({family_id}): {url}")
                return {'FINISHED'}

            if is_api_provider_value(self.provider):
                ensure_default_api_providers(prefs)
                provider = find_api_provider(prefs, provider_id_from_value(self.provider))
                if not provider:
                    self.report({'ERROR'}, "未找到 API Provider")
                    return {'CANCELLED'}
                if not provider.api_key:
                    self.report({'ERROR'}, f"{provider.name} API Key 为空")
                    return {'CANCELLED'}
                if provider.protocol == 'GEMINI':
                    # SECURITY: Gemini API Key 通过 URL query param 传递
                    url = AI_OT_FetchModels._models_url(provider.base_url, "gemini")
                    resp = requests.get(f"{url}?key={provider.api_key}", timeout=10)
                else:
                    url = AI_OT_FetchModels._models_url(provider.base_url, "openai")
                    resp = requests.get(
                        url,
                        headers=AI_OT_FetchModels._model_headers(provider.api_key, "openai"),
                        timeout=10,
                    )
                if resp.status_code >= 400:
                    self.report({'WARNING'}, f"{provider.name} /models 返回 HTTP {resp.status_code}")
                    return {'FINISHED'}
                self.report({'INFO'}, f"{provider.name} /models 响应正常。生成/对话端点可能需要配置正确的默认模型。")
                return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"连接测试失败: {e}")
            return {'CANCELLED'}

        return {'CANCELLED'}


class AI_OT_AddAPIProvider(bpy.types.Operator):
    bl_idname = "ai_concept.add_api_provider"
    bl_label = "新增API"
    bl_description = "Add a new API provider configuration"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        prefs = context.preferences.addons[__package__.split('.')[0]].preferences
        ensure_default_api_providers(prefs)
        provider = prefs.api_providers.add()
        provider.provider_id = _new_provider_id("custom")
        provider.name = "Custom API"
        provider.protocol = 'OPENAI_COMPATIBLE'
        provider.base_url = "https://api.openai.com/v1"
        provider.default_image_model = "gpt-image-2"
        provider.default_text_model = "gpt-5.5"
        provider.default_vision_model = "gpt-4o"
        prefs.active_api_provider_index = len(prefs.api_providers) - 1
        context.preferences.is_dirty = True
        return {'FINISHED'}


class AI_OT_RemoveAPIProvider(bpy.types.Operator):
    bl_idname = "ai_concept.remove_api_provider"
    bl_label = "删除 API"
    bl_description = "删除此 API Provider 配置"
    bl_options = {'REGISTER', 'INTERNAL'}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        prefs = context.preferences.addons[__package__.split('.')[0]].preferences
        ensure_default_api_providers(prefs)
        idx = self.index if self.index >= 0 else prefs.active_api_provider_index
        if idx < 0 or idx >= len(prefs.api_providers):
            return {'CANCELLED'}
        prefs.api_providers.remove(idx)
        prefs.active_api_provider_index = min(max(0, idx - 1), max(0, len(prefs.api_providers) - 1))
        context.preferences.is_dirty = True
        return {'FINISHED'}


class AI_OT_FetchModels(bpy.types.Operator):
    bl_idname = "ai_concept.fetch_models"
    bl_label = "获取模型列表"
    bl_description = "从 API 拉取可用模型列表。支持 OpenAI / Gemini / ModelScope 等多种协议"
    bl_options = {'REGISTER', 'INTERNAL'}

    # 模型名称关键词分类（参考 Infinite-Canvas）
    IMAGE_KEYWORDS = [
        "image", "dalle", "dall-e", "imagen", "flux", "stable", "sdxl",
        "midjourney", "banana", "z-image", "qwen-image",
        "klein", "seedream", "text-to-image", "gpt-image"
    ]

    TEXT_KEYWORDS = [
        "gpt-4", "gpt-5", "chat", "llm", "qwen", "deepseek", "llama",
        "instruct", "text", "coder", "glm", "yi-", "baichuan", "gemini"
    ]
    VIDEO_KEYWORDS = ["video", "wan", "sora", "kling", "animate"]
    VISION_KEYWORDS = [
        "gpt-4o", "gpt-4.1", "claude-3", "claude-3-5", "claude-3-7",
        "gemini-1.5", "gemini-2", "gemini-pro-vision",
        "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
        "llava", "internvl", "minicpm-v", "yi-vl", "glm-4v", "cogvlm"
    ]

    MODELSCOPE_DEFAULT_MODELS = [
        "Tongyi-MAI/Z-Image-Turbo",
        "Qwen/Qwen-Image-2512",
        "Qwen/Qwen-Image-Edit-2511",
        "black-forest-labs/FLUX.2-klein-9B",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]

    # 兜底模型列表
    DEFAULT_OPENAI_MODELS = ["gpt-image-2", "gpt-image-1", "dall-e-3", "dall-e-2", "gpt-4o", "gpt-4o-mini"]
    DEFAULT_GEMINI_MODELS = ["gemini-2.5-flash-image", "gemini-2.0-flash-exp-image-generation", "gemini-2.5-flash"]

    provider_id: bpy.props.StringProperty(default="")
    provider_index: bpy.props.IntProperty(default=-1)

    @staticmethod
    def _looks_like_html(text: str) -> bool:
        t = (text or "").strip()
        return t.startswith("<") or "<html" in t.lower() or "<!doctype" in t.lower()

    @classmethod
    def _classify_model(cls, mid: str) -> str:
        lc = mid.lower()
        for k in cls.IMAGE_KEYWORDS:
            if k in lc:
                return "IMAGE"
        for k in cls.VIDEO_KEYWORDS:
            if k in lc:
                return "VIDEO"
        for k in cls.TEXT_KEYWORDS:
            if k in lc:
                return "TEXT"
        return "OTHER"

    @classmethod
    def _is_vision_model(cls, mid: str) -> bool:
        lc = mid.lower()
        return any(k in lc for k in cls.VISION_KEYWORDS)

    @classmethod
    def _filter_image_models(cls, models: list) -> list:
        return [m for m in models if cls._classify_model(m) == "IMAGE"]

    @classmethod
    def _pick_first_available(cls, candidates: list, available: list) -> str:
        available_set = {m.lower() for m in available}
        for c in candidates:
            if c.lower() in available_set:
                return c
        return ""

    @classmethod
    def _auto_select_best_models(cls, provider, models: list, vision_candidates: list):
        """根据 provider 类型和可用模型列表，自动设置最佳 image/text/vision 默认值。"""
        base_url = (provider.base_url or "").lower()
        is_modelscope = "modelscope" in base_url
        is_gemini = provider.protocol == 'GEMINI' or "googleapis" in base_url

        if is_modelscope:
            image_candidates = [
                "Tongyi-MAI/Z-Image-Turbo",
                "Qwen/Qwen-Image-2512",
                "Qwen/Qwen-Image-Edit-2511",
                "black-forest-labs/FLUX.2-klein-9B",
            ]
            text_candidates = [
                "Qwen/Qwen3-235B-A22B-Instruct-2507",
                "Qwen/Qwen3-235B-A22B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "Qwen/Qwen2-72B-Instruct",
                "Qwen/Qwen3-32B-Instruct",
            ]
            vision_candidates_ordered = [
                "Qwen/Qwen3-VL-72B-Instruct",
                "Qwen/Qwen3-VL-8B-Instruct",
                "Qwen/Qwen2-VL-72B-Instruct",
                "Qwen/Qwen2.5-VL-32B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ]
        elif is_gemini:
            image_candidates = [
                "gemini-2.5-flash-image",
                "gemini-2.0-flash-exp-image-generation",
            ]
            text_candidates = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
            ]
            vision_candidates_ordered = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
        else:
            # OpenAI-compatible / 官方 OpenAI / 第三方代理
            image_candidates = [
                "gpt-image-2",
                "gpt-image-1",
                "dall-e-3",
                "dall-e-2",
            ]
            text_candidates = [
                "gpt-5.5",
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]
            vision_candidates_ordered = [
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4o-mini",
                "claude-3-opus",
                "claude-3-5-sonnet",
                "claude-3-sonnet",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "qwen-vl-max",
                "qwen2-vl-72b-instruct",
            ]

        provider.default_image_model = cls._pick_first_available(image_candidates, models) or provider.default_image_model
        provider.default_text_model = cls._pick_first_available(text_candidates, models) or provider.default_text_model
        provider.default_vision_model = cls._pick_first_available(vision_candidates_ordered, vision_candidates) or provider.default_vision_model

    @staticmethod
    def _models_url(base_url: str, protocol: str = "openai") -> str:
        base = base_url.rstrip("/")
        if base.endswith("/models"):
            return base
        if protocol == "gemini":
            return f"{base}/models" if base.endswith("/v1beta") else f"{base}/v1beta/models"
        return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"

    @staticmethod
    def _model_headers(api_key: str, protocol: str = "openai") -> dict:
        if protocol == "gemini":
            return {"Accept": "application/json"}  # Gemini 用 query param 传 key
        return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    def _fetch_openai_models(self, url: str, api_key: str) -> list:
        """请求 OpenAI 兼容 /models，返回原始模型 ID 列表。"""
        requests = _requests_module()
        headers = self._model_headers(api_key, "openai")
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location") or resp.headers.get("location") or ""
            raise RuntimeError(f"发生跳转{('：' + loc) if loc else ''}，请检查 Base URL 是否为 API 地址")
        if self._looks_like_html(resp.text):
            raise RuntimeError("返回网页 HTML，该端点不支持 /models（常见于第三方代理）")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        ct = resp.headers.get("Content-Type", "")
        if "json" not in ct.lower():
            raise RuntimeError(f"返回非 JSON ({ct}): {resp.text[:300]}")
        data = resp.json()
        models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
        return models

    def _fetch_gemini_models(self, url: str, api_key: str) -> list:
        """请求 Gemini /v1beta/models，返回模型名列表。"""
        requests = _requests_module()
        req_url = f"{self._models_url(url, 'gemini')}?key={api_key}"
        resp = requests.get(req_url, timeout=15)
        if self._looks_like_html(resp.text):
            raise RuntimeError("返回网页 HTML，该端点不支持 /models")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        models = [m["name"].split("/")[-1] for m in data.get("models", []) if isinstance(m, dict) and "name" in m]
        return models

    def _handle_fetch(self, provider):
        """根据 provider 协议选择正确的拉取策略，返回 (models_list, message)。"""
        if provider.protocol == 'GEMINI':
            base_url = provider.base_url
            api_key = provider.api_key
            models = self._fetch_gemini_models(base_url, api_key)
            msg = f"Gemini /v1beta/models 返回 {len(models)} 个模型"
        else:
            url = self._models_url(provider.base_url, "openai")
            models = self._fetch_openai_models(url, provider.api_key)
            msg = f"OpenAI-compatible /v1/models 返回 {len(models)} 个模型"
        return models, msg

    def execute(self, context):
        addon_pkg = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_pkg].preferences
        ensure_default_api_providers(prefs)

        provider = None
        if self.provider_id:
            provider = find_api_provider(prefs, self.provider_id)
        elif 0 <= self.provider_index < len(prefs.api_providers):
            provider = prefs.api_providers[self.provider_index]
        elif 0 <= prefs.active_api_provider_index < len(prefs.api_providers):
            provider = prefs.api_providers[prefs.active_api_provider_index]

        if not provider:
            self.report({'ERROR'}, "未找到 API Provider")
            return {'CANCELLED'}
        if not provider.api_key:
            self.report({'ERROR'}, f"{provider.name} API Key 为空")
            return {'CANCELLED'}

        try:
            models, msg = self._handle_fetch(provider)
        except Exception as e:
            if "modelscope" in provider.base_url.lower():
                models = self.MODELSCOPE_DEFAULT_MODELS
                msg = f"ModelScope 拉取失败 ({e})，已使用内置默认模型列表"
            elif provider.protocol == 'GEMINI':
                models = self.DEFAULT_GEMINI_MODELS
                msg = f"Gemini 拉取失败 ({e})，已使用默认模型列表"
            else:
                models = self.DEFAULT_OPENAI_MODELS
                msg = f"{provider.name} 拉取失败 ({e})，已使用默认模型列表"

        provider.models.clear()
        vision_candidates = []
        for model_id in sorted(set(models)):
            item = provider.models.add()
            item.model_id = model_id
            item.label = model_id
            item.model_kind = self._classify_model(model_id)
            if self._is_vision_model(model_id):
                vision_candidates.append(model_id)

        # 自动选择最佳默认模型
        self._auto_select_best_models(provider, models, vision_candidates)

        info_msg = f"{provider.name}: {msg}，已记录 {len(provider.models)} 个模型"
        if provider.default_vision_model:
            info_msg += f"，vision={provider.default_vision_model}"
        self.report({'INFO'}, info_msg)

        context.preferences.is_dirty = True

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

        return {'FINISHED'}


class AI_OT_RefreshComfyUIModels(bpy.types.Operator):
    """扫描本地 ComfyUI 的 models 目录，刷新可用模型缓存。"""
    bl_idname = "ai_concept.refresh_comfyui_models"
    bl_label = "刷新本地模型"
    bl_description = "扫描 ComfyUI/models 目录，更新本地模型下拉列表"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        addon_pkg = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_pkg].preferences
        install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()

        prefs.comfyui_models.clear()
        scanned = comfyui_installer.scan_comfyui_models(install_path)
        total = 0
        for subdir, files in sorted(scanned.items()):
            for filename in files:
                item = prefs.comfyui_models.add()
                item.model_id = filename
                item.label = filename
                item.model_kind = subdir
                item.model_family = _classify_local_model(filename, subdir)
                total += 1

        self.report({'INFO'}, f"已扫描到 {total} 个本地模型")
        context.preferences.is_dirty = True
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
        return {'FINISHED'}


class AI_OT_RepairDependencies(bpy.types.Operator):
    bl_idname = "ai_concept.repair_dependencies"
    bl_label = "一键修复环境依赖"
    bl_description = "安装当前功能所需的 Python 组件"
    bl_options = {'REGISTER', 'INTERNAL'}

    profile: bpy.props.EnumProperty(
        name="修复范围",
        items=[
            ('API', "API 增强组件", ""),
            ('COMFYUI', "本地 ComfyUI 增强组件", ""),
            ('LOCAL_PBR', "本地 PBR 组件", ""),
            ('INSTALLER', "安装器增强组件", ""),
            ('ALL', "全部推荐组件", ""),
        ],
        default='API',
    )

    def execute(self, context):
        addon_pkg = __package__.split('.')[0]
        addon_mod = importlib.import_module(addon_pkg)
        if hasattr(addon_mod, "_inject_vendor_site"):
            addon_mod._inject_vendor_site()

        missing = dependency_missing_for_profile(self.profile)
        if not missing:
            self.report({'INFO'}, "所需组件已就绪")
            return {'FINISHED'}

        missing_display = [f"{import_name} ({desc})" for import_name, _pip_name, desc in missing]

        def verifier():
            return [f"{i} ({d})" for i, _p, d in dependency_missing_for_profile(self.profile)]

        ok, msg = addon_mod._auto_install_dependencies(missing_display, verifier=verifier)
        if ok:
            if self.profile in {'LOCAL_PBR', 'ALL'}:
                msg += "。基础组件修复后请重新启用插件或重启 Blender。"
            self.report({'INFO'}, msg)
            return {'FINISHED'}
        self.report({'ERROR'}, msg)
        return {'CANCELLED'}


class CTAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    api_providers: bpy.props.CollectionProperty(type=AIAPIProviderItem)
    active_api_provider_index: bpy.props.IntProperty(name="当前 API Provider", default=0)

    comfyui_models: bpy.props.CollectionProperty(type=AIComfyUIModelItem)

    comfyui_url: bpy.props.StringProperty(
        name="ComfyUI 地址",
        default="http://127.0.0.1:8188",
    )

    comfyui_path: bpy.props.StringProperty(
        name="ComfyUI 安装路径",
        description="ComfyUI 根目录。留空时使用插件目录下的 comfyui_portable",
        subtype='DIR_PATH',
        default="",
    )

    # OpenAI-compatible image APIs (旧版兼容字段)
    gpt_api_key: bpy.props.StringProperty(
        name="API Key（旧版）",
        description="旧版 OpenAI 兼容 API Key。建议使用上方 API Provider 管理",
        subtype='PASSWORD',
    )
    gpt_model: bpy.props.StringProperty(
        name="默认图像模型（旧版）",
        description="旧版默认模型。建议使用上方 API Provider 管理",
        default="gpt-image-2",
    )
    gpt_base_url: bpy.props.StringProperty(
        name="Base URL（旧版）",
        description="旧版 API 地址。建议使用上方 API Provider 管理",
        default="https://api.openai.com/v1",
    )
    prompt_optimize_api_key: bpy.props.StringProperty(
        name="提示词优化 API Key（旧版）",
        description="旧版提示词优化密钥。建议使用上方 API Provider 管理",
        subtype='PASSWORD',
    )
    prompt_optimize_base_url: bpy.props.StringProperty(
        name="提示词优化 Base URL（旧版）",
        description="旧版提示词优化地址。建议使用上方 API Provider 管理",
        default="",
    )
    prompt_optimize_model: bpy.props.StringProperty(
        name="提示词优化模型（旧版）",
        description="旧版提示词优化文本模型。建议使用上方 API Provider 管理",
        default="gpt-5.5",
    )

    # Gemini (旧版兼容槽位)
    nanobanana_api_key: bpy.props.StringProperty(
        name="Gemini API Key（旧版）",
        description="旧版 Google AI Studio / Gemini API Key。建议使用上方 API Provider 管理",
        subtype='PASSWORD',
    )
    nanobanana_url: bpy.props.StringProperty(
        name="Gemini API 地址（旧版）",
        description="旧版 Gemini API 端点。建议使用上方 API Provider 管理",
        default="https://generativelanguage.googleapis.com/v1beta",
    )
    nanobanana_model: bpy.props.StringProperty(
        name="默认 Gemini 模型（旧版）",
        description="旧版默认 Gemini 图像生成模型。建议使用上方 API Provider 管理",
        default="gemini-2.5-flash-image",
    )

    asset_output_path: bpy.props.StringProperty(
        name="资源输出目录",
        description="必填。所有生成的贴图将保存到此目录。首次安装默认填充为系统 Pictures 文件夹",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        ensure_default_api_providers(self)
        ensure_default_asset_output_path(self)

        install_path = self.comfyui_path or comfyui_installer.get_default_install_path()
        installed = comfyui_installer.is_comfyui_installed(install_path)
        has_configured_comfyui_path = bool((self.comfyui_path or "").strip())

        ctype = comfyui_installer.get_comfyui_type(install_path)
        if sys.platform != 'win32' and ctype != "desktop" and not installed:
            warn_box = layout.box()
            warn_box.alert = True
            warn_box.label(
                text="本地 ComfyUI 自动安装仅支持 Windows 便携版；macOS/Linux 或桌面版请手动启动 ComfyUI",
                icon='ERROR',
            )

        box = layout.box()
        box.label(text="本地 ComfyUI", icon='NODETREE')
        box.prop(self, "comfyui_url")
        box.prop(self, "comfyui_path")

        row = box.row(align=True)
        if not installed and not has_configured_comfyui_path:
            row.operator("ai_concept.get_comfyui", text="获取 ComfyUI", icon='URL')
        op = row.operator("ai_concept.test_provider_connection", text="测试连接", icon='CHECKMARK')
        op.provider = 'COMFYUI'
        if installed and has_configured_comfyui_path:
            op2 = row.operator("ai_concept.repair_dependencies", text="一键修复环境依赖", icon='IMPORT')
            op2.profile = 'COMFYUI'

        status_row = box.row()
        if installed:
            status_row.label(text=f"已检测到: {install_path}", icon='CHECKMARK')
        elif has_configured_comfyui_path:
            status_row.label(text="未检测到 ComfyUI，请确认路径是否正确或 ComfyUI 是否已启动", icon='ERROR')
        else:
            status_row.label(text="未检测到 ComfyUI，请通过「获取 ComfyUI」下载并解压", icon='ERROR')

        # 模型管理：只显示缺失的模型，已存在时不显示列表
        model_box = layout.box()
        model_box.label(text="模型管理", icon='PACKAGE')
        missing_models = [
            m for m in comfyui_installer.get_model_registry()
            if comfyui_installer.find_model_file(install_path, m) is None
        ]
        if not missing_models:
            row = model_box.row(align=True)
            row.label(text="所有必需模型已就绪", icon='CHECKMARK')
        else:
            model_box.label(text="请自行下载缺失模型并放到对应目录", icon='INFO')
            row = model_box.row(align=True)
            row.label(text="模型")
            row.label(text="存放目录")
            row.label(text="下载页")
            for model in missing_models:
                row = model_box.row(align=True)
                row.label(text=f"{model['label']} ({model['size']})")
                row.label(text=f"{model['dir']}/{model['filename']}")
                icon_id = icon_manager.get_icon_id("download_arrow")
                if icon_id:
                    op = row.operator("ai_concept.open_model_download_page", text="", icon_value=icon_id)
                else:
                    op = row.operator("ai_concept.open_model_download_page", text="", icon='DOWNARROW_HLT')
                op.model_id = model["id"]

        api_box = layout.box()
        api_box.label(text="API 提供商", icon='URL')
        for idx, provider in enumerate(self.api_providers):
            if not provider.provider_id:
                provider.provider_id = _new_provider_id("api")

            provider_box = api_box.box()
            header = provider_box.row(align=True)
            header.prop(provider, "name", text="")
            header.prop(provider, "protocol", text="")
            remove = header.operator("ai_concept.remove_api_provider", text="", icon='X')
            remove.index = idx

            provider_box.prop(provider, "api_key")
            provider_box.prop(provider, "base_url")

            if provider.protocol == 'GEMINI':
                warn_row = provider_box.row()
                warn_row.alert = True
                warn_row.label(
                    text="⚠ Gemini API Key 通过 URL 参数传递，可能被中间代理/日志记录。请使用低权限专属 Key。",
                    icon='ERROR',
                )

            row = provider_box.row()
            row.prop(provider, "default_image_model")
            row.prop(provider, "default_text_model")
            row.prop(provider, "default_vision_model")

            image_count = sum(1 for model in provider.models if model.model_kind == 'IMAGE')
            text_count = sum(1 for model in provider.models if model.model_kind == 'TEXT')
            other_count = len(provider.models) - image_count - text_count
            provider_box.label(text=f"模型: {image_count} 图像 / {text_count} 文本 / {other_count} 其他", icon='PRESET')

            row = provider_box.row(align=True)
            test = row.operator("ai_concept.test_provider_connection", text="测试", icon='CHECKMARK')
            test.provider = _provider_value(provider.provider_id)
            fetch = row.operator("ai_concept.fetch_models", text="获取模型", icon='FILE_REFRESH')
            fetch.provider_id = provider.provider_id

        api_box.operator("ai_concept.add_api_provider", text="新增 API", icon='ADD')

        output_box = layout.box()
        output_box.label(text="输出", icon='FILE_FOLDER')
        output_box.prop(self, "asset_output_path")



classes = [
    AIComfyUIModelItem,
    AIAPIModelItem,
    AIAPIProviderItem,
    CTAddonPreferences,
    AI_OT_TestProviderConnection,
    AI_OT_AddAPIProvider,
    AI_OT_RemoveAPIProvider,
    AI_OT_FetchModels,
    AI_OT_RefreshComfyUIModels,
    AI_OT_RepairDependencies,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            # 开发/刷新插件时可能出现重复注册，跳过已注册的类
            if "already registered" in str(e):
                log.debug("%s already registered, skipping.", cls.__name__)
                continue
            raise


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
