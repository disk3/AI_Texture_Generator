import bpy
import time
import os
import sys

from .sd_backend import comfyui_installer
from .utils.logger import get_logger

log = get_logger(__name__)

API_PROVIDER_PREFIX = "API:"
MAX_PANEL_IMAGE_MODELS = 8


def resolve_asset_output_path(prefs) -> str:
    """解析插件的资源输出路径为绝对路径。

    返回空字符串表示用户尚未设置输出目录。
    """
    asset_output_path = getattr(prefs, "asset_output_path", "")
    if asset_output_path.startswith("//"):
        return bpy.path.abspath(asset_output_path)
    if not asset_output_path:
        return ""
    return os.path.abspath(asset_output_path)


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
    bl_description = "测试选中的图像后端连接状态"
    bl_options = {'REGISTER', 'INTERNAL'}

    provider: bpy.props.StringProperty(default='COMFYUI')

    def execute(self, context):
        import requests

        addon_pkg = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_pkg].preferences

        try:
            if self.provider == 'COMFYUI':
                url = prefs.comfyui_url.rstrip("/")
                resp = requests.get(f"{url}/system_stats", timeout=5)
                resp.raise_for_status()
                self.report({'INFO'}, f"ComfyUI 已连接: {url}")
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
        # 标记 preferences 已修改并保存，避免重启后丢失
        context.preferences.is_dirty = True
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            log.debug("wm.save_userpref failed (preferences may already be saved)")
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
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            log.debug("wm.save_userpref failed (preferences may already be saved)")
        return {'FINISHED'}


class AI_OT_FetchModels(bpy.types.Operator):
    bl_idname = "ai_concept.fetch_models"
    bl_label = "获取模型列表"
    bl_description = "从 API 拉取可用模型列表。支持 OpenAI / Gemini / ModelScope 等多种协议"
    bl_options = {'REGISTER', 'INTERNAL'}

    # 模型名称关键词分类（参考 Infinite-Canvas）
    IMAGE_KEYWORDS = [
        "image", "dalle", "dall-e", "imagen", "flux", "stable", "sdxl",
        "midjourney", "banana", "ideogram", "z-image", "qwen-image",
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
        import requests
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
        import requests
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

        # 拉取的模型列表需要持久化保存
        context.preferences.is_dirty = True
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            log.debug("wm.save_userpref failed during model fetch (prefs may be auto-saved)")

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

        return {'FINISHED'}


class CTAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    api_providers: bpy.props.CollectionProperty(type=AIAPIProviderItem)
    active_api_provider_index: bpy.props.IntProperty(name="当前 API Provider", default=0)

    comfyui_url: bpy.props.StringProperty(
        name="ComfyUI 地址",
        default="http://127.0.0.1:8188",
    )

    comfyui_controlnet_tile_model: bpy.props.StringProperty(
        name="ControlNet Tile 模型",
        description="ComfyUI 中 ControlNet Tile 模型文件名，例如 control_v11f1e_sd15_tile.pth。有参考图时会注入此模型",
        default="control_v11f1e_sd15_tile.pth",
    )

    comfyui_path: bpy.props.StringProperty(
        name="ComfyUI 安装路径",
        description="ComfyUI 根目录。留空时使用插件目录下的 comfyui_portable",
        subtype='DIR_PATH',
        default="",
    )

    huggingface_token: bpy.props.StringProperty(
        name="HuggingFace Token",
        description="可选。用于自动下载 gated 模型；留空则只能下载公开模型",
        subtype='PASSWORD',
        default="",
    )

    use_china_mirror: bpy.props.BoolProperty(
        name="使用国内镜像下载",
        description="优先使用 hf-mirror.com、GitHub 代理等国内可访问镜像。如果已在国外或已翻墙可关闭",
        default=True,
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
        description="必填。所有生成的贴图将保存到此目录",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        ensure_default_api_providers(self)

        install_path = self.comfyui_path or comfyui_installer.get_default_install_path()
        installed = comfyui_installer.is_comfyui_installed(install_path)

        if sys.platform != 'win32' and not installed:
            warn_box = layout.box()
            warn_box.alert = True
            warn_box.label(
                text="本地 ComfyUI 自动安装/启动仅支持 Windows；macOS/Linux 请手动配置 ComfyUI",
                icon='ERROR',
            )

        box = layout.box()
        box.label(text="本地 ComfyUI", icon='NODETREE')
        box.prop(self, "comfyui_url")
        box.prop(self, "comfyui_path")

        row = box.row()
        if installed:
            row.label(text=f"已检测到: {install_path}", icon='CHECKMARK')
        else:
            row.label(text="未检测到 ComfyUI", icon='ERROR')
            row = box.row()
            row.operator("ai_concept.install_comfyui", text="自动下载安装", icon='IMPORT')

        # 仅安装完成且指定了路径时才显示测试与 ControlNet Tile 配置
        if installed and self.comfyui_path:
            row = box.row(align=True)
            op = row.operator("ai_concept.test_provider_connection", text="测试", icon='CHECKMARK')
            op.provider = 'COMFYUI'
            box.prop(self, "comfyui_controlnet_tile_model")

        # Model Manager：仅在 ComfyUI 安装完成后显示
        if installed:
            model_box = layout.box()
            model_box.label(text="模型管理", icon='PACKAGE')
            model_box.prop(self, "use_china_mirror")
            model_box.prop(self, "huggingface_token")
            model_box.label(text="公开模型可直接下载；gated 模型需填 HuggingFace Token 或手动下载")
            for model in comfyui_installer.get_model_registry():
                actual_path = comfyui_installer.find_model_file(install_path, model)
                is_installed = actual_path is not None
                col = model_box.column(align=True)
                row = col.row(align=True)
                icon = 'CHECKMARK' if is_installed else 'CANCEL'
                gated_text = " [gated]" if model.get("gated") else ""
                row.label(text=f"{model['label']} ({model['size']}){gated_text}", icon=icon)
                if not is_installed:
                    if model.get("url"):
                        op = row.operator("ai_concept.download_model", text="下载", icon='IMPORT')
                        op.model_id = model["id"]
                op2 = row.operator("ai_concept.open_model_download_page", text="网页", icon='URL')
                op2.model_id = model["id"]
                # 显示模型应存放路径或实际找到的路径
                path_row = col.row()
                path_row.scale_y = 0.8
                if is_installed and actual_path:
                    rel_path = os.path.relpath(actual_path, install_path).replace("\\", "/")
                    path_row.label(text=f"  ✓ {rel_path}", icon='FILE_FOLDER')
                else:
                    path_row.label(text=f"  → {model['dir']}/{model['filename']}", icon='FILE_FOLDER')

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
    AIAPIModelItem,
    AIAPIProviderItem,
    CTAddonPreferences,
    AI_OT_TestProviderConnection,
    AI_OT_AddAPIProvider,
    AI_OT_RemoveAPIProvider,
    AI_OT_FetchModels,
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
