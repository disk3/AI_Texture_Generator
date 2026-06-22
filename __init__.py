bl_info = {
    "name": "AI 材质生成",
    "author": "Xiong Meng Han",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D 视图 > 侧边栏 (N) > AI 材质",
    "description": "从提示词或参考图一键生成 PBR 材质贴图（支持本地 ComfyUI 与在线 API）",
    "doc_url": "https://github.com/xiongmenghan/BlenderAIGC",
    "tracker_url": "https://github.com/xiongmenghan/BlenderAIGC/issues",
    "category": "3D View",
    "support": "COMMUNITY",
}

import atexit
import bpy
import sys

from .utils.logger import get_logger

log = get_logger(__name__)
modules = []

# ── 运行时依赖声明 ──────────────────────────────────────────────────────────
# Blender 内置 Python 已提供: bpy, mathutils, bmesh, math, os, json, io, base64,
#   queue, threading, time, uuid, random, subprocess, shutil, tempfile, urllib,
#   logging, ctypes, re, webbrowser, signal, abc, dataclasses, typing
# 以下为需要用户自行安装的第三方包（安装到 Blender 的 site-packages）:
_REQUIRED_PACKAGES = {
    "requests":        "HTTP 客户端（API 调用、ComfyUI 通信）",
    "PIL":             "图像处理（贴图读写、格式转换）",
    "numpy":           "数值计算（贴图像素处理）",
    "websocket":       "WebSocket 客户端（ComfyUI 实时通信）",
}
_OPTIONAL_PACKAGES = {
    "cv2":             "OpenCV（法线贴图 Sobel 算法，无此包将回退到简化实现）",
    "py7zr":           "7z 解压库（ComfyUI 安装，无此包将尝试系统 7z 命令）",
}


def _check_runtime_dependencies() -> list:
    """检查必需的第三方依赖是否可用，返回缺失包名列表。"""
    missing = []
    for pkg, desc in _REQUIRED_PACKAGES.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"{pkg} ({desc})")
    # 可选包仅记录日志，不阻止加载
    for pkg, desc in _OPTIONAL_PACKAGES.items():
        try:
            __import__(pkg)
        except ImportError:
            log.debug("Optional package '%s' not found: %s", pkg, desc)
    return missing


def _show_dependency_warning(missing: list):
    """通过 Blender 弹窗提示用户安装缺失的依赖。"""
    import textwrap
    lines = [
        "AI 材质生成插件检测到缺失的 Python 包：",
        "",
    ]
    for m in missing:
        lines.append(f"  • {m}")
    lines.extend([
        "",
        "请将以下包安装到 Blender 的 Python 环境中：",
        "",
        f"  {sys.executable} -m pip install requests Pillow numpy websocket-client",
        "",
        "可选（推荐安装以获得最佳效果）：",
        "",
        f"  {sys.executable} -m pip install opencv-python py7zr",
        "",
        "安装后请重启 Blender 并重新启用插件。",
    ])

    def _draw_warning(self_dummy, context):
        layout = self_dummy.layout
        for line in textwrap.wrap("\n".join(lines), width=90):
            layout.label(text=line)

    bpy.context.window_manager.popup_menu(_draw_warning, title="依赖缺失", icon='ERROR')


def _resolve_comfyui_url() -> str:
    """Safely read the configured ComfyUI URL from add-on preferences."""
    try:
        addon_pkg = __name__.split('.')[0]
        addon = bpy.context.preferences.addons.get(addon_pkg)
        if addon and hasattr(addon.preferences, "comfyui_url"):
            return addon.preferences.comfyui_url or "http://127.0.0.1:8188"
    except Exception:
        log.debug("Could not read ComfyUI URL from preferences")
    return "http://127.0.0.1:8188"


def _shutdown_on_exit():
    """Blender 进程退出时的兜底清理：强制关闭插件启动的 ComfyUI。

    此函数被设计为幂等：多次调用不会重复关闭或引发异常。
    """
    try:
        url = _resolve_comfyui_url()
        from .sd_backend.comfyui_launcher import shutdown_comfyui
        shutdown_comfyui(base_url=url)
    except Exception:
        log.debug("atexit shutdown failed (may be expected during unregister)")

    # 防止 unregister() 再次触发 atexit
    try:
        atexit.unregister(_shutdown_on_exit)
    except (ValueError, Exception):
        pass


def register():
    # ── 依赖检查 ──
    missing = _check_runtime_dependencies()
    if missing:
        log.error("Missing dependencies: %s", ", ".join(missing))
        _show_dependency_warning(missing)
        raise RuntimeError(
            "AI 材质生成插件缺少必需的 Python 包，请安装后重启 Blender。\n"
            f"缺失: {', '.join(missing)}\n"
            f"运行: {sys.executable} -m pip install requests Pillow numpy websocket-client"
        )

    from . import preferences, operators, panels, properties
    from .core import orchestrator
    from .sd_backend import comfyui_client
    from .utils import async_bridge
    from .ui import preview_manager

    modules.clear()
    modules.extend([
        preferences,
        properties,
        operators,
        panels,
        orchestrator,
        comfyui_client,
        async_bridge,
        preview_manager,
    ])

    for mod in modules:
        if hasattr(mod, "register"):
            mod.register()

    bpy.types.Scene.ai_concept_props = bpy.props.PointerProperty(
        type=properties.CTProperties
    )

    # 注册进程退出兜底清理
    atexit.register(_shutdown_on_exit)
    log.info("AI Texture to PBR registered")


def unregister():
    # 关闭由插件自动启动的 ComfyUI 进程，避免残留后台进程
    try:
        url = _resolve_comfyui_url()
        from .sd_backend.comfyui_launcher import shutdown_comfyui
        shutdown_comfyui(base_url=url)
    except Exception:
        log.debug("ComfyUI shutdown failed during unregister (may be expected)")

    if hasattr(bpy.types.Scene, "ai_concept_props"):
        del bpy.types.Scene.ai_concept_props

    for mod in reversed(modules):
        if hasattr(mod, "unregister"):
            mod.unregister()

    modules.clear()

    # 注销 atexit 处理器，避免重复执行
    try:
        atexit.unregister(_shutdown_on_exit)
    except (ValueError, Exception):
        pass

    log.info("AI Texture to PBR unregistered")


if __name__ == "__main__":
    register()
