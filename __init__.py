bl_info = {
    "name": "AI 材质生成",
    "author": "Xiong Meng Han",
    "version": (1, 1, 6),
    "blender": (3, 6, 0),
    "location": "3D 视图 > 侧边栏 (N) > AI 材质",
    "description": "从提示词或参考图一键生成 PBR 材质贴图（支持本地 ComfyUI 与在线 API）",
    "doc_url": "https://github.com/disk3/AI_Texture_Generator",
    "tracker_url": "https://github.com/disk3/AI_Texture_Generator/issues",
    "category": "3D View",
    "support": "COMMUNITY",
}

import atexit
import bpy
import os
import sys

from .utils.logger import get_logger

log = get_logger(__name__)
modules = []


def _addon_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _vendor_site_dir() -> str:
    version_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    return os.path.join(_addon_dir(), "vendor", version_tag)


def _inject_vendor_site() -> str:
    """Add the add-on managed dependency directory to sys.path."""
    vendor_dir = _vendor_site_dir()
    if os.path.isdir(vendor_dir):
        try:
            import site
            site.addsitedir(vendor_dir)
        except Exception:
            pass
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
    return vendor_dir


_inject_vendor_site()

# ── 运行时依赖声明 ──────────────────────────────────────────────────────────
# Blender 内置 Python 已提供: bpy, mathutils, bmesh, math, os, json, io, base64,
#   queue, threading, time, uuid, random, subprocess, shutil, tempfile, urllib,
#   logging, ctypes, re, webbrowser, signal, abc, dataclasses, typing
# 以下为需要用户自行安装的第三方包（安装到 Blender 的 site-packages）:
# 基础依赖：本地 PBR（法线、无缝、高度图反推）必需。Blender 4.4+/4.5 已内置 numpy。
_REQUIRED_PACKAGES = {
    "numpy":           "数值计算（贴图像素处理）",
}
# 后端增强依赖：API 已有标准库/Blender fallback；这些包主要用于本地 ComfyUI
# 兼容性、实时进度和剪贴板支持。本地 PBR/API 用户无需安装。
# 若使用 Windows ComfyUI 便携版，插件会优先尝试复用其 python_embeded 中的包。
_BACKEND_PACKAGES = {
    "PIL":             "图像处理增强（剪贴板/解码加速）",
    "requests":        "HTTP 客户端增强（更好的 ComfyUI 兼容性）",
}
_OPTIONAL_PACKAGES = {
    "cv2":             "OpenCV（法线贴图 Sobel 算法，无此包将回退到简化实现）",
    "py7zr":           "7z 解压库（ComfyUI 安装，无此包将尝试系统 7z 命令）",
    "websocket":       "WebSocket 客户端（ComfyUI 实时进度，无此包将回退到 HTTP polling）",
}


def _check_runtime_dependencies() -> list:
    """检查必需的第三方依赖是否可用，返回缺失包名列表。"""
    missing = []
    for pkg, desc in _REQUIRED_PACKAGES.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"{pkg} ({desc})")
    return missing


def _check_backend_dependencies() -> list:
    """检查 ComfyUI / API 后端所需的依赖，返回缺失包名列表。"""
    missing = []
    for pkg, desc in _BACKEND_PACKAGES.items():
        # PIL 的导入名是 PIL，但 pip 包名是 Pillow
        import_name = "PIL" if pkg == "PIL" else pkg
        try:
            __import__(import_name)
        except ImportError:
            missing.append(f"{pkg} ({desc})")
    return missing


def ensure_backend_dependencies(comfyui_path: str = "") -> tuple[bool, str]:
    """安装本地 ComfyUI 环境增强依赖。

    逻辑：
      1. 若 Blender 当前 Python 已有 PIL/requests，直接返回成功。
      2. 若提供了 ComfyUI 路径（尤其是 Windows 便携版），尝试注入其
         python_embeded/Lib/site-packages 中的依赖。
      3. 仍缺失时尝试自动 pip install 安装。
      4. 全部失败则返回错误提示，由调用方弹窗/报告。

    返回 (success, message)。
    """
    missing = _check_backend_dependencies()
    if not missing:
        return True, "增强依赖已就绪"

    # 1) 尝试复用 ComfyUI 便携版依赖
    if comfyui_path:
        try:
            from .sd_backend.comfyui_env_resolver import inject_comfyui_packages
            if inject_comfyui_packages(comfyui_path):
                missing = _check_backend_dependencies()
                if not missing:
                    return True, "已从 ComfyUI 复用增强依赖"
        except Exception as e:
            log.debug("Failed to inject ComfyUI packages: %s", e)

    # 2) 尝试自动安装
    log.info("ComfyUI environment dependencies missing, trying auto-install: %s", ", ".join(missing))
    ok, msg = _auto_install_dependencies(missing, verifier=_check_backend_dependencies)
    if ok:
        return True, msg

    # 3) 失败时返回手动命令
    return False, (
        "缺少本地 ComfyUI 环境依赖，自动安装失败。\n"
        f"{msg}\n"
        f"请手动运行: {sys.executable} -m pip install --target \"{_vendor_site_dir()}\" Pillow requests\n\n"
        "或配置 Windows ComfyUI 便携版路径，让插件自动复用其纯 Python 依赖。"
    )


def _format_missing_packages(missing: list) -> str:
    """生成依赖安装命令提示。"""
    names = [m.split()[0] for m in missing]
    # 将 PIL 映射到 pip 包名 Pillow
    pip_names = []
    for name in names:
        if name == "PIL":
            pip_names.append("Pillow")
        elif name == "websocket":
            pip_names.append("websocket-client")
        else:
            pip_names.append(name)
    return (
        "请将以下包安装到插件托管依赖目录中：\n\n"
        f"  {sys.executable} -m pip install --target \"{_vendor_site_dir()}\" {' '.join(pip_names)}\n\n"
        "注意：安装命令必须对应当前 Blender 内置的 Python（上方路径），不能安装到系统 Python。\n"
        "安装后请完全关闭并重启 Blender，再重新启用插件。"
    )


def _show_dependency_warning(missing: list, title: str = "AI 材质生成插件检测到缺失的 Python 包"):
    """通过 Blender 弹窗提示用户安装缺失的依赖。"""
    import textwrap
    lines = [
        title,
        "",
    ]
    for m in missing:
        lines.append(f"  • {m}")
    lines.extend([
        "",
        _format_missing_packages(missing),
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


def _auto_install_dependencies(missing: list, verifier=None) -> tuple[bool, str]:
    """尝试用 Blender 内置 pip 自动安装缺失的依赖。

    策略：
      1. 安装到插件托管目录 ``vendor/pyXY``，并在启用插件时注入 sys.path。
      2. 如果 pip 未初始化，先调用 ``ensurepip``。
      3. 依次尝试清华 / 阿里 / 官方 PyPI 镜像。

    返回 (success, message)：
      - success=True 表示安装后所有必需依赖已可用。
      - success=False 时 message 包含失败原因或提示。
    """
    import subprocess

    names = [m.split()[0] for m in missing]
    pip_names = []
    for name in names:
        if name == "PIL":
            pip_names.append("Pillow")
        elif name == "websocket":
            pip_names.append("websocket-client")
        else:
            pip_names.append(name)

    log.info("Auto-installing dependencies: %s", pip_names)
    target_dir = _vendor_site_dir()
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        return False, f"无法创建插件依赖目录: {target_dir}\n{e}"
    _inject_vendor_site()

    # 1) 确保 pip 可用
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception as e:
        log.info("pip not available, trying ensurepip: %s", e)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if proc.returncode != 0:
                return False, f"无法初始化 pip: {proc.stderr or proc.stdout}"
        except Exception as ee:
            return False, f"无法初始化 pip: {ee}"

    def _run_pip(extra_args: list) -> tuple[bool, str]:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            target_dir,
        ] + extra_args + pip_names
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if proc.returncode == 0:
                return True, proc.stdout
            return False, proc.stderr or proc.stdout
        except Exception as e:
            return False, str(e)

    # 2) 依次尝试安装策略
    strategies = [
        (["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], "插件目录 + 清华镜像"),
        (["-i", "https://mirrors.aliyun.com/pypi/simple/"], "插件目录 + 阿里镜像"),
        ([], "插件目录 + PyPI 官方"),
    ]

    last_error = ""
    for extra_args, desc in strategies:
        log.info("Trying install strategy: %s", desc)
        ok, out = _run_pip(extra_args)
        if ok:
            log.info("pip install stdout: %s", out)
            break
        last_error = f"[{desc}] {out}"
        log.warning("Install strategy failed: %s", last_error)
    else:
        return False, f"所有自动安装方式均失败，最后错误：\n{last_error}"

    check_missing = verifier or _check_runtime_dependencies

    # 3) 安装后再次检查
    try:
        import importlib
        _inject_vendor_site()
        importlib.invalidate_caches()
    except Exception:
        pass

    if not check_missing():
        return True, f"依赖已安装到插件目录: {target_dir}"

    # 4) 最后再刷新一次 site 路径后检查
    try:
        import importlib
        import site
        site.addsitedir(target_dir)
        importlib.invalidate_caches()
        if not check_missing():
            return True, f"依赖已安装到插件目录: {target_dir}"
    except Exception:
        pass

    return False, (
        "依赖已安装到插件目录，但当前 Blender 仍无法导入。"
        "请完全关闭 Blender 后重新打开并启用插件；如果仍失败，可能是当前 Blender Python 版本没有可用的二进制 wheel。"
    )


def register():
    _inject_vendor_site()
    # ── 基础依赖检查（本地 PBR 必需：numpy）──
    # 不在启用插件时自动 pip install。美术用户通过 Preferences / 面板里的
    # “一键修复”显式安装，避免插件启用阶段修改 Blender Python 环境。
    missing = _check_runtime_dependencies()
    runtime_ready = not missing
    if missing:
        log.warning("Missing runtime dependencies: %s", ", ".join(missing))
        _show_dependency_warning(
            missing,
            "AI 材质生成插件缺少基础组件，请在插件设置中点击“一键修复”",
        )

    # 可选依赖缺失时仅警告，不阻止启用
    try:
        __import__("PIL")
    except ImportError:
        log.warning(
            "Pillow 未安装，将使用 Blender/标准库降级图像解码；"
            "剪贴板和部分图像格式兼容性会受限。本地 ComfyUI 用户可在对应设置中一键修复环境依赖。"
        )

    from . import preferences, panels, properties
    from .utils import async_bridge
    from .ui import preview_manager

    modules.clear()
    modules.extend([
        preferences,
        properties,
        panels,
        async_bridge,
        preview_manager,
    ])

    if runtime_ready:
        from . import operators
        from .core import orchestrator
        modules.extend([
            operators,
            orchestrator,
        ])
    else:
        log.warning("Generation operators were not registered because runtime dependencies are missing")

    # 后端模块（ComfyUI 客户端）可选加载
    if runtime_ready:
        try:
            from .sd_backend import comfyui_client
            modules.append(comfyui_client)
        except ImportError as e:
            log.warning("ComfyUI 客户端未能加载：%s", e)

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
