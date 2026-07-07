"""复用本地 ComfyUI 便携版中的 Python 依赖。

Blender 插件与 ComfyUI 是两个独立进程，默认不共享 site-packages。
当用户已经拥有 Windows ComfyUI 便携版时，可以把它 python_embeded 里的
site-packages 注入 Blender Python 的 sys.path，避免用户在 Blender 中重复安装。
"""

import os
import sys
import site
import subprocess
from typing import List, Optional

from ..utils.logger import get_logger

log = get_logger(__name__)

# 插件后端需要、且可以从 ComfyUI 复用的包
_TARGET_PACKAGES = {
    "Pillow": ["PIL"],
    "requests": ["requests"],
    "websocket-client": ["websocket"],
}


def _get_blender_python_version() -> tuple:
    """返回 Blender 当前 Python 的 (major, minor) 版本元组。"""
    return sys.version_info[:2]


def _get_comfyui_python_version(python_exe: str) -> Optional[tuple]:
    """查询指定 Python 解释器的 (major, minor) 版本。"""
    try:
        result = subprocess.run(
            [python_exe, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            return (int(parts[0]), int(parts[1]))
    except Exception as e:
        log.debug("Failed to detect ComfyUI Python version: %s", e)
    return None


def get_comfyui_type(comfyui_path: str) -> Optional[str]:
    """检测 ComfyUI 路径类型：portable / desktop / None。"""
    if not comfyui_path or not os.path.isdir(comfyui_path):
        return None
    main_py = os.path.join(comfyui_path, "ComfyUI", "main.py")
    if not os.path.isfile(main_py):
        return None
    portable_python = os.path.join(comfyui_path, "python_embeded", "python.exe")
    return "portable" if os.path.isfile(portable_python) else "desktop"


def resolve_comfyui_site_packages(comfyui_path: str) -> List[str]:
    """返回 ComfyUI 路径下所有可用的 site-packages 目录。"""
    paths = []
    ctype = get_comfyui_type(comfyui_path)
    if ctype == "portable":
        sp = os.path.join(comfyui_path, "python_embeded", "Lib", "site-packages")
        if os.path.isdir(sp):
            paths.append(sp)
    elif ctype == "desktop":
        # 尝试常见 venv 位置
        for venv in ("venv", ".venv"):
            for sub in ("Lib", "lib"):
                sp = os.path.join(comfyui_path, venv, sub, "site-packages")
                if os.path.isdir(sp):
                    paths.append(sp)
        # 也尝试通过 ComfyUI 主 Python 定位
        for exe in ("python.exe", "python3.exe", "python", "python3"):
            try:
                result = subprocess.run(
                    [exe, "-c", "import site; print(site.getsitepackages()[0])"],
                    capture_output=True, text=True, timeout=5,
                    cwd=comfyui_path,
                )
                if result.returncode == 0:
                    sp = result.stdout.strip()
                    if os.path.isdir(sp) and sp not in paths:
                        paths.append(sp)
            except Exception:
                continue
    return paths


def _get_portable_dll_dir(comfyui_path: str) -> Optional[str]:
    """返回 ComfyUI 便携版的 DLLs 目录（Windows C 扩展可能需要）。"""
    dll_dir = os.path.join(comfyui_path, "python_embeded", "DLLs")
    if os.path.isdir(dll_dir):
        return dll_dir
    return None


def has_backend_packages() -> bool:
    """检查 Blender 当前 Python 是否已有后端依赖。"""
    for import_names in _TARGET_PACKAGES.values():
        for name in import_names:
            try:
                __import__(name)
            except ImportError:
                return False
    return True


def _add_sitedir(sitedir: str) -> bool:
    """使用 site.addsitedir 把目录加入 sys.path，并处理 .pth 文件。

    放在 sys.path 末尾，避免覆盖 Blender 自带的同名包。
    """
    if not os.path.isdir(sitedir):
        return False
    try:
        site.addsitedir(sitedir)
        log.debug("Injected ComfyUI site-packages dir: %s", sitedir)
        return True
    except Exception as e:
        log.warning("Failed to add ComfyUI sitedir %s: %s", sitedir, e)
        return False


def inject_comfyui_packages(comfyui_path: str) -> bool:
    """尝试将 ComfyUI 的 site-packages 注入 Blender Python。

    返回 True 表示注入后所有目标包都可 import；False 表示失败或不需要。
    """
    if has_backend_packages():
        return True

    ctype = get_comfyui_type(comfyui_path)
    if not ctype:
        log.debug("Not a valid ComfyUI path: %s", comfyui_path)
        return False

    # 版本检查：优先用便携版 python_embeded/python.exe
    python_exe = None
    if ctype == "portable":
        python_exe = os.path.join(comfyui_path, "python_embeded", "python.exe")
    else:
        for venv in ("venv", ".venv"):
            for sub in ("Scripts", "bin"):
                candidate = os.path.join(comfyui_path, venv, sub, "python.exe")
                if os.path.isfile(candidate):
                    python_exe = candidate
                    break
            if python_exe:
                break

    if python_exe:
        comfy_py_version = _get_comfyui_python_version(python_exe)
        blender_py_version = _get_blender_python_version()
        if comfy_py_version and comfy_py_version != blender_py_version:
            log.warning(
                "ComfyUI Python %s.%s differs from Blender Python %s.%s; "
                "skipping dependency injection to avoid binary incompatibility.",
                comfy_py_version[0], comfy_py_version[1],
                blender_py_version[0], blender_py_version[1],
            )
            return False

    site_packages_list = resolve_comfyui_site_packages(comfyui_path)
    if not site_packages_list:
        log.debug("No site-packages found for ComfyUI path: %s", comfyui_path)
        return False

    injected = False
    for sp in site_packages_list:
        if _add_sitedir(sp):
            injected = True

    # Windows 便携版：补充 DLLs 目录，部分 C 扩展需要找到 python3x.dll 等
    if ctype == "portable":
        dll_dir = _get_portable_dll_dir(comfyui_path)
        if dll_dir and dll_dir not in sys.path:
            sys.path.append(dll_dir)
            log.debug("Injected ComfyUI DLLs dir: %s", dll_dir)
            injected = True

    if not injected:
        log.debug("No ComfyUI site-packages could be injected")
        return False

    import importlib
    importlib.invalidate_caches()

    return has_backend_packages()
