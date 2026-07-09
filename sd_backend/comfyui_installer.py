"""ComfyUI 自动安装器与模型下载管理。

负责：
1. 检测 ComfyUI portable 是否已安装
2. 从 GitHub Release 下载并安装 ComfyUI portable
3. 自动安装必需的 custom nodes
4. 提供模型注册表、缺失检测、逐一下载

原则：
- 安装/启用插件时不主动执行任何网络请求
- 只有用户明确选择 Local ComfyUI 并点击安装后才触发
"""

import os
import json
import subprocess
from typing import Dict, List, Optional

from ..utils.logger import get_logger

log = get_logger(__name__)


# =============================================================================
# 配置
# =============================================================================

MODEL_REGISTRY: List[Dict] = [
    {
        "id": "ae_vae",
        "label": "FLUX VAE (ae.sft)",
        "dir": "models/vae",
        "filename": "ae.sft",
        "family": "zimage",
        "page": "https://hf-mirror.com/black-forest-labs/FLUX.1-dev/tree/main",
        "url": "https://hf-mirror.com/black-forest-labs/FLUX.1-dev/resolve/main/ae.sft",
        "size": "~300 MB",
        "gated": True,
        "mirrors": [
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.sft",
        ],
    },
    {
        "id": "zimage_turbo",
        "label": "Z-Image Turbo",
        "dir": "models/unet",
        "filename": "z_image_turbo_bf16.safetensors",
        "family": "zimage",
        "page": "https://modelscope.cn/models/Tongyi-MAI/ZImage-Turbo",
        "url": "https://hf-mirror.com/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "size": "~6 GB",
        "gated": False,
        "mirrors": [
            "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        ],
    },
    {
        "id": "chord_v1",
        "label": "CHORD v1",
        "dir": "models/checkpoints",
        "filename": "chord_v1.safetensors",
        "family": "chord",
        "page": "https://hf-mirror.com/Ubisoft/ubisoft-laforge-chord/tree/main",
        "url": "https://hf-mirror.com/Ubisoft/ubisoft-laforge-chord/resolve/main/chord_v1.safetensors",
        "size": "~2 GB",
        "gated": True,
        "mirrors": [
            "https://huggingface.co/Ubisoft/ubisoft-laforge-chord/resolve/main/chord_v1.safetensors",
        ],
    },
    {
        "id": "seedvr2_dit_3b",
        "label": "SeedVR2 DiT 3B",
        "dir": "models/seedvr2",
        "filename": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "alt_filenames": [
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_3b-Q4_K_M.gguf",
            "seedvr2_ema_3b-Q8_0.gguf",
        ],
        "family": "seedvr2",
        "page": "https://hf-mirror.com/numz/SeedVR2_comfyUI/tree/main",
        "url": "https://hf-mirror.com/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "size": "~6 GB",
        "gated": True,
        "mirrors": [
            "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        ],
    },
    {
        "id": "seedvr2_vae",
        "label": "SeedVR2 VAE",
        "dir": "models/seedvr2",
        "filename": "ema_vae_fp16.safetensors",
        "family": "seedvr2",
        "page": "https://hf-mirror.com/numz/SeedVR2_comfyUI/tree/main",
        "url": "https://hf-mirror.com/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors",
        "size": "~300 MB",
        "gated": True,
        "mirrors": [
            "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors",
        ],
    },
]


# =============================================================================
# 路径与检测
# =============================================================================

def _addon_root() -> str:
    """返回插件根目录。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_default_install_path() -> str:
    """默认安装路径：插件根目录下的 comfyui_portable。"""
    return os.path.join(_addon_root(), "comfyui_portable")


def get_comfyui_type(path: str) -> Optional[str]:
    """判断 ComfyUI 安装类型。

    返回：
        "portable" — 便携版（存在 ComfyUI/main.py + python_embeded/python.exe）
        "desktop"  — 桌面版/git 版（存在 ComfyUI/main.py，使用系统 Python）
        None       — 不是有效 ComfyUI 目录
    """
    if not path or not os.path.isdir(path):
        return None
    main_py = os.path.join(path, "ComfyUI", "main.py")
    if not os.path.isfile(main_py):
        return None
    python_exe = os.path.join(path, "python_embeded", "python.exe")
    return "portable" if os.path.isfile(python_exe) else "desktop"


def is_comfyui_installed(path: str) -> bool:
    """检查指定路径是否是可用的 ComfyUI 安装（支持便携版与桌面版）。"""
    return get_comfyui_type(path) is not None


def _is_valid_python(python_exe: str) -> bool:
    """验证给定路径/命令的 Python 可执行文件可用。"""
    try:
        result = subprocess.run(
            [python_exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_python_exe(comfyui_path: str) -> Optional[str]:
    """返回启动 ComfyUI 可用的 Python 可执行文件路径。

    便携版返回 embedded python；桌面版优先查找 venv/python_embeded，
    最后回退到系统 python/python3。
    """
    ctype = get_comfyui_type(comfyui_path)
    if ctype == "portable":
        return os.path.join(comfyui_path, "python_embeded", "python.exe")

    if ctype == "desktop":
        # 桌面版常见 Python 环境（按优先级）
        candidates = []
        if os.name == "nt":
            candidates.extend([
                os.path.join(comfyui_path, "venv", "Scripts", "python.exe"),
                os.path.join(comfyui_path, ".venv", "Scripts", "python.exe"),
                os.path.join(comfyui_path, "python_embeded", "python.exe"),
            ])
        else:
            candidates.extend([
                os.path.join(comfyui_path, "venv", "bin", "python"),
                os.path.join(comfyui_path, ".venv", "bin", "python"),
                os.path.join(comfyui_path, "venv", "bin", "python3"),
                os.path.join(comfyui_path, ".venv", "bin", "python3"),
            ])
        for exe in candidates:
            if _is_valid_python(exe):
                return exe

        # 回退到系统 Python
        for exe in ("python.exe", "python3.exe", "python", "python3"):
            if _is_valid_python(exe):
                return exe
    return None


# 扫描时会纳入的模型子目录
_MODEL_SCAN_SUBDIRS = ("checkpoints", "unet")


def scan_comfyui_models(path: str) -> Dict[str, List[str]]:
    """扫描指定 ComfyUI 安装目录下的模型文件。

    只扫描 ComfyUI/models 下的常见模型子目录（checkpoints/unet 等）。

    返回：
        {子目录名: [文件名列表]}，例如
        {'checkpoints': ['chord_v1.safetensors'], 'unet': ['z_image_turbo_bf16.safetensors']}
    """
    if not path or not os.path.isdir(path):
        return {}
    models_dir = os.path.join(path, "ComfyUI", "models")
    if not os.path.isdir(models_dir):
        return {}
    result: Dict[str, List[str]] = {}
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        files = []
        for name in os.listdir(subdir_path):
            if name.startswith("."):
                continue
            lower = name.lower()
            if lower.endswith((".txt", ".md", ".json", ".ini", ".py", ".cache")):
                continue
            file_path = os.path.join(subdir_path, name)
            if os.path.isfile(file_path):
                files.append(name)
        if files:
            files.sort(key=str.lower)
            result[subdir] = files
    return result


# =============================================================================
# 模型管理
# =============================================================================

def _model_search_roots(comfyui_path: str) -> List[str]:
    """返回可能需要搜索的模型根目录。

    同时支持：
    - 便携版 / 桌面版实例：<comfyui_path>/ComfyUI/models/...
    - 桌面版共享模型：<comfyui_path>/ComfyUI-Shared/models/...
    - 若 comfyui_path 本身是实例子目录，也尝试父目录旁的 ComfyUI-Shared
    """
    if not comfyui_path:
        return []
    roots = [
        os.path.join(comfyui_path, "ComfyUI"),
        os.path.join(comfyui_path, "ComfyUI-Shared"),
    ]
    parent = os.path.dirname(comfyui_path)
    if parent:
        roots.append(os.path.join(parent, "ComfyUI-Shared"))

    seen = set()
    result = []
    for r in roots:
        key = os.path.normcase(r)
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


def _desktop_config_candidates(comfyui_path: str) -> List[str]:
    """桌面版 ComfyUI 可能使用的 extra model paths 配置文件候选。"""
    candidates = []
    appdata = os.environ.get('APPDATA', '')
    localappdata = os.environ.get('LOCALAPPDATA', '')
    if appdata:
        candidates.append(os.path.join(appdata, 'ComfyUI', 'extra_models_config.yaml'))
        candidates.append(os.path.join(appdata, 'Comfy Desktop', 'shared_model_paths.yaml'))
    if localappdata:
        candidates.append(os.path.join(localappdata, 'Comfy-Desktop', 'shared_model_paths.yaml'))
    if comfyui_path:
        candidates.append(os.path.join(comfyui_path, 'extra_model_paths.yaml'))
        candidates.append(os.path.join(comfyui_path, 'extra_models_config.yaml'))
        parent = os.path.dirname(comfyui_path)
        if parent:
            candidates.append(os.path.join(parent, 'extra_model_paths.yaml'))
            candidates.append(os.path.join(parent, 'extra_models_config.yaml'))
    return candidates


def _parse_extra_model_paths_simple(path: str) -> List[str]:
    """无 PyYAML 时的极简 YAML 解析：提取 base_path 与相对路径并拼成绝对目录。"""
    roots = []
    base = ""
    block_key_indent = None
    block_lines = []

    def _flush_block():
        nonlocal block_key_indent
        for sub in block_lines:
            sub = sub.strip()
            if sub:
                roots.append(os.path.normpath(os.path.join(base, sub)))
        block_lines.clear()
        block_key_indent = None

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip() or line.strip().startswith('#'):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            # base_path 行
            if stripped.startswith('base_path:'):
                _flush_block()
                base = stripped.split(':', 1)[1].strip().strip('"\'')
                continue

            # 新的顶层 profile/section
            if indent == 0 and stripped.endswith(':') and ':' not in stripped[:-1]:
                _flush_block()
                base = ""
                continue

            # key: value 或 key: | 多行块
            if ':' in stripped:
                _flush_block()
                key, val = stripped.split(':', 1)
                key = key.strip()
                val = val.strip().strip('"\'')
                if key in ('is_default',):
                    continue
                if val in ('|', '>'):
                    block_key_indent = indent
                elif base and val:
                    roots.append(os.path.normpath(os.path.join(base, val)))
                continue

            # 多行块标量内的行
            if block_key_indent is not None and indent > block_key_indent and base:
                block_lines.append(stripped)
            else:
                _flush_block()

    _flush_block()
    return roots


def _parse_extra_model_paths_yaml(path: str) -> List[str]:
    """解析桌面版 extra model paths YAML，返回所有模型目录的绝对路径。"""
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return _parse_extra_model_paths_simple(path)

    roots = []
    for section in data.values():
        if not isinstance(section, dict):
            continue
        base = section.get('base_path', '')
        if not base:
            continue
        for key, val in section.items():
            if key in ('base_path', 'is_default'):
                continue
            if isinstance(val, str):
                for sub in val.splitlines():
                    sub = sub.strip()
                    if sub:
                        roots.append(os.path.normpath(os.path.join(base, sub)))
            elif isinstance(val, list):
                for sub in val:
                    if isinstance(sub, str) and sub.strip():
                        roots.append(os.path.normpath(os.path.join(base, sub.strip())))
    return roots


def _desktop_extra_model_dirs(comfyui_path: str) -> List[str]:
    """从桌面版配置文件中读取所有模型搜索目录。"""
    dirs = []
    for cfg_path in _desktop_config_candidates(comfyui_path):
        if not os.path.isfile(cfg_path):
            continue
        try:
            parsed = _parse_extra_model_paths_yaml(cfg_path)
            dirs.extend(parsed)
            log.debug("Parsed extra model paths from %s: %s", cfg_path, parsed)
        except Exception as e:
            log.debug("Could not parse %s: %s", cfg_path, e)
    return dirs


def get_model_registry() -> List[Dict]:
    return list(MODEL_REGISTRY)


def find_model_file(comfyui_path: str, model: Dict) -> Optional[str]:
    """查找模型文件的实际路径。

    支持便携版、桌面版实例、桌面版 ComfyUI-Shared 共享目录，
    以及通过 extra_model_paths.yaml / shared_model_paths.yaml 配置的任意共享目录。
    优先匹配精确路径，未找到时按文件名递归查找已知模型目录。
    """
    candidates = [model["filename"]] + list(model.get("alt_filenames", []))
    rel_dir = model["dir"]

    for root in _model_search_roots(comfyui_path):
        # 标准路径：root/models/<category>/filename
        target_dir = os.path.join(root, rel_dir)
        for name in candidates:
            exact = os.path.join(target_dir, name)
            if os.path.isfile(exact):
                return exact

        # 桌面版共享目录有时没有中间的 models/ 层级
        if rel_dir.startswith("models/"):
            alt_dir = os.path.join(root, rel_dir[7:])
            for name in candidates:
                exact = os.path.join(alt_dir, name)
                if os.path.isfile(exact):
                    return exact

        # 递归回退
        if "ComfyUI-Shared" in root and os.path.isdir(root):
            walk_root = root
        else:
            walk_root = os.path.join(root, "models")
        if not os.path.isdir(walk_root):
            continue
        for walk_dir, _dirs, files in os.walk(walk_root):
            for name in candidates:
                if name in files:
                    return os.path.join(walk_dir, name)

    # 最后扫描桌面版 extra_model_paths.yaml 中配置的任意共享目录
    for extra_dir in _desktop_extra_model_dirs(comfyui_path):
        if not os.path.isdir(extra_dir):
            continue
        # 若该目录名正好与模型类别对应，先直接尝试
        target_dir = os.path.join(extra_dir, rel_dir)
        for name in candidates:
            exact = os.path.join(target_dir, name)
            if os.path.isfile(exact):
                return exact
        if rel_dir.startswith("models/"):
            target_dir = os.path.join(extra_dir, rel_dir[7:])
            for name in candidates:
                exact = os.path.join(target_dir, name)
                if os.path.isfile(exact):
                    return exact
        # 递归回退
        for walk_dir, _dirs, files in os.walk(extra_dir):
            for name in candidates:
                if name in files:
                    return os.path.join(walk_dir, name)
    return None


