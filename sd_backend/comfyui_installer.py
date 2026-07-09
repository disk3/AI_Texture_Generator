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
import shutil
import tempfile
import subprocess
import urllib.request
import urllib.parse
import hashlib
from typing import Callable, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

log = get_logger(__name__)


# =============================================================================
# 配置
# =============================================================================

COMFYUI_REPO_OWNER = "comfyanonymous"
COMFYUI_REPO_NAME = "ComfyUI"

# 必需 custom nodes：仓库地址 -> 目录名，或 {"folder": "...", "commit": "..."}
REQUIRED_CUSTOM_NODES = {
    "ubisoft/ComfyUI-Chord": "ComfyUI-Chord",
    "numz/ComfyUI-SeedVR2_VideoUpscaler": "ComfyUI-SeedVR2_VideoUpscaler",
}

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


def resolve_comfyui_path(prefs_or_path) -> str:
    """解析有效的 ComfyUI 根目录路径。"""
    if isinstance(prefs_or_path, str):
        path = prefs_or_path
    else:
        path = getattr(prefs_or_path, "comfyui_path", "")
    return path or get_default_install_path()


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


def _run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """运行子进程，返回 (returncode, stdout, stderr)。"""
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    }
    if cwd:
        kwargs["cwd"] = cwd
    try:
        proc = subprocess.run(cmd, **kwargs, text=True, encoding="utf-8", errors="ignore")
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return -1, "", str(e)


# =============================================================================
# GitHub Release 下载
# =============================================================================

def _fetch_url(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "BlenderAI-ComfyUI-Installer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_json(url: str, timeout: int = 30) -> Dict:
    return json.loads(_fetch_url(url, timeout).decode("utf-8", errors="ignore"))


def _find_portable_asset(release_data: Dict) -> Optional[Dict]:
    """从 release JSON 中找到 windows portable nvidia 7z 资源。"""
    assets = release_data.get("assets", [])
    candidates = []
    for asset in assets:
        name = (asset.get("name") or "").lower()
        if "portable" in name and name.endswith(".7z"):
            score = 0
            if "nvidia" in name or "gpu" in name:
                score += 10
            if "cuda" in name:
                score += 5
            candidates.append((score, asset))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


DEFAULT_COMFYUI_VERSION = "v0.25.0"

# 国内 GitHub Release 代理前缀（按可用性排序）
_GITHUB_PROXY_PREFIXES = [
    "https://ghproxy.com/",
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
]


def _github_proxy_urls(base_url: str) -> List[str]:
    """为单个 GitHub URL 生成带代理的镜像 URL 列表。"""
    urls = [base_url]
    for prefix in _GITHUB_PROXY_PREFIXES:
        urls.append(prefix + base_url)
    return urls


def _portable_download_url(version: str) -> str:
    """构造 ComfyUI portable nvidia 7z 下载链接。"""
    return f"https://github.com/Comfy-Org/ComfyUI/releases/download/{version}/ComfyUI_windows_portable_nvidia.7z"


def get_comfyui_download_url(use_mirror: bool = True) -> Tuple[str, str]:
    """获取最新 ComfyUI portable 下载链接与版本名。

    优先调用 GitHub API 获取最新 tag；API 限流或失败时回退到固定默认版本。
    当 use_mirror=True 时，返回的 URL 会包含国内 GitHub 代理，方便墙内用户下载。
    """
    api_url = f"https://api.github.com/repos/{COMFYUI_REPO_OWNER}/{COMFYUI_REPO_NAME}/releases/latest"
    try:
        data = _fetch_json(api_url, timeout=15)
        asset = _find_portable_asset(data)
        version = data.get("tag_name", DEFAULT_COMFYUI_VERSION)
        if asset:
            url = asset["browser_download_url"]
            return (url, version) if not use_mirror else (_github_proxy_urls(url), version)
        # API 返回了但没有找到 portable asset，使用 tag 构造固定 URL
        url = _portable_download_url(version)
        return (url, version) if not use_mirror else (_github_proxy_urls(url), version)
    except Exception:
        log.debug("GitHub API latest release fetch failed, using default version %s", DEFAULT_COMFYUI_VERSION)

    version = DEFAULT_COMFYUI_VERSION
    url = _portable_download_url(version)
    return (url, version) if not use_mirror else (_github_proxy_urls(url), version)


# =============================================================================
# 7z 解压
# =============================================================================

def _find_7z_executable() -> Optional[str]:
    candidates = ["7z", "7za"]
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ])
    for name in candidates:
        path = shutil.which(name) or (name if os.path.isfile(name) else None)
        if path and os.path.isfile(path):
            return path
    return None


def _try_py7zr(archive_path: str, target_path: str) -> bool:
    try:
        import py7zr
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=target_path)
        return True
    except Exception:
        log.debug("py7zr not available or extraction failed, falling back to 7z CLI")
        return False


def extract_7z(archive_path: str, target_path: str, progress_cb: Optional[Callable] = None) -> bool:
    """解压 7z 到目标目录。"""
    if progress_cb:
        progress_cb("extract", 0.0, "准备解压...")

    os.makedirs(target_path, exist_ok=True)

    if _try_py7zr(archive_path, target_path):
        if progress_cb:
            progress_cb("extract", 1.0, "py7zr 解压完成")
        return True
    if progress_cb:
        progress_cb("extract", 0.1, "py7zr 不可用，尝试 7z 命令...")

    exe7z = _find_7z_executable()
    if exe7z:
        code, out, err = _run_subprocess([exe7z, "x", archive_path, f"-o{target_path}", "-y"])
        if code == 0:
            if progress_cb:
                progress_cb("extract", 1.0, "7z 命令解压完成")
            return True
        if progress_cb:
            progress_cb("extract", 0.2, f"7z 命令失败: {err[:200]}")

    return False


# =============================================================================
# 磁盘空间预检
# =============================================================================

# ComfyUI 本体约 15 GB，模型约 10 GB，总计约 30 GB（留余量）
MIN_FREE_SPACE_BYTES_FULL = 30 * 1024 ** 3   # 完整安装
MIN_FREE_SPACE_BYTES_PORTABLE = 18 * 1024 ** 3  # 仅本体（无模型）


def _get_free_disk_space(path: str) -> int:
    """返回 path 所在分区的可用字节数；获取失败返回 -1。"""
    try:
        import shutil
        usage = shutil.disk_usage(os.path.dirname(path) if not os.path.isdir(path) else path)
        return usage.free
    except Exception:
        log.debug("Failed to query disk space for: %s", path)
        return -1


def _parse_model_size_bytes(size_str: str) -> int:
    """将模型尺寸字符串（如 '~17 GB', '300 MB'）转为字节数。"""
    import re
    size_str = (size_str or "").strip().replace("~", "").lower().replace(",", "")
    match = re.match(r"([\d.]+)\s*(gb|mb|tb|kb)", size_str, re.IGNORECASE)
    if not match:
        return 0
    value, unit = float(match.group(1)), match.group(2).upper()
    multipliers = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024}
    return int(value * multipliers.get(unit, 0))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_disk_space(target_path: str, required_bytes: int, progress_cb=None) -> bool:
    """检查目标路径所在磁盘是否有足够空间。空间不足则回调错误并返回 False。"""
    free = _get_free_disk_space(target_path)
    if free < 0:
        return True  # 无法获取时放行，避免误拦
    if free < required_bytes:
        free_gb = free / (1024 ** 3)
        need_gb = required_bytes / (1024 ** 3)
        msg = (
            f"磁盘空间不足（可用 {free_gb:.1f} GB，需要至少 {need_gb:.1f} GB）。"
            f"请清理磁盘后重试。"
        )
        if progress_cb:
            progress_cb("install", 0.0, f"错误: {msg}")
        log.error(msg)
        return False
    return True


# =============================================================================
# 下载与安装
# =============================================================================

def _is_hf_domain(url: str) -> bool:
    """判断 URL 是否属于 HuggingFace 官方或镜像域名。"""
    if not url:
        return False
    domain = urllib.parse.urlparse(url).netloc.lower()
    return domain in ("huggingface.co", "www.huggingface.co", "hf-mirror.com", "www.hf-mirror.com")


def _download_single_url(url: str, dest_path: str, progress_cb: Optional[Callable], hf_token: str = "") -> bool:
    """尝试从单个 URL 流式下载文件并汇报进度。"""
    headers = {"User-Agent": "BlenderAI-ComfyUI-Installer/1.0"}
    if hf_token and _is_hf_domain(url):
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)
    part_path = f"{dest_path}.part"
    if os.path.isfile(part_path):
        try:
            os.remove(part_path)
        except OSError:
            log.debug("Failed to remove stale partial download: %s", part_path)

    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024
        downloaded = 0
        with open(part_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total and progress_cb:
                    progress_cb("download", downloaded / total, f"已下载 {downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB")
        if total and downloaded < total:
            raise RuntimeError(f"下载不完整: {downloaded}/{total} bytes")
    os.replace(part_path, dest_path)
    return True


def _download_with_progress(
    urls,
    dest_path: str,
    progress_cb: Optional[Callable],
    hf_token: str = "",
    expected_size: int = 0,
    expected_sha256: str = "",
    skip_existing: bool = True,
) -> bool:
    """流式下载文件并汇报进度。支持 URL 列表自动 fallback。"""
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for u in urls if u]
    if not urls:
        if progress_cb:
            progress_cb("download", 0.0, "下载失败: 没有可用的下载地址")
        return False

    if os.path.isfile(dest_path) and skip_existing:
        size = os.path.getsize(dest_path)
        hash_ok = True
        if expected_sha256:
            try:
                hash_ok = _sha256_file(dest_path).lower() == expected_sha256.lower()
            except Exception:
                hash_ok = False
        if (expected_size and size < expected_size * 0.9) or not hash_ok:
            try:
                os.remove(dest_path)
            except OSError:
                log.debug("Failed to remove incomplete existing file: %s", dest_path)
        else:
            if progress_cb:
                progress_cb("download", 1.0, f"文件已存在，跳过下载: {dest_path}")
            return True

    if os.path.isfile(dest_path) and not skip_existing:
        if progress_cb:
            progress_cb("download", 0.0, f"重新下载并覆盖: {dest_path}")
        try:
            os.remove(dest_path)
        except OSError:
            log.debug("Failed to remove existing file before redownload: %s", dest_path)

    last_error = ""
    for idx, url in enumerate(urls):
        if progress_cb:
            if len(urls) > 1:
                progress_cb("download", 0.0, f"尝试镜像 {idx + 1}/{len(urls)}，目标: {dest_path}")
            else:
                progress_cb("download", 0.0, f"目标路径: {dest_path}")

        try:
            if _download_single_url(url, dest_path, progress_cb, hf_token):
                if expected_sha256 and _sha256_file(dest_path).lower() != expected_sha256.lower():
                    try:
                        os.remove(dest_path)
                    except OSError:
                        log.debug("Failed to remove hash-mismatched file: %s", dest_path)
                    last_error = "SHA256 校验失败"
                    continue
                return True
        except urllib.error.HTTPError as e:
            if os.path.isfile(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    log.debug("Failed to remove partial download: %s", dest_path)
            last_error = f"HTTP {e.code}"
            if e.code == 401:
                last_error = "需要 HuggingFace 登录/授权"
            elif e.code == 403:
                last_error = "需要 HuggingFace 授权同意，请先访问网页接受许可"
            elif e.code == 404:
                last_error = "文件不存在"
            # 继续尝试下一个镜像
        except Exception as e:
            if os.path.isfile(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    log.debug("Failed to remove partial download: %s", dest_path)
            last_error = str(e)
            # 继续尝试下一个镜像

    msg = f"下载失败: {last_error}"
    # 区分网络层错误和 HTTP 授权/鉴权错误
    is_network_error = any(kw in last_error for kw in ["10060", "10061", "timed out", "timeout", "连接", "Connection", "Network", "Name or service not known"])
    if is_network_error:
        msg += "。网络连接失败，请检查能否访问 hf-mirror.com/huggingface.co，或关闭国内镜像后重试。"
    elif hf_token:
        msg += "。已尝试所有可用镜像，请检查 Token 是否已接受模型许可，或点击网页手动下载。"
    else:
        msg += "。已尝试所有可用镜像，请填写 HuggingFace Token（gated 模型）或点击网页手动下载。"
    if progress_cb:
        progress_cb("download", 0.0, msg)
    return False


def install_comfyui_portable(target_path: str, progress_cb: Optional[Callable] = None, use_mirror: bool = True) -> bool:
    """从 GitHub Release 下载并安装 ComfyUI portable。"""
    if is_comfyui_installed(target_path):
        if progress_cb:
            progress_cb("install", 0.0, "ComfyUI 已安装，跳过本体安装")
        return True

    # ── 磁盘空间预检 ──
    if not _check_disk_space(
        target_path,
        MIN_FREE_SPACE_BYTES_FULL if use_mirror else MIN_FREE_SPACE_BYTES_PORTABLE + (2 * 1024 ** 3),
        progress_cb,
    ):
        return False

    os.makedirs(target_path, exist_ok=True)

    try:
        if progress_cb:
            progress_cb("download", 0.0, "正在获取 ComfyUI Release 下载链接...")
        url, version = get_comfyui_download_url(use_mirror=use_mirror)
        if progress_cb:
            progress_cb("download", 0.0, f"准备下载 {version}")

        archive_path = os.path.join(tempfile.gettempdir(), f"comfyui_windows_portable_{version}.7z")
        if not _download_with_progress(url, archive_path, progress_cb, skip_existing=False):
            raise RuntimeError("ComfyUI portable 下载失败，请检查网络连接或关闭国内镜像后重试")

        if progress_cb:
            progress_cb("extract", 0.0, "开始解压...")
        if not extract_7z(archive_path, target_path, progress_cb):
            raise RuntimeError("7z 解压失败，请确认已安装 7-Zip 或 py7zr")

        try:
            os.remove(archive_path)
        except OSError:
            log.debug("Failed to remove temp archive: %s", archive_path)

        # 处理 7z 根目录多一层的情况
        if not is_comfyui_installed(target_path):
            subdirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
            for d in subdirs:
                candidate = os.path.join(target_path, d)
                if is_comfyui_installed(candidate):
                    for item in os.listdir(candidate):
                        src = os.path.join(candidate, item)
                        dst = os.path.join(target_path, item)
                        if os.path.exists(dst):
                            shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                        shutil.move(src, dst)
                    shutil.rmtree(candidate)
                    break

        if not is_comfyui_installed(target_path):
            raise RuntimeError("解压后未找到 ComfyUI/main.py 或 python_embeded/python.exe")

        if progress_cb:
            progress_cb("install", 1.0, "ComfyUI 本体安装完成")
        return True
    except Exception as e:
        if progress_cb:
            progress_cb("install", 0.0, f"安装失败: {e}")
        return False


def _is_git_available() -> bool:
    return shutil.which("git") is not None


def _git_clone_urls(repo: str, use_mirror: bool = True) -> List[str]:
    """生成 custom node 仓库的 git clone URL 列表。"""
    base = f"https://github.com/{repo}.git"
    if not use_mirror:
        return [base]
    return [base] + [f"{prefix}{base}" for prefix in _GITHUB_PROXY_PREFIXES]


def install_custom_nodes(comfyui_path: str, progress_cb: Optional[Callable] = None, use_mirror: bool = True) -> bool:
    """git clone 必需节点并安装依赖。"""
    custom_nodes_dir = os.path.join(comfyui_path, "ComfyUI", "custom_nodes")
    os.makedirs(custom_nodes_dir, exist_ok=True)
    python_exe = get_python_exe(comfyui_path)

    if not _is_git_available():
        if progress_cb:
            progress_cb("nodes", 0.0, "未找到 git，无法自动安装 custom nodes，请手动安装并配置 PATH")
        return False

    total = len(REQUIRED_CUSTOM_NODES)
    for idx, (repo, node_spec) in enumerate(REQUIRED_CUSTOM_NODES.items(), 1):
        if isinstance(node_spec, dict):
            folder_name = node_spec.get("folder") or repo.rsplit("/", 1)[-1]
            pinned_commit = node_spec.get("commit", "")
        else:
            folder_name = str(node_spec)
            pinned_commit = ""
        if progress_cb:
            progress_cb("nodes", idx / total, f"安装节点 {folder_name}...")

        target = os.path.join(custom_nodes_dir, folder_name)
        if os.path.isdir(os.path.join(target, ".git")):
            _run_subprocess(["git", "pull"], cwd=target)
        else:
            if os.path.isdir(target):
                shutil.rmtree(target)
            clone_ok = False
            last_err = ""
            for url in _git_clone_urls(repo, use_mirror=use_mirror):
                code, out, err = _run_subprocess(["git", "clone", "--depth", "1", url, target])
                if code == 0:
                    clone_ok = True
                    break
                last_err = err[:200]
            if not clone_ok:
                if progress_cb:
                    progress_cb("nodes", idx / total, f"git clone {repo} 失败: {last_err}")
                continue

        if pinned_commit:
            code, out, err = _run_subprocess(["git", "checkout", pinned_commit], cwd=target)
            if code != 0:
                if progress_cb:
                    progress_cb("nodes", idx / total, f"{folder_name} checkout {pinned_commit} 失败: {err[:200]}")
                continue

        req_file = os.path.join(target, "requirements.txt")
        if os.path.isfile(req_file):
            code, out, err = _run_subprocess([python_exe, "-m", "pip", "install", "-r", req_file])
            if code != 0 and progress_cb:
                progress_cb("nodes", idx / total, f"{folder_name} 依赖安装警告: {err[:200]}")

    if progress_cb:
        progress_cb("nodes", 1.0, "节点安装完成")
    return True


def install_comfyui(target_path: str, progress_cb: Optional[Callable] = None, use_mirror: bool = True) -> bool:
    """完整安装：ComfyUI portable + custom nodes。"""
    if not install_comfyui_portable(target_path, progress_cb, use_mirror=use_mirror):
        return False
    if not install_custom_nodes(target_path, progress_cb, use_mirror=use_mirror):
        return False
    if progress_cb:
        progress_cb("done", 1.0, "ComfyUI 安装完成，请在模型管理器中下载所需模型")
    return True


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


def check_missing_models(comfyui_path: str) -> List[Dict]:
    missing = []
    for model in MODEL_REGISTRY:
        if find_model_file(comfyui_path, model) is None:
            missing.append(model)
    return missing


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


def is_model_installed(comfyui_path: str, model_id: str) -> bool:
    for model in MODEL_REGISTRY:
        if model["id"] == model_id:
            return find_model_file(comfyui_path, model) is not None
    return False


def download_model(comfyui_path: str, model_id: str, progress_cb: Optional[Callable] = None, hf_token: str = "", use_mirror: bool = True) -> bool:
    """下载单个模型文件到 ComfyUI 对应目录。"""
    model = next((m for m in MODEL_REGISTRY if m["id"] == model_id), None)
    if not model:
        if progress_cb:
            progress_cb("model", 0.0, f"未知模型: {model_id}")
        return False

    url = model.get("url") or ""
    if not url:
        if progress_cb:
            progress_cb("model", 0.0, f"模型 {model['label']} 没有配置自动下载链接，请使用网页按钮手动下载")
        return False

    base_dir = _model_base_dir(comfyui_path)
    target_dir = os.path.join(base_dir, model["dir"])
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, model["filename"])

    # ── 磁盘空间预检 ──
    size_str = model.get("size", "0")
    size_bytes = _parse_model_size_bytes(size_str)
    if size_bytes > 0:
        if not _check_disk_space(dest_path, size_bytes + (2 * 1024 ** 3), progress_cb):
            return False

    existing_path = find_model_file(comfyui_path, model)
    if existing_path:
        file_size = os.path.getsize(existing_path)
        expected_sha256 = model.get("sha256", "")
        hash_ok = True
        if expected_sha256:
            try:
                hash_ok = _sha256_file(existing_path).lower() == expected_sha256.lower()
            except Exception:
                hash_ok = False
        if file_size > 0 and (not size_bytes or file_size >= size_bytes * 0.9) and hash_ok:
            if progress_cb:
                progress_cb("model", 1.0, f"{model['label']} 已存在: {existing_path}")
            return True
        else:
            # 空文件或明显不完整的文件视为不存在，重新下载
            try:
                os.remove(existing_path)
            except OSError:
                log.debug("Failed to remove empty model file: %s", existing_path)

    if progress_cb:
        progress_cb("model", 0.0, f"准备下载 {model['label']} 到 {dest_path}")

    mirror_urls = list(model.get("mirrors", []))
    if use_mirror:
        urls = [url] + mirror_urls
    else:
        # 关闭国内镜像时优先走官方源（model["mirrors"] 中保存的是官方/备选源）
        urls = mirror_urls + [url]
    # 去重同时保持顺序
    seen = set()
    unique_urls = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return _download_with_progress(
        unique_urls,
        dest_path,
        progress_cb,
        hf_token=hf_token,
        expected_size=size_bytes,
        expected_sha256=model.get("sha256", ""),
    )
