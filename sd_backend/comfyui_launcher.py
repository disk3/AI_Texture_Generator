"""ComfyUI 自动启动器。

检测 ComfyUI 是否运行，若未运行则自动启动本地安装的实例。
"""

import os
import re
import signal
import subprocess
import time

from ..utils.logger import get_logger

log = get_logger(__name__)


def _find_launch_script(comfyui_path: str) -> str:
    """在 ComfyUI 安装目录中查找可用的启动脚本。"""
    if not comfyui_path or not os.path.isdir(comfyui_path):
        return ""

    # 优先级：5070Ti 修复脚本 > 快速 FP16 > 标准 NVIDIA GPU > CPU
    candidates = [
        "fix_5070ti_portable.bat",
        "run_nvidia_gpu_fast_fp16_accumulation.bat",
        "run_nvidia_gpu.bat",
        "run_cpu.bat",
    ]
    for name in candidates:
        script = os.path.join(comfyui_path, name)
        if os.path.isfile(script):
            return script
    return ""


def _sanitize_auto_launch(args: list) -> list:
    """过滤掉会导致浏览器自动弹出的 --auto-launch 参数及其变体。"""
    cleaned = []
    for a in args:
        al = a.lower()
        if al in ("--auto-launch", "--autolaunch") or al.startswith(("--auto-launch=", "--autolaunch=")):
            continue
        cleaned.append(a)
    return cleaned


def _parse_bat_args(bat_path: str) -> list:
    """从 bat 脚本中提取传给 ComfyUI/main.py 的参数，过滤 --auto-launch。"""
    try:
        with open(bat_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        log.debug("Failed to read batch file: %s", bat_path)
        return []

    for line in lines:
        line = line.strip()
        if not line or line.startswith(("::", "rem ", "@echo", "echo ", "pause", "if ", "cd ", "set ")):
            continue
        if "python" in line.lower() and "main.py" in line.lower():
            if line.startswith("@"):
                line = line[1:].strip()
            parts = line.split()
            main_idx = -1
            for i, p in enumerate(parts):
                if "main.py" in p:
                    main_idx = i
                    break
            if main_idx == -1:
                continue
            args = parts[main_idx + 1:]
            args = [
                a for a in args
                if a.lower() not in ("--auto-launch", "--autolaunch", "pause", "%*", ">nul", "2>nul", ">>")
                and not a.startswith((">", "%"))
            ]
            return args
    return []


def _build_launch_cmd(comfyui_path: str) -> list:
    """构建启动命令列表（优先直接调用 python，避免 bat 脚本里的 --auto-launch 弹出浏览器）。

    支持两种目录结构：
      - 便携版：<root>/ComfyUI/main.py + <root>/python_embeded/python.exe
      - 桌面版：<root>/ComfyUI/main.py（使用系统 Python）
    """
    from .comfyui_installer import get_comfyui_type, get_python_exe

    ctype = get_comfyui_type(comfyui_path)
    if ctype is None:
        return []

    main_py = os.path.join(comfyui_path, "ComfyUI", "main.py")
    python_exe = get_python_exe(comfyui_path)
    if not python_exe:
        return []

    args = [python_exe, "-s", main_py, "--disable-auto-launch"]
    if ctype == "portable":
        args.append("--windows-standalone-build")

    # fallback: 从 bat 脚本提取额外参数
    bat = _find_launch_script(comfyui_path)
    if bat:
        extra_args = _parse_bat_args(bat)
        # 过滤掉会与上面冲突的参数
        seen = set(args)
        for a in extra_args:
            if a not in seen:
                args.append(a)

    return _sanitize_auto_launch(args)


_comfyui_process = None


def launch_comfyui(comfyui_path: str) -> bool:
    """启动本地 ComfyUI 实例，返回是否成功提交启动。"""
    global _comfyui_process

    cmd = _build_launch_cmd(comfyui_path)
    if not cmd:
        return False

    # 纯后台运行：无窗口、不阻塞 Blender、不弹浏览器
    try:
        cmd = _sanitize_auto_launch(cmd)
        kwargs = {
            "cwd": comfyui_path,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP 让我们可以用 CTRL_BREAK_EVENT 结束整个进程树
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = si
        else:
            # Linux/macOS：创建新会话，退出时可直接 killpg 整个进程组
            kwargs["start_new_session"] = True
        _comfyui_process = subprocess.Popen(cmd, **kwargs)
        return True
    except Exception:
        log.warning("Failed to launch ComfyUI process")
        return False


def _kill_by_port(port: int):
    """按端口查找并终止占用进程（Windows）。

    使用纯 Python 过滤 netstat 输出，避免 shell 管道拼接。
    """
    if os.name != "nt" or not port:
        return
    if not isinstance(port, int) or port <= 0 or port > 65535:
        log.warning("Invalid port number: %s", port)
        return
    port_str = str(port)
    try:
        # 直接获取完整 netstat 输出，在 Python 中过滤
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        pids = set()
        for line in result.stdout.splitlines():
            # 匹配 LISTENING 行中端口匹配的条目
            parts = line.strip().split()
            if len(parts) >= 5:
                addr = parts[1]
                if addr.endswith(f":{port_str}"):
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        for pid in pids:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
    except Exception as e:
        log.warning("Failed to kill process by port %d: %s", port, e)


def _extract_port(base_url: str) -> int:
    """从 http://host:port 中提取端口号。"""
    try:
        match = re.search(r":(\d+)(?:/|$)", base_url or "")
        if match:
            return int(match.group(1))
    except (ValueError, Exception):
        log.debug("Failed to extract port from URL: %s", base_url)
    return 8188


def shutdown_comfyui(base_url: str = ""):
    """关闭由本插件自动启动的 ComfyUI 进程及其子进程。"""
    global _comfyui_process

    port = _extract_port(base_url)

    if _comfyui_process is not None:
        try:
            pid = _comfyui_process.pid
            if os.name == "nt":
                # Windows: 先发送 CTRL_BREAK_EVENT 结束整个进程组
                try:
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                except (AttributeError, OSError):
                    log.debug("CTRL_BREAK_EVENT not available")
                # 再 taskkill /T 确保终止
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (AttributeError, OSError):
                    log.debug("killpg not available")
                try:
                    _comfyui_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _comfyui_process.kill()
        except Exception as e:
            log.warning("Failed to shutdown ComfyUI process: %s", e)
        finally:
            _comfyui_process = None

    # 即使 _comfyui_process 已丢失或无法终止，也按端口清理残留进程
    _kill_by_port(port)


def wait_for_comfyui(base_url: str, timeout: int = 120) -> bool:
    """轮询等待 ComfyUI 启动成功。"""
    import requests
    url = base_url.rstrip("/") + "/system_stats"
    for _ in range(timeout):
        try:
            # connect 1s / read 4s，避免启动初期响应慢被误判为失败
            resp = requests.get(url, timeout=(1, 4))
            if resp.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    log.warning("ComfyUI did not start within %ds", timeout)
    return False


def is_comfyui_running(base_url: str) -> bool:
    """检查 ComfyUI 是否已响应。"""
    import requests
    try:
        resp = requests.get(base_url.rstrip("/") + "/system_stats", timeout=(1, 5))
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
