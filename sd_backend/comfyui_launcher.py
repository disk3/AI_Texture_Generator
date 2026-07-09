"""ComfyUI 自动启动器。

检测 ComfyUI 是否运行，若未运行则自动启动本地安装的实例。
"""

import os
import re
import signal
import subprocess
import time
import traceback

from ..utils.logger import get_logger

log = get_logger(__name__)


def _read_log_tail(log_path: str, lines: int = 30) -> str:
    """读取日志文件最后 N 行，用于快速诊断启动失败原因。"""
    if not os.path.isfile(log_path):
        return "(no log file)"
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"(failed to read log: {e})"


def _requests_module():
    try:
        import requests
        return requests
    except ImportError:
        from ..utils import simple_requests
        return simple_requests


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
    """过滤掉会导致浏览器自动弹出的 --auto-launch 参数，保留 --disable-auto-launch。"""
    cleaned = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        al = a.lower()
        # 保留显式禁用 auto-launch 的参数
        if al in ("--disable-auto-launch", "--disable-autolaunch"):
            cleaned.append(a)
            continue
        if al in ("--auto-launch", "--autolaunch"):
            # 形如 --auto-launch value 时，跳过后面的值（只要不是下一个参数）
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                skip_next = True
            continue
        if al.startswith(("--auto-launch=", "--autolaunch=")):
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

    args = [python_exe, "-s", main_py]
    if ctype == "portable":
        args.append("--windows-standalone-build")

    # 禁止 ComfyUI 启动后自动弹出浏览器
    args.append("--disable-auto-launch")

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
        log.warning("无法为 %s 构建 ComfyUI 启动命令（路径无效或找不到 python）", comfyui_path)
        return False

    log_path = os.path.join(comfyui_path, "comfyui_launcher.log")
    log.info("Launching ComfyUI: %s", " ".join(cmd))
    log.info("ComfyUI stdout/stderr -> %s", log_path)

    try:
        cmd = _sanitize_auto_launch(cmd)
        log_file = open(log_path, "a", encoding="utf-8", errors="ignore")
        log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launch command: {' '.join(cmd)}\n")
        log_file.flush()

        kwargs = {
            "cwd": comfyui_path,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = si
        else:
            kwargs["start_new_session"] = True

        _comfyui_process = subprocess.Popen(cmd, **kwargs)

        # 给进程 2 秒时间，若已退出则读取日志并提示
        time.sleep(2)
        early_code = _comfyui_process.poll()
        if early_code is not None:
            log_file.flush()
            tail = _read_log_tail(log_path, 30)
            log.warning(
                "ComfyUI process exited early (return code %d). Last log lines:\n%s",
                early_code, tail,
            )
            _comfyui_process = None
            return False

        return True
    except Exception:
        log.warning("Failed to launch ComfyUI process:\n%s", traceback.format_exc())
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
            timeout=10,
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
                timeout=10,
            )
    except Exception as e:
        log.warning("Failed to kill process by port %d: %s", port, e)


def _extract_port(base_url: str) -> int:
    """从 http://host:port 中提取端口号。"""
    try:
        match = re.search(r":(\d+)(?:/|$)", base_url or "")
        if match:
            return int(match.group(1))
    except (ValueError, OSError):
        log.debug("Failed to extract port from URL: %s", base_url)
    return 8188


def shutdown_comfyui(base_url: str = "", force_by_port: bool = False):
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
                    timeout=10,
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

    if force_by_port:
        _kill_by_port(port)


def wait_for_comfyui(base_url: str, timeout: int = 120, comfyui_path: str = "") -> bool:
    """轮询等待 ComfyUI 启动成功。"""
    requests = _requests_module()
    url = base_url.rstrip("/") + "/system_stats"
    for _ in range(timeout):
        try:
            # connect 1s / read 4s，避免启动初期响应慢被误判为失败
            resp = requests.get(url, timeout=(1, 4))
            if resp.status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
            pass
        time.sleep(1)

    log_path = os.path.join(comfyui_path, "comfyui_launcher.log") if comfyui_path else ""
    tail = _read_log_tail(log_path, 30) if log_path else "(no log path)"
    log.warning(
        "ComfyUI did not start within %ds. Last log lines:\n%s",
        timeout, tail,
    )
    return False


def is_comfyui_running(base_url: str) -> bool:
    """检查 ComfyUI 是否已响应。"""
    requests = _requests_module()
    try:
        resp = requests.get(base_url.rstrip("/") + "/system_stats", timeout=(1, 5))
        return resp.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
        return False
