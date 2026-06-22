import queue
import threading
import bpy

from .logger import get_logger

log = get_logger(__name__)
_result_queue = queue.Queue()
_orchestrator = None
_orchestrator_lock = threading.Lock()


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                from ..core.orchestrator import GenerationOrchestrator
                _orchestrator = GenerationOrchestrator()
    return _orchestrator


def thread_safe_callback(result: dict):
    _result_queue.put(result)


def blender_timer_poll() -> float:
    try:
        while True:
            result = _result_queue.get_nowait()
            _apply_result_in_main_thread(result)
    except queue.Empty:
        pass
    return 0.1


def _apply_result_in_main_thread(result: dict):
    if bpy.context is None:
        # Scene may be loading or closing; skip this update
        return
    try:
        props = bpy.context.scene.ai_concept_props
    except (AttributeError, KeyError):
        log.debug("Scene properties not available (scene may be closing)")
        return

    status = result.get("status")

    if status == "progress":
        props.progress = result.get("progress", 0.0)
        props.status_message = result.get("message", "")
    elif status == "done":
        props.is_generating = False
        props.progress = 1.0
        props.status_message = "完成"
    elif status == "error":
        props.is_generating = False
        props.progress = 0.0
        props.status_message = f"错误: {result.get('message', '')}"
    elif status == "cancelled":
        props.is_generating = False
        props.progress = 0.0
        props.status_message = "已取消"


def register():
    bpy.app.timers.register(blender_timer_poll, first_interval=0.5, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(blender_timer_poll):
        bpy.app.timers.unregister(blender_timer_poll)
