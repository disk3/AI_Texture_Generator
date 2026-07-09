import os
import bpy

from ..utils.logger import get_logger

log = get_logger(__name__)

_preview_collection = None


def _ensure_preview_collection():
    """懒加载并缓存自定义图标集合。"""
    global _preview_collection
    if _preview_collection is not None:
        return _preview_collection

    pcoll = bpy.utils.previews.new()
    icon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    icon_path = os.path.join(icon_dir, "download_arrow.png")
    if os.path.isfile(icon_path):
        try:
            pcoll.load("download_arrow", icon_path, 'IMAGE')
            log.debug("Loaded custom icon: %s", icon_path)
        except Exception as e:
            log.warning("Could not load custom icon %s: %s", icon_path, e)
    else:
        log.warning("Custom icon not found: %s", icon_path)

    _preview_collection = pcoll
    return _preview_collection


def get_icon_id(name: str) -> int:
    """返回已加载自定义图标的 icon_id；未找到返回 0。"""
    pcoll = _ensure_preview_collection()
    icon = pcoll.get(name)
    return icon.icon_id if icon else 0


def register():
    _ensure_preview_collection()


def unregister():
    global _preview_collection
    if _preview_collection is not None:
        try:
            bpy.utils.previews.remove(_preview_collection)
        except Exception:
            pass
        _preview_collection = None
