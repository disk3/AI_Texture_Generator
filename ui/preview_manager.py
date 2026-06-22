import bpy
import bpy.utils.previews
import os

_preview_collection = {}


def get_preview_collection():
    return _preview_collection.get("ai_concept")


def register_preview_collection():
    unregister_preview_collection()
    pcoll = bpy.utils.previews.new()
    _preview_collection["ai_concept"] = pcoll


def unregister_preview_collection():
    pcoll = _preview_collection.pop("ai_concept", None)
    if pcoll is not None:
        bpy.utils.previews.remove(pcoll)


def load_preview(name: str, filepath: str) -> int:
    """Load an image file into the preview collection and return its icon_id."""
    pcoll = get_preview_collection()
    if pcoll is None:
        return 0

    if name in pcoll:
        return pcoll[name].icon_id

    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        return 0

    try:
        pcoll.load(name, abs_path, 'IMAGE')
        return pcoll[name].icon_id
    except Exception:
        # Image may be corrupt, missing, or unsupported format
        return 0


def get_icon_id(name: str) -> int:
    pcoll = get_preview_collection()
    if pcoll is not None and name in pcoll:
        return pcoll[name].icon_id
    return 0


def remove_preview(name: str):
    pcoll = get_preview_collection()
    if pcoll is not None and name in pcoll:
        del pcoll[name]


def clear_previews():
    pcoll = get_preview_collection()
    if pcoll is not None:
        pcoll.clear()


def register():
    register_preview_collection()


def unregister():
    unregister_preview_collection()
