"""ComfyUI 工作流模型族规范与动态子图合并工具。

本模块不包含任何 ComfyUI 运行时依赖，仅操作 workflow JSON 字典，
方便在单元测试中直接导入验证。
"""

import json
import os
from typing import Dict, Any, Tuple


#: 模型族规范注册表。新增模型族时，只需在这里追加条目并创建对应 JSON 模板。
WORKFLOW_FAMILIES: Dict[str, Dict[str, Any]] = {
    "zimage": {
        "label": "Z-Image / Lumina2",
        "gen_template": "gen_zimage.json",
        "post_lowres_template": "post_pbr_lowres.json",
        "post_hires_template": "post_pbr_hires.json",
        "seamless_method": "PPS",
        "model_kind_filter": {"unet", "diffusion_models"},
        "nodes": {
            # 生图子图节点
            "main_model":      {"tpl": "gen",  "id": "1",   "class": "UNETLoader",         "field": "unet_name"},
            "text_encoder":    {"tpl": "gen",  "id": "3",   "class": "CLIPLoader",         "field": "clip_name"},
            "positive":        {"tpl": "gen",  "id": "4",   "class": "CLIPTextEncode",     "field": "text"},
            "latent":          {"tpl": "gen",  "id": "6",   "class": "EmptySD3LatentImage", "fields": ["width", "height"]},
            "sampler":         {"tpl": "gen",  "id": "7",   "class": "KSampler",           "fields": ["seed", "cfg", "steps", "sampler_name", "scheduler", "denoise"]},
            "vae":             {"tpl": "gen",  "id": "8",   "class": "VAELoader",          "field": "vae_name"},
            "vae_decode":      {"tpl": "gen",  "id": "9",   "class": "VAEDecode"},
            # 后处理子图节点
            "post_input":      {"tpl": "post", "id": "11",  "class": "ResizeAndPadImage",  "field": "image"},
            "chord_model":     {"tpl": "post", "id": "12",  "class": "ChordLoadModel",     "field": "ckpt_name"},
            "seedvr2_basecolor":{"tpl": "post", "id": "111", "class": "SeedVR2VideoUpscaler", "field": "resolution"},
            "seedvr2_normal":  {"tpl": "post", "id": "121", "class": "SeedVR2VideoUpscaler", "field": "resolution"},
        },
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 9,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "shift": None,  # Z-Image 的 shift 固化在 ModelSamplingAuraFlow 节点中
        },
    },
}


def get_family_spec(family_id: str) -> Dict[str, Any]:
    """返回指定模型族的规范；未知族时回退到 zimage（保持现有行为）。"""
    return WORKFLOW_FAMILIES.get(family_id, WORKFLOW_FAMILIES["zimage"])


def get_family_min_version(family_id: str) -> str:
    """返回指定模型族所需的 ComfyUI 核心最低版本；无要求时返回空字符串。"""
    return get_family_spec(family_id).get("min_comfyui_version", "")


def get_family_seamless_method(family_id: str) -> str:
    """返回指定模型族在 ComfyUI/CHORD 流程中默认使用的本地无缝化算法。

    当前统一使用 PPS：不偏移、不裁剪原图，仅让对向边界一致，避免 PBR 通道间错位。
    """
    return get_family_spec(family_id).get("seamless_method", "PPS")


def get_workflows_dir() -> str:
    """返回 workflows/ 目录的绝对路径。"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows")


def load_workflow_template(name: str) -> Dict[str, Any]:
    """加载 workflows/ 目录下的 JSON 模板。"""
    path = os.path.join(get_workflows_dir(), name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unique_post_id(original_id: str, used_ids: set) -> str:
    """为后处理子图节点生成不与生图子图冲突的新 ID。"""
    candidate = f"post_{original_id}"
    if candidate not in used_ids:
        return candidate
    # 极小概率冲突时追加数字
    suffix = 1
    while f"{candidate}_{suffix}" in used_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def merge_workflow_subgraphs(
    gen_wf: Dict[str, Any],
    post_wf: Dict[str, Any],
    gen_output_node_id: str,
    post_input_node_id: str,
    post_input_field: str = "image",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """把生图子图和后处理子图合并成一个完整 workflow。

    Args:
        gen_wf: 生图子图（节点 ID 保持不变）。
        post_wf: 后处理子图（节点 ID 会被加上 ``post_`` 前缀以避免冲突）。
        gen_output_node_id: 生图子图输出图像的节点 ID（取 output slot 0）。
        post_input_node_id: 后处理子图接收图像的节点 ID。
        post_input_field: 后处理子图接收图像的输入字段名，默认 ``image``。

    Returns:
        (merged_workflow, post_id_map)
        ``post_id_map`` 将后处理子图原始节点 ID 映射到合并后的新 ID。
    """
    merged = {k: _deep_copy_node(v) for k, v in gen_wf.items()}
    used_ids = set(merged.keys())

    # 1. 建立后处理节点 ID 映射
    post_id_map: Dict[str, str] = {}
    for node_id in post_wf:
        post_id_map[node_id] = _unique_post_id(node_id, used_ids)
        used_ids.add(post_id_map[node_id])

    # 2. 复制后处理节点，并 remap 内部引用
    for node_id, node_def in post_wf.items():
        new_id = post_id_map[node_id]
        merged[new_id] = _deep_copy_node(node_def)
        _remap_node_inputs(merged[new_id], post_id_map)

    # 3. 把生图输出接到后处理输入
    new_post_input_id = post_id_map[post_input_node_id]
    merged[new_post_input_id]["inputs"][post_input_field] = [gen_output_node_id, 0]

    # 4. 后处理子图模板里通常硬编码了来自旧生图输出的引用（如 ["9", 0]），
    #    当实际生图输出节点不是 "9" 时，需要把后处理子图里所有指向该占位符的
    #    引用统一替换成真正的生图输出节点。
    placeholder_ref = _copy_input_value(
        post_wf.get(post_input_node_id, {}).get("inputs", {}).get(post_input_field)
    )
    actual_ref = [gen_output_node_id, 0]
    if placeholder_ref and placeholder_ref != actual_ref:
        for node_id in post_wf:
            _replace_input_reference(merged[post_id_map[node_id]], placeholder_ref, actual_ref)

    return merged, post_id_map


def _deep_copy_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """浅拷贝节点定义；inputs 内部可能含列表引用，需要独立副本。"""
    copied = dict(node)
    copied["inputs"] = {k: _copy_input_value(v) for k, v in node.get("inputs", {}).items()}
    return copied


def _copy_input_value(value: Any) -> Any:
    """拷贝输入值；对 [node_id, slot] 引用返回新列表。"""
    if isinstance(value, list):
        return list(value)
    return value


def _remap_node_inputs(node_def: Dict[str, Any], id_map: Dict[str, str]) -> None:
    """将节点输入中指向后处理子图内部的引用替换为新 ID。"""
    inputs = node_def.get("inputs", {})
    for key, value in inputs.items():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in id_map:
            inputs[key] = [id_map[value[0]], value[1]]


def _replace_input_reference(node_def: Dict[str, Any], old_ref: list, new_ref: list) -> None:
    """将节点输入中等于 old_ref 的引用统一替换为 new_ref。"""
    inputs = node_def.get("inputs", {})
    for key, value in inputs.items():
        if isinstance(value, list) and len(value) == 2 and value == old_ref:
            inputs[key] = list(new_ref)


def resolve_node_id(
    family_spec: Dict[str, Any],
    logical_name: str,
    post_id_map: Dict[str, str],
) -> str:
    """根据逻辑名取得合并后 workflow 中的实际节点 ID。"""
    node_spec = family_spec["nodes"][logical_name]
    original_id = node_spec["id"]
    if node_spec["tpl"] == "post":
        return post_id_map[original_id]
    return original_id
