import os
import bpy

from . import preferences
from .ui import preview_manager
from .sd_backend import comfyui_installer
from .utils.logger import get_logger

log = get_logger(__name__)


def _prop_row(box, data, prop, label, factor=0.35):
    """绘制左对齐标签 + 右侧控件的统一行。"""
    row = box.row()
    split = row.split(factor=factor)
    col = split.column()
    col.alignment = 'LEFT'
    col.label(text=label)
    split.prop(data, prop, text="")


def _addon_prefs(context):
    addon_pkg = __package__.split('.')[0]
    prefs = context.preferences.addons[addon_pkg].preferences
    preferences.ensure_default_api_providers(prefs)
    return prefs


def _provider_status(prefs, provider):
    if provider == 'LOCAL_COMFYUI':
        if not prefs.comfyui_url:
            return "请在偏好设置中填写 ComfyUI 地址", 'ERROR'
        install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()
        ctype = comfyui_installer.get_comfyui_type(install_path)
        if ctype is None:
            return "ComfyUI 未安装，请在偏好设置中安装", 'ERROR'
        if ctype == "desktop":
            return "ComfyUI 桌面版已配置，请确保已手动启动", 'CHECKMARK'
        if prefs.comfyui_path:
            return "ComfyUI 便携版已配置，自动启动已启用", 'CHECKMARK'
        return "ComfyUI 便携版已安装，自动启动已启用", 'CHECKMARK'

    if preferences.is_api_provider_value(provider):
        api_provider = preferences.find_api_provider(prefs, preferences.provider_id_from_value(provider))
        if not api_provider:
            return "未找到 API Provider", 'ERROR'
        if not api_provider.api_key:
            return f"请在偏好设置中填写 {api_provider.name} API Key", 'ERROR'
        if not api_provider.base_url:
            return f"请在偏好设置中填写 {api_provider.name} Base URL", 'ERROR'
        return f"{api_provider.name} 已配置", 'CHECKMARK'

    return "Provider 未配置", 'ERROR'


def _draw_api_model_selector(box, props, prefs, provider):
    if not preferences.is_api_provider_value(provider):
        return
    provider_id = preferences.provider_id_from_value(provider)
    api_provider = preferences.find_api_provider(prefs, provider_id)
    row = box.row(align=True)
    row.prop(props, "api_image_model", text="图像模型")
    fetch = row.operator("ai_concept.fetch_models", text="", icon='FILE_REFRESH')
    fetch.provider_id = provider_id
    if api_provider and props.api_image_model == 'DEFAULT':
        box.label(text=f"默认: {api_provider.default_image_model}", icon='INFO')


def _is_local_comfyui_ready(prefs) -> bool:
    install_path = prefs.comfyui_path or comfyui_installer.get_default_install_path()
    return comfyui_installer.is_comfyui_installed(install_path)


def _draw_generation_backend(box, props, prefs):
    provider = props.texture_generator
    row = box.row(align=True)
    row.prop(props, "texture_generator", text="后端")
    op = row.operator("ai_concept.test_provider_connection", text="", icon='CHECKMARK')
    op.provider = 'COMFYUI' if provider == 'LOCAL_COMFYUI' else provider

    status_text, status_icon = _provider_status(prefs, provider)
    box.label(text=status_text, icon=status_icon)

    if provider == 'LOCAL_COMFYUI':
        row = box.row(align=True)
        row.prop(props, "local_comfyui_model", text="主模型")
        row.operator("ai_concept.refresh_comfyui_models", text="", icon='FILE_REFRESH')

    if preferences.is_api_provider_value(provider):
        _draw_api_model_selector(box, props, prefs, provider)


class AI_PT_TexturePanel(bpy.types.Panel):
    bl_label = "AI 材质生成"
    bl_idname = "AI_PT_texture_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AI 材质"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ai_concept_props
        prefs = _addon_prefs(context)

        runtime_missing = preferences.dependency_missing_for_profile('LOCAL_PBR')
        if runtime_missing:
            box = layout.box()
            box.alert = True
            box.label(text="基础组件缺失，生成工具暂不可用", icon='ERROR')
            op = box.operator("ai_concept.repair_dependencies", text="一键修复基础组件", icon='IMPORT')
            op.profile = 'LOCAL_PBR'
            box.label(text="修复完成后请重新启用插件或重启 Blender")
            return

        # Generation Backend
        box = layout.box()
        box.label(text="生成后端", icon='PREFERENCES')
        _draw_generation_backend(box, props, prefs)

        # Material Config
        box = layout.box()
        box.label(text="材质配置", icon='MATERIAL')
        _prop_row(box, props, "texture_category", "分类")
        _prop_row(box, props, "texture_subcategory", "子分类")
        _prop_row(box, props, "texture_color", "颜色")
        _prop_row(box, props, "texture_finish", "表面处理")
        _prop_row(box, props, "texture_feature", "特征")
        _prop_row(box, props, "texture_orientation", "视角")
        _prop_row(box, props, "texture_usecase", "用途")

        # Reference Image
        box = layout.box()
        box.label(text="使用参考图", icon='IMAGE_REFERENCE')

        ref_icon_id = preview_manager.get_icon_id(props.reference_image.name) if props.reference_image else 0
        if ref_icon_id:
            col = box.column(align=True)
            col.template_icon(ref_icon_id, scale=14.0)
            row = col.row(align=True)
            row.alignment = 'CENTER'
            op = row.operator("ai_concept.enlarge_preview", text="", icon='FULLSCREEN_ENTER')
            op.image_name = props.reference_image.name
            row.operator("ai_concept.load_reference_image", text="更换", icon='FILEBROWSER')
            row.operator("ai_concept.caption_reference_image", text="反推提示词", icon='VIEWZOOM')
            row.operator("ai_concept.clear_reference_image", text="", icon='X')
            _prop_row(box, props, "reference_denoise", "重绘幅度")
            if props.texture_generator == 'LOCAL_COMFYUI' and _is_local_comfyui_ready(prefs):
                _prop_row(box, props, "reference_tile_strength", "纹理保持")
        else:
            box.separator(factor=4.0)
            col = box.column(align=True)
            col.alignment = 'CENTER'
            col.menu("AI_MT_reference_image_menu", text="添加参考图", icon='ADD')

        # Prompt
        box = layout.box()
        box.label(text="提示词", icon='TEXT')
        row = box.row(align=True)
        row.prop(props, "prompt", text="")
        lock_icon = 'LOCKED' if props.lock_prompt else 'UNLOCKED'
        row.prop(props, "lock_prompt", text="", icon=lock_icon, toggle=True)
        row.operator("ai_concept.edit_prompt_in_text_editor", text="", icon='TEXT')
        row.operator("ai_concept.optimize_prompt", text="", icon='MODIFIER')
        # 高级：负面提示词（折叠明细）
        box.prop(
            props, "negative_prompt",
            text="负面提示词",
            icon='GHOST_DISABLED',
            emboss=False,
        )

        # Output Size & Parameters
        box = layout.box()
        box.label(text="输出参数", icon='SETTINGS')
        row = box.row(align=True)
        row.prop(props, "width", text="宽")
        row.prop(props, "height", text="高")
        # 种子只在 LOCAL_COMFYUI 后端下有意义
        if props.texture_generator == 'LOCAL_COMFYUI':
            row = box.row(align=True)
            row.prop(props, "seed", text="种子")

        # 本地离线 PBR 烘焙开关：只要检测到本地 ComfyUI 就绪就显示
        # 这样即使使用 API 后端，也可以选择跳过 CHORD、用本地算法生成 PBR
        if _is_local_comfyui_ready(prefs):
            row = box.row(align=True)
            row.prop(props, "use_local_pbr", text="本地离线 PBR 烘焙")

        # 本地 PBR 法线参数（紧接在「不用 ComfyUI」开关下方）
        show_local_normal = props.use_local_pbr
        try:
            show_local_normal = show_local_normal or not _is_local_comfyui_ready(prefs)
        except Exception:
            show_local_normal = True
        if show_local_normal:
            try:
                box = layout.box()
                box.label(text="本地法线参数", icon='MATERIAL')
                _prop_row(box, props, "normal_strength", "强度")
                _prop_row(box, props, "normal_detail", "细节")
                box.prop(props, "normal_invert", text="反转绿色通道")
            except Exception as e:
                box = layout.box()
                box.label(text=f"法线参数显示错误: {e}", icon='ERROR')

        # Maps
        box = layout.box()
        box.label(text="贴图输出", icon='TEXTURE')
        row = box.row(align=True)
        row.prop(props, "output_basecolor", text="基础色")
        row.prop(props, "output_normal", text="法线")
        row.prop(props, "output_roughness", text="粗糙度")
        row = box.row(align=True)
        row.prop(props, "output_metalness", text="金属度")
        row.prop(props, "output_height", text="高度")

        # Status & Progress（只读展示，使用 label 替代可编辑的 prop）
        box = layout.box()
        box.label(text="状态", icon='INFO')
        row = box.row()
        row.label(text=props.status_message or "就绪")
        if props.is_generating:
            pct = int(props.progress * 100)
            box.label(text=f"进度: {pct}%", icon='TIME')
        elif props.progress >= 1.0:
            box.label(text="进度: 100% ✓", icon='CHECKMARK')

        row = box.row(align=True)
        if props.is_generating:
            row.operator("ai_concept.stop", text="停止", icon='PAUSE')
        else:
            row.operator("ai_concept.generate", text="生成 PBR 贴图", icon='PLAY')

        # Channel Pack（仅在有 PBR 结果时显示）
        has_pbr_result = any(
            item.result_type == 'pbr' and len(item.pbr_maps) > 0
            for item in props.results
        )
        if has_pbr_result:
            box = layout.box()
            box.label(text="通道打包", icon='IMAGE_RGB')
            row = box.row(align=True)
            row.prop(props, "pack_channel_r", text="R")
            row.prop(props, "pack_channel_g", text="G")
            row.prop(props, "pack_channel_b", text="B")
            row.prop(props, "pack_channel_a", text="A")
            box.operator("ai_concept.channel_pack_rebuild", text="重建通道打包材质", icon='MATERIAL')

        # Results
        if props.results:
            box = layout.box()
            box.label(text=f"贴图预览（共 {len(props.results)} 个结果）", icon='RENDER_RESULT')

            for idx, item in enumerate(props.results):
                result_box = box.box()
                row = result_box.row(align=True)
                icon_expand = 'TRIA_DOWN' if item.is_expanded else 'TRIA_RIGHT'
                row.prop(item, "is_expanded", text="", icon=icon_expand, emboss=False)
                row.label(text=f"{idx+1}. {item.name}")
                row.label(text=item.timestamp)
                op = row.operator("ai_concept.remove_result", text="", icon='X')
                op.index = idx

                if not item.is_expanded:
                    continue

                maps = {m.map_type: m.image_name for m in item.pbr_maps}
                is_packed = 'packed' in maps

                display_types = ['diffuse']
                if is_packed:
                    display_types.append('packed')
                else:
                    if 'roughness' in maps:
                        display_types.append('roughness')
                    if 'metallic' in maps:
                        display_types.append('metallic')
                if 'normal' in maps:
                    display_types.append('normal')
                if 'height' in maps:
                    display_types.append('height')

                MAP_LABELS = {
                    'diffuse': '基础色',
                    'packed': '通道打包',
                    'roughness': '粗糙度',
                    'metallic': '金属度',
                    'normal': '法线',
                    'height': '高度',
                }

                for i in range(0, len(display_types), 2):
                    row = result_box.row(align=True)
                    row.alignment = 'CENTER'
                    for mt in display_types[i:i+2]:
                        col = row.column(align=True)
                        image_name = maps.get(mt, '')
                        icon_id = preview_manager.get_icon_id(image_name)
                        # 预览未加载时尝试从输出目录重新加载
                        if not icon_id and image_name:
                            output_dir = prefs.asset_output_path
                            if output_dir:
                                filepath = os.path.join(output_dir, f"{image_name}.png")
                                icon_id = preview_manager.load_preview(image_name, filepath)
                        if icon_id:
                            col.template_icon(icon_id, scale=6.0)
                        else:
                            col.label(text=MAP_LABELS.get(mt, mt), icon='IMAGE_DATA')
                        lbl = MAP_LABELS.get(mt, mt)
                        col.label(text=lbl)
                        btn_row = col.row(align=True)
                        btn_row.alignment = 'CENTER'
                        op = btn_row.operator("ai_concept.enlarge_preview", text="", icon='FULLSCREEN_ENTER')
                        op.image_name = maps.get(mt, '')
                        op2 = btn_row.operator("ai_concept.open_image_folder", text="", icon='FILEBROWSER')
                        op2.image_name = maps.get(mt, '')

            row = box.row()
            row.operator("ai_concept.clear_results", text="清空全部", icon='TRASH')


classes = [AI_PT_TexturePanel]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError as e:
            if "already registered" in str(e):
                log.debug("%s already registered, skipping.", cls.__name__)
                continue
            raise


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
