import bpy
from .preferences import (
    get_api_image_model_items,
    get_api_text_model_items,
    get_texture_provider_items,
)
from .utils.logger import get_logger

log = get_logger(__name__)

# =============================================================================
# 工业材质规范映射表（中文）
# =============================================================================

CATEGORY_ITEMS = [
    ('Conc', '混凝土', ''),
    ('Cem', '水泥', ''),
    ('Asph', '沥青', ''),
    ('Cer', '陶瓷/瓷砖', ''),
    ('Met', '金属', ''),
    ('Wd', '木材', ''),
    ('Stn', '石材', ''),
    ('Plas', '塑料', ''),
    ('Glas', '玻璃', ''),
    ('Veg', '植被', ''),
    ('Comp', '复合材料', ''),
    ('Coat', '涂层/涂料', ''),
    ('Rub', '橡胶', ''),
    ('Fab', '织物', ''),
    ('Soil', '土壤/沙子', ''),
    ('Insul', '保温材料', ''),
    ('Seal', '密封剂', ''),
    ('Gyp', '石膏', ''),
    ('StnPaint', '石材漆', ''),
]

SUBCATEGORY_BY_CATEGORY = {
    'Conc': [
        ('Nil', '--- 无 ---', ''),
        ('ExpAgg', '裸露骨料', ''),
        ('Stmp', '压花', ''),
        ('BrmFin', '扫帚饰面', ''),
        ('SmCst', '光滑浇筑', ''),
        ('RghCst', '粗糙浇筑', ''),
        ('ConcBlk', '混凝土砌块', ''),
    ],
    'Cem': [
        ('Nil', '--- 无 ---', ''),
        ('BrmFin', '扫帚饰面', ''),
        ('SmCst', '光滑浇筑', ''),
    ],
    'Asph': [
        ('Nil', '--- 无 ---', ''),
        ('Pors', '多孔', ''),
        ('Dens', '密级配', ''),
    ],
    'Cer': [
        ('Nil', '--- 无 ---', ''),
        ('Mosc', '马赛克', ''),
        ('Trraz', '水磨石', ''),
        ('Crkl', '开片釉', ''),
        ('Brick', '砖纹', ''),
        ('Clink', '熔岩砖', ''),
    ],
    'Met': [
        ('Nil', '--- 无 ---', ''),
        ('Crrug', '波纹板', ''),
        ('Perf', '穿孔板', ''),
        ('ChkPlt', '花纹钢板', ''),
        ('ExpMsh', '扩张网', ''),
        ('DiaPlt', '菱形花纹板', ''),
    ],
    'Wd': [
        ('Nil', '--- 无 ---', ''),
        ('Grain', '直纹', ''),
        ('Knot', '多结疤', ''),
        ('Rclm', '风化旧木', ''),
    ],
    'Stn': [
        ('Nil', '--- 无 ---', ''),
        ('Gran', '花岗岩', ''),
        ('Mrbl', '大理石', ''),
        ('Lmstn', '石灰石', ''),
        ('Slate', '板岩', ''),
        ('Rubl', '毛石', ''),
    ],
    'Plas': [
        ('Nil', '--- 无 ---', ''),
        ('FibReinf', '纤维增强', ''),
    ],
    'Glas': [
        ('Nil', '--- 无 ---', ''),
        ('Clear', '透明', ''),
        ('FrstGls', '磨砂玻璃', ''),
        ('Tint', '有色玻璃', ''),
    ],
    'Veg': [
        ('Nil', '--- 无 ---', ''),
        ('Grass', '草地', ''),
        ('Shrub', '灌木', ''),
    ],
    'Comp': [
        ('Nil', '--- 无 ---', ''),
        ('SMC', 'SMC', ''),
        ('FRP', 'FRP', ''),
    ],
    'Coat': [
        ('Nil', '--- 无 ---', ''),
        ('WarnStr', '警示条纹', ''),
        ('Latex', '乳胶漆', ''),
        ('Alkyd', '醇酸漆', ''),
    ],
    'Rub': [
        ('Nil', '--- 无 ---', ''),
        ('RubMat', '橡胶垫', ''),
        ('RubSeal', '橡胶密封条', ''),
    ],
    'Fab': [
        ('Nil', '--- 无 ---', ''),
        ('Woven', '机织', ''),
        ('NonWov', '无纺', ''),
    ],
    'Soil': [
        ('Nil', '--- 无 ---', ''),
        ('Grav', '砾石', ''),
        ('Sand', '沙子', ''),
        ('Clay', '黏土', ''),
    ],
    'Insul': [
        ('Nil', '--- 无 ---', ''),
        ('RkWool', '岩棉', ''),
        ('GlsWool', '玻璃棉', ''),
    ],
    'Seal': [
        ('Nil', '--- 无 ---', ''),
        ('SilSeal', '硅酮密封胶', ''),
        ('Epoxy', '环氧树脂', ''),
    ],
    'Gyp': [
        ('Nil', '--- 无 ---', ''),
        ('PlaBrd', '石膏板', ''),
        ('PlaCoat', '石膏涂层', ''),
    ],
    'StnPaint': [
        ('Nil', '--- 无 ---', ''),
        ('NatStn', '天然石饰面', ''),
        ('RckFlk', '岩片', ''),
    ],
}

COLOR_ITEMS = [
    ('Nil', '--- 无 ---', ''),
    ('Wht', '白色', ''),
    ('Blk', '黑色', ''),
    ('Gray', '灰色', ''),
    ('LtGray', '浅灰', ''),
    ('DkGray', '深灰', ''),
    ('Silv', '银色', ''),
    ('Red', '红色', ''),
    ('DkRed', '暗红', ''),
    ('Ylw', '黄色', ''),
    ('Blue', '蓝色', ''),
    ('Grn', '绿色', ''),
    ('Cyan', '青色', ''),
    ('Brwn', '棕色', ''),
    ('Orng', '橙色', ''),
    ('Purp', '紫色', ''),
    ('Pink', '粉色', ''),
    ('Beig', '米色', ''),
    ('Crem', '奶油色', ''),
    ('Tan', '棕褐色', ''),
    ('Taupe', '灰褐色', ''),
    ('OffWht', '米白', ''),
    ('Multi', '多彩', ''),
    ('TrrzPat', '水磨石图案', ''),
]

FINISH_ITEMS = [
    ('Smo', '光滑', ''),
    ('Rgh', '粗糙', ''),
    ('Pol', '抛光', ''),
    ('Brsh', '拉丝', ''),
    ('Mat', '哑光', ''),
    ('Gloss', '光泽', ''),
    ('Tex', '纹理', ''),
    ('Sbl', '喷砂', ''),
]

FEATURE_ITEMS = [
    ('Nil', '--- 无 ---', ''),
    ('AntiSk', '防滑', ''),
    ('Watprf', '防水', ''),
    ('Frst', '磨砂', ''),
    ('Emb', '压花', ''),
    ('Transp', '透明', ''),
    ('Reinf', '加强', ''),
    ('Refl', '反光', ''),
]

ORIENTATION_ITEMS = [
    ('TopDown', '俯视', ''),
    ('SideView', '侧视', ''),
    ('FrontView', '正视', ''),
]

USECASE_ITEMS = [
    ('Floor', '地面', ''),
    ('Wall', '墙面', ''),
    ('Roof', '屋顶', ''),
    ('Pavement', '人行道', ''),
    ('Structural', '结构', ''),
    ('Decorative', '装饰', ''),
    ('Insulation', '保温', ''),
    ('Foundation', '基础', ''),
]

# =============================================================================
# Prompt 映射表（中文）
# =============================================================================

CATEGORY_MAP = {
    'Conc': '混凝土',
    'Cem': '水泥砂浆',
    'Asph': '沥青',
    'Cer': '陶瓷砖',
    'Met': '金属',
    'Wd': '木材',
    'Stn': '石材',
    'Plas': '塑料',
    'Glas': '玻璃',
    'Veg': '植被',
    'Comp': '复合材料',
    'Coat': '涂料涂层',
    'Rub': '橡胶',
    'Fab': '织物',
    'Soil': '土壤',
    'Insul': '保温材料',
    'Seal': '密封剂',
    'Gyp': '石膏',
    'StnPaint': '石材漆',
}

SUBCATEGORY_MAP = {
    'Nil': '',
    'ExpAgg': '裸露骨料',
    'Stmp': '压花',
    'BrmFin': '扫帚饰面',
    'SmCst': '光滑浇筑',
    'RghCst': '粗糙浇筑',
    'ConcBlk': '混凝土砌块',
    'Pors': '多孔',
    'Dens': '密级配',
    'Mosc': '马赛克',
    'Trraz': '水磨石',
    'Crkl': '开片釉',
    'Brick': '砖纹',
    'Clink': '熔岩砖',
    'Crrug': '波纹板',
    'Perf': '穿孔板',
    'ChkPlt': '花纹钢板',
    'ExpMsh': '扩张网',
    'DiaPlt': '菱形花纹板',
    'Grain': '直纹',
    'Knot': '多结疤',
    'Rclm': '风化旧木',
    'Gran': '花岗岩',
    'Mrbl': '大理石',
    'Lmstn': '石灰石',
    'Slate': '板岩',
    'Rubl': '毛石',
    'FibReinf': '纤维增强',
    'Clear': '透明',
    'FrstGls': '磨砂玻璃',
    'Tint': '有色玻璃',
    'SMC': 'SMC',
    'FRP': 'FRP',
    'WarnStr': '警示条纹',
    'Latex': '乳胶漆',
    'Alkyd': '醇酸漆',
    'RubMat': '橡胶垫',
    'RubSeal': '橡胶密封条',
    'Woven': '机织',
    'NonWov': '无纺',
    'Grav': '砾石',
    'Sand': '沙子',
    'Clay': '黏土',
    'RkWool': '岩棉',
    'GlsWool': '玻璃棉',
    'SilSeal': '硅酮密封胶',
    'Epoxy': '环氧树脂',
    'Grass': '草地',
    'Shrub': '灌木',
    'PlaBrd': '石膏板',
    'PlaCoat': '石膏涂层',
    'NatStn': '天然石饰面',
    'RckFlk': '岩片',
}

COLOR_MAP = {
    'Nil': '',
    'Wht': '白色',
    'Blk': '黑色',
    'Gray': '灰色',
    'LtGray': '浅灰',
    'DkGray': '深灰',
    'Silv': '银色',
    'Red': '红色',
    'DkRed': '暗红',
    'Ylw': '黄色',
    'Blue': '蓝色',
    'Grn': '绿色',
    'Cyan': '青色',
    'Brwn': '棕色',
    'Orng': '橙色',
    'Purp': '紫色',
    'Pink': '粉色',
    'Beig': '米色',
    'Crem': '奶油色',
    'Tan': '棕褐色',
    'Taupe': '灰褐色',
    'OffWht': '米白',
    'Multi': '多彩',
    'TrrzPat': '水磨石图案',
}

FINISH_MAP = {
    'Smo': '光滑',
    'Rgh': '粗糙',
    'Pol': '抛光',
    'Brsh': '拉丝',
    'Mat': '哑光',
    'Gloss': '光泽',
    'Tex': '纹理',
    'Sbl': '喷砂',
}

FEATURE_MAP = {
    'Nil': '',
    'AntiSk': '防滑',
    'Watprf': '防水',
    'Frst': '磨砂',
    'Emb': '压花',
    'Transp': '透明',
    'Reinf': '加强',
    'Refl': '反光',
}

ORIENTATION_MAP = {
    'TopDown': '俯视视角',
    'SideView': '侧视视角',
    'FrontView': '正视视角',
}

USECASE_MAP = {
    'Floor': '地面铺装',
    'Wall': '墙面覆层',
    'Roof': '屋顶表面',
    'Pavement': '户外道路',
    'Structural': '结构构件',
    'Decorative': '装饰表面',
    'Insulation': '保温隔热',
    'Foundation': '基础基底',
}


# =============================================================================
# Prompt 构建函数
# =============================================================================

def build_texture_prompt(props) -> str:
    """根据材质配置属性生成自然语言中文 prompt（适配 Zimage + Qwen3 文本编码器）。

    不再使用逗号拼接的关键词列表，而是生成通顺的中文描述句，
    让 Qwen3-4B 能更准确地理解语义意图。
    注意：此函数只根据 Material Config 生成基础 prompt；参考图提示词由调用方按需追加。
    """
    cat = CATEGORY_MAP.get(props.texture_category, '')
    sub = SUBCATEGORY_MAP.get(props.texture_subcategory, '')
    col = COLOR_MAP.get(props.texture_color, '')
    fin = FINISH_MAP.get(props.texture_finish, '')
    feat = FEATURE_MAP.get(props.texture_feature, '')
    ori = ORIENTATION_MAP.get(props.texture_orientation, '')
    use = USECASE_MAP.get(props.texture_usecase, '')

    # 主体描述：颜色 + 子分类 + 分类
    main = cat
    if sub and sub != '无':
        main = f"{sub}{cat}"
    if col and col != '无':
        if col.endswith('色') or col.endswith('图案'):
            main = f"{col}的{main}"
        else:
            main = f"{col}色的{main}"

    # 属性修饰
    attrs = []
    if fin:
        attrs.append(f"{fin}表面")
    if feat and feat != '无':
        attrs.append(f"具有{feat}特性")
    if use:
        attrs.append(f"用于{use}")

    attr_str = "，".join(attrs)
    if attr_str:
        desc = f"{main}纹理，{attr_str}"
    else:
        desc = f"{main}纹理"

    view_desc = f"{ori}拍摄" if ori else "正投影材质视角"

    return f"{desc}。{view_desc}，无缝拼接，照片扫描超写实，平光照明，无阴影，无高光"


def _build_preservation_prompt(props) -> str:
    """构建参考图保留提示词前缀（中文），不包含 Material Config 内容。"""
    parts = [
        "以参考图作为强视觉参考和主要依据",
        "尽量保留参考图的图案布局、颜色比例、纹理细节、材质类型和表面质感",
        "只做轻微清理、去噪、补边和无缝化，生成无缝可平铺 PBR 材质贴图",
        "照片级真实感，平光照明，无阴影，UV就绪。",
    ]
    return "，".join(parts)


# =============================================================================
# 回调函数
# =============================================================================

def _get_feature_items(self, context):
    return FEATURE_ITEMS


def _get_subcategory_items(self, context):
    """根据当前 Category 返回对应的 Subcategory 列表。"""
    cat = self.texture_category
    items = SUBCATEGORY_BY_CATEGORY.get(cat, [('Nil', '--- 无 ---', '')])
    return items


def _update_texture_category(self, context):
    """Category 改变时，重置 Subcategory 为第一个有效值，并同步更新 prompt。"""
    items = SUBCATEGORY_BY_CATEGORY.get(self.texture_category, [('Nil', '--- 无 ---', '')])
    if items:
        self.texture_subcategory = items[0][0]
    # 确保 prompt 也同步更新
    _update_texture_prompt(self, context)


def _update_texture_prompt(self, context):
    """材质配置改变时，自动同步更新 prompt。

    - 已锁定且有手动输入的 prompt → 不覆盖（保护用户编辑）
    - 已锁定但 prompt 为空 → 自动填充（首次使用的兜底）
    - 未锁定 → 始终自动生成
    - 有参考图时 → 不自动修改（避免覆盖用户针对参考图的 prompt）
    """
    if getattr(self, "lock_prompt", False) and self.prompt.strip():
        return

    # 有参考图时不自动修改 prompt
    if getattr(self, "reference_image", None) is not None:
        return

    # 无参考图：根据 Material Config 生成模板 prompt
    self.prompt = build_texture_prompt(self)


# =============================================================================
# PropertyGroup
# =============================================================================

class AIPBRMapItem(bpy.types.PropertyGroup):
    map_type: bpy.props.StringProperty(name="Map Type")
    image_name: bpy.props.StringProperty(name="Image Name")


class CTResultItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="名称")
    image_name: bpy.props.StringProperty(name="图像名")
    timestamp: bpy.props.StringProperty(name="时间")
    result_type: bpy.props.StringProperty(name="Result Type", default="pbr")
    pbr_maps: bpy.props.CollectionProperty(type=AIPBRMapItem)
    is_expanded: bpy.props.BoolProperty(name="Expand", default=False)


class CTProperties(bpy.types.PropertyGroup):
    # -------------------------------------------------------------------------
    # 生成后端
    # -------------------------------------------------------------------------
    texture_generator: bpy.props.EnumProperty(
        name="生成后端",
        description="选择本次 Texture-to-PBR 生成使用的后端",
        items=get_texture_provider_items,
    )

    # -------------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------------
    prompt: bpy.props.StringProperty(name="提示词", default="")
    negative_prompt: bpy.props.StringProperty(name="负面提示词", default="low quality, blurry")

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    SIZE_ITEMS = [
        ('512', "512", ""),
        ('1024', "1024", ""),
        ('2048', "2048", ""),
        ('4096', "4096", ""),
    ]
    width: bpy.props.EnumProperty(name="宽", items=SIZE_ITEMS, default='1024')
    height: bpy.props.EnumProperty(name="高", items=SIZE_ITEMS, default='1024')
    seed: bpy.props.IntProperty(
        name="Seed",
        description="-1 = 随机种子；固定值可复现相同结果",
        default=-1, min=-1, max=2147483647,
    )
    batch_size: bpy.props.IntProperty(name="批量数量", default=1, min=1, max=4)

    # 快速模式：跳过 SeedVR2 超分 + Flux-Fill 修缝，Z-Image 1024² 直接送 CHORD
    # 可大幅降低显存占用和生成时间。最终输出始终按用户选择的尺寸缩放。
    fast_mode: bpy.props.BoolProperty(
        name="快速模式",
        description="开启：省显存、速度快，适合低显存卡和日常出图。关闭：质量更高但需要更高显存",
        default=False,
    )

    # -------------------------------------------------------------------------
    # TEXTURE 模式 — 工业材质配置
    # -------------------------------------------------------------------------
    texture_category: bpy.props.EnumProperty(
        name="分类",
        translation_context='AI_TEX',
        items=CATEGORY_ITEMS,
        default='Conc',
        update=_update_texture_category,
    )
    texture_subcategory: bpy.props.EnumProperty(
        name="子分类",
        translation_context='AI_TEX',
        items=_get_subcategory_items,
        update=_update_texture_prompt,
    )
    texture_color: bpy.props.EnumProperty(
        name="颜色",
        translation_context='AI_TEX',
        items=COLOR_ITEMS,
        default='Gray',
        update=_update_texture_prompt,
    )
    texture_finish: bpy.props.EnumProperty(
        name="表面处理",
        translation_context='AI_TEX',
        items=FINISH_ITEMS,
        default='Rgh',
        update=_update_texture_prompt,
    )
    texture_feature: bpy.props.EnumProperty(
        name="特征",
        translation_context='AI_TEX',
        items=FEATURE_ITEMS,
        default='Nil',
        update=_update_texture_prompt,
    )
    texture_orientation: bpy.props.EnumProperty(
        name="视角",
        translation_context='AI_TEX',
        items=ORIENTATION_ITEMS,
        default='TopDown',
        update=_update_texture_prompt,
    )
    texture_usecase: bpy.props.EnumProperty(
        name="用途",
        translation_context='AI_TEX',
        items=USECASE_ITEMS,
        default='Floor',
        update=_update_texture_prompt,
    )

    # API 后端具体模型选择（由 API Provider 的"Fetch Models"按钮填充）
    api_image_model: bpy.props.EnumProperty(
        name="Image Model",
        items=get_api_image_model_items,
    )
    api_text_model: bpy.props.EnumProperty(
        name="Text Model",
        items=get_api_text_model_items,
    )

    # PBR 提取参数已删除：默认始终使用 CHORD，Blender 算法仅作为 CHORD 失败时的 fallback。

    # Maps 输出选择
    output_basecolor: bpy.props.BoolProperty(name="Basecolor", default=True, description="输出 Base Color / Diffuse 贴图")
    output_normal: bpy.props.BoolProperty(name="Normal", default=True, description="输出 Normal 贴图")
    output_roughness: bpy.props.BoolProperty(name="Roughness", default=True, description="输出 Roughness 贴图")
    output_metalness: bpy.props.BoolProperty(name="Metalness", default=True, description="输出 Metallic 贴图")
    output_height: bpy.props.BoolProperty(name="Height", default=True, description="输出 Height / Displacement 贴图")

    # 通道打包（Channel Packed）
    enable_channel_pack: bpy.props.BoolProperty(
        name="Channel Packed",
        default=False,
        description="Pack multiple grayscale maps into a single RGB texture to reduce texture slots"
    )
    _PACK_CHANNEL_ITEMS = [
        ('NONE', "None", ""),
        ('AO', "AO", ""),
        ('ROUGHNESS', "Roughness", ""),
        ('METALLIC', "Metallic", ""),
        ('HEIGHT', "Height", ""),
    ]

    pack_channel_r: bpy.props.EnumProperty(
        name="R Channel",
        items=_PACK_CHANNEL_ITEMS,
        default='ROUGHNESS',
    )
    pack_channel_g: bpy.props.EnumProperty(
        name="G Channel",
        items=_PACK_CHANNEL_ITEMS,
        default='AO',
    )
    pack_channel_b: bpy.props.EnumProperty(
        name="B Channel",
        items=_PACK_CHANNEL_ITEMS,
        default='METALLIC',
    )
    pack_channel_a: bpy.props.EnumProperty(
        name="A Channel",
        items=_PACK_CHANNEL_ITEMS,
        default='HEIGHT',
    )

    # 参考图（img2img）
    reference_tile_strength: bpy.props.FloatProperty(
        name="纹理保持",
        default=0.5,
        min=0.0,
        max=1.0,
        description="参考图纹理对生成的约束强度",
    )
    reference_image: bpy.props.PointerProperty(
        name="参考图",
        type=bpy.types.Image,
        description="用于指导生成的参考贴图（建议 1024×1024 或更高）",
    )
    reference_denoise: bpy.props.FloatProperty(
        name="重绘幅度",
        default=0.65,
        min=0.05,
        max=1.0,
        description="0.05=几乎不变，1.0=完全重新生成"
    )

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------
    is_generating: bpy.props.BoolProperty(name="正在生成", default=False)
    progress: bpy.props.FloatProperty(name="进度", default=0.0, min=0.0, max=1.0)
    status_message: bpy.props.StringProperty(name="状态", default="就绪")

    # 锁定 prompt，防止 material config 更改时自动覆盖手动编辑的内容
    # 默认锁定：多数用户手动编辑 prompt 后不希望被 Material Config 自动覆盖。
    # 状态随 .blend 文件保存，重启后无需重新设置。
    lock_prompt: bpy.props.BoolProperty(
        name="锁定提示词",
        description="锁定后，修改 Material Config 不再自动同步更新 Prompt。关闭锁定则每次调整材质配置自动生成新 Prompt",
        default=False,
    )

    # 文本编辑器编辑 prompt 时临时保存之前的区域类型和拆分区域的指针
    prompt_editor_previous_area_type: bpy.props.StringProperty(name="Previous Area Type", default="")
    prompt_editor_area_ptr: bpy.props.StringProperty(name="Editor Area Ptr", default="")
    prompt_editor_new_area_ptr: bpy.props.StringProperty(name="New Editor Area Ptr", default="")

    results: bpy.props.CollectionProperty(type=CTResultItem)
    active_result_index: bpy.props.IntProperty(name="Active Result", default=-1)


classes = [AIPBRMapItem, CTResultItem, CTProperties]


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
