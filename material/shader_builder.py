import bpy


class ShaderBuilder:
    @staticmethod
    def build_principled_bsdf(
        name: str,
        images: dict,
        pack_config: dict | None = None,
        output_config: dict | None = None,
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            nodes.remove(node)

        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (400, 0)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)

        links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

        # UV Mapping 基础设施（所有贴图共用）
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-800, 300)

        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-600, 300)
        links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

        output_config = output_config or {
            'diffuse': True,
            'normal': True,
            'roughness': True,
            'metallic': True,
            'height': True,
        }

        has_packed = 'packed' in images
        pack_config = pack_config or {'r': 'ROUGHNESS', 'g': 'AO', 'b': 'METALLIC'}
        uses_ao = any(value == 'AO' for value in pack_config.values())
        mix_rgb = None
        has_displacement = False

        # Diffuse
        if 'diffuse' in images and output_config.get('diffuse', True):
            tex_diffuse = nodes.new(type='ShaderNodeTexImage')
            tex_diffuse.location = (-700, 200)
            tex_diffuse.image = images['diffuse']
            tex_diffuse.image.colorspace_settings.name = 'sRGB'
            links.new(mapping.outputs['Vector'], tex_diffuse.inputs['Vector'])

            # Color Factor: diffuse (A) * white (B) → Base Color
            # 默认 Color2 为白色、Fac 为 1.0，避免默认灰度让贴图发灰
            color_factor = nodes.new(type='ShaderNodeMixRGB')
            color_factor.name = 'Color Factor'
            color_factor.label = 'Color Factor'
            color_factor.location = (-500, 200)
            color_factor.blend_type = 'MULTIPLY'
            color_factor.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)
            color_factor.inputs['Fac'].default_value = 1.0
            links.new(tex_diffuse.outputs['Color'], color_factor.inputs['Color1'])

            if has_packed and uses_ao:
                # packed 模式下：color_factor * AO → Base Color
                mix_rgb = nodes.new(type='ShaderNodeMixRGB')
                mix_rgb.name = 'AO Multiply'
                mix_rgb.label = 'AO Multiply'
                mix_rgb.location = (-300, 200)
                mix_rgb.blend_type = 'MULTIPLY'
                links.new(color_factor.outputs['Color'], mix_rgb.inputs['Color1'])
                links.new(mix_rgb.outputs['Color'], bsdf.inputs['Base Color'])
            else:
                links.new(color_factor.outputs['Color'], bsdf.inputs['Base Color'])

        # Normal
        if 'normal' in images and output_config.get('normal', True):
            tex_normal = nodes.new(type='ShaderNodeTexImage')
            tex_normal.location = (-600, 0)
            tex_normal.image = images['normal']
            tex_normal.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_normal.inputs['Vector'])

            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-200, 0)
            links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

        # Channel Packed 模式
        if has_packed and (output_config.get('roughness', True) or output_config.get('metallic', True)):
            tex_packed = nodes.new(type='ShaderNodeTexImage')
            tex_packed.location = (-600, -200)
            tex_packed.image = images['packed']
            tex_packed.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_packed.inputs['Vector'])

            sep = nodes.new(type='ShaderNodeSeparateColor')
            sep.location = (-400, -200)
            sep.mode = 'RGB'
            links.new(tex_packed.outputs['Color'], sep.inputs['Color'])

            channel_outputs = {
                'r': sep.outputs['Red'],
                'g': sep.outputs['Green'],
                'b': sep.outputs['Blue'],
                'a': tex_packed.outputs['Alpha'],
            }
            height_output = None
            for channel_name, content in pack_config.items():
                channel_output = channel_outputs.get(channel_name.lower())
                if channel_output is None:
                    continue
                if content == 'ROUGHNESS' and output_config.get('roughness', True):
                    links.new(channel_output, bsdf.inputs['Roughness'])
                elif content == 'METALLIC' and output_config.get('metallic', True):
                    links.new(channel_output, bsdf.inputs['Metallic'])
                elif content == 'AO' and mix_rgb is not None:
                    links.new(channel_output, mix_rgb.inputs['Color2'])
                elif content == 'HEIGHT':
                    height_output = channel_output

            # 如果 packed 中有 HEIGHT 通道，直接用它驱动 Displacement
            has_displacement = False
            if height_output and output_config.get('height', True):
                disp = nodes.new(type='ShaderNodeDisplacement')
                disp.location = (-200, -600)
                disp.inputs['Scale'].default_value = 0.1
                links.new(height_output, disp.inputs['Height'])
                links.new(disp.outputs['Displacement'], output_node.inputs['Displacement'])
                has_displacement = True

        # 非合并模式
        else:
            if 'roughness' in images and output_config.get('roughness', True):
                tex_rough = nodes.new(type='ShaderNodeTexImage')
                tex_rough.location = (-600, -200)
                tex_rough.image = images['roughness']
                tex_rough.image.colorspace_settings.name = 'Non-Color'
                links.new(mapping.outputs['Vector'], tex_rough.inputs['Vector'])
                links.new(tex_rough.outputs['Color'], bsdf.inputs['Roughness'])

            if 'metallic' in images and output_config.get('metallic', True):
                tex_metal = nodes.new(type='ShaderNodeTexImage')
                tex_metal.location = (-600, -400)
                tex_metal.image = images['metallic']
                tex_metal.image.colorspace_settings.name = 'Non-Color'
                links.new(mapping.outputs['Vector'], tex_metal.inputs['Vector'])
                links.new(tex_metal.outputs['Color'], bsdf.inputs['Metallic'])

        # Height / Displacement（仅在 packed 中未创建 displacement 时处理独立 height 贴图）
        if not has_displacement and 'height' in images and output_config.get('height', True):
            tex_height = nodes.new(type='ShaderNodeTexImage')
            tex_height.location = (-600, -600)
            tex_height.image = images['height']
            tex_height.image.colorspace_settings.name = 'Non-Color'
            links.new(mapping.outputs['Vector'], tex_height.inputs['Vector'])

            disp = nodes.new(type='ShaderNodeDisplacement')
            disp.location = (-200, -600)
            disp.inputs['Scale'].default_value = 0.1
            links.new(tex_height.outputs['Color'], disp.inputs['Height'])
            links.new(disp.outputs['Displacement'], output_node.inputs['Displacement'])

        return mat

