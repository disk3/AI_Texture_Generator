import bpy
import bmesh
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageDraw


@dataclass
class UVValidationResult:
    is_valid: bool
    overlapping_faces: List[Tuple[int, int]]
    zero_area_faces: List[int]
    unwrapped_faces: List[int]
    out_of_bounds_ratio: float
    isolated_verts: List[int]
    error_messages: List[str]


class UVExtractor:
    def __init__(self, resolution: int = 1024):
        self.resolution = resolution

    def validate_uv(self, obj: bpy.types.Object, uv_layer_name: str = "") -> UVValidationResult:
        if obj.type != 'MESH':
            return UVValidationResult(False, [], [], [], 0.0, [], ["Object is not a mesh"])

        mesh = obj.data
        bm = bmesh.from_edit_mesh(mesh) if obj.mode == 'EDIT' else bmesh.new()
        if obj.mode != 'EDIT':
            bm.from_mesh(mesh)

        bm.faces.ensure_lookup_table()

        uv_layer = (bm.loops.layers.uv.get(uv_layer_name) or
                    bm.loops.layers.uv.active)
        if uv_layer is None:
            result = UVValidationResult(
                False, [], [], list(range(len(bm.faces))),
                0.0, [], ["未找到 UV 层"]
            )
            if obj.mode != 'EDIT':
                bm.free()
            return result

        overlapping = []
        zero_area = []
        unwrapped = []
        total_uv_area = 0.0

        for face in bm.faces:
            uv_coords = [loop[uv_layer].uv for loop in face.loops]
            if len(uv_coords) < 3:
                unwrapped.append(face.index)
                continue

            area = 0.0
            for i in range(len(uv_coords)):
                x1, y1 = uv_coords[i]
                x2, y2 = uv_coords[(i + 1) % len(uv_coords)]
                area += x1 * y2 - x2 * y1
            area = abs(area) * 0.5

            if area < 1e-10:
                zero_area.append(face.index)
            else:
                total_uv_area += area

        if len(bm.faces) < 1000:
            face_uv_boxes = []
            for face in bm.faces:
                uvs = [loop[uv_layer].uv for loop in face.loops]
                if len(uvs) < 3:
                    continue
                us = [v.x for v in uvs]
                vs = [v.y for v in uvs]
                face_uv_boxes.append((face.index, (min(us), min(vs), max(us), max(vs))))

            for i in range(len(face_uv_boxes)):
                for j in range(i + 1, len(face_uv_boxes)):
                    idx_a, box_a = face_uv_boxes[i]
                    idx_b, box_b = face_uv_boxes[j]
                    if (box_a[0] < box_b[2] and box_a[2] > box_b[0] and
                            box_a[1] < box_b[3] and box_a[3] > box_b[1]):
                        overlapping.append((idx_a, idx_b))

        out_of_bounds_area = 0.0
        for face in bm.faces:
            uvs = [loop[uv_layer].uv for loop in face.loops]
            if len(uvs) < 3:
                continue
            for v in uvs:
                if v.x < 0 or v.x > 1 or v.y < 0 or v.y > 1:
                    break

        oob_ratio = out_of_bounds_area / (total_uv_area + 1e-10)

        if obj.mode != 'EDIT':
            bm.free()

        is_valid = len(overlapping) == 0 and len(zero_area) == 0 and len(unwrapped) == 0

        messages = []
        if overlapping:
            messages.append(f"检测到 {len(overlapping)} 对重叠的 UV 面")
        if zero_area:
            messages.append(f"检测到 {len(zero_area)} 个零面积 UV 面")
        if unwrapped:
            messages.append(f"检测到 {len(unwrapped)} 个未展开 UV 的面")
        if oob_ratio > 0.01:
            messages.append(f"UV 越界比例: {oob_ratio:.2%}（警告）")

        return UVValidationResult(
            is_valid=is_valid,
            overlapping_faces=overlapping,
            zero_area_faces=zero_area,
            unwrapped_faces=unwrapped,
            out_of_bounds_ratio=oob_ratio,
            isolated_verts=[],
            error_messages=messages,
        )

    def select_problematic_faces(self, obj: bpy.types.Object, face_indices: List[int]):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        for idx in face_indices:
            if idx < len(bm.faces):
                bm.faces[idx].select = True
        bmesh.update_edit_mesh(obj.data)

    def render_uv_layout(self, obj: bpy.types.Object, image_size: int = 1024) -> Image.Image:
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            bm.free()
            raise RuntimeError("Mesh has no active UV layer")

        img = Image.new('RGB', (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for face in bm.faces:
            uvs = [(loop[uv_layer].uv.x * image_size,
                    (1.0 - loop[uv_layer].uv.y) * image_size)
                   for loop in face.loops]
            if len(uvs) >= 3:
                draw.polygon(uvs, fill=(200, 200, 200), outline=(0, 0, 0))

        bm.free()
        return img
