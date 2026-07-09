import os
import tempfile

import bpy
import bmesh
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from ..utils.image_utils import blender_image_to_numpy


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
            area = 0.0
            for i in range(len(uvs)):
                x1, y1 = uvs[i]
                x2, y2 = uvs[(i + 1) % len(uvs)]
                area += x1 * y2 - x2 * y1
            area = abs(area) * 0.5
            for v in uvs:
                if v.x < 0 or v.x > 1 or v.y < 0 or v.y > 1:
                    out_of_bounds_area += area
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

    def render_uv_layout(self, obj: bpy.types.Object, image_size: int = 1024) -> np.ndarray:
        """渲染 UV 布局图为 numpy RGB 数组。"""
        uv_layer = obj.data.uv_layers.active
        if uv_layer is None:
            raise RuntimeError("Mesh has no active UV layer")

        # 方案 1：优先使用 Blender 内置的 UV 导出，不依赖 Pillow
        try:
            return self._render_uv_layout_via_export(obj, image_size)
        except Exception as e:
            # 方案 2：失败时回退到 numpy 自绘
            return self._render_uv_layout_via_numpy(obj, image_size)

    def _render_uv_layout_via_export(self, obj: bpy.types.Object, image_size: int) -> np.ndarray:
        """使用 bpy.ops.uv.export_layout 导出 UV 布局，然后读回 numpy。"""
        import bpy

        tmp_name = f"_AI_UVLayout_{obj.name}"

        original_mode = obj.mode
        original_active = bpy.context.view_layer.objects.active
        original_selected = list(bpy.context.selected_objects)
        tmp_path = os.path.join(tempfile.gettempdir(), f"{tmp_name}.png")
        try:
            bpy.context.view_layer.objects.active = obj
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')

            if obj not in bpy.context.selected_objects:
                bpy.ops.mesh.select_all(action='DESELECT')
                obj.select_set(True)

            bpy.ops.uv.export_layout(
                filepath=tmp_path,
                size=(image_size, image_size),
                opacity=1.0,
            )

            try:
                from PIL import Image
                with Image.open(tmp_path) as pil_img:
                    arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
            except ImportError:
                loaded = bpy.data.images.load(tmp_path)
                try:
                    arr = blender_image_to_numpy(loaded)
                finally:
                    bpy.data.images.remove(loaded)
            return arr
        finally:
            try:
                if obj.mode != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass
            try:
                bpy.ops.object.select_all(action='DESELECT')
                for selected in original_selected:
                    if selected.name in bpy.data.objects:
                        selected.select_set(True)
                if original_active and original_active.name in bpy.data.objects:
                    bpy.context.view_layer.objects.active = original_active
            except Exception:
                pass
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _render_uv_layout_via_numpy(self, obj: bpy.types.Object, image_size: int) -> np.ndarray:
        """使用 numpy 自绘 UV 多边形（无 Pillow 回退）。"""
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            bm.free()
            raise RuntimeError("Mesh has no active UV layer")

        # 白色背景
        canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

        for face in bm.faces:
            uvs = [(loop[uv_layer].uv.x * image_size,
                    (1.0 - loop[uv_layer].uv.y) * image_size)
                   for loop in face.loops]
            if len(uvs) >= 3:
                self._draw_polygon(canvas, uvs, fill=(200, 200, 200), outline=(0, 0, 0))

        bm.free()
        return canvas

    @staticmethod
    def _draw_polygon(canvas: np.ndarray, pts: List[Tuple[float, float]],
                      fill: Tuple[int, int, int], outline: Tuple[int, int, int]):
        """在 numpy 画布上绘制填充多边形（简单扫描线算法）。"""
        h, w = canvas.shape[:2]
        pts = np.array(pts, dtype=np.float32)
        if len(pts) < 3:
            return

        # 三角剖分（fan）
        for i in range(1, len(pts) - 1):
            UVExtractor._draw_triangle(canvas, pts[0], pts[i], pts[i + 1], fill)

        # 画轮廓线
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            UVExtractor._draw_line(canvas, p1, p2, outline)

    @staticmethod
    def _draw_triangle(canvas: np.ndarray, p0, p1, p2, color: Tuple[int, int, int]):
        """重心坐标法填充三角形。"""
        h, w = canvas.shape[:2]
        pts = np.array([p0, p1, p2], dtype=np.float32)
        min_x = max(0, int(np.floor(pts[:, 0].min())))
        max_x = min(w - 1, int(np.ceil(pts[:, 0].max())))
        min_y = max(0, int(np.floor(pts[:, 1].min())))
        max_y = min(h - 1, int(np.ceil(pts[:, 1].max())))

        if min_x > max_x or min_y > max_y:
            return

        # 计算重心坐标
        v0 = pts[2] - pts[0]
        v1 = pts[1] - pts[0]
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                v2 = p - pts[0]
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)
                v = (d11 * d20 - d01 * d21) / denom
                u = (d00 * d21 - d01 * d20) / denom
                if v >= 0 and u >= 0 and u + v <= 1:
                    canvas[y, x] = color

    @staticmethod
    def _draw_line(canvas: np.ndarray, p1, p2, color: Tuple[int, int, int]):
        """Bresenham 画线。"""
        h, w = canvas.shape[:2]
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= x1 < w and 0 <= y1 < h:
                canvas[y1, x1] = color
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
