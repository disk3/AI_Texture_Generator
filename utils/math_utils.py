import math
import mathutils


def compute_plane_size(fov_y_rad: float, distance: float, aspect: float) -> tuple:
    height = 2.0 * distance * math.tan(fov_y_rad / 2.0)
    width = height * aspect
    return width, height


def camera_direction(camera: mathutils.Matrix) -> mathutils.Vector:
    return camera.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
