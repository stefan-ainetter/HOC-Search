import numpy as np
import torch
from pytorch3d.transforms import Transform3d
from pytorch3d.structures import Meshes, Pointclouds

from utils import Rx, Ry, Rz

def rotate_points_around_y(points, rot_degree):
    rot_radians = rot_degree * np.pi / 180.

    rot_mat = Ry(rot_radians)

    rot_mat = torch.from_numpy(rot_mat).to(points.device)

    transform_func = Transform3d(device=points.device).rotate(R=torch.tensor(rot_mat))

    points = transform_func.transform_points(points)

    return points

def rotate_mesh_around_y(mesh,rot_degree):
    rotation_transform = Transform3d().rotate_axis_angle(angle=rot_degree, axis='Y', degrees=True)
    tverts = rotation_transform.transform_points(mesh.verts_list()[0])
    faces = mesh.faces_list()[0]

    tmesh = Meshes(
        verts=[tverts],
        faces=[faces]
    )
    return tmesh