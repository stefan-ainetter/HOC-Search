import numpy as np
import cv2
import os
import open3d as o3d
import copy
import torch
from pytorch3d.structures import Meshes
from misc.line_mesh import LineMesh
from pytorch3d.io import IO
from pytorch3d.transforms import Rotate, Transform3d, Translate, RotateAxisAngle, Scale

from pytorch3d.io import load_obj
from utils import SEMANTIC_IDX2NAME, COLOR_DETECTRON2


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def normalize_mesh(verts_init, faces, device):
    mesh = Meshes(
        verts=[verts_init],
        faces=[faces],
    )
    bbox = mesh.get_bounding_boxes().squeeze(dim=0)
    bbox = bbox.cpu().detach().numpy()

    center = torch.tensor(bbox.mean(1)).float().to(device)
    vector_x = np.array([bbox[0, 1] - bbox[0, 0], 0, 0])
    vector_y = np.array([0, bbox[1, 1] - bbox[1, 0], 0])
    vector_z = np.array([0, 0, bbox[2, 1] - bbox[2, 0]])

    coeff_x = np.linalg.norm(vector_x)
    coeff_y = np.linalg.norm(vector_y)
    coeff_z = np.linalg.norm(vector_z)

    mesh = mesh.offset_verts(-center)
    transform_func = Transform3d().scale(x=(1 / coeff_x), y=(1 / coeff_y), z=1 / coeff_z).to(device)
    tverts = transform_func.transform_points(mesh.verts_list()[0]).unsqueeze(dim=0)

    mesh = Meshes(
        verts=[tverts.squeeze(dim=0)],
        faces=[faces]
    )

    return mesh, tverts


def load_textured_cad_model(model_path, cad_transform_base, cls_name, device='cpu'):
    try:
        sem_id = list(SEMANTIC_IDX2NAME.keys())[
            list(SEMANTIC_IDX2NAME.values()).index(cls_name)]
    except:
        sem_id = 0
    mesh_color = COLOR_DETECTRON2[sem_id]

    verts, faces_, _ = load_obj(model_path, load_textures=False, device=device)
    faces = faces_.verts_idx

    tverts_final = cad_transform_base.transform_points(verts)

    if device == 'cpu':
        tverts_final_ary = tverts_final.squeeze(dim=0).detach().numpy()
        faces_ary = faces.detach().numpy().astype(np.int32)

    else:
        tverts_final_ary = tverts_final.squeeze(dim=0).cpu().detach().numpy()
        faces_ary = faces.cpu().detach().numpy().astype(np.int32)

    cad_model_o3d = o3d.geometry.TriangleMesh()
    cad_model_o3d.vertices = o3d.utility.Vector3dVector(tverts_final_ary)
    cad_model_o3d.triangles = o3d.utility.Vector3iVector(faces_ary)
    vertex_n = np.array(cad_model_o3d.vertices).shape[0]
    vertex_colors = np.ones((vertex_n, 3)) * mesh_color
    cad_model_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return cad_model_o3d


def load_textured_cad_model_normalized(model_path, cad_transform_base, cls_name, device='cpu'):
    try:
        sem_id = list(SEMANTIC_IDX2NAME.keys())[
            list(SEMANTIC_IDX2NAME.values()).index(cls_name)]
    except:
        sem_id = 0
    mesh_color = COLOR_DETECTRON2[sem_id]

    verts, faces_, _ = load_obj(model_path, load_textures=False, device=device)
    faces = faces_.verts_idx

    _, tverts = normalize_mesh(verts, faces, device)

    tverts_final = cad_transform_base.transform_points(tverts)

    if device == 'cpu':
        tverts_final_ary = tverts_final.squeeze(dim=0).detach().numpy()
        faces_ary = faces.detach().numpy().astype(np.int32)

    else:
        tverts_final_ary = tverts_final.squeeze(dim=0).cpu().detach().numpy()
        faces_ary = faces.cpu().detach().numpy().astype(np.int32)

    cad_model_o3d = o3d.geometry.TriangleMesh()
    cad_model_o3d.vertices = o3d.utility.Vector3dVector(tverts_final_ary)
    cad_model_o3d.triangles = o3d.utility.Vector3iVector(faces_ary)
    vertex_n = np.array(cad_model_o3d.vertices).shape[0]
    vertex_colors = np.ones((vertex_n, 3)) * mesh_color
    cad_model_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return cad_model_o3d


def compose_cad_transforms(box_item, rotations_list, num_scales, use_45_deg_rot=False):
    transform_dict = {}
    transform_list = []
    transform_cnt = 0
    scale_func_list = []

    scale_func_xyz = box_item.transform_dict['scale_transform']
    scale_tensor = torch.diagonal(scale_func_xyz.get_matrix()[0]).detach().cpu()
    scale_func_list.append(scale_func_xyz)
    if num_scales == 2:
        scale_func_zyx = Transform3d().scale(x=torch.as_tensor(scale_tensor[2]).float(),
                                             y=torch.as_tensor(scale_tensor[1]).float(),
                                             z=torch.as_tensor(scale_tensor[0]).float()
                                             )
        scale_func_list.append(scale_func_zyx)

    translation_func = box_item.transform_dict['translate_transform']

    rotate_func_tmp = box_item.transform_dict['rotate_transform']

    for scale_func_id, scale_func in enumerate(scale_func_list):
        for rot_idx, rot in enumerate(rotations_list):
            transform_dict_tmp = {}
            if scale_func_id == 1:
                rotation_transform = Transform3d().rotate_axis_angle(angle=rot + 90, axis='Y', degrees=True)
            else:
                rotation_transform = Transform3d().rotate_axis_angle(angle=rot, axis='Y', degrees=True)

            rotate_func_init = rotation_transform.compose(rotate_func_tmp)

            final_transform = scale_func.compose(rotate_func_init, translation_func)
            transform_dict_tmp['scale_transform'] = scale_func
            transform_dict_tmp['rotate_transform'] = rotate_func_init
            transform_dict_tmp['translate_transform'] = translation_func
            transform_dict[transform_cnt] = transform_dict_tmp
            transform_list.append(final_transform)
            transform_cnt += 1

    if use_45_deg_rot:
        list_size = copy.deepcopy(len(transform_list))
        for transform_id in range(list_size):
            transform_dict_entry = transform_dict[transform_id]

            rotate_func_init_ = transform_dict_entry['rotate_transform'].clone()
            scale_func = transform_dict_entry['scale_transform'].clone()
            translation_func = transform_dict_entry['translate_transform'].clone()
            for rot_45 in [-45, 45]:
                transform_dict_tmp = {}

                rot_func_45_deg = Transform3d().rotate_axis_angle(angle=rot_45, axis='Y', degrees=True)
                rotate_func = rot_func_45_deg.compose(rotate_func_init_)

                final_transform_ = scale_func.compose(rotate_func, translation_func)

                transform_dict_tmp['scale_transform'] = scale_func
                transform_dict_tmp['rotate_transform'] = rotate_func
                transform_dict_tmp['translate_transform'] = translation_func
                transform_dict[transform_cnt] = transform_dict_tmp
                transform_list.append(final_transform_)
                transform_cnt += 1

    return transform_list, transform_dict


def load_depth_img(path):
    depth_image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
    return depth_image


def load_rgb_img(path):
    rgb_img = cv2.imread(path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img / 255.
    return rgb_img


def save_depth_img(depth, depth_out_path, filename=None):
    if depth.ndim == 4:
        depth = depth[:, :, :, 0]

    depth_im_out = None

    for cnt, depth_im in enumerate(depth):
        depth_norm = ((depth_im / np.max(depth_im)) * 255).astype('uint8')

        if depth_im_out is None:
            depth_im_out = depth_norm
        else:
            depth_im_out = np.vstack((depth_im_out, depth_norm))

    if filename is None:
        cv2.imwrite(os.path.join(depth_out_path, 'depth_rendered.png'), depth_im_out)
    else:
        cv2.imwrite(os.path.join(depth_out_path, filename + '.png'), depth_im_out)
    return


def save_normals_img(normals, depth_out_path):
    normals_im_out = None

    for cnt, normals_img in enumerate(normals):
        img = normals_img[:, :, 0, :]

        img = (((img + 1.) / 2.) * 255).astype('uint8')

        if normals_im_out is None:
            normals_im_out = img
        else:
            normals_im_out = np.vstack((normals_im_out, img))

    cv2.imwrite(os.path.join(depth_out_path, 'normals_rendered.png'), normals_im_out)

    return


def save_rgb_img(normals, depth_out_path, filename):
    normals_im_out = None

    for cnt, normals_img in enumerate(normals):
        img = normals_img

        rgb_img = (img * 255).astype('uint8')
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if normals_im_out is None:
            normals_im_out = rgb_img
        else:
            normals_im_out = np.vstack((normals_im_out, rgb_img))

    if filename is None:
        cv2.imwrite(os.path.join(depth_out_path, 'depth_rendered.png'), normals_im_out)
    else:
        cv2.imwrite(os.path.join(depth_out_path, filename + '.png'), normals_im_out)
    return


def cut_meshes(mesh_o3d, indices_list, inst_label, scene_name):
    mesh_o3d_obj = copy.deepcopy(mesh_o3d)
    mesh_o3d_obj = mesh_o3d_obj.select_by_index(indices_list)
    mesh_o3d.remove_vertices_by_index(indices_list)

    face_list_bg = np.asarray(mesh_o3d.triangles)
    face_list_obj = np.asarray(mesh_o3d_obj.triangles)

    mesh_bg = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_bg))],
    )

    mesh_obj = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d_obj.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_obj))],
    )

    return mesh_bg, mesh_obj


def cut_meshes_o3d(mesh_o3d, indices_list, inst_label, scene_name):
    mesh_o3d_obj = copy.deepcopy(mesh_o3d)
    mesh_o3d_obj = mesh_o3d_obj.select_by_index(indices_list)
    mesh_o3d.remove_vertices_by_index(indices_list)

    face_list_bg = np.asarray(mesh_o3d.triangles)
    face_list_obj = np.asarray(mesh_o3d_obj.triangles)

    mesh_bg = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_bg))],
    )

    mesh_obj = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d_obj.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_obj))],
    )

    return mesh_o3d, mesh_o3d_obj


def drawOpen3dCylLines(bbListIn, col=None):
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    line_sets = []

    for bb in bbListIn:
        points = bb
        if col is None:
            col = [0, 0, 1]
        colors = [col for i in range(len(lines))]

        line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        line_sets = line_mesh1_geoms[0]
        for l in line_mesh1_geoms[1:]:
            line_sets = line_sets + l

    return line_sets


def normalize_point(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_bdb_from_corners(corners, planexy=[3, 2, 6, 7]):
    """
    get coeffs, basis, centroid from corners
    :param corners: 8x3 numpy array
        corners of a 3D bounding box
    :return: bounding box parameters
    """
    up_max = np.max(corners[:, 1])
    up_min = np.min(corners[:, 1])

    points_2d = corners[planexy, :]
    points_2d = points_2d[np.argsort(points_2d[:, 0]), :]

    vector2 = np.array([points_2d[1, 0] - points_2d[0, 0], 0, points_2d[1, 2] - points_2d[0, 2]])
    vector1 = np.array([points_2d[2, 0] - points_2d[0, 0], 0, points_2d[2, 2] - points_2d[0, 2]])

    coeff1 = np.linalg.norm(vector1)
    coeff2 = np.linalg.norm(vector2)
    vector1 = normalize_point(vector1)
    vector2 = np.cross(vector1, [0, 1, 0])
    centroid = np.array(
        [points_2d[0, 0] + points_2d[3, 0], float(up_max) + float(up_min), points_2d[0, 2] + points_2d[3, 2]]) * 0.5

    basis = np.array([vector1, [0, 1, 0], vector2])
    coeffs = np.array([coeff1, up_max - up_min, coeff2]) * 0.5
    return centroid, basis.T, coeffs


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[5, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[6, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[7, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners
