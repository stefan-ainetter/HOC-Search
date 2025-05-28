import json
import os
import sys
import numpy as np
import open3d as o3d
import torch
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer
)
import pytorch3d
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes

from HOC_search.Torch3DRenderer.SimpleShader import SimpleShader
from HOC_search.Torch3DRenderer.pytorch3d_rasterizer_custom import MeshRendererViewSelection
from HOC_search.utils_CAD_retrieval import get_bdb_from_corners, get_corners_of_bb3d_no_index, drawOpen3dCylLines
from HOC_search.ScanNetAnnotation import ScanNetAnnotation, ObjectAnnotation

from utils import (COLOR_DETECTRON2, Rz, transform_ScanNet_to_py3D, alignPclMesh,
                   shapenet_category_dict, MSEG_SEMANTIC_IDX2NAME)
import pickle
from colmap import Image, Camera
import cv2

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def camera_to_intrinsic(camera):
    '''
    camera object to intrinsic matrix
    fx 0  cx
    0  fy cy
    0  0  1
    '''
    return np.array([
        [camera.params[0], 0, camera.params[2]],
        [0, camera.params[1], camera.params[3]],
        [0, 0, 1]
    ])


def read_json(filename):
    with open(filename, 'r') as infile:
        return json.load(infile)


def view_selection_new_pose(scene_name, tmesh, frame_id_pose_dict, new_intrinsics, dist_params, img_scale, max_views,
                            silhouette_thres, inst_label_list):
    n_views = 1
    height = 1440.
    width = 1920.

    view_selection_dict = {}
    img_list = []
    img_path_list = []
    depth_path_list = []
    R_list = []
    T_list = []
    frame_name_list = []
    path_list = []

    raster_settings = RasterizationSettings(
        image_size=(int(height * img_scale), int(width * img_scale)),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )
    for frame_name, pose_dict in frame_id_pose_dict.items():

        frame_name_list.append(frame_name)
        intrinsics = new_intrinsics

        R_world_to_cam = pose_dict['R'].cpu().numpy()
        T_world_to_cam = pose_dict['T'].cpu().numpy()

        if R_world_to_cam is None or T_world_to_cam is None:
            img_list.append(np.ones((int(height * img_scale), int(width * img_scale))) * -1)
            depth_path_list.append('')
            img_path_list.append('')
            path_list.append('')
            R_list.append(np.zeros((1, 3, 3), dtype=np.float64))
            T_list.append(np.zeros((1, 3), dtype=np.float64))
            continue

        R_list.append(R_world_to_cam)
        T_list.append(T_world_to_cam)

        R = torch.tensor(R_world_to_cam).to(device)
        T = torch.tensor(T_world_to_cam).to(device)

        px, py = (intrinsics[0, 2] * img_scale), (intrinsics[1, 2] * img_scale)
        principal_point = torch.tensor([px, py])[None].type(torch.FloatTensor).to(device)
        principal_point = principal_point.repeat(n_views, 1)
        fx, fy = ((intrinsics[0, 0] * img_scale)), ((intrinsics[1, 1] * img_scale))
        focal_length = torch.tensor([fx, fy])[None].type(torch.FloatTensor).to(device)
        focal_length = focal_length.repeat(n_views, 1)

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            device=device, T=T, R=R,
            image_size=((int(height * img_scale), int(width * img_scale)),))

        renderer = MeshRendererViewSelection(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SimpleShader(
                device=device,
                cameras=cameras,
            )
        )
        tmesh = tmesh.extend(n_views)
        img, fragments = renderer(meshes_world=tmesh.to(device))
        valid_pix = fragments.pix_to_face.repeat(1, 1, 1, 4)

        img[valid_pix < 0] = -1.
        img_ = img.cpu().detach().numpy()[0, :, :, 0]
        img = np.round(img_ * np.max(inst_label_list))

        # cv2.imshow('image window', img)
        # # add wait key. window waits until user presses a key
        # cv2.waitKey(0)
        # # and finally destroy/close all open windows
        # cv2.destroyAllWindows()

        img_list.append(img)

    img_ary = np.asarray(img_list)

    for label in inst_label_list:
        view_params_dict = {}

        mask = np.zeros_like(img_ary)
        mask[img_ary == label] = 1
        label_cnt = np.sum(mask, axis=(1, 2))
        label_max = np.max(label_cnt)
        label_norm = label_cnt / label_max
        best_view_idx = np.where(label_norm > silhouette_thres)
        if len(best_view_idx[0]) < 1:
            best_view_idx = np.where(label_norm > 0.)

        if len(best_view_idx[0]) < 4:
            continue

        if len(best_view_idx[0]) < max_views:
            views = best_view_idx[0]
        else:
            views_select = np.linspace(0, len(best_view_idx[0]) - 1, max_views).astype(int)
            views = best_view_idx[0][views_select].astype(int)
        view_params_dict['views'] = views
        view_params_dict['R'] = np.asarray(R_list)[views, :, :]
        view_params_dict['T'] = np.asarray(T_list)[views, :]
        view_params_dict['frame_ids'] = np.asarray(frame_name_list)[views].tolist()

        view_params_dict['intrinsics'] = intrinsics
        view_params_dict['dist_params'] = dist_params
        view_selection_dict[int(label)] = view_params_dict

    return view_selection_dict


def parse_cls_label(cls_label_scannetpp):
    cls_label_shapenet = None

    if cls_label_scannetpp in ['table', 'desk', 'office table', 'rolling table', 'experiment bench', 'laboratory bench',
                               'office desk', 'work bench', 'coffee table', 'conference table', 'dining table',
                               'high table', 'ping pong table', 'bedside table', 'computer desk', 'foosball table',
                               'babyfoot table',
                               'side table', 'workbench', 'table football', 'trolley table', 'serving trolley',
                               'bar table', 'computer table', 'tv table', 'work station', 'study table', 'footstool',
                               'tv console', 'tv trolley',
                               'center table', 'folded table', 'short table', 'sidetable', 'folding table',
                               'laptop table', 'joined tables']:
        cls_label_shapenet = 'table'
    elif cls_label_scannetpp in ['standing lamp', 'floor lamp', 'light', 'table lamp',
                                 'lamp', 'desk lamp', 'studio light', 'bedside lamp',
                                 'ring light', 'door lamp', 'desk light', 'monitor light',
                                 'kitchen light']:
        cls_label_shapenet = 'lamp'
    elif cls_label_scannetpp in ['chair', 'office chair', 'stool', 'sofa chair', 'beanbag', 'arm chair', 'lounge chair',
                                 'armchair', 'dining chair', 'office visitor chair', 'seat', 'chairs', 'rolling chair',
                                 'high stool',
                                 'wheelchair', 'ottoman chair', 'recliner', 'barber chair', 'papasan chair',
                                 'toilet seat', 'easy chair', 'step stool', 'deck chair', 'high chair', 'office  chair',
                                 'medical stool',
                                 'barstool', 'linked retractable seats', 'folding chair']:
        cls_label_shapenet = 'chair'
    elif cls_label_scannetpp in ['wall cabinet', 'locker', 'storage cabinet', 'cabinet', 'kitchen cabinet', 'wardrobe',
                                 'cupboard', 'office cabinet', 'laboratory cabinet', 'kitchen unit', 'file cabinet',
                                 'open cabinet', 'closet', 'bathroom cabinet', 'bath cabinet',
                                 'clothes cabinet', 'fitted wardrobe',
                                 'small cabinet', 'bottle crate', 'foldable closet', 'tool rack',
                                 'power cabinet', 'shoe cabinet', 'drawer', 'bedside cabinet',
                                 'shelf trolley', 'wall unit',
                                 'switchboard cabinet', 'bedside shelf', 'file storage', 'shoe box', 'mirror cabinet',
                                 'first aid cabinet', 'mobile tv stand', 'tv stand', 'television stand',
                                 'monitor stand']:
        cls_label_shapenet = 'cabinet'
    elif cls_label_scannetpp in ['bookshelf', 'storage rack', 'shoe rack', 'book shelf', 'storage shelf',
                                 'kitchen shelf', 'bathroom shelf', 'shoes holder', 'garage shelf',
                                 'kitchen storage rack', 'bathroom rack',
                                 'glass shelf', 'clothes rack', 'shower rug', 'wall shelf', 'recesssed shelf',
                                 'recessed shelve', 'shelve', 'wine rack']:
        cls_label_shapenet = 'bookshelf'
    elif cls_label_scannetpp in ['monitor', 'tv', 'projector screen', 'television', 'flat panel display',
                                 'tv screen', 'tv mount', 'computer monitor',
                                 'screen']:
        cls_label_shapenet = 'display'
    elif cls_label_scannetpp in ['oven', 'stove', 'oven range', 'kitchen stove']:
        cls_label_shapenet = 'stove'
    elif cls_label_scannetpp in ['sofa', 'couch', 'l-shaped sofa', 'floor sofa', 'folding sofa', 'floor couch']:
        cls_label_shapenet = 'sofa'
    elif cls_label_scannetpp in ['bed', 'loft bed', 'canopy bed', 'camping bed']:
        cls_label_shapenet = 'bed'
    elif cls_label_scannetpp in ['pillow', 'cushion', 'floor cushion', 'sit-up pillow', 'sofa cushion', 'long pillow',
                                 'seat cushion', 'chair cushion']:
        cls_label_shapenet = 'pillow'
    elif cls_label_scannetpp in ['plant pot', 'pot', 'vase', 'flower pot']:
        cls_label_shapenet = 'flowerpot'
    elif cls_label_scannetpp in ['trash can', 'trash bin', 'bucket', 'bin', 'trashcan', 'dustbin', 'jerry can',
                                 'garbage bin', 'storage bin', 'wastebin', 'recycle bin']:
        cls_label_shapenet = 'trash bin'
    elif cls_label_scannetpp in ['sink', 'kitchen sink', 'bathroom sink', 'shower sink']:
        cls_label_shapenet = 'bathtub'
    elif cls_label_scannetpp in ['suitcase', 'backpack', 'bagpack', 'luggage bag', 'suit bag', 'rucksack',
                                 'package bag']:
        cls_label_shapenet = 'bag'
    elif cls_label_scannetpp in ['refrigerator', 'fridge', 'mini fridge', 'lab fridge', 'freezer']:
        cls_label_shapenet = 'refridgerator'
    elif cls_label_scannetpp in ['bathtub', 'bath tub', 'basin', 'washbasin', 'wash basin']:
        cls_label_shapenet = 'bathtub'
    elif cls_label_scannetpp in ['keyboard']:
        cls_label_shapenet = 'keyboard'
    elif cls_label_scannetpp in ['toilet', 'urinal']:
        cls_label_shapenet = 'toilet'
    elif cls_label_scannetpp in ['printer', 'photocopy machine', 'copier', '3d printer', 'overhead projector',
                                 'copy machine', 'paper shredder', 'multifunction printer', 'photocopier',
                                 'scanner']:
        cls_label_shapenet = 'printer'
    elif cls_label_scannetpp in ['bench', 'foot rest', 'piano stool', 'bench stool', 'shoe stool', 'piano chair',
                                 'high bench']:
        cls_label_shapenet = 'bench'
    elif cls_label_scannetpp in ['microwave', 'microwave oven', 'toaster oven', 'mini oven']:
        cls_label_shapenet = 'microwaves'
    elif cls_label_scannetpp in ['basket', 'laundry basket', 'shopping basket']:
        cls_label_shapenet = 'basket'
    elif cls_label_scannetpp in ['washing machine']:
        cls_label_shapenet = 'washer'
    elif cls_label_scannetpp in ['dishwasher']:
        cls_label_shapenet = 'dishwasher'
    elif cls_label_scannetpp in ['laptop']:
        cls_label_shapenet = 'laptop'
    elif cls_label_scannetpp in ['nightstand', 'night stand']:
        cls_label_shapenet = 'cabinet'
    elif cls_label_scannetpp in ['dresser']:
        cls_label_shapenet = 'cabinet'
    elif cls_label_scannetpp in ['bicycle', 'bike']:
        cls_label_shapenet = 'motorbike'
    elif cls_label_scannetpp in ['guitar', 'guitar bag', 'guitar case', 'electric guitar']:
        cls_label_shapenet = 'guitar'
    elif cls_label_scannetpp in ['clock', 'wall clock', 'table clock']:
        cls_label_shapenet = 'clock'
    elif cls_label_scannetpp in ['piano']:
        cls_label_shapenet = 'piano'
    elif cls_label_scannetpp in ['bowl']:
        cls_label_shapenet = 'bowl'

    return cls_label_shapenet


def parse_py3d_transform(center, basis, scale):
    transform_dict = {}
    scale_transform = Transform3d().scale(x=torch.as_tensor(scale[0]).float(),
                                          y=torch.as_tensor(scale[1]).float(),
                                          z=torch.as_tensor(scale[2]).float()
                                          )

    translation_transform = Transform3d().translate(x=torch.as_tensor(center[0]).float(),
                                                    y=torch.as_tensor(center[1]).float(),
                                                    z=torch.as_tensor(center[2]).float()
                                                    )

    rotate_transform = Transform3d().rotate(R=torch.as_tensor(basis).float())

    final_transform = scale_transform.compose(rotate_transform, translation_transform)

    transform_dict['scale_transform'] = scale_transform
    transform_dict['rotate_transform'] = rotate_transform
    transform_dict['translate_transform'] = translation_transform

    return final_transform, transform_dict


def main():
    # parameters for view selection
    img_scale = .4
    max_views = 30
    silhouette_thres = 0.3

    scannetpp_path = os.path.join(parent, 'data', 'ScanNetpp')
    data_folder = 'data'

    scene_list = ['30966f4c6e']

    for scene_cnt, scene_name in enumerate(scene_list):
        print(scene_name)

        prepro_out_path = os.path.join(parent, 'output', 'scene_preprocessing', scene_name)
        os.makedirs(prepro_out_path, exist_ok=True)

        mesh_path = os.path.join(scannetpp_path, data_folder, scene_name, 'scans', 'mesh_aligned_0.05.ply')
        anno_json = os.path.join(scannetpp_path, data_folder, scene_name, 'scans', 'segments_anno.json')
        pose_json = os.path.join(scannetpp_path, data_folder, scene_name, 'iphone', 'pose_intrinsic_imu.json')
        colmap_path = os.path.join(scannetpp_path, data_folder, scene_name, 'iphone', 'colmap')

        if not os.path.exists(mesh_path) or not os.path.exists(anno_json) or not os.path.exists(pose_json):
            continue

        cameras = read_cameras_text(os.path.join(colmap_path, "cameras.txt"))
        images = read_images_text(os.path.join(colmap_path, "images.txt"))
        T_mat = transform_ScanNet_to_py3D()

        frame_id_pose_dict = {}
        new_intrinsics = camera_to_intrinsic(cameras[1])
        dist_params = cameras[1].params[4:]
        for image_id, image in images.items():
            pose_dict = {}
            world_to_camera = image.world_to_camera
            frame_id = image.name[:-4]

            M_frame_inv = world_to_camera
            T_nviews = np.dot(M_frame_inv, np.linalg.inv(T_mat))
            T2 = np.eye(4)
            T2[:3, :3] = Rz(np.deg2rad(180))
            T_nviews = np.dot(T2, T_nviews)

            rot_py3d = np.copy(T_nviews[:3, :3].T)
            T_nviews[:3, :3] = rot_py3d

            R_world_to_cam = np.expand_dims(T_nviews, axis=0)[:, 0:3, 0:3]
            T_world_to_cam = np.expand_dims(T_nviews, axis=0)[:, 0:3, 3]

            R = torch.tensor(R_world_to_cam)
            T = torch.tensor(T_world_to_cam)
            pose_dict['R'] = R
            pose_dict['T'] = T

            frame_id_pose_dict[frame_id] = pose_dict

        annotations = read_json(anno_json)
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        mesh_o3d = alignPclMesh(mesh_o3d, T=T_mat)

        tmesh = Meshes(
            verts=[torch.tensor(np.asarray(mesh_o3d.vertices)[:, :3].astype(np.float32))],
            faces=[torch.tensor(np.asarray(mesh_o3d.triangles))]
        )

        points = np.copy(np.asarray(mesh_o3d.vertices))
        pcl_color_instance = np.zeros((points.shape[0], 3))
        pcl_color_semantic = np.zeros((points.shape[0], 3))
        semantic_label_map = np.zeros((points.shape[0], 1))
        inst_label_map = np.zeros((points.shape[0], 1))
        inst_label_list = []

        # prepare semantic and instance segmenation point clouds
        valid_annotations = []
        valid_annotation_classes = []

        for annotation in annotations['segGroups']:
            inst_id = annotation['objectId']
            shapenet_cls_label = parse_cls_label(annotation['label'])
            if shapenet_cls_label is None:
                continue

            sem_label = list(MSEG_SEMANTIC_IDX2NAME.keys())[list(MSEG_SEMANTIC_IDX2NAME.values()).index(
                shapenet_cls_label)]

            color_id = inst_id
            color_id = color_id % COLOR_DETECTRON2.shape[0]
            pcl_color_instance[annotation['segments']] = COLOR_DETECTRON2[int(color_id), :]
            inst_label_map[annotation['segments']] = inst_id
            inst_label_list.append(inst_id)

            pcl_color_semantic[annotation['segments']] = COLOR_DETECTRON2[int(sem_label), :]
            semantic_label_map[annotation['segments']] = int(sem_label)

            valid_annotations.append(annotation)
            valid_annotation_classes.append(annotation['label'])

        inst_label_map[inst_label_map == -100] = -255.

        inst_label_norm = inst_label_map / np.max(np.unique(inst_label_list))
        tex = torch.tensor(inst_label_norm.astype(np.float32)).unsqueeze(dim=0)

        tex = tex.repeat(1, 1, 3)
        tmesh.textures = pytorch3d.renderer.mesh.textures.TexturesVertex(verts_features=tex)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(pcl_color_semantic)
        path_tmp = os.path.join(prepro_out_path, 'pcl_sem_seg.ply')
        tmp = o3d.io.write_point_cloud(path_tmp, pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(pcl_color_instance)
        path_tmp = os.path.join(prepro_out_path, 'pcl_inst_seg.ply')
        tmp = o3d.io.write_point_cloud(path_tmp, pcd)

        path_tmp = os.path.join(prepro_out_path, 'mesh_py3d_textured.ply')
        tmp = o3d.io.write_triangle_mesh(path_tmp, mesh_o3d)

        pkl_out_path = os.path.join(prepro_out_path)
        if not os.path.exists(pkl_out_path):
            os.makedirs(pkl_out_path)

        view_sel_path = os.path.join(prepro_out_path, scene_name + 'view_selection.pkl')

        print('Start view selection')
        if os.path.exists(view_sel_path):
            pkl_file = open(view_sel_path, 'rb')
            view_selection_dict = pickle.load(pkl_file)
            pkl_file.close()
        else:
            view_selection_dict = view_selection_new_pose(scene_name, tmesh, frame_id_pose_dict, new_intrinsics,
                                                          dist_params, img_scale, max_views, silhouette_thres,
                                                          inst_label_list)
        if view_selection_dict is None:
            continue

        out_file = open(view_sel_path, 'wb')
        pickle.dump(view_selection_dict, out_file)
        out_file.close()

        print('View selection done')

        all_boxes_selected = None
        obj_3d_list = []
        for obj_count, annotation in enumerate(valid_annotations):

            if obj_count < 0:
                continue

            shapenet_cls_label = parse_cls_label(annotation['label'])
            object_id = int(annotation['objectId'])

            if shapenet_cls_label is None:
                continue

            scannetpp_cls_label = annotation['label']
            print(shapenet_cls_label)

            obb = annotation['obb']
            center = np.array(obb['centroid'])
            rot = np.array(obb['normalizedAxes']).reshape(3, 3).T
            extents = np.array(obb['axesLengths'])
            bbox = o3d.geometry.OrientedBoundingBox(center, rot, extents)
            bbox.rotate(T_mat[0:3, 0:3], center=(0, 0, 0))

            box_final = get_corners_of_bb3d_no_index(bbox.R.T, bbox.extent / 2, bbox.center)
            center_final, basis_final, coeffs_final = get_bdb_from_corners(box_final)
            box_final = get_corners_of_bb3d_no_index(basis_final.T, coeffs_final, center_final)

            line_color = [0, 1, 0]
            lineSets_selected = drawOpen3dCylLines([box_final], line_color)
            if all_boxes_selected is None:
                all_boxes_selected = lineSets_selected
            else:
                all_boxes_selected += lineSets_selected

            transform3d, transform_dict = parse_py3d_transform(center_final, basis_final, coeffs_final * 2)

            for annotation in annotations['segGroups']:
                if object_id == int(annotation['objectId']):

                    if object_id in view_selection_dict:
                        view_params = view_selection_dict[object_id]
                        catid_cad = shapenet_category_dict[shapenet_cls_label]
                        obj_instance = ObjectAnnotation(object_id, shapenet_cls_label, scannet_category_label=None,
                                                        view_params=view_params,
                                                        transform3d=transform3d, transform_dict=transform_dict,
                                                        catid_cad=catid_cad,
                                                        scan2cad_annotation_dict=annotation)

                        obj_3d_list.append(obj_instance)

            print(scannetpp_cls_label)
            print(scene_name)
            print(object_id)
            print('---------------')

        bpaResult = o3d.io.write_triangle_mesh(os.path.join(prepro_out_path, 'bbox_all' + ".ply"),
                                               all_boxes_selected)

        scene_obj = ScanNetAnnotation(scene_name, obj_3d_list, inst_label_map, scene_type=None)

        if scene_obj is not None:
            pkl_out_file = open(os.path.join(pkl_out_path, scene_name + '.pkl'), 'wb')
            pickle.dump(scene_obj, pkl_out_file)
            pkl_out_file.close()

        print('Extracting scene information done for ' + str(scene_name))


if __name__ == "__main__":
    main()
