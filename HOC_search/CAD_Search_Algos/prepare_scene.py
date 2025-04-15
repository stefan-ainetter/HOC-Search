import os
import numpy as np
from HOC_search.utils_CAD_retrieval import cut_meshes
from utils import transform_ScanNet_to_py3D, alignPclMesh, transform_ARKIT_to_py3D
from HOC_search.load_ScanNet_data import load_axis_alignment_mat
import open3d as o3d
import copy
from HOC_search.utils_CAD_retrieval import load_depth_img, load_rgb_img
import torch.nn.functional as F
import cv2
import torch
from HOC_search.Torch3DRenderer.Torch3DRenderer import initialize_renderer, initialize_renderer_scannetpp
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_batch

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))


class Prepare_Scene():
    def __init__(self, config, config_general, data_split, parent_dir, device):
        self.config = config
        self.config_general = config_general
        self.data_split = data_split
        self.parent = parent_dir
        self.dataset_name = self.config_general['dataset']
        self.device = device
        self.all_obj_idx_list = None
        self.num_scales = None
        self.rotations = None

    def load_scene_list_scannetpp(self):

        scannetpp_base_path = self.config_general['dataset_base_path']
        base_path = self.parent

        dataset_name = self.config_general['dataset']

        if self.data_split == '':
            scene_list = os.listdir(os.path.join(base_path, self.config['data_folder']))
        else:
            if dataset_name == 'ScanNetpp':
                data_split_path = os.path.join(scannetpp_base_path, 'ScanNetpp_splits', self.data_split)
            else:
                print('data_splits only available for ScanNet dataset; current dataset: ' + str(dataset_name))
                assert False
            if not os.path.exists(data_split_path):
                print('data_split file not found: ' + str(self.data_split))
                assert False
            text_file = open(data_split_path, "r")
            scene_list_init = text_file.readlines()
            scene_list = []
            for scene_name in scene_list_init:
                scene_name = scene_name.rstrip()
                scene_list.append(scene_name)

        return scene_list

    def load_scene_list(self):

        scenes_prepro_folder = os.path.join(self.config_general['dataset_base_path'], 'preprocessed')
        dataset_name = self.config_general['dataset']

        if self.data_split == '':
            scene_list = os.listdir(scenes_prepro_folder)
        else:
            if dataset_name == 'ScanNet':
                data_split_path = os.path.join(self.parent, 'data/ScanNet_splits', self.data_split)
            else:
                print('data_splits only available for ScanNet dataset; current dataset: ' + str(dataset_name))
                assert False
            if not os.path.exists(data_split_path):
                print('data_split file not found: ' + str(self.data_split))
                assert False
            text_file = open(data_split_path, "r")
            scene_list_init = text_file.readlines()
            scene_list = []
            for scene_name in scene_list_init:
                scene_name = scene_name.rstrip()
                scene_list.append(scene_name)

        return scene_list

    def prepare_scene(self, scene_obj):
        print('prepare_scene; device =' + str(self.device))
        print(torch.cuda.is_available())
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())

        self.all_obj_idx_list = []

        SCANNET_base_path = os.path.join(parent, self.config_general['data_path'])
        dataset_name = self.config_general['dataset']

        if dataset_name == 'ScanNet':
            T_mat = transform_ScanNet_to_py3D()
            meta_file_path = os.path.join(SCANNET_base_path, dataset_name, 'scans', scene_obj.scene_name,
                                          scene_obj.scene_name + '.txt')
            align_mat = load_axis_alignment_mat(meta_file_path=meta_file_path)
            align_mat = np.reshape(np.asarray(align_mat), (4, 4))

            scene_path = os.path.join(SCANNET_base_path, dataset_name, 'scans', scene_obj.scene_name,
                                      scene_obj.scene_name + '_vh_clean_2.ply')
            mesh_scene = o3d.io.read_triangle_mesh(scene_path)
            mesh_scene = alignPclMesh(mesh_scene, axis_align_matrix=align_mat, T=T_mat)

        elif dataset_name == 'ARKit_Scenes':
            T_mat = transform_ARKIT_to_py3D()
            align_mat = np.eye(4)

            assert False, 'Not implemented yet'

        elif dataset_name == 'ScanNetpp':
            T_mat = transform_ScanNet_to_py3D()

            scene_path = os.path.join(SCANNET_base_path, dataset_name, 'data', scene_obj.scene_name,
                                      'scans', 'mesh_aligned_0.05.ply')
            mesh_scene = o3d.io.read_triangle_mesh(scene_path)
            mesh_scene = alignPclMesh(mesh_scene, T=T_mat)

        else:
            assert False

        return mesh_scene

    def prepare_GT_data(self, scene_mesh, mesh_bg, renderer, n_views, device):
        with torch.no_grad():
            scene_mesh = scene_mesh.extend(n_views * self.config.getint('batch_size') * 1)

            fragments_gt = renderer(meshes_world=scene_mesh.to(device))

            depth_gt = fragments_gt.zbuf
            mask_depth_valid_render_gt = torch.zeros_like(depth_gt)
            mask_depth_valid_render_gt[fragments_gt.pix_to_face != -1] = 1

            depth_gt[fragments_gt.pix_to_face == -1] = torch.max(depth_gt[fragments_gt.pix_to_face > 0])
            max_depth_gt = torch.max(depth_gt)

            mesh_bg = mesh_bg.extend(n_views * self.config.getint('batch_size') * 1)

            fragments_bg = renderer(meshes_world=mesh_bg.to(device))
            depth_bg = fragments_bg.zbuf
            depth_bg[fragments_bg.pix_to_face == -1] = max_depth_gt

            mask_gt = torch.zeros_like(depth_gt)
            mask_bg = torch.zeros_like(depth_gt)
            mask_gt[depth_gt < depth_bg] = 1.
            mask_bg[depth_gt >= depth_bg] = 1.

            del fragments_gt, fragments_bg
        return depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, mesh_bg

    def prepare_box_item_for_rendering_scannetpp(self, box_item, indices_inst_seg, mesh_scene, scene_name, num_scales,
                                                 rotations):

        self.num_scales = num_scales
        self.rotations = rotations

        n_views = self.config.getint('n_views')
        indices = indices_inst_seg
        self.all_obj_idx_list = self.all_obj_idx_list + indices
        inst_label = box_item.object_id

        mesh_tmp = copy.deepcopy(mesh_scene)
        mesh_bg, mesh_obj = cut_meshes(mesh_tmp, indices, inst_label, scene_name)

        scene_mesh = Meshes(
            verts=[torch.tensor(np.asarray(mesh_scene.vertices)).float()],
            faces=[torch.tensor(np.asarray(mesh_scene.triangles))],
        )

        view_parameters = box_item.view_params

        if len(view_parameters['views']) < n_views:
            n_views = len(view_parameters['views'])

        views_select = np.linspace(0, len(view_parameters['views']) - 1, n_views).astype(int)

        R = view_parameters['R'][views_select].squeeze(axis=1)
        T = view_parameters['T'][views_select].squeeze(axis=1)
        intrinsics = view_parameters['intrinsics']

        radial_params = view_parameters['dist_params']

        frame_id_list = np.asarray(view_parameters['frame_ids'])[views_select].tolist()
        depth_imgs = []

        for depth_cnt, depth_frame_name in enumerate(frame_id_list):
            depth_img_path_new = os.path.join(self.parent, 'data', self.dataset_name, 'data', scene_name, 'iphone',
                                              'depth',
                                              str(depth_frame_name) + '.png')

            depth_img = load_depth_img(depth_img_path_new)
            depth_imgs.append(depth_img)

            depth_imgs_ary = np.asarray(depth_imgs)
            depth_imgs_ary = np.expand_dims(depth_imgs_ary, axis=1)
            depth_sensor = torch.from_numpy(depth_imgs_ary).to(self.device)

        depth_sensor = depth_sensor.permute((0, 2, 3, 1)).to(self.device)
        depth_sensor = torch.repeat_interleave(depth_sensor,
                                               (self.num_scales *
                                                len(self.rotations)), dim=0)
        mask_depth_valid_sensor = torch.zeros_like(depth_sensor).to(self.device)
        mask_depth_valid_sensor[depth_sensor > 0] = 1

        depth_gt_tensor = None
        depth_bg_tensor = None
        mask_gt_tensor = None
        mask_depth_valid_render_gt_tensor = None
        mesh_bg_list = []

        for view_cnt, (R_cnt, T_cnt) in enumerate(zip(R, T)):
            R_cnt = np.expand_dims(R_cnt, axis=0)
            T_cnt = np.expand_dims(T_cnt, axis=0)

            renderer_scene = initialize_renderer_scannetpp(1, self.config.getfloat('img_scale'), R_cnt, T_cnt,
                                                           intrinsics,
                                                           radial_params,
                                                           self.config.getint('batch_size'),
                                                           1,
                                                           self.device,
                                                           self.config_general.getfloat('img_height'),
                                                           self.config_general.getfloat('img_width'))

            depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, mesh_bg = self.prepare_GT_data(scene_mesh,
                                                                                                    mesh_bg,
                                                                                                    renderer_scene,
                                                                                                    1,
                                                                                                    self.device)

            del renderer_scene

            if depth_gt_tensor is None:
                depth_gt_tensor = copy.deepcopy(depth_gt)
                depth_bg_tensor = copy.deepcopy(depth_bg)
                mask_gt_tensor = copy.deepcopy(mask_gt)
                mask_depth_valid_render_gt_tensor = copy.deepcopy(mask_depth_valid_render_gt)
            else:
                depth_gt_tensor = torch.cat([depth_gt_tensor, depth_gt], dim=0)
                depth_bg_tensor = torch.cat([depth_bg_tensor, depth_bg], dim=0)
                mask_gt_tensor = torch.cat([mask_gt_tensor, mask_gt], dim=0)
                mask_depth_valid_render_gt_tensor = torch.cat([mask_depth_valid_render_gt_tensor,
                                                               mask_depth_valid_render_gt], dim=0)

            mesh_bg_list.append(mesh_bg)

        del scene_mesh
        torch.cuda.empty_cache()

        if self.config_general.getboolean('use_2d_inst_from_RGB') and box_item.use_2d_rgb_mask:
            mask_list = []
            mask_gt_tensor = None
            if box_item.cls_name in self.config_general.getstruct('inst_seg_2d_labels_list'):
                inst_seg_2d_path = os.path.join(scenes_prepro_folder, scene_name, 'mask2d_final')
                # print(inst_seg_2d_path)
                if not os.path.exists(inst_seg_2d_path):
                    assert False
                    return None

                for frame_id in frame_id_list:
                    mask = cv2.imread(
                        os.path.join(inst_seg_2d_path, str(int(inst_label)) + '_' + str(frame_id) + '.png'))
                    if mask is None:
                        assert False

                    mask[np.where(mask == 255)] = 1.
                    mask_list.append(mask[:, :, 0])

                mask_imgs_ary = np.asarray(mask_list)
                mask_imgs_ary = np.expand_dims(mask_imgs_ary, axis=1)
                masks_gt = torch.from_numpy(mask_imgs_ary).to(self.device)
                masks_gt = F.interpolate(masks_gt,
                                         scale_factor=(
                                             self.config.getfloat('img_scale'), self.config.getfloat('img_scale')),
                                         )
                masks_gt = masks_gt.permute((0, 2, 3, 1)).to(self.device)
                mask_gt_tensor = torch.repeat_interleave(masks_gt, (self.num_scales *
                                                                    len(self.rotations)), dim=0)

        mesh_bg_tensor = join_meshes_as_batch(mesh_bg_list)
        max_depth_gt = torch.max(depth_gt_tensor)

        renderer = initialize_renderer(n_views, self.config.getfloat('img_scale'), R, T, intrinsics,
                                       self.config.getint('batch_size'),
                                       len(self.rotations) * self.num_scales,
                                       self.device,
                                       self.config_general.getfloat('img_height'),
                                       self.config_general.getfloat('img_width'))

        return n_views, mesh_bg_tensor, renderer, depth_gt_tensor, depth_bg_tensor, mask_gt_tensor, \
            mask_depth_valid_render_gt_tensor, \
            max_depth_gt, mesh_obj, depth_sensor, mask_depth_valid_sensor

    def prepare_box_item_for_rendering(self, box_item, inst_seg_3d, mesh_scene, scene_name, num_scales, rotations):

        self.num_scales = num_scales
        self.rotations = rotations

        n_views = self.config.getint('n_views')
        dataset_base_path = self.config_general['dataset_base_path']

        indices = np.where(inst_seg_3d == box_item.object_id)[0].tolist()
        self.all_obj_idx_list = self.all_obj_idx_list + indices
        inst_label = box_item.object_id

        mesh_tmp = copy.deepcopy(mesh_scene)
        mesh_bg, mesh_obj = cut_meshes(mesh_tmp, indices, inst_label, scene_name)

        scene_mesh = Meshes(
            verts=[torch.tensor(np.asarray(mesh_scene.vertices)).float()],
            faces=[torch.tensor(np.asarray(mesh_scene.triangles))],
        )

        view_parameters = box_item.view_params

        if len(view_parameters['views']) < n_views:
            n_views = len(view_parameters['views'])

        views_select = np.linspace(0, len(view_parameters['views']) - 1, n_views).astype(int)
        R = view_parameters['R'][views_select].squeeze(axis=1)
        T = view_parameters['T'][views_select].squeeze(axis=1)
        intrinsics = view_parameters['intrinsics']

        frame_id_list = np.asarray(view_parameters['frame_ids'])[views_select].tolist()
        depth_imgs = []

        if self.dataset_name == 'ScanNet':
            for depth_cnt, depth_frame_name in enumerate(frame_id_list):
                depth_img_path_new = os.path.join(parent, dataset_base_path, 'extracted', scene_name, 'depths',
                                                  'frame-' + str(depth_frame_name).zfill(6) + '.depth.pgm')

                depth_img = load_depth_img(depth_img_path_new)
                depth_imgs.append(depth_img)

                depth_imgs_ary = np.asarray(depth_imgs)
                depth_imgs_ary = np.expand_dims(depth_imgs_ary, axis=1)
                depth_sensor = torch.from_numpy(depth_imgs_ary).to(self.device)
                depth_sensor = F.interpolate(depth_sensor,
                                             scale_factor=(
                                                 self.config.getfloat('img_scale'), self.config.getfloat('img_scale')),
                                             )
        elif self.dataset_name == 'ARKitScenes':
            for depth_cnt, depth_frame_name in enumerate(frame_id_list):
                depth_img_path_new = os.path.join(self.parent, data_path, 'scans', scene_name, scene_name + '_frames',
                                                  'lowres_depth', depth_frame_name)
                depth_img = load_depth_img(depth_img_path_new)
                depth_imgs.append(depth_img)
        else:
            assert False

        depth_sensor = depth_sensor.permute((0, 2, 3, 1)).to(self.device)
        depth_sensor = torch.repeat_interleave(depth_sensor,
                                               (self.num_scales *
                                                len(self.rotations)), dim=0)
        mask_depth_valid_sensor = torch.zeros_like(depth_sensor).to(self.device)
        mask_depth_valid_sensor[depth_sensor > 0] = 1

        depth_gt_tensor = None
        depth_bg_tensor = None
        mask_gt_tensor = None
        mask_depth_valid_render_gt_tensor = None
        mesh_bg_list = []

        for view_cnt, (R_cnt, T_cnt) in enumerate(zip(R, T)):
            R_cnt = np.expand_dims(R_cnt, axis=0)
            T_cnt = np.expand_dims(T_cnt, axis=0)

            renderer_scene = initialize_renderer(1, self.config.getfloat('img_scale'), R_cnt, T_cnt, intrinsics,
                                                 self.config.getint('batch_size'),
                                                 1,
                                                 self.device,
                                                 self.config_general.getfloat('img_height'),
                                                 self.config_general.getfloat('img_width'))

            depth_gt, depth_bg, mask_gt, mask_depth_valid_render_gt, mesh_bg = self.prepare_GT_data(scene_mesh,
                                                                                                    mesh_bg,
                                                                                                    renderer_scene,
                                                                                                    1,
                                                                                                    self.device)
            del renderer_scene

            if depth_gt_tensor is None:
                depth_gt_tensor = copy.deepcopy(depth_gt)
                depth_bg_tensor = copy.deepcopy(depth_bg)
                mask_gt_tensor = copy.deepcopy(mask_gt)
                mask_depth_valid_render_gt_tensor = copy.deepcopy(mask_depth_valid_render_gt)
            else:
                depth_gt_tensor = torch.cat([depth_gt_tensor, depth_gt], dim=0)
                depth_bg_tensor = torch.cat([depth_bg_tensor, depth_bg], dim=0)
                mask_gt_tensor = torch.cat([mask_gt_tensor, mask_gt], dim=0)
                mask_depth_valid_render_gt_tensor = torch.cat([mask_depth_valid_render_gt_tensor,
                                                               mask_depth_valid_render_gt], dim=0)

            mesh_bg_list.append(mesh_bg)

        del scene_mesh
        torch.cuda.empty_cache()

        if self.config_general.getboolean('use_2d_inst_from_RGB') and box_item.use_2d_rgb_mask:
            mask_list = []
            mask_gt_tensor = None
            if box_item.cls_name in self.config_general.getstruct('inst_seg_2d_labels_list'):
                inst_seg_2d_path = os.path.join(scenes_prepro_folder, scene_name, 'mask2d_final')
                if not os.path.exists(inst_seg_2d_path):
                    assert False
                    return None

                for frame_id in frame_id_list:
                    mask = cv2.imread(
                        os.path.join(inst_seg_2d_path, str(int(inst_label)) + '_' + str(frame_id) + '.png'))
                    if mask is None:
                        assert False

                    mask[np.where(mask == 255)] = 1.
                    mask_list.append(mask[:, :, 0])

                mask_imgs_ary = np.asarray(mask_list)
                mask_imgs_ary = np.expand_dims(mask_imgs_ary, axis=1)
                masks_gt = torch.from_numpy(mask_imgs_ary).to(self.device)
                masks_gt = F.interpolate(masks_gt,
                                         scale_factor=(
                                             self.config.getfloat('img_scale'), self.config.getfloat('img_scale')),
                                         )
                masks_gt = masks_gt.permute((0, 2, 3, 1)).to(self.device)
                mask_gt_tensor = torch.repeat_interleave(masks_gt, (self.num_scales *
                                                                    len(self.rotations)), dim=0)

        mesh_bg_tensor = join_meshes_as_batch(mesh_bg_list)
        max_depth_gt = torch.max(depth_gt_tensor)

        renderer = initialize_renderer(n_views, self.config.getfloat('img_scale'), R, T, intrinsics,
                                       self.config.getint('batch_size'),
                                       len(self.rotations) * self.num_scales,
                                       self.device,
                                       self.config_general.getfloat('img_height'),
                                       self.config_general.getfloat('img_width'))

        return n_views, mesh_bg_tensor, renderer, depth_gt_tensor, depth_bg_tensor, mask_gt_tensor, \
            mask_depth_valid_render_gt_tensor, \
            max_depth_gt, mesh_obj, depth_sensor, mask_depth_valid_sensor
