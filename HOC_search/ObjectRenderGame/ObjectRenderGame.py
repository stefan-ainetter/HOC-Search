import os.path

import torch
from ordered_set import OrderedSet
from typing import List
import open3d as o3d
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from MonteScene.ProposalGame import ProposalGame

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Rotate, Transform3d, Translate, RotateAxisAngle, Scale
from HOC_search.ObjectGame.ObjectClusterTree import ObjectClusterTree
from HOC_search.ObjectGame.colormap import MyColormap
from HOC_search.ObjectGame import ObjectGame
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex
)

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures.meshes import join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.structures.pointclouds import join_pointclouds_as_batch
from HOC_search.losses import chamfer_distance_one_way

class ObjectRenderGame(ObjectGame):
    """
    Implementation of scene game.
    """

    def __init__(self, config, ex_config):
        super().__init__(config, ex_config)

    def __str__(self):
        return "ObjectRenderGame()"

    def initialize_game(self, config, ex_config):
        """
        Initialize game-specific attributes.

        :return:
        """
        self.shapenet_path = ex_config['shapenet_path']
        self.w_sil = ex_config['w_sil']
        self.w_depth = ex_config['w_depth']
        self.w_sensor = ex_config['w_sensor']
        self.w_chamfer = ex_config['w_chamfer']

        self.device = ex_config['device']

        self.obj_category = ex_config['obj_category']
        self.obj_cluster_tree = ex_config['obj_cluster_tree']

        self.prop_seq = []
        self.all_mcts_results_dict = {'score_list':[],'obj_id_list':[],'orientation_list': [], 'iteration_list': [],
                                      'deg_rot45_index_list':[],'best_transform_list':[]}

        self.current_rot_45deg_index = 0
        self.best_transform = None

        self.loader = ex_config['dataset']

        self.renderer = ex_config['renderer']

        self.num_renderings = ex_config['n_views']
        self.num_samples_per_pcl = ex_config['num_samples_per_pcl']

        self.num_orientations_per_mesh = ex_config['num_orientations_per_mesh']
        self.num_scales = ex_config['num_scales']
        self.num_45_deg_rotations = ex_config['num_45_deg_rotations']

        self.target_object = {
            'depth_GT': ex_config['depth_GT'],
            'mask_GT': ex_config['mask_GT'],
            'max_depth_GT': ex_config['max_depth_GT'],
            'depth_bg': ex_config['depth_bg'],
            'depth_sensor': ex_config['depth_sensor'],
            'mask_depth_valid_render_GT': ex_config['mask_depth_valid_render_GT'],
            'mask_depth_valid_sensor': ex_config['mask_depth_valid_sensor'],
            'label': ex_config['obj_category'],
            'target_pcl': ex_config['target_pcl'],
            'cad_transformations': ex_config['cad_transformations'],
            'transform_dict': ex_config['transform_dict'],
        }

    def calc_score_from_proposals(self, prop_seq=None, props_optimizer=None):
        """
        Calculate score from proposals

        :param prop_seq: Sequence of proposals. If None, uses self.prop_seq instead
        :type prop_seq: List[Proposal]
        :param props_optimizer: Optimizer. Enables optimization during score calculation. If None, optimization
        step is not performed
        :type props_optimizer: PropsOptimizer

        :return: score
        :rtype: torch.Tensor
        """

        if prop_seq is None:
            prop_seq = self.prop_seq
        assert len(prop_seq)

        score = self.calc_loss_from_proposals(prop_seq)

        return score

    def calc_loss_from_proposals(self, prop_seq=None):
        """
        Calculate loss from proposals

        :param prop_seq: Sequence of proposals. If None, uses self.prop_seq instead
        :type prop_seq: List[Proposal]
        :param props_optimizer: Optimizer. Enables optimization during score calculation. If None, optimization
        step is not performed
        :type props_optimizer: PropsOptimizer

        :return: loss
        :rtype: torch.Tensor
        """

        if prop_seq is None:
            prop_seq = self.prop_seq
        assert len(prop_seq)

        leaf_prop = self.prop_seq[-1]
        base_rotation_idx = self.prop_seq[0].rotation_degree
        print('Rotation branch = ' + str(base_rotation_idx))
        objects_cluster = leaf_prop.objects_cluster["cluster"]

        score = self.calc_render_loss(objects_cluster,base_rotation_idx)

        return score

    def calc_chamfer_loss(self,objects_cluster,base_rotation_deg):

        target_pcl = self.target_object['pcl_normalized']


        shapenet_id = objects_cluster[0]['mid'].split('/')[-1]
        model_path = os.path.join(self.shapenet_rw.shapenet_corev2_path, objects_cluster[0]['sid'], shapenet_id,
                                  'models',
                                  'model_normalized.obj')
        verts, faces, _ = load_obj(model_path, load_textures=False, device=self.device)

        tverts_normalized, faces_normalized = self.normalize_mesh(verts, faces.verts_idx)

        base_rotation = Transform3d().rotate_axis_angle(angle=base_rotation_deg, axis='Y', degrees=True)
        scale_func = self.target_object['scale_func']

        cad_transform = base_rotation.compose(scale_func).to(self.device)
        tverts = cad_transform.transform_points(tverts_normalized)

        cad_mesh = Meshes(
            verts=[tverts.squeeze(dim=0)],
            faces=[faces.verts_idx],
        )

        cad_points_sampled, cad_normals_sampled = sample_points_from_meshes(cad_mesh,
                                                                    num_samples=self.num_samples_per_pcl,
                                                                    return_normals=True,
                                                                    return_textures=False)

        loss_chamfer, loss_normals = chamfer_distance_one_way(x=target_pcl.points_padded(), y=cad_points_sampled,
                                                              x_normals=target_pcl.points_padded(),
                                                              y_normals=cad_normals_sampled,
                                                              point_reduction='mean',
                                                              batch_reduction=None)

        loss_final = loss_chamfer[0]

        return loss_final


    def render_loss(self,cad_mesh,cad_pcl):

        with torch.no_grad():

            # L1 norm for point distance here
            chamfer_dist_x = chamfer_distance_one_way(x=self.target_object['target_pcl'].points_padded(),
                                                      y=cad_pcl.points_padded(),
                                                      point_reduction='mean',
                                                      batch_reduction=None,
                                                      norm=1)


            cad_mesh = cad_mesh.extend(self.num_renderings)

            fragments = self.renderer(meshes_world=cad_mesh)

            depth_pred = fragments.zbuf
            depth_pred[fragments.pix_to_face == -1] = self.target_object['max_depth_GT']
            mask_depth_valid_render_pred = torch.zeros_like(depth_pred).to(self.device)
            mask_depth_valid_render_pred[fragments.pix_to_face != -1] = 1

            mask_pred = torch.zeros_like(depth_pred)
            mask_depth_bg = torch.zeros_like(depth_pred)

            mask_pred[depth_pred < self.target_object['depth_bg']] = 1.
            mask_depth_bg[depth_pred >= self.target_object['depth_bg']] = 1.

            mask_combined = torch.zeros_like(mask_pred)
            mask_combined[mask_pred == 1] = 1.
            mask_combined[self.target_object['mask_GT'] == 1] = 1.

            depth_final = depth_pred * mask_pred + self.target_object['depth_bg'] * mask_depth_bg

            loss_sil, loss_depth, loss_sensor = loss_IOU_render_sensor(mask_pred, self.target_object['mask_GT'],
                                                                       self.target_object['depth_GT'], depth_final,
                                                                       mask_combined, self.target_object['depth_sensor'],
                                                                       self.target_object['mask_depth_valid_sensor'],
                                                                       self.target_object['mask_depth_valid_render_GT'],
                                                                       mask_depth_valid_render_pred)

            loss_sil *= self.w_sil
            loss_depth *= self.w_depth
            loss_sensor *= self.w_sensor
            chamfer_dist_x *= self.w_chamfer

            loss_render = loss_sil + loss_depth + loss_sensor + chamfer_dist_x

            loss_final = loss_render
            loss_final = torch.sum(loss_final, dim=0)

        return loss_final


    def calc_render_loss(self,objects_cluster,base_rotation_idx):

        shapenet_id = objects_cluster[0]['mid'].split('/')[-1]
        batch = self.loader[(shapenet_id,base_rotation_idx)]

        cad_mesh_batch = batch['mesh'].to(device=self.device)
        cad_pcl_batch = batch['pcl'].to(device=self.device)
        loss_list_tmp = []
        for cad_mesh,cad_pcl in zip(cad_mesh_batch,cad_pcl_batch):

            loss_init = self.render_loss(cad_mesh,cad_pcl)
            loss_list_tmp.append(loss_init[0])

        min_idx = int(torch.argmin(torch.Tensor(loss_list_tmp)))
        self.current_rot_45deg_index = min_idx

        score = self.convert_loss_to_score(loss_list_tmp[min_idx])

        if min_idx == 0:
            transform_id = base_rotation_idx
        elif min_idx == 1:
            transform_id = 4 + base_rotation_idx * 2
        elif min_idx == 2:
            transform_id = (4 + base_rotation_idx * 2) + 1
        else:
            assert False

        self.current_rot_45deg_index = min_idx
        self.best_transform = self.target_object['cad_transformations'][transform_id]

        return score

    def convert_loss_to_score(self, loss):
        """
        Convert loss to score

        :param loss: loss value
        :type loss: torch.Tensor

        :return: score
        :rtype: torch.Tensor
        """

        score = - loss #* 0.1

        score = score.cpu().numpy()

        return float(score)

    def o3d_visualize_gf(self, prop):
        """
        """

        objects_cluster = prop.objects_cluster["cluster"]
        cluster_centroid = prop.objects_cluster["centroid"]

        z_tensor = torch.from_numpy(cluster_centroid).to(device=self.device)
        z_tensor = z_tensor[None]

        target_points = self.target_object['pcl_normalized'].points_padded()

        sigmas = self.shapegf_recon_net.sigmas[-1:]
        total_grad, total_abs_grad = self.shapegf_recon_net.calc_gf_from_z_and_points(z_tensor, target_points, sigmas=sigmas)


        total_abs_grad_np = total_abs_grad.cpu().numpy()[0]
        total_abs_grad_np_norm = np.linalg.norm(total_abs_grad_np, axis=-1)
        total_abs_grad_np_norm = total_abs_grad_np_norm / np.max(total_abs_grad_np_norm)
        total_grad_color = MyColormap.get_color_from_value(total_abs_grad_np_norm)[:,:-1]


        target_points_np = target_points[0].cpu().numpy()
        target_points_o3d = o3d.geometry.PointCloud()
        target_points_o3d.points = o3d.utility.Vector3dVector(target_points_np)
        # target_points_o3d.colors = o3d.utility.Vector3dVector(np.zeros_like(target_points_np) + 0.5)
        target_points_o3d.colors = o3d.utility.Vector3dVector(total_grad_color)
        target_points_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                                center=(0, 0, 0))
        # target_points_o3d.translate((-3, 0, 0))

        prop_rec, _ = self.shapegf_recon_net.reconstruct_from_z(z_tensor, num_points=20000)

        prop_rec_pts_np = prop_rec[0].detach().cpu() /2

        prop_rec_pts_o3d = o3d.geometry.PointCloud()
        prop_rec_pts_o3d.points = o3d.utility.Vector3dVector(prop_rec_pts_np)
        prop_rec_pts_o3d.colors = o3d.utility.Vector3dVector(np.zeros_like(prop_rec_pts_np) + 0.5)

        prop_rec_pts_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                           center=(0, 0, 0))
        # prop_rec_pts_o3d.translate((-3, 0, 0))

        o3d.visualization.draw_geometries([prop_rec_pts_o3d, target_points_o3d])