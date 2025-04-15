import os.path
import torch
from typing import List
import open3d as o3d
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Transform3d
from HOC_search.ObjectGame.colormap import MyColormap
from HOC_search.ObjectGame import ObjectGame
from HOC_search.losses import chamfer_distance_one_way, loss_IOU_render_sensor

class ObjectRenderRefineGame(ObjectGame):
    """
    Implementation of scene game.
    """

    def __init__(self, config, ex_config):
        super().__init__(config, ex_config)

    def __str__(self):
        return "ObjectRenderRefineGame()"

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

        self.iteration_cnt = 0

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
        self.refine_model = ex_config['refine_model']

        self.num_renderings = ex_config['n_views']
        self.num_samples_per_pcl = ex_config['num_samples_per_pcl']

        self.num_orientations_per_mesh = ex_config['num_orientations_per_mesh']
        self.num_scales = ex_config['num_scales']
        self.num_45_deg_rotations = ex_config['num_45_deg_rotations']
        self.inter_optim_steps = ex_config['inter_optim_steps']


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
            'cad_transformations_refined': [None] * len(ex_config['cad_transformations']),
            'transform_dict_refined': {}
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

        self.best_transform = None
        self.current_rot_45deg_index = None

        score = self.calc_render_loss(objects_cluster,base_rotation_idx)

        if self.best_transform is None or self.current_rot_45deg_index is None:
            assert False

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


    def render_loss(self,cad_mesh_rend,cad_pcl):

        with torch.no_grad():

            # L1 norm for point distance here
            chamfer_dist_x = 0.
            if self.w_chamfer != 0.:
                chamfer_dist_x = chamfer_distance_one_way(x=self.target_object['target_pcl'].points_padded(),
                                                          y=cad_pcl.points_padded(),
                                                          point_reduction='mean',
                                                          batch_reduction=None,
                                                          norm=1)
                chamfer_dist_x *= self.w_chamfer


            cad_mesh_rend = cad_mesh_rend.extend(self.num_renderings)

            fragments = self.renderer(meshes_world=cad_mesh_rend)

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

            loss_render = loss_sil + loss_depth + loss_sensor

            loss_final = torch.mean(loss_render, dim=0)
            loss_final = loss_final + chamfer_dist_x

        return loss_final

    def calc_render_loss(self,objects_cluster,base_rotation_idx):

        self.iteration_cnt +=1

        shapenet_id = objects_cluster[0]['mid'].split('/')[-1]
        batch = self.loader[(shapenet_id,base_rotation_idx)]
        transform_id = None
        min_idx = None

        cad_mesh_batch = batch['mesh'].to(device=self.device)
        cad_pcl_batch = batch['pcl'].to(device=self.device)
        loss_list_tmp = []
        for cad_mesh_,cad_pcl in zip(cad_mesh_batch,cad_pcl_batch):

            loss_init = self.render_loss(cad_mesh_,cad_pcl)
            loss_list_tmp.append(loss_init[0])

        min_idx = int(torch.argmin(torch.Tensor(loss_list_tmp)))
        self.current_rot_45deg_index = min_idx

        score_init = self.convert_loss_to_score(loss_list_tmp[min_idx])
        score_transform = -np.inf

        if min_idx == 0:
            transform_id = base_rotation_idx
        elif min_idx == 1:
            transform_id = 4 + base_rotation_idx * 2
        elif min_idx == 2:
            transform_id = (4 + base_rotation_idx * 2) + 1
        else:
            assert False

        transform_init = self.target_object['cad_transformations'][transform_id].clone().to(self.device)
        transform_init_inverse = self.target_object['cad_transformations'][transform_id].clone().to(self.device)
        transform_init_inverse = transform_init_inverse.inverse()

        if len(self.all_mcts_results_dict['score_list']) == 0:
            self.best_transform = transform_init
            return score_init

        score = score_init
        transform_dict = self.target_object['transform_dict'][transform_id]
        scale_func = transform_dict['scale_transform'].clone().to(self.device)
        rotate_func = transform_dict['rotate_transform'].clone().to(self.device)
        translate_func = transform_dict['translate_transform'].clone().to(self.device)

        self.best_transform = transform_init
        epsilon = 1.1

        if score > (np.max(self.all_mcts_results_dict['score_list']) * epsilon):

            start_refine_iter = 0
            curr_iter = len(self.all_mcts_results_dict['score_list'])
            if curr_iter > start_refine_iter:
                print(transform_id)

                transform_refined_new, scale_func_refined, rot_func_refined, translate_func_refined = \
                    self.perform_pose_refine(batch['path'], scale_func, rotate_func,
                                             translate_func,curr_iter,num_refine_iterations=self.inter_optim_steps)

                if transform_refined_new is not None:
                    cad_mesh_refine_ = cad_mesh_batch[min_idx].clone().to(self.device)

                    tverts = transform_init_inverse.transform_points(cad_mesh_refine_.verts_list()[0])

                    tverts_final = transform_refined_new.transform_points(tverts)
                    faces = cad_mesh_refine_.faces_list()[0]

                    cad_mesh_refine = Meshes(
                        verts=[tverts_final.to(self.device)],
                        faces=[faces.to(self.device)],
                    )

                    points_sampled, normals_sampled = sample_points_from_meshes(cad_mesh_refine,
                                                                                num_samples=self.num_samples_per_pcl,
                                                                                return_normals=True,
                                                                                return_textures=False)
                    cad_pcl = Pointclouds(points=points_sampled)

                    loss_final_refined = self.render_loss(cad_mesh_refine, cad_pcl)
                    score_after_refinement = self.convert_loss_to_score(loss_final_refined[0])

                    #if score_after_refinement > score:
                    if score_after_refinement > (np.max(self.all_mcts_results_dict['score_list'])):

                        score = score_after_refinement

                        self.target_object['cad_transformations_refined'][transform_id] = transform_refined_new
                        self.best_transform = transform_refined_new

                        self.loader.cad_transformations[transform_id] = transform_refined_new.to('cpu')

                        transform_dict_tmp = {}
                        transform_dict_tmp['scale_transform'] = scale_func_refined.get_matrix().detach()
                        transform_dict_tmp['rotate_transform'] = rot_func_refined.get_matrix().detach()
                        transform_dict_tmp['translate_transform'] = translate_func_refined.get_matrix().detach()

                        self.target_object['transform_dict_refined'][transform_id] = transform_dict_tmp

        return score

    def perform_pose_refine(self,model_path,scale_func_init, rotate_func_init, translate_func_init,curr_iter,
                            num_refine_iterations=126):
        #shapenet_dir = self.shapenet_path
        #model_path = os.path.join(shapenet_dir, self.obj_category, str(shapenet_id) + '.obj')
        model = self.refine_model

        verts_, faces_, _ = load_obj(
            model_path,
            create_texture_atlas=False,
            load_textures=False,
        )
        verts = verts_
        faces = faces_.verts_idx


        mesh = Meshes(
            verts=[verts.squeeze(dim=0)],
            faces=[faces]
        ).to(device=self.device)

        model.reset_model(mesh, scale_func_init, rotate_func_init, translate_func_init)
        adam_wd = 0.01
        adam_lr = 0.0002
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=adam_wd)

        transform_tmp = None
        scale_tmp = None
        rot_tmp = None
        translate_tmp = None
        tmesh_final = None
        min_loss = None

        with torch.enable_grad():
            for i in range(num_refine_iterations):
                optimizer.zero_grad()
                loss_sil, loss_depth, loss_sensor, tmesh, depth_final = model()

                if loss_sil is None:
                    break

                loss_sil *= self.w_sil
                loss_depth *= self.w_depth
                loss_sensor *= self.w_sensor

                points_cad_sampled_ = sample_points_from_meshes(tmesh, num_samples=self.num_samples_per_pcl,
                                                                return_normals=False, return_textures=False)
                pcl_CAD_ = Pointclouds(points=points_cad_sampled_).to(device=self.device)

                # L1 norm for point distance here
                chamfer_dist_x = 0.
                if self.w_chamfer != 0.:
                    chamfer_dist_x = chamfer_distance_one_way(x=self.target_object['target_pcl'].points_padded(),
                                                              y=pcl_CAD_.points_padded(),
                                                              point_reduction='mean',
                                                              batch_reduction=None,
                                                              norm=1)
                    chamfer_dist_x *= self.w_chamfer

                loss = loss_sil + loss_depth + loss_sensor
                loss_render = torch.mean(loss, dim=0) #+ chamfer_dist_x
                loss_refine = loss_render + chamfer_dist_x

                #loss_init = self.render_loss(tmesh,pcl_CAD_)

                if i == 0:
                    min_loss = loss_refine[0].item()

                loss_refine.backward()
                optimizer.step()

                if i % 25 == 0:
                    print('Optimization step: ' + str(i))
                    print('Optimizing (loss %.4f)' % loss_refine.item())

                if (loss_refine.item() < min_loss):
                    min_loss = loss_refine.item()
                    tmesh_final = tmesh
                    transform_tmp = model.transform_refine.clone()
                    scale_tmp = model.scale_func_refined.clone()
                    rot_tmp = model.rot_func_refined.clone()
                    translate_tmp = model.translate_func_refined.clone()

                if loss_refine.item() > (min_loss * 1.1):
                    break


        if tmesh_final is None:
            return None,None,None,None
        else:
            return transform_tmp, scale_tmp, rot_tmp, translate_tmp

    def convert_loss_to_score(self, loss):
        """
        Convert loss to score

        :param loss: loss value
        :type loss: torch.Tensor

        :return: score
        :rtype: torch.Tensor
        """

        score = - loss * 5.

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
        target_points_o3d.colors = o3d.utility.Vector3dVector(total_grad_color)
        target_points_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                                center=(0, 0, 0))

        prop_rec, _ = self.shapegf_recon_net.reconstruct_from_z(z_tensor, num_points=20000)

        prop_rec_pts_np = prop_rec[0].detach().cpu() /2

        prop_rec_pts_o3d = o3d.geometry.PointCloud()
        prop_rec_pts_o3d.points = o3d.utility.Vector3dVector(prop_rec_pts_np)
        prop_rec_pts_o3d.colors = o3d.utility.Vector3dVector(np.zeros_like(prop_rec_pts_np) + 0.5)

        prop_rec_pts_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                           center=(0, 0, 0))

        o3d.visualization.draw_geometries([prop_rec_pts_o3d, target_points_o3d])