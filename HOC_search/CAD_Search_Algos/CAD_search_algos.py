import copy
import os

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds
from HOC_search.shapenet_core import ShapeNetCore_MCSS_45deg, ShapeNetCore_MCSS_45deg_unknown_category

from HOC_search.ObjectRenderGame.ObjectRenderGame import ObjectRenderGame
from HOC_search.ObjectRenderGame.ObjectRenderRefineGame import ObjectRenderRefineGame
from HOC_search.ObjectRenderGame.ObjectRenderGameUnknownCategory import ObjectRenderGameUnknownCategory
from HOC_search.ObjectRenderGame.ObjectsGameLogger.ObjectsGameLogger import ObjectsGameLogger
from HOC_search.ObjectRenderGame.ObjectsGameLogger.ObjectsGameLoggerUnknownCategory import ObjectsGameLoggerUnknownCategory
from MonteScene.MonteCarloTreeSearch import MonteCarloSceneSearch
from HOC_search.PoseRefinementModel import PoseRefineModel

class CAD_Search_Algos(object):

    def __init__(self,config,config_general,renderer,box_item,n_views,device,num_45_deg_rotations,mesh_obj,max_depth_GT,
                 depth_bg,mask_GT, depth_GT, depth_sensor, mask_depth_valid_sensor,
                 mask_depth_valid_render_GT,cad_transformations,mesh_bg,weights_dict,transform_dict=None):

        self.config = config
        self.config_general = config_general
        self.renderer = renderer
        self.box_item = box_item
        self.shapenet_path = config_general['shapenet_path']
        self.n_views = n_views
        self.img_scale = config.getfloat('img_scale')
        self.batch_size = config.getint('batch_size')
        self.rotations = [0]
        self.num_scales = 1
        self.num_orientations_per_mesh = len(self.rotations)
        self.num_45_deg_rotations = num_45_deg_rotations
        self.inter_optim_steps = config.getint('inter_optim_steps')


        self.w_sil = weights_dict['weight_sil']#config.getfloat('weight_sil')
        self.w_depth = weights_dict['weight_depth']#config.getfloat('weight_depth')
        self.w_sensor = weights_dict['weight_sensor']#config.getfloat('weight_sensor')
        self.w_chamfer = weights_dict['weight_chamfer']#config.getfloat('weight_chamfer')

        self.num_sampled_points = config.getint('num_sampled_points')
        self.num_workers = config.getint('num_workers')

        self.mesh_obj = mesh_obj
        self.max_depth_GT = max_depth_GT

        self.mesh_bg = mesh_bg.extend(N=self.num_orientations_per_mesh * self.num_scales)
        self.depth_bg = depth_bg.repeat_interleave(repeats=self.num_orientations_per_mesh * self.num_scales, dim=0)
        self.mask_GT = mask_GT.repeat_interleave(repeats=self.num_orientations_per_mesh * self.num_scales, dim=0)
        self.depth_GT = depth_GT.repeat_interleave(repeats=self.num_orientations_per_mesh * self.num_scales, dim=0)


        self.depth_sensor = depth_sensor
        self.mask_depth_valid_sensor = mask_depth_valid_sensor
        self.mask_depth_valid_render_GT = \
            mask_depth_valid_render_GT.repeat_interleave(repeats=self.num_orientations_per_mesh *
                                                                 self.num_scales, dim=0)

        self.cad_transformations = cad_transformations
        self.transform_dict = transform_dict
        self.shapenet_dataset = None

        self.device = device

        if config.getint('num_mcss_iter') is None:
            self.num_mcss_iter = 500
        else:
            self.num_mcss_iter = config.getint('num_mcss_iter')

    def run_MCSS_search(self,obj_cluster_tree,config_mcss,log_path):

        if self.num_45_deg_rotations == 3:
            use_45deg_rot = True
        else:
            use_45deg_rot = False

        shapenet_dataset = ShapeNetCore_MCSS_45deg(self.shapenet_path, load_textures=False, synsets=self.box_item.catid_cad,
                                                   cls_label=self.box_item.category_label,
                                             cad_transformations=self.cad_transformations, num_views=self.n_views,
                                             num_sampled_points=self.num_sampled_points,use_45deg_rot=use_45deg_rot)

        points_GT_sampled = sample_points_from_meshes(self.mesh_obj, num_samples=self.num_sampled_points,
                                                      return_normals=False, return_textures=False)

        pcl_GT_ = Pointclouds(points=points_GT_sampled).to(device=self.device)

        ex_config_dict = {}
        ex_config_dict['n_views'] = self.n_views
        ex_config_dict['im_scale'] = self.img_scale
        ex_config_dict['batch_size'] = self.batch_size
        ex_config_dict['num_orientations_per_mesh'] = self.num_orientations_per_mesh
        ex_config_dict['num_scales'] = self.num_scales
        ex_config_dict['num_45_deg_rotations'] = self.num_45_deg_rotations
        ex_config_dict['device'] = self.device
        ex_config_dict['obj_category'] = self.box_item.category_label
        ex_config_dict['target_pcl'] = pcl_GT_
        ex_config_dict['log_path'] = log_path
        ex_config_dict['num_samples_per_pcl'] = self.num_sampled_points
        ex_config_dict['cad_transformations'] = self.cad_transformations
        ex_config_dict['renderer'] = self.renderer
        ex_config_dict['depth_GT'] = self.depth_GT
        ex_config_dict['depth_bg'] = self.depth_bg
        ex_config_dict['mask_GT'] = self.mask_GT
        ex_config_dict['mask_depth_valid_render_GT'] = self.mask_depth_valid_render_GT
        ex_config_dict['max_depth_GT'] = self.max_depth_GT
        ex_config_dict['depth_sensor'] = self.depth_sensor
        ex_config_dict['mask_depth_valid_sensor'] = self.mask_depth_valid_sensor
        ex_config_dict['dataset'] = shapenet_dataset
        ex_config_dict['w_sil'] = self.w_sil
        ex_config_dict['w_depth'] = self.w_depth
        ex_config_dict['w_sensor'] = self.w_sensor
        ex_config_dict['w_chamfer'] = self.w_chamfer
        ex_config_dict['shapenet_path'] = self.shapenet_path
        ex_config_dict['obj_cluster_tree'] = obj_cluster_tree
        ex_config_dict['transform_dict'] = self.transform_dict

        # # Instantiate Game
        game = ObjectRenderGame(config_mcss, ex_config_dict)

        # Create Logger
        mcts_logger = ObjectsGameLogger(game, config_mcss.logger, ex_config_dict)

        # Create MCTS
        mcts = MonteCarloSceneSearch(game, mcts_logger=mcts_logger, tree=game.obj_cluster_tree,
                                     settings=config_mcss.montescene)

        if self.num_mcss_iter >= len(shapenet_dataset.obj_ids) * self.num_scales * self.num_orientations_per_mesh:
            mcts.settings.mcts.num_iters = len(shapenet_dataset.obj_ids) * self.num_scales * self.num_orientations_per_mesh - 1
        else:
            mcts.settings.mcts.num_iters = self.num_mcss_iter

        mcts.run()
        loss_list = game.all_mcts_results_dict['score_list']
        obj_id_list = game.all_mcts_results_dict['obj_id_list']
        orientation_list = game.all_mcts_results_dict['orientation_list']
        iteration_list = game.all_mcts_results_dict['iteration_list']

        loss_list, obj_id_list, orientation_list, iteration_list = \
            zip(*sorted(zip(loss_list, obj_id_list, orientation_list, iteration_list), reverse=True))

        return loss_list, obj_id_list, orientation_list, iteration_list, game


    def run_MCSS_search_unknown_category(self,obj_cluster_tree,config_mcss,log_path):

        if self.num_45_deg_rotations == 3:
            use_45deg_rot = True
        else:
            use_45deg_rot = False

        shapenet_dataset = ShapeNetCore_MCSS_45deg_unknown_category(self.shapenet_path, load_textures=False,
                                                                    synsets=self.box_item.catid_cad,cls_label=None,
                                             cad_transformations=self.cad_transformations, num_views=self.n_views,
                                             num_sampled_points=self.num_sampled_points,use_45deg_rot=use_45deg_rot)

        points_GT_sampled = sample_points_from_meshes(self.mesh_obj, num_samples=self.num_sampled_points,
                                                      return_normals=False, return_textures=False)

        pcl_GT_ = Pointclouds(points=points_GT_sampled).to(device=self.device)

        ex_config_dict = {}
        ex_config_dict['n_views'] = self.n_views
        ex_config_dict['im_scale'] = self.img_scale
        ex_config_dict['batch_size'] = self.batch_size
        ex_config_dict['num_orientations_per_mesh'] = self.num_orientations_per_mesh
        ex_config_dict['num_scales'] = self.num_scales
        ex_config_dict['num_45_deg_rotations'] = self.num_45_deg_rotations
        ex_config_dict['device'] = self.device
        ex_config_dict['obj_category'] = None
        ex_config_dict['target_pcl'] = pcl_GT_
        ex_config_dict['log_path'] = log_path
        ex_config_dict['num_samples_per_pcl'] = self.num_sampled_points
        ex_config_dict['cad_transformations'] = self.cad_transformations
        ex_config_dict['renderer'] = self.renderer
        ex_config_dict['depth_GT'] = self.depth_GT
        ex_config_dict['depth_bg'] = self.depth_bg
        ex_config_dict['mask_GT'] = self.mask_GT
        ex_config_dict['mask_depth_valid_render_GT'] = self.mask_depth_valid_render_GT
        ex_config_dict['max_depth_GT'] = self.max_depth_GT
        ex_config_dict['depth_sensor'] = self.depth_sensor
        ex_config_dict['mask_depth_valid_sensor'] = self.mask_depth_valid_sensor
        ex_config_dict['dataset'] = shapenet_dataset
        ex_config_dict['w_sil'] = self.w_sil
        ex_config_dict['w_depth'] = self.w_depth
        ex_config_dict['w_sensor'] = self.w_sensor
        ex_config_dict['w_chamfer'] = self.w_chamfer
        ex_config_dict['shapenet_path'] = self.shapenet_path
        ex_config_dict['obj_cluster_tree'] = obj_cluster_tree
        ex_config_dict['transform_dict'] = self.transform_dict

        # # Instantiate Game
        game = ObjectRenderGameUnknownCategory(config_mcss, ex_config_dict)

        # Create Logger
        mcts_logger = ObjectsGameLoggerUnknownCategory(game, config_mcss.logger, ex_config_dict)

        # Create MCTS
        mcts = MonteCarloSceneSearch(game, mcts_logger=mcts_logger, tree=game.obj_cluster_tree,
                                     settings=config_mcss.montescene)

        mcts.settings.mcts.num_iters = self.num_mcss_iter

        mcts.run()
        loss_list = game.all_mcts_results_dict['score_list']
        obj_id_list = game.all_mcts_results_dict['obj_id_list']
        orientation_list = game.all_mcts_results_dict['orientation_list']
        iteration_list = game.all_mcts_results_dict['iteration_list']
        category_list = game.all_mcts_results_dict['category_list']

        loss_list, obj_id_list, orientation_list, iteration_list,category_list = \
            zip(*sorted(zip(loss_list, obj_id_list, orientation_list, iteration_list,category_list), reverse=True))

        return loss_list, obj_id_list, orientation_list, iteration_list, category_list, game


    def run_MCSS_search_refine(self,obj_cluster_tree,config_mcss,log_path,depth_out_path):

        if self.num_45_deg_rotations == 3:
            use_45deg_rot = True
        else:
            use_45deg_rot = False

        shapenet_dataset = ShapeNetCore_MCSS_45deg(self.shapenet_path, load_textures=False,
                                                   synsets=self.box_item.catid_cad,
                                                   cls_label=self.box_item.category_label,
                                             cad_transformations=self.cad_transformations, num_views=self.n_views,
                                             num_sampled_points=self.num_sampled_points,use_45deg_rot=use_45deg_rot)

        # Initialize a model using the renderer, mesh and reference image
        model = PoseRefineModel(bg_mesh=self.mesh_bg, obj_mesh=None,
                                scale_func_init=None,
                                rotate_func_init=None,
                                translate_func_init=None,
                                renderer=self.renderer, device=self.device,
                                num_views=self.n_views, depth_GT=self.depth_GT, mask_GT=self.mask_GT,
                                depth_bg=self.depth_bg,
                                max_depth_GT=self.max_depth_GT, depth_sensor=self.depth_sensor,
                                mask_depth_valid_sensor=self.mask_depth_valid_sensor,
                                mask_depth_valid_render_GT=self.mask_depth_valid_render_GT,
                                depth_out_path=depth_out_path)


        points_GT_sampled = sample_points_from_meshes(self.mesh_obj, num_samples=self.num_sampled_points,
                                                      return_normals=False, return_textures=False)

        pcl_GT_ = Pointclouds(points=points_GT_sampled).to(device=self.device)

        ex_config_dict = {}
        ex_config_dict['n_views'] = self.n_views
        ex_config_dict['im_scale'] = self.img_scale
        ex_config_dict['batch_size'] = self.batch_size
        ex_config_dict['num_orientations_per_mesh'] = self.num_orientations_per_mesh
        ex_config_dict['num_scales'] = self.num_scales
        ex_config_dict['num_45_deg_rotations'] = self.num_45_deg_rotations
        ex_config_dict['device'] = self.device
        ex_config_dict['obj_category'] = self.box_item.category_label
        ex_config_dict['target_pcl'] = pcl_GT_
        ex_config_dict['log_path'] = log_path
        ex_config_dict['num_samples_per_pcl'] = self.num_sampled_points
        ex_config_dict['cad_transformations'] = self.cad_transformations
        ex_config_dict['renderer'] = self.renderer
        ex_config_dict['depth_GT'] = self.depth_GT
        ex_config_dict['depth_bg'] = self.depth_bg
        ex_config_dict['mask_GT'] = self.mask_GT
        ex_config_dict['mask_depth_valid_render_GT'] = self.mask_depth_valid_render_GT
        ex_config_dict['max_depth_GT'] = self.max_depth_GT
        ex_config_dict['depth_sensor'] = self.depth_sensor
        ex_config_dict['mask_depth_valid_sensor'] = self.mask_depth_valid_sensor
        ex_config_dict['dataset'] = shapenet_dataset
        ex_config_dict['w_sil'] = self.w_sil
        ex_config_dict['w_depth'] = self.w_depth
        ex_config_dict['w_sensor'] = self.w_sensor
        ex_config_dict['w_chamfer'] = self.w_chamfer
        ex_config_dict['shapenet_path'] = self.shapenet_path
        ex_config_dict['obj_cluster_tree'] = obj_cluster_tree
        ex_config_dict['refine_model'] = model
        ex_config_dict['transform_dict'] = self.transform_dict
        ex_config_dict['inter_optim_steps'] = self.inter_optim_steps


        # # Instantiate Game
        game = ObjectRenderRefineGame(config_mcss, ex_config_dict)

        # Create Logger
        mcts_logger = ObjectsGameLogger(game, config_mcss.logger, ex_config_dict)

        # Create MCTS
        mcts = MonteCarloSceneSearch(game, mcts_logger=mcts_logger, tree=game.obj_cluster_tree,
                                     settings=config_mcss.montescene)

        if self.num_mcss_iter >= len(shapenet_dataset.obj_ids):
            mcts.settings.mcts.num_iters = len(shapenet_dataset.obj_ids)
        else:
            mcts.settings.mcts.num_iters = self.num_mcss_iter

        mcts.run()
        loss_list = game.all_mcts_results_dict['score_list']
        obj_id_list = game.all_mcts_results_dict['obj_id_list']
        orientation_list = game.all_mcts_results_dict['orientation_list']
        iteration_list = game.all_mcts_results_dict['iteration_list']
        rot_45deg_list = game.all_mcts_results_dict['deg_rot45_index_list']
        transform_list = game.all_mcts_results_dict['best_transform_list']

        loss_list, obj_id_list, orientation_list, iteration_list,rot_45deg_list,transform_list = \
            zip(*sorted(zip(loss_list, obj_id_list, orientation_list, iteration_list,rot_45deg_list,transform_list),
                        reverse=True))

        return loss_list, obj_id_list, orientation_list, iteration_list,rot_45deg_list, transform_list, game
