# coding: utf-8
import argparse
import os
import sys
import yaml
import torch

parser = argparse.ArgumentParser(description="HOC-Search for the ScanNet dataset")
parser.add_argument("--config", type=str,
                    default='ScanNet_HOC_Search.ini',
                    help="Path to configuration file")
parser.add_argument("--data_split", type=str, default="", help="data split")
parser.add_argument("--device", type=str, default="0", help="device")

os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.transforms import Transform3d
import pickle
from config import load_config
from config.utils import *
from HOC_search.ObjectGame.ObjectClusterTree import ObjectClusterTree
from HOC_search.CAD_Search_Algos.CAD_search_algos import *
from HOC_search.CAD_Search_Algos.prepare_scene import Prepare_Scene
from HOC_search.utils_CAD_retrieval import compose_cad_transforms, load_textured_cad_model, \
    load_textured_cad_model_normalized
import open3d as o3d
from HOC_search.ScanNetAnnotation import *
import copy


def main(args):
    # Setup
    if args.device == '':
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    else:
        device = 'cuda:' + str(torch.cuda.current_device())

    config_path = os.path.join(parent, 'config', args.config)
    config = load_config(config_path)['CAD_retrieval']
    config_general = load_config(config_path)['general']

    fast_search_classes = config_general.getstruct('fast_search_classes')
    rotation_degrees = config.getstruct('rotation_degrees')
    num_refine_iterations = config.getint('final_optim_steps')

    config_general['shapenet_path'] = os.path.join(parent, config_general['shapenet_path'])
    config_general['shapenet_core_path'] = os.path.join(parent, config_general['shapenet_core_path'])
    config_general['dataset_base_path'] = os.path.join(parent, config_general['dataset_base_path'])
    config['data_folder'] = os.path.join(parent, config['data_folder'])

    # Read MCSS config
    config_path = os.path.join(parent, "config/config_MCSS.yaml")
    with open(config_path, 'r') as f:
        config_mcss = yaml.safe_load(f)

    config_mcss = convert_dict2namespace(config_mcss)
    config_mcss.cluster_tree.rotation_degrees = rotation_degrees

    data_split = args.data_split
    prepare_scene_obj = Prepare_Scene(config, config_general, data_split, parent, device)
    scene_list = prepare_scene_obj.load_scene_list()

    for scene_cnt, scene_name in enumerate(scene_list):
        print(scene_name)

        out_path = os.path.join(parent, config['out_folder'], scene_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        pkl_out_path = os.path.join(out_path, scene_name + '.pkl')
        if os.path.exists(pkl_out_path):
            continue

        data_path = os.path.join(parent, config['data_folder'], scene_name, scene_name + '.pkl')
        if not os.path.exists(data_path):
            continue

        pkl_file = open(data_path, 'rb')
        scene_obj = pickle.load(pkl_file)

        mesh_scene = prepare_scene_obj.prepare_scene(scene_obj)

        # Save GT mesh in PyTorch3D coordinate system
        path_tmp = os.path.join(out_path, scene_name + '_mesh_py3D.ply')
        tmp = o3d.io.write_triangle_mesh(path_tmp, mesh_scene)

        obj_mesh_all = None

        weights_dict = {'weight_sil': config.getfloat('weight_sil'),
                        'weight_depth': config.getfloat('weight_depth'),
                        'weight_sensor': config.getfloat('weight_sensor'),
                        'weight_chamfer': config.getfloat('weight_chamfer')}

        for count, box_item in enumerate(scene_obj.obj_annotation_list):

            print(box_item.category_label)

            if box_item.category_label not in fast_search_classes:
                continue

            depth_out_path = os.path.join(out_path, str(count) + '_' + box_item.category_label)
            if not os.path.exists(depth_out_path):
                os.makedirs(depth_out_path)
            log_path = os.path.join(out_path, 'mcss_log_dir')
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            use_45_deg_rot_bool = config.getboolean('use_45_deg_rot')

            if use_45_deg_rot_bool:
                num_45_deg_rotations = 3
            else:
                num_45_deg_rotations = 1

            n_views_selected, mesh_bg, renderer, depth_GT, depth_bg, mask_GT, mask_depth_valid_render_GT, \
                max_depth_GT, mesh_obj, depth_sensor, \
                mask_depth_valid_sensor = prepare_scene_obj.prepare_box_item_for_rendering(box_item,
                                                                                           scene_obj.inst_seg_3d,
                                                                                           mesh_scene,
                                                                                           scene_name,
                                                                                           1, [0])
            cad_transformations, transform_dict = compose_cad_transforms(box_item,
                                                                         config.getstruct('rotations'),
                                                                         config.getint('num_scales'),
                                                                         use_45_deg_rot=use_45_deg_rot_bool)

            if mesh_obj.isempty():
                continue

            ret_obj = CAD_Search_Algos(config, config_general, renderer, box_item, n_views_selected, device,
                                       num_45_deg_rotations, mesh_obj,
                                       max_depth_GT, depth_bg, mask_GT, depth_GT, depth_sensor,
                                       mask_depth_valid_sensor,
                                       mask_depth_valid_render_GT, cad_transformations, mesh_bg,
                                       weights_dict, transform_dict)

            obj_cluster_tree = ObjectClusterTree([box_item.category_label], config_mcss.cluster_tree,
                                                 config_mcss.montescene)

            loss_list, obj_id_list, orientation_list, iteration_list, rot_45deg_list, transform_list, game = \
                ret_obj.run_MCSS_search_refine(obj_cluster_tree,
                                               config_mcss,
                                               log_path,
                                               depth_out_path)

            out_folder = os.path.join(out_path, str(count) + '_' + str(box_item.category_label))
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            cad_transformations_list = []
            transform_dict_list = []

            for cad_index in range(5):

                if cad_index >= len(obj_id_list):
                    break

                transform_dict_tmp = {}

                obj_id = obj_id_list[cad_index]
                min_idx = rot_45deg_list[cad_index]
                base_rotation_idx = orientation_list[cad_index]
                if min_idx == 0:
                    transform_id = base_rotation_idx
                elif min_idx == 1:
                    transform_id = 4 + base_rotation_idx * 2
                elif min_idx == 2:
                    transform_id = (4 + base_rotation_idx * 2) + 1
                else:
                    assert False

                if game.target_object['cad_transformations_refined'][transform_id] is not None:
                    transform_dict_transform = game.target_object['transform_dict_refined'][transform_id]
                    scale_matrix = transform_dict_transform['scale_transform'].clone()
                    scale_func = Transform3d(matrix=scale_matrix).to(device)
                    rotate_matrix = transform_dict_transform['rotate_transform'].clone()
                    rotate_func = Transform3d(matrix=rotate_matrix).to(device)

                    translate_matrix = transform_dict_transform['translate_transform'].clone()
                    translate_func = Transform3d(matrix=translate_matrix).to(device)
                else:
                    transform_dict = game.target_object['transform_dict'][transform_id]
                    scale_func = transform_dict['scale_transform'].clone().to(device)
                    rotate_func = transform_dict['rotate_transform'].clone().to(device)
                    translate_func = transform_dict['translate_transform'].clone().to(device)

                if cad_index == 0:
                    model_path = os.path.join(config_general['shapenet_path'], box_item.catid_cad, obj_id, 'models',
                                              'model_normalized.obj')
                    transform_refined_new, scale_func_refined, rot_func_refined, translate_func_refined = \
                        game.perform_pose_refine(model_path, scale_func, rotate_func,
                                                 translate_func, 0, num_refine_iterations=num_refine_iterations)

                    if transform_refined_new is None:
                        cad_transform = transform_list[cad_index].to('cpu')
                        transform_dict_tmp['scale_transform'] = scale_func
                        transform_dict_tmp['rotate_transform'] = rotate_func
                        transform_dict_tmp['translate_transform'] = translate_func
                    else:
                        cad_transform = transform_refined_new.to('cpu')
                        transform_dict_tmp['scale_transform'] = scale_func_refined
                        transform_dict_tmp['rotate_transform'] = rot_func_refined
                        transform_dict_tmp['translate_transform'] = translate_func_refined

                else:
                    cad_transform = transform_list[cad_index].to('cpu')
                    transform_dict_tmp['scale_transform'] = scale_func
                    transform_dict_tmp['rotate_transform'] = rotate_func
                    transform_dict_tmp['translate_transform'] = translate_func

                # synset_id = shapenet_category_dict.get(box_item.category_label)

                obj_id = obj_id_list[cad_index]
                # TODO use model path of normalized ShapeNet models, modify line below:
                model_path = os.path.join(config_general['shapenet_core_path'], box_item.catid_cad, obj_id, 'models',
                                          'model_normalized.obj')
                # noinspection PyTypeChecker
                cad_save_path = os.path.join(out_folder, str(cad_index) + '_' +
                                             str(iteration_list[cad_index]) + '_' +
                                             str(obj_id_list[cad_index]) + '_' +
                                             str(orientation_list[cad_index]) + '_' +
                                             str(loss_list[cad_index]) + ".ply")

                cad_transformations_list.append(cad_transform)
                transform_dict_list.append(transform_dict_tmp)
                cad_model_o3d = load_textured_cad_model_normalized(model_path, cad_transform, box_item.category_label)

                tmp = o3d.io.write_triangle_mesh(cad_save_path, cad_model_o3d)

                if cad_index == 0:
                    if obj_mesh_all is None:
                        obj_mesh_all = cad_model_o3d
                    else:
                        obj_mesh_all += cad_model_o3d

            del ret_obj
            del cad_transformations, transform_dict

        mesh_full_bg = copy.deepcopy(mesh_scene)
        mesh_full_bg.remove_vertices_by_index(prepare_scene_obj.all_obj_idx_list)
        path_tmp = os.path.join(out_path, scene_name + '_mesh_full_bg.ply')
        tmp = o3d.io.write_triangle_mesh(path_tmp, mesh_full_bg)

        if obj_mesh_all is not None:
            tmp = o3d.io.write_triangle_mesh(os.path.join(out_path, scene_name + "_cad_retrieval.ply"), obj_mesh_all)


if __name__ == "__main__":
    main(parser.parse_args())
