import numpy as np
import os

def load_intrinsics(depth_intrinsics_path):

    cam_K = np.zeros((3, 3))
    cam_K[2, 2] = 1
    cam_K_depth = np.zeros_like(cam_K)
    scene_type = None
    with open(depth_intrinsics_path, 'r') as cam_params:
        for cam_param in cam_params:
            param_split = cam_param.split(' ')
            if param_split[0] == 'fx_color':
                cam_K[0, 0] = float(param_split[2])
            if param_split[0] == 'fy_color':
                cam_K[1, 1] = float(param_split[2])
            if param_split[0] == 'mx_color':
                cam_K[0, 2] = float(param_split[2])
            if param_split[0] == 'my_color':
                cam_K[1, 2] = float(param_split[2])

            if param_split[0] == 'fx_depth':
                cam_K_depth[0, 0] = float(param_split[2])
            if param_split[0] == 'fy_depth':
                cam_K_depth[1, 1] = float(param_split[2])
            if param_split[0] == 'mx_depth':
                cam_K_depth[0, 2] = float(param_split[2])
            if param_split[0] == 'my_depth':
                cam_K_depth[1, 2] = float(param_split[2])

            if param_split[0] == 'sceneType':
                scene_type = ' '.join(param_split[2:]).rstrip()

    return cam_K,cam_K_depth,scene_type

def load_axis_alignment_mat(meta_file_path):
    metaFile = meta_file_path  # includes axisAlignment info for the train set scans.
    assert os.path.exists(metaFile), '%s' % metaFile
    axis_align_matrix = np.identity(4)
    if os.path.isfile(metaFile):
        with open(metaFile) as f:
            lines = f.readlines()

        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                                     for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break

    return axis_align_matrix

