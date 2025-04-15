import torch
import numpy as np
from pytorch3d.transforms import Transform3d

from utils import Rx, Ry, Rz
from .utils import rotate_points_around_y

from .RotationProposal import RotationProposal

class TargetObject(object):
    def __init__(self, ex_config, shapenet_rw: None):

        loader = shapenet_rw.get_obj_loader(ex_config.obj_category)

        self.device = shapenet_rw.device

        target_obj_sid = ex_config.sid
        target_obj_mid = ex_config.mid
        target_obj_pcd = loader.dataset.get_item_from_sid_mid(target_obj_sid, target_obj_mid)['te_points']
        target_obj_pcd = target_obj_pcd.cpu().numpy()

        target_obj_pcd = torch.from_numpy(target_obj_pcd).to(self.device)
        target_obj_pcd = target_obj_pcd[None]

        self.obj_category = ex_config.obj_category
        self.sid = target_obj_sid
        self.mid = target_obj_mid
        self.points3d = target_obj_pcd

        if hasattr(ex_config, 'rotation_degree'):
            self.points3d = self.rotate_around_y(ex_config.rotation_degree)

    def rotate_around_y(self, rot_degree):
        rot_radians = rot_degree * np.pi / 180.

        rot_mat = Ry(rot_radians)

        rot_mat = torch.from_numpy(rot_mat).to(self.device)

        transform_func = Transform3d(device=self.device).rotate(R=torch.tensor(rot_mat))

        rot_target_obj_pcd = transform_func.transform_points(self.points3d)

        return rot_target_obj_pcd

    def transform_using_prop_seq(self, prop_seq):
        target_points = self.points3d
        for p in prop_seq:
            if isinstance(p, RotationProposal):
                print("Rotate:", -p.rotation_degree)
                target_points = rotate_points_around_y(target_points, -p.rotation_degree)

        return target_points






