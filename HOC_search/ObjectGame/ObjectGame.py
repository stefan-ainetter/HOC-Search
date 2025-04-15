from abc import ABC, abstractmethod
import torch
from ordered_set import OrderedSet
from typing import List
import open3d as o3d
import numpy as np

#from MonteScene.Proposal import Proposal
from MonteScene.ProposalGame import ProposalGame
from MonteScene.constants import NodesTypes

from .Objective import Objective
from utils import Rx, Ry, Rz

from HOC_search.ObjectGame.ObjectClusterTree import ObjectClusterTree
from HOC_search.ObjectGame.colormap import MyColormap
from .TargetObject import TargetObject
from .RotationProposal import RotationProposal
from .utils import rotate_points_around_y

class ObjectGame(ProposalGame):
    """
    Implementation of scene game.
    """
    def __init__(self, config, ex_config):
        args_list = [config, ex_config]
        super().__init__(args_list)

    def __str__(self):
        return "ObjectGame()"

    def initialize_game(self, config, ex_config):
        """
        Initialize game-specific attributes.

        :return:
        """

        self.shapenet_rw = ShapeNetRW(config.shapenet)


        self.device = self.shapenet_rw.device

        self.obj_category = ex_config.obj_category
        self.obj_cluster_tree = ObjectClusterTree(self.obj_category, config.cluster_tree, config.montescene)

        self.prop_seq = []

        self.shapegf_recon_net = self.shapenet_rw.load_shapegf_recon_model(self.obj_category)
        self.loader = self.shapenet_rw.get_obj_loader(self.obj_category)

        self.target_object = TargetObject(ex_config=ex_config, shapenet_rw=self.shapenet_rw)

        self.objective = Objective(self.shapenet_rw, self.shapegf_recon_net)

    def generate_proposals(self):
        """
        Generate proposals for the game

        :return: a set of generated proposals
        :rtype: OrderedSet
        """

        # Nothing happens
        pass

    def restart_game(self):
        self.pool_curr = [] # This game does not use a pool
        self.prop_seq = []

    def set_state(self, pool_curr, prop_seq):
        self.prop_seq = prop_seq

    def step(self, prop):
        """
        Take a single step in the game.

        :param prop:
        :type prop: Proposal
        :return:
        """

        if prop.type not in NodesTypes.SPECIAL_NODES_LIST:
            self.prop_seq.append(prop)


    def get_prop_points_from_prop_seq(self, prop_seq):

        leaf_cluster_prop = prop_seq[-1]
        objects_cluster = leaf_cluster_prop.objects_cluster["cluster"]
        cluster_centroid = leaf_cluster_prop.objects_cluster["centroid"]

        pf_sid = objects_cluster[0]['sid']
        pf_mid = objects_cluster[0]['mid']
        prop_points = self.loader.dataset.get_item_from_sid_mid(pf_sid, pf_mid)['te_points'].to(device=self.device)
        prop_points = prop_points[None]

        return prop_points

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

        loss = self.calc_loss_from_proposals(prop_seq)
        score = self.convert_loss_to_score(loss)

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

        leaf_cluster_prop = prop_seq[-1]
        target_points = self.target_object.transform_using_prop_seq(prop_seq)
        prop_points = self.get_prop_points_from_prop_seq(prop_seq)

        target2prop_total_grad, target2prop_total_abs_grad = \
            self.objective.calc_gf_from_prop_and_target_points(leaf_cluster_prop, target_points)

        prop2target_total_grad, prop2target_total_abs_grad = \
            self.objective.calc_gf_from_target_points_and_prop_points(target_points, prop_points)

        loss = target2prop_total_abs_grad.mean() + prop2target_total_abs_grad.mean()

        return loss


    def convert_loss_to_score(self, loss):
        """
        Convert loss to score

        :param loss: loss value
        :type loss: torch.Tensor

        :return: score
        :rtype: torch.Tensor
        """

        score = - loss * 0.1

        score = score.cpu().numpy()

        return score

    def reconstruct_prop_shapegf(self, prop, vis_o3d=False):
        objects_cluster = prop.objects_cluster["cluster"]
        cluster_centroid = prop.objects_cluster["centroid"]

        print("Cluster with %d objects" % (len(objects_cluster)))

        z_tensor = torch.from_numpy(cluster_centroid).to(device=self.device)
        z_tensor = z_tensor[None]

        rec, rec_list = self.shapegf_recon_net.reconstruct_from_z(z_tensor, num_points=20000)

        if vis_o3d:
            rec_pts_np = rec[0].detach().cpu()

            rec_pts_o3d = o3d.geometry.PointCloud()
            rec_pts_o3d.points = o3d.utility.Vector3dVector(rec_pts_np)
            rec_pts_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                               center=(0, 0, 0))
            rec_pts_o3d.translate((0, 0, -3))

            o3d.visualization.draw_geometries([rec_pts_o3d])

        return rec

    def o3d_visualize_gf(self, prop_seq):
        """
        """

        leaf_cluster_prop = prop_seq[-1]
        target_points = self.target_object.transform_using_prop_seq(prop_seq)
        prop_points = self.get_prop_points_from_prop_seq(prop_seq)
        target2prop_total_grad, target2prop_total_abs_grad = \
            self.objective.calc_gf_from_prop_and_target_points(leaf_cluster_prop, target_points)

        total_abs_grad_np = target2prop_total_abs_grad.cpu().numpy()[0]
        total_abs_grad_np_norm = np.linalg.norm(total_abs_grad_np, axis=-1)
        total_abs_grad_np_norm = total_abs_grad_np_norm / np.max(total_abs_grad_np_norm)
        total_grad_color = MyColormap.get_color_from_value(total_abs_grad_np_norm)[:,:-1]


        target_points_np = target_points[0].cpu().numpy()
        target_points_o3d = o3d.geometry.PointCloud()
        target_points_o3d.points = o3d.utility.Vector3dVector(target_points_np)
        target_points_o3d.colors = o3d.utility.Vector3dVector(total_grad_color)
        target_points_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                                center=(0, 0, 0))

        prop_rec_pts_np = prop_points[0].detach().cpu()

        prop_rec_pts_o3d = o3d.geometry.PointCloud()
        prop_rec_pts_o3d.points = o3d.utility.Vector3dVector(prop_rec_pts_np)
        prop_rec_pts_o3d.colors = o3d.utility.Vector3dVector(np.zeros_like(prop_rec_pts_np) + 0.5)

        prop_rec_pts_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                           center=(0, 0, 0))

        o3d.visualization.draw_geometries([prop_rec_pts_o3d, target_points_o3d])