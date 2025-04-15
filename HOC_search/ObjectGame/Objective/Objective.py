import torch

from ..TargetObject import TargetObject

class Objective(object):
    def __init__(self, shapenet_rw: None, shapegf_recon_net):
        """

        :param shapenet_rw:
        :param shapegf_recon_net:
        """
        self.shapenet_rw = shapenet_rw
        self.shapegf_recon_net = shapegf_recon_net

    def calc_gf_from_prop_and_target_points(self, leaf_cluster_prop, target_points):

        # objects_cluster = leaf_cluster_prop.objects_cluster["cluster"]
        cluster_centroid = leaf_cluster_prop.objects_cluster["centroid"]

        prop_z_tensor = torch.from_numpy(cluster_centroid).to(device=target_points.device)
        prop_z_tensor = prop_z_tensor[None]


        sigmas = self.shapegf_recon_net.sigmas[-1:]
        target2prop_total_grad, target2prop_total_abs_grad = self.shapegf_recon_net.calc_gf_from_z_and_points(
            prop_z_tensor, target_points, sigmas=sigmas)

        return target2prop_total_grad, target2prop_total_abs_grad

    def calc_gf_from_target_points_and_prop_points(self, target_points, prop_points):
        target_z = self.shapegf_recon_net.get_z(target_points)

        sigmas = self.shapegf_recon_net.sigmas[-1:]
        prop2target_total_grad, prop2target_total_abs_grad = \
            self.shapegf_recon_net.calc_gf_from_z_and_points(target_z, prop_points, sigmas=sigmas)

        return prop2target_total_grad, prop2target_total_abs_grad


