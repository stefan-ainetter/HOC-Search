import torch
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points
from typing import Union


def loss_pose_refine(mask_pred, mask_GT, depth_GT, depth_final, mask_combined, depth_sensor,
                     mask_depth_valid_sensor, mask_depth_valid_render_GT, mask_depth_valid_render_pred,
                     device):
    loss_sil = 1 - (torch.sum((mask_pred * mask_GT), dim=(1, 2)) /
                    torch.sum((mask_pred + mask_GT - (mask_pred * mask_GT)), dim=(1, 2)))

    loss_depth = ((depth_GT - depth_final) * mask_depth_valid_render_GT * mask_depth_valid_render_pred).abs().mean(
        dim=(1, 2))
    loss_sensor = ((depth_sensor - depth_final) * mask_depth_valid_sensor * mask_depth_valid_render_pred).abs().mean(
        dim=(1, 2))

    return loss_sil, loss_depth, loss_sensor


def loss_IOU_render_sensor(mask_pred, mask_GT, depth_GT, depth_final, mask_combined, depth_sensor,
                           mask_depth_valid_sensor, mask_depth_valid_render_GT, mask_depth_valid_render_pred):
    loss_sil = 1 - (torch.sum((mask_pred * mask_GT), dim=(1, 2)) /
                    torch.sum((mask_pred + mask_GT - (mask_pred * mask_GT)), dim=(1, 2)))

    loss_depth = ((depth_GT - depth_final) * mask_depth_valid_render_GT * mask_depth_valid_render_pred).abs().mean(
        dim=(1, 2))
    loss_sensor = ((depth_sensor - depth_final) * mask_depth_valid_sensor * mask_depth_valid_render_pred).abs().mean(
        dim=(1, 2))

    return loss_sil, loss_depth, loss_sensor


def chamfer_distance_one_way(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
        norm: int = 2,
):
    """
    One-way Chamfer distance between two pointclouds x and y.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths

    if batch_reduction is not None:
        cham_x = cham_x.sum()

        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div

    return cham_x
