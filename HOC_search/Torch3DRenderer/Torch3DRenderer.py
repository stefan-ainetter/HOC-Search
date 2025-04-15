import torch
import math
import pytorch3d
from pytorch3d.renderer.cameras import PerspectiveCameras
from HOC_search.Torch3DRenderer.pytorch3d_rasterizer_custom import MeshRendererScannet
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer
)

from .SimpleShader import SimpleShader


def initialize_renderer(n_views, img_scale, R, T, intrinsics, batch_size, num_orientations_per_mesh, device, img_height,
                        img_width, bin_size=None):
    raster_settings = RasterizationSettings(
        image_size=(math.ceil(img_height * img_scale), math.ceil(img_width * img_scale)),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )
    R_world_to_cam = R
    T_world_to_cam = T

    R = torch.as_tensor(R_world_to_cam).to(device)
    R = R.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    T = torch.as_tensor(T_world_to_cam).to(device)
    T = T.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    px, py = (intrinsics[0, 2] * img_scale), (intrinsics[1, 2] * img_scale)
    principal_point = torch.as_tensor([px, py])[None].type(torch.FloatTensor).to(device)
    principal_point = principal_point.repeat(n_views, 1)
    principal_point = principal_point.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    fx, fy = ((intrinsics[0, 0] * img_scale)), ((intrinsics[1, 1] * img_scale))
    focal_length = torch.as_tensor([fx, fy])[None].type(torch.FloatTensor).to(device)
    focal_length = focal_length.repeat(n_views, 1)
    focal_length = focal_length.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        device=device, T=T, R=R,
        image_size=((math.ceil(img_height * img_scale), math.ceil(img_width * img_scale)),))

    renderer = MeshRendererScannet(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SimpleShader(
            device=device,
            cameras=cameras
        )
    )

    del R, T, principal_point, focal_length
    return renderer


def initialize_renderer_scannetpp(n_views, img_scale, R, T, intrinsics, radial_params_, batch_size,
                                  num_orientations_per_mesh, device, img_height,
                                  img_width, bin_size=None):
    raster_settings = RasterizationSettings(
        image_size=(math.ceil(img_height * img_scale), math.ceil(img_width * img_scale)),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )

    R_world_to_cam = R
    T_world_to_cam = T

    R = torch.as_tensor(R_world_to_cam).to(device)
    R = R.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    T = torch.as_tensor(T_world_to_cam).to(device)
    T = T.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    px, py = (intrinsics[0, 2] * img_scale), (intrinsics[1, 2] * img_scale)
    principal_point = torch.as_tensor([px, py])[None].type(torch.FloatTensor).to(device)
    principal_point = principal_point.repeat(n_views, 1)
    principal_point = principal_point.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    fx, fy = ((intrinsics[0, 0] * img_scale)), ((intrinsics[1, 1] * img_scale))
    focal_length = torch.as_tensor([fx, fy])[None].type(torch.FloatTensor).to(device)
    focal_length = focal_length.repeat(n_views, 1)
    focal_length = focal_length.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)

    # radial_params = torch.tensor([[radial_params_[0], radial_params_[1], radial_params_[2],radial_params_[3],0.,0.],],dtype=torch.float32,)
    # radial_params = radial_params.repeat(n_views, 1)
    # radial_params = radial_params.repeat_interleave(repeats=num_orientations_per_mesh * batch_size, dim=0)
    #
    # cameras = FishEyeCameras(
    #     focal_length=focal_length,
    #     principal_point=principal_point,
    #     device=device, T=T, R=R, radial_params=radial_params,
    #     use_thin_prism=False,
    #     use_radial=True,
    #     use_tangential=False,
    #     world_coordinates=True,
    #     image_size=((math.ceil(img_height * img_scale), math.ceil(img_width * img_scale)),))

    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        device=device, T=T, R=R,
        image_size=((math.ceil(img_height * img_scale), math.ceil(img_width * img_scale)),))

    renderer = MeshRendererScannet(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SimpleShader(
            device=device,
            cameras=cameras
        )
    )

    del R, T, principal_point, focal_length
    return renderer


def prepare_GT_data(scene_mesh, mesh_bg, renderer, device):
    fragments_GT = renderer(meshes_world=scene_mesh.to(device))
    depth_GT = fragments_GT.zbuf

    mask_depth_valid_render_GT = torch.zeros_like(depth_GT)
    mask_depth_valid_render_GT[fragments_GT.pix_to_face != -1] = 1

    depth_GT[fragments_GT.pix_to_face == -1] = torch.max(depth_GT[fragments_GT.pix_to_face > 0])
    max_depth_GT = torch.max(depth_GT)

    fragments_bg = renderer(meshes_world=mesh_bg.to(device))
    depth_bg = fragments_bg.zbuf
    depth_bg[fragments_bg.pix_to_face == -1] = max_depth_GT

    mask_GT = torch.zeros_like(depth_GT)
    mask_bg = torch.zeros_like(depth_GT)
    mask_GT[depth_GT < depth_bg] = 1.
    mask_bg[depth_GT >= depth_bg] = 1.

    del fragments_GT, fragments_bg
    return depth_GT, depth_bg, mask_GT, mask_depth_valid_render_GT, max_depth_GT
