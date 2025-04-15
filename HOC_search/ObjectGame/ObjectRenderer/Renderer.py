import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    BlendParams
)

from ..ObjectProposal import ObjectProposal


class ObjectRenderer(object):
    def __init__(self, params):
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=params['image_size'],
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
        )

        # Initialize rasterizer by using a MeshRasterizer class
        rasterizer = MeshRasterizer(
            # cameras=cameras,
            raster_settings=raster_settings
        )

        # The textured phong shader interpolates the texture uv coordinates for
        # each vertex, and samples from a texture image.
        # shader = SoftPhongShader(device=device, cameras=cameras)
        shader = SoftSilhouetteShader(blend_params=blend_params)

        # Create a mesh renderer by composing a rasterizer and a shader
        self.renderer = MeshRenderer(rasterizer, shader)

    def render_mesh(self, mesh: Meshes, cameras: FoVPerspectiveCameras):

        batch_size = len(cameras)
        meshes = mesh.extend(batch_size)

        images = self.renderer(meshes, cameras=cameras)

        return images

    def render_obj_prop(self, prop: ObjectProposal, cameras: FoVPerspectiveCameras):

        batch_size = len(cameras)
        meshes = prop.get_mesh().extend(batch_size)

        images = self.renderer(meshes, cameras=cameras)

        return images

