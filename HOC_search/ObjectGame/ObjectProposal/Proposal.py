import json
import os

from pytorch3d.io import load_obj
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

from MonteScene.Proposal import Proposal
from MonteScene.Tree.constants import NodesTypes

class ObjectProposal(Proposal):
    def __init__(self, model_id, model_folder_path, device):

        super().__init__(model_id, NodesTypes.OTHERNODE)

        model_images_path = os.path.join(model_folder_path, 'images')
        model_models_path = os.path.join(model_folder_path, 'models')

        model_obj_path = os.path.join(model_models_path, 'model_normalized.obj')
        model_normalized_json_path = os.path.join(model_models_path, 'model_normalized.json')

        with open(model_normalized_json_path, "r") as f:
            model_attributes = json.load(f)

        verts, faces, aux = load_obj(
            model_obj_path,
            device=device,
            load_textures=False
        )

        # atlas = aux.texture_atlas

        # Initialize the mesh with vertices, faces, and textures.
        # Created Meshes object
        # color = torch.ones(1, verts.size(0), 3, device=device)

        # print(color.shape)
        # print(verts.shape)

        model_mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx])
        # textures=TexturesVertex(verts_features=color))
        # textures=TexturesAtlas(atlas=[atlas]), )

        print('We have {0} vertices and {1} faces.'.format(verts.shape[0], faces.verts_idx.shape[0]))

        self.model_attributes = model_attributes
        self.model_mesh = model_mesh

    def get_mesh(self):
        return self.model_mesh
