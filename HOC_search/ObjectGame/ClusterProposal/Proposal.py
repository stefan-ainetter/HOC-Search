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
from MonteScene.constants import NodesTypes

class ClusterProposal(Proposal):
    def __init__(self, id, objects_cluster):

        super().__init__(id, NodesTypes.OTHERNODE)

        self.objects_cluster = objects_cluster


