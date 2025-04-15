# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import os
from typing import Dict
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.structures.pointclouds import join_pointclouds_as_batch

from pytorch3d.structures.meshes import join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes

from pytorch3d.datasets.shapenet_base import ShapeNetBase


class ShapeNetCore_MCSS_45deg(ShapeNetBase):  # pragma: no cover
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
            self,
            data_dir,
            synsets=None,
            load_textures: bool = False,
            texture_resolution: int = 4,
            transform=None,
            rotations=[0],
            num_views=1,
            cls_label=None,
            cad_transformations=None,
            num_sampled_points=None,
            use_45deg_rot=False

    ) -> None:
        """
        Store each object's synset id and models id from data_dir.

        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and version 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        if cls_label is None:
            assert False
        self.cls_label = cls_label
        self.model_paths = []
        self.param_file_paths = []
        self.transform = transform
        self.rotations = rotations
        self.num_views = num_views
        self.cad_transformations = cad_transformations
        self.num_sampled_points = num_sampled_points
        self.use_45deg_rot = use_45deg_rot
        self.synset_id = synsets

        self.CAD_paths, self.obj_ids = self._load_paths()

    def _load_paths(self):
        cls_path = os.path.join(self.shapenet_dir, self.synset_id)
        obj_ids = os.listdir(cls_path)
        paths = [os.path.join(cls_path, x.strip(), 'models', 'model_normalized.obj') for x in obj_ids]

        obj_ids_valid = []
        paths_valid = []
        for obj_id, path in zip(obj_ids, paths):
            if os.path.exists(path):
                paths_valid.append(path)
                obj_ids_valid.append(obj_id)

        return paths_valid, obj_ids_valid

    def __len__(self) -> int:
        return len(self.CAD_paths)

    def __getitem__(self, idx: tuple) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model_id = str(idx[0]) + '.obj'
        base_rotation_idx = idx[1]
        model = {}
        model_path = os.path.join(self.shapenet_dir, self.synset_id, str(idx[0]), 'models', 'model_normalized.obj')
        verts_init, faces, _ = self._load_mesh(model_path)

        if not torch.isfinite(verts_init).all():
            raise ValueError("Meshes contain nan or inf.")

        mesh_list = []
        pcl_list = []

        if self.use_45deg_rot:
            transform_list = [self.cad_transformations[base_rotation_idx]]
            # tree rotations are self.cad_transformations[0:4]
            start_id = 4 + base_rotation_idx * 2  # base_rotation_idx * 4
            transform_list.append(self.cad_transformations[start_id])
            transform_list.append(self.cad_transformations[start_id + 1])
        else:
            transform_list = [self.cad_transformations[base_rotation_idx]]

        for transform_id, transform in enumerate(transform_list):

            verts_new = copy.deepcopy(verts_init)
            tverts_ = transform.transform_points(verts_new)
            mesh = Meshes(
                verts=[tverts_.squeeze(dim=0)],
                faces=[faces],
            )

            verts_test_nan = mesh.verts_packed()
            if not torch.isfinite(verts_test_nan).all():
                raise ValueError("Meshes contain nan or inf.")

            points_sampled, normals_sampled = sample_points_from_meshes(mesh, num_samples=self.num_sampled_points,
                                                                        return_normals=True,
                                                                        return_textures=False)
            pcl = Pointclouds(points=points_sampled)
            pcl_list.append(pcl)

            mesh_list.append(mesh)

        mesh_all = join_meshes_as_batch(mesh_list, include_textures=False)
        model['path'] = model_path
        model['cls_label'] = self.cls_label
        model['id'] = idx
        model['mesh'] = mesh_all
        model['pcl'] = join_pointclouds_as_batch(pcl_list)
        model['obj_id'] = model_id
        model['cad_transform'] = transform_list

        model['mesh_list'] = mesh_all
        model['pcl_list'] = pcl_list
        model['cad_transform_list'] = transform_list

        return model


class ShapeNetCore_MCSS_45deg_unknown_category(ShapeNetBase):  # pragma: no cover
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
            self,
            data_dir,
            synsets=None,
            load_textures: bool = False,
            texture_resolution: int = 4,
            transform=None,
            rotations=[0],
            num_views=1,
            cls_label=None,
            cad_transformations=None,
            num_sampled_points=None,
            use_45deg_rot=False

    ) -> None:
        """
        Store each object's synset id and models id from data_dir.

        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and version 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.cls_label = cls_label
        self.model_paths = []
        self.param_file_paths = []
        self.transform = transform
        self.rotations = rotations
        self.num_views = num_views
        self.cad_transformations = cad_transformations
        self.num_sampled_points = num_sampled_points
        self.use_45deg_rot = use_45deg_rot
        self.synset_id = synsets

    def _load_paths(self):
        cls_path = os.path.join(self.shapenet_dir, self.synset_id)
        obj_ids = os.listdir(cls_path)
        paths = [os.path.join(cls_path, x.strip(), 'models', 'model_normalized.obj') for x in obj_ids]

        obj_ids_valid = []
        paths_valid = []
        for obj_id, path in zip(obj_ids, paths):
            if os.path.exists(path):
                paths_valid.append(path)
                obj_ids_valid.append(obj_id)

        return paths_valid, obj_ids_valid

    def __len__(self) -> int:
        return len(self.CAD_paths)

    def __getitem__(self, idx: tuple) -> Dict:
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model_id = str(idx[0]) + '.obj'
        base_rotation_idx = idx[1]
        model = {}

        if self.cls_label is None:
            cls_label = idx[2]
        else:
            cls_label = self.cls_label

        model_path = os.path.join(self.shapenet_dir, self.synset_id, str(idx[0]), 'models', 'model_normalized.obj')

        verts_init, faces, _ = self._load_mesh(model_path)

        if not torch.isfinite(verts_init).all():
            raise ValueError("Meshes contain nan or inf.")

        mesh_list = []
        pcl_list = []

        if self.use_45deg_rot:
            transform_list = [self.cad_transformations[base_rotation_idx]]
            # tree rotations are self.cad_transformations[0:4]
            start_id = 4 + base_rotation_idx * 2  # base_rotation_idx * 4
            transform_list.append(self.cad_transformations[start_id])
            transform_list.append(self.cad_transformations[start_id + 1])
        else:
            transform_list = [self.cad_transformations[base_rotation_idx]]

        for transform_id, transform in enumerate(transform_list):

            verts_new = copy.deepcopy(verts_init)
            tverts_ = transform.transform_points(verts_new)
            mesh = Meshes(
                verts=[tverts_.squeeze(dim=0)],
                faces=[faces],
            )

            verts_test_nan = mesh.verts_packed()
            if not torch.isfinite(verts_test_nan).all():
                raise ValueError("Meshes contain nan or inf.")

            points_sampled, normals_sampled = sample_points_from_meshes(mesh, num_samples=self.num_sampled_points,
                                                                        return_normals=True,
                                                                        return_textures=False)
            pcl = Pointclouds(points=points_sampled)
            pcl_list.append(pcl)

            mesh_list.append(mesh)

        mesh_all = join_meshes_as_batch(mesh_list, include_textures=False)

        model['path'] = model_path
        model['cls_label'] = self.cls_label
        model['id'] = idx
        model['mesh'] = mesh_all
        model['pcl'] = join_pointclouds_as_batch(pcl_list)
        model['obj_id'] = model_id
        model['cad_transform'] = transform_list

        model['mesh_list'] = mesh_all
        model['pcl_list'] = pcl_list
        model['cad_transform_list'] = transform_list

        return model
