#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn


class MeshRendererScannet(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """

        fragments = self.rasterizer(meshes_world, **kwargs)

        return fragments


class MeshRendererViewSelection(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs):
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """

        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments
