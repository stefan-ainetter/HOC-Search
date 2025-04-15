import numpy as np
from graphviz import Digraph
from abc import ABC, abstractmethod
from typing import List
import os
import shutil

from MonteScene.Tree.Tree import Tree
from MonteScene.Proposal.Prop import Proposal
from MonteScene.MonteCarloTreeSearch.MCTSLogger import MCTSLogger
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import IO
from ObjectGame.ObjectGame import ObjectGame
from ObjectGame.ClusterProposal import ClusterProposal

class ObjectsGameLogger(MCTSLogger):
    """
    Objects Game Logger

    Attributes:
          game: ObjectGame instance

    """
    def __init__(self, game, log_config, ex_config):
        """

        :param game: ObjectGame instance
        :type game: ObjectGame
        """

        self.game = game

        self.reset_logger()
        self.log_config = log_config
        self.ex_config = ex_config

        if os.path.exists(ex_config.log_path):
            shutil.rmtree(ex_config.log_path)
        os.makedirs(ex_config.log_path)

    def print_to_log(self, print_str):
        print(print_str)

    def reset_logger(self):
        """
        Reset logging variables

        :return:
        """

        self.scores_list = []
        self.tree_depth_list = []
        self.iters = []

        self.best_score = -np.inf
        self.best_prop_seq_str = "_inv"

    def export_solution(self, best_props_list):
        """
        Export final solution.

        :param best_props_list: List of best proposals
        :type best_props_list:  List[Proposal]
        :return:
        """

        # TODO Export solution. Should be called from log_final
        raise NotImplementedError()

    def log_final(self, mc_tree):
        """
        Final log performed after the search

        :param mc_tree: final tree
        :type mc_tree: Tree
        :return:
        """

        # TODO Implement logging for the last iteration on MCTS
        # TODO maybe visualize best solution
        # TODO maybe visualize plots
        # TODO maybe drawGraph
        # TODO maybe export_solution

        best_props_list, _ = mc_tree.get_best_path()
        best_prop_seq_str = "_".join([prop.id for prop in best_props_list])

        best_object_prop = best_props_list[-1]  # type: ClusterProposal

        best_object = best_object_prop.objects_cluster['cluster'][0]
        best_object_mid = best_object['mid']
        best_object_z = best_object['z']

        print("Best object model id: %s " % (best_object_mid))
        print("At iteration %d" % self.best_iter)

        # best_object_rec = self.game.reconstruct_prop_shapegf(best_object_prop, vis_o3d=True)
        #self.save_best_obj_as_ply(best_object_prop.objects_cluster['cluster'])
        #self.game.o3d_visualize_gf(best_object_prop)

    def save_best_obj_as_ply(self, objects_cluster):
        shapenet_id = objects_cluster[0]['mid'].split('/')[-1]
        model_path = os.path.join(self.game.shapenet_rw.shapenet_corev2_path, objects_cluster[0]['sid'], shapenet_id,
                                  'models',
                                  'model_normalized.obj')
        verts, faces, _ = load_obj(model_path, load_textures=False, device=self.game.device)

        tverts_normalized, faces_normalized = self.game.normalize_mesh(verts, faces.verts_idx)
        cad_transform = self.game.target_object['base_rotate'].compose(self.game.target_object['scale_func'],
                                                                       self.game.target_object['rotate_func'],
                                                                       self.game.target_object['translate_func']).to(
            self.game.device)
        tverts = cad_transform.transform_points(tverts_normalized)

        cad_mesh = Meshes(
            verts=[tverts.squeeze(dim=0)],
            faces=[faces_normalized],
        )

        mesh_out_path = os.path.join('/home/stefan/PycharmProjects/MCSS2/debug_out', 'cad_mesh_final.ply')

        IO().save_mesh(data=cad_mesh,
                       path=mesh_out_path)

    def log_mcts(self, iter, last_score, last_tree_depth, mc_tree):
        """
        Log MCTS progress

        :param iter: iteration
        :type iter: int
        :param last_score: last score
        :type last_score: float
        :param last_tree_depth: last tree depth
        :type last_tree_depth: int
        :param mc_tree: current tree
        :type mc_tree: Tree
        :return:
        """

        if last_score > self.best_score:

            best_props_list, _ = mc_tree.get_best_path()
            best_prop_seq_str = "_".join([prop.id for prop in best_props_list])

            # assert best_prop_seq_str != self.best_prop_seq_str, \
            #     "%.3f >= %.3f and %s == %s" % (last_score, self.best_score, best_prop_seq_str, self.best_prop_seq_str)

            self.best_score = last_score
            self.best_iter = iter

            self.best_prop_seq_str = best_prop_seq_str
            self.best_props_list = best_props_list

            best_object_prop = best_props_list[-1] # type: ClusterProposal

            best_object = best_object_prop.objects_cluster['cluster'][0]
            best_object_mid = best_object['mid']
            best_object_z = best_object['z']

            print("Best object model id: %s " % (best_object_mid))
            print("At iteration %d" % iter)

            # best_object_rec = self.game.reconstruct_prop_shapegf(best_object_prop, vis_o3d=True)

            #self.game.o3d_visualize_gf(best_object_prop)

            # Todo Export best ply

        if iter % self.log_config.export_graph_every == self.log_config.export_graph_every - 1:

            tree_file_path = os.path.join(self.ex_config['log_path'], "%s" % str(iter))
            self.drawGraph(mc_tree, tree_file_path, K=2)






