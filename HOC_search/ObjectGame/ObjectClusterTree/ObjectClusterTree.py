import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import torch
import pickle
import os
import copy
from MonteScene.Tree import Tree
from .utils import *
from MonteScene.Proposal import Proposal
from MonteScene.constants import NodesTypes
from MonteScene.Tree.Node import Node
from HOC_search.ObjectGame.ClusterProposal import ClusterProposal
from HOC_search.ObjectGame.RotationProposal import RotationProposal
from HOC_search.ObjectGame.CategoryProposal import CategoryProposal


class ObjectClusterTree(Tree):
    @staticmethod
    def export_tree_for_obj_cat(shapenet_rw, obj_category, cluster_tree_config, settings):
        """

        :param shapenet_rw:
        :type shapenet_rw: ShapeNetRW
        :param obj_category:
        :type obj_category: str
        :return:
        """

        z_obj_list = shapenet_rw.load_zs_for_all_objects(obj_category=obj_category)

        nodes_per_level = cluster_tree_config.nodes_per_level
        avg_nodes_per_cluster = cluster_tree_config.avg_nodes_per_cluster
        min_clusters_n = cluster_tree_config.min_clusters_n
        min_nodes_per_cluster = cluster_tree_config.min_nodes_per_cluster

        update_mode = settings.mcts.ucb_score_type
        node_stack = Stack()

        root_prop = Proposal(NodesTypes.NODE_STR_DICT[NodesTypes.ROOTNODE], NodesTypes.ROOTNODE)
        root_node = Node(root_prop, parent=None)

        node_stack.push(root_node)

        total_tree_nodes = 1
        leaves_num = 0
        max_leaf_cluster_size = 0
        max_tree_depth = 0
        avg_tree_depth = 0
        while True:
            curr_node = node_stack.pop() # type: Node
            if curr_node is None:
                break

            tree_level = curr_node.depth

            if curr_node.prop.type == NodesTypes.ROOTNODE:
                objects_cluster = z_obj_list
            else:
                objects_cluster = curr_node.prop.objects_cluster["cluster"]

            if tree_level < len(nodes_per_level):
                n_clusters = nodes_per_level[tree_level]
            else:
                n_clusters = nodes_per_level[-1]

            if len(objects_cluster) / n_clusters < avg_nodes_per_cluster:
                n_clusters = len(objects_cluster) // avg_nodes_per_cluster

            if n_clusters < min_clusters_n:
                continue

            sids = [obj['sid'] for obj in objects_cluster]
            mids = [obj['mid'] for obj in objects_cluster]
            zs = [obj['z'] for obj in objects_cluster]
            sids = np.asarray(sids)
            mids = np.asarray(mids)
            zs = np.asarray(zs)

            kmeans = KMeans(n_clusters=n_clusters, n_init=100, max_iter=1000, random_state=0).fit(zs)

            obj2cluster_ids = kmeans.labels_
            unique_cluster_ids, objects_per_cluster = np.unique(obj2cluster_ids, return_counts=True)

            curr_node.all_children_created = True
            for cid in unique_cluster_ids:
                c_objs_inds = np.where(obj2cluster_ids == cid)[0]

                c_cluster_size = c_objs_inds.shape[0]

                c_sids = sids[c_objs_inds]
                c_mids = mids[c_objs_inds]
                c_zs = zs[c_objs_inds, :]
                c_centroid = kmeans.cluster_centers_[cid]

                c_objects_cluster = [{
                    'sid': c_sids[c_o_ind],
                    'mid': c_mids[c_o_ind],
                    'z': c_zs[c_o_ind]
                } for c_o_ind in range(c_cluster_size)]

                c_objects_cluster = {
                    "cluster": c_objects_cluster,
                    'centroid': c_centroid
                }

                # print("Tree level %d, child id %d, child cluster size %d" % (tree_level, cid, c_cluster_size))

                c_prop = ClusterProposal(total_tree_nodes, c_objects_cluster)
                c_node = Node(c_prop, parent=curr_node)

                curr_node.append_child(c_node)
                total_tree_nodes += 1

                if c_cluster_size > min_nodes_per_cluster:
                    node_stack.push(c_node)
                else:
                    leaves_num += 1
                    print("%d-th Cluster leaf, Tree level %d, child id %d, child cluster size %d" %
                          (leaves_num, tree_level, cid, c_cluster_size))

                    if c_cluster_size > max_leaf_cluster_size:
                        max_leaf_cluster_size = c_cluster_size
                    if c_node.depth > max_tree_depth:
                        max_tree_depth = c_node.depth
                    avg_tree_depth += c_node.depth

            print("Max leaf cluster size was %d." %(max_leaf_cluster_size))
            print("Max tree depth was %d." %(max_tree_depth))

        avg_tree_depth = avg_tree_depth / float(leaves_num)
        print("Avg tree depth was %d." %(avg_tree_depth))

        tree_folder_path, tree_file_path = ObjectClusterTree.get_tree_path(obj_category, cluster_tree_config)
        os.makedirs(tree_folder_path, exist_ok=True)
        with open(tree_file_path, "wb") as f:
            pickle.dump(root_node, f)

    @staticmethod
    def get_tree_path(obj_category, cluster_tree_config):
        tree_folder_path = cluster_tree_config.tree_path % obj_category
        tree_file_path = tree_folder_path + "tree.pickle"
        return tree_folder_path, tree_file_path


    def __init__(self, obj_category_list, cluster_tree_config, settings):
        assert not settings.tree.add_esc_nodes, "Skipping nodes is not allowed for this tree"

        super().__init__(settings)
        self.add_esc_nodes = False

        if len(obj_category_list) == 1:
            obj_category = obj_category_list[0]
            self.obj_category = obj_category
            tree_folder_path, tree_file_path = ObjectClusterTree.get_tree_path(obj_category, cluster_tree_config)

            with open(tree_file_path, "rb") as f:
                root_node = pickle.load(f)
            #self.root_node = root_node

            rotation_degrees = cluster_tree_config.rotation_degrees
            if rotation_degrees:
                root_node = self.add_rotation_nodes_on_top(root_node,rotation_degrees)

            self.root_node = root_node

        else:

            root_prop = Proposal(NodesTypes.NODE_STR_DICT[NodesTypes.ROOTNODE], NodesTypes.ROOTNODE)
            final_root_node = Node(root_prop, parent=None)

            for obj_category in obj_category_list:

                new_category_prop_id = "CAT_" + str(obj_category)
                new_root_prop = CategoryProposal(new_category_prop_id, obj_category)

                self.obj_category = None
                tree_folder_path, tree_file_path = ObjectClusterTree.get_tree_path(obj_category, cluster_tree_config)

                with open(tree_file_path, "rb") as f:
                    root_node = pickle.load(f)
                #self.root_node = root_node

                rotation_degrees = cluster_tree_config.rotation_degrees
                if rotation_degrees:
                    root_node = self.add_rotation_nodes_on_top(root_node, rotation_degrees)

                cat_root_node = copy.deepcopy(root_node)  # type: Node
                cat_root_node.set_prop(new_root_prop)
                cat_root_node.change_node_id_and_children_id(new_category_prop_id + "_" + final_root_node.id)

                final_root_node.append_child(cat_root_node)


            self.root_node = final_root_node

    def add_rotation_nodes_on_top(self,root_node, rotation_degrees):
        root_prop = Proposal(NodesTypes.NODE_STR_DICT[NodesTypes.ROOTNODE], NodesTypes.ROOTNODE)
        new_root_node = Node(root_prop, parent=None)

        for rot_d in rotation_degrees:
            new_rot_prop_id = "ROT_" + str(rot_d)
            new_root_prop = RotationProposal(new_rot_prop_id, rot_d)

            rot_root_node = copy.deepcopy(root_node) # type: Node
            rot_root_node.set_prop(new_root_prop)
            rot_root_node.change_node_id_and_children_id(new_rot_prop_id + "_" + new_root_node.id)

            new_root_node.append_child(rot_root_node)

        return new_root_node


    def get_node_children(self, node, game=None):
        """
        Get children nodes.

        :param node: node to get the children from
        :type node: Node
        :param game: Game instance, unused and kept for Tree class compatibility. Can be kept as None
        :type game: Game

        :return: list of children nodes
        :rtype: List[Node]
        """
        # Endnode has no children
        if  node.prop.type == NodesTypes.ENDNODE:
            return []

        # Initialize list of children nodes if necessary
        if node.children_nodes is None:
            end_prop = Tree.generate_new_end_prop()

            end_node = Node(end_prop, node)
            node.children_nodes = [end_node]
            node.all_children_created = True

            end_node.children_nodes = []
            end_node.all_children_created = True

        return node.children_nodes

    def visualize_tree_centroids(self, shapenet_rw, traverse_depth_first=False, subtree_selection=[3,4,0,4], mid_subpath=None):
        """

        :param shapenet_rw:
        :type shapenet_rw: ShapeNetRW
        :param traverse_depth_first:
        :param subtree_select: list of children indices that describe the path to a wanted subtree
        :param mid_subpath: visualize path to the mid object
        :return:
        """
        if traverse_depth_first:
            nodes_to_visit = Stack()
        else:
            nodes_to_visit = Queue()

        if not subtree_selection:
            nodes_to_visit.push_list(self.root_node.children_nodes)
        else:
            subtree_node = self.root_node
            for child_node_ind in subtree_selection:
                subtree_node = subtree_node.children_nodes[child_node_ind]

            nodes_to_visit.push(subtree_node)

        shapegf_recon_net = shapenet_rw.load_shapegf_recon_model(self.obj_category)

        total_tree_nodes = 1
        leaves_num = 0
        max_leaf_cluster_size = 0
        max_tree_depth = 0
        avg_tree_depth = 0
        while True:
            curr_node = nodes_to_visit.pop()  # type: Node
            if curr_node is None:
                break

            tree_level = curr_node.depth

            print("Visualize current node with cluster id %s, at tree level %d, from parent id %s " %
                  (curr_node.prop.id, tree_level, curr_node.parent.prop.id))

            objects_cluster = curr_node.prop.objects_cluster["cluster"]
            cluster_centroid = curr_node.prop.objects_cluster["centroid"]

            if mid_subpath is not None and \
                    not mid_subpath in [objects_cluster[o_ind]["mid"] for o_ind in range(len(objects_cluster))]:
                continue

            print("Cluster with %d objects" % (len(objects_cluster)))


            if curr_node.children_nodes is not None:
                nodes_to_visit.push_list(curr_node.children_nodes)

            z_tensor = torch.from_numpy(cluster_centroid).to(device=shapenet_rw.device)
            z_tensor = z_tensor[None]

            rec, rec_list = shapegf_recon_net.reconstruct_from_z(z_tensor, num_points=20000)

            rec_pts_np = rec[0].detach().cpu()

            rec_pts_o3d = o3d.geometry.PointCloud()
            rec_pts_o3d.points = o3d.utility.Vector3dVector(rec_pts_np)
            rec_pts_o3d.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0)),
                               center=(0, 0, 0))
            rec_pts_o3d.translate((0, 0, -3))

            o3d.visualization.draw_geometries([rec_pts_o3d])




