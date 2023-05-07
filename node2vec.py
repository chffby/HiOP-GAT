import numpy as np
import sys
sys.path.append('../')
import numpy.random as npr
import random
import math
# from gensim.models import Word2Vec


class Node2vec:
    def __init__(self, G,entity_embeddings, relation_embeddings,current_batch_2hop_indices):
        self.G = G
        self.nhop_head = current_batch_2hop_indices[:, 0].tolist()
        self.nhop_tail = current_batch_2hop_indices[:, 3].tolist()
        self.nhop_rela = current_batch_2hop_indices[:, 2].tolist()
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        # self.node_walk, self.edge_walk = self.node2vecWalk()

    def euclidean_distance(self,node,rel_node,entity_embeddings,relation_embeddings,neighbord_node_list,neighbord_edge_list):
        Aentity = entity_embeddings[node]
        Arelation = relation_embeddings[rel_node]
        d1list = []
        d2list = []
        nodes_info, edges_info = {}, {}
        for node_neig in neighbord_node_list:
            d1 = 0
            Bentity = entity_embeddings[node_neig]
            for (a, b) in zip(Aentity, Bentity):
                d1 += math.sqrt(sum([(a - b) ** 2 ]))
            d1list.append(d1)
        for node_neig in neighbord_edge_list:
            d2 = 0
            Brelation = entity_embeddings[node_neig]
            for (a, b) in zip(Arelation, Brelation):
                d2 += math.sqrt(sum([(a - b) ** 2 ]))
            d2list.append(d2)
        dlist = np.sum([d1list, d2list], axis=0).tolist()
        norm = sum(dlist)  # 求和
        normalized_probs = [float(n) / norm for n in dlist]  # 归一化
        nodes_info[node] = self.alias_setup(normalized_probs)
        # for node_neig in neighbord_node_list:
        #     edges_info[node_neig] = self.get_alias_edge(node_neig[0], node_neig[1])
        # self.nodes_info = nodes_info
        walk = self.alias_draw(nodes_info[node][0], nodes_info[node][1])
        # self.edges_info = edges_info
        # return d1list.index(min(d1list))
        return walk
        # return dlist.index(min(dlist))

    # def euclidean_distance(self,node,rel_node,entity_embeddings,relation_embeddings,neighbord_node_list,neighbord_edge_list):
    #     Aentity = entity_embeddings[node]
    #     # Arelation = relation_embeddings[rel_node]
    #     Aarrentity = np.array(Aentity.numpy())
    #     d1list = []
    #     # d2list = []
    #     for node_neig in neighbord_node_list:
    #         d1 = 0
    #         Bentity = entity_embeddings[node_neig]
    #         Barrentity = np.array(Bentity.numpy())
    #         d1 = np.linalg.norm(Aarrentity+Barrentity, ord=2)
    #         d1list.append(d1)
    #     # dlist = d1list.tolist()
    #     return d1list.index(min(d1list))

    def euclidean_distance(self,node,rel_node,entity_embeddings,relation_embeddings,neighbord_node_list,neighbord_edge_list):
        Aentity = entity_embeddings[node]
        # Arelation = relation_embeddings[rel_node]
        Aarrentity = np.array(Aentity.numpy())
        d1list = []
        nodes_info ,edges_info= {},{}
        # d2list = []
        if len(neighbord_node_list) == 1: return 0
        for node_neig in neighbord_node_list:
            d1 = 0
            Bentity = entity_embeddings[node_neig]
            for (a, b) in zip(Aentity, Bentity):
                d1 += math.sqrt(sum([(a - b) ** 2 ]))
            Barrentity = np.array(Bentity.numpy())
            d1 = np.linalg.norm(Aarrentity+Barrentity, ord=2)
            d1list.append(d1)
        # dlist = d1list.tolist()
        norm = sum(d1list)  # 求和
        normalized_probs = [float(n) / norm for n in d1list]  # 归一化
        nodes_info[node] = self.alias_setup(normalized_probs)
        # for node_neig in neighbord_node_list:
        #     edges_info[node_neig] = self.get_alias_edge(node_neig[0], node_neig[1])
        # self.nodes_info = nodes_info
        walk = self.alias_draw(nodes_info[node][0], nodes_info[node][1])
        # self.edges_info = edges_info
        # return d1list.index(min(d1list))
        return walk

    def alias_setup(self, probs):
        """
        :probs: v到所有x的概率
        :return: Alias数组与Prob数组
        """
        K = len(probs)
        q = np.zeros(K)  # 对应Prob数组
        J = np.zeros(K, dtype=np.int)  # 对应Alias数组
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []  # 存储比1小的列
        larger = []  # 存储比1大的列
        for kk, prob in enumerate(probs):
            q[kk] = K * prob  # 概率
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.

        # 通过拼凑，将各个类别都凑为1
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large  # 填充Alias数组
            q[large] = q[large] - (1.0 - q[small])  # 将大的分到小的上

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def get_alias_edge(self, t, v):
        """
        Get the alias edge setup lists for a given edge.
        """
        g = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for v_nbr in sorted(g.neighbors(v)):
            if v_nbr == t:
                unnormalized_probs.append(g[v][v_nbr]['weight'] / p)
            elif g.has_edge(v_nbr, t):
                unnormalized_probs.append(g[v][v_nbr]['weight'])
            else:
                unnormalized_probs.append(g[v][v_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)

    def alias_draw(self, J, q):
        """
        输入: Prob数组和Alias数组
        输出: 一次采样结果
        """
        K = len(J)
        # Draw from the overall uniform mixture.
        kk = int(np.floor(npr.rand() * K))  # 随机取一列

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if npr.rand() < q[kk]:  # 比较
            return kk
        else:
            return J[kk]

    def node2vecWalk(self, G, entity_embeddings, relation_embeddings, current_batch_2hop_indices):
        walk = []
        edge_walk = []
        node_walk = []
        # batch_node2vec_entity = []
        # batch_node2vec_edge = []
        nhop_head = current_batch_2hop_indices[:, 0].tolist()
        nhop_tail = current_batch_2hop_indices[:, 3].tolist()
        nhop_rela = current_batch_2hop_indices[:, 2].tolist()
        tail_walk = []
        i_nhop_rela = 0

        for node in nhop_tail:
            tail_walk.append(node)
            edge_node = 0
            walk = [node]
            edge = []
            edge.append(nhop_rela[i_nhop_rela])
            walk_len = 0
            curr_walk = tail_walk[-1]
            while walk_len <= 6:
                if curr_walk in G.keys():
                    neighbord_node_list = list(G[curr_walk].keys())
                    neighbord_edge_list = list(G[curr_walk].values())
                    next_walk = self.euclidean_distance(curr_walk, nhop_rela[edge_node], entity_embeddings,
                                                        relation_embeddings, neighbord_node_list, neighbord_edge_list)
                    # next_walk = np.random.choice(len(neighbord_node_list))
                    # next_walk = (neighbord_node_list[self.alias_draw(nodes_info[curr_walk][0], nodes_info[curr_walk][1])])
                    walk.append(neighbord_node_list[next_walk])
                    edge.append(neighbord_edge_list[next_walk])
                    curr_walk = walk[-1]
                else:
                    walk.append(np.random.choice(nhop_tail))
                    edge.append(random.randint(0, 233))
                    curr_walk = walk[-1]
                walk_len += 1
            i_nhop_rela += 1
            edge_node += 1
            node_walk.append(walk)
            edge_walk.append(edge)
        return node_walk, edge_walk
    # def euclidean_distance(self,node,rel_node,entity_embeddings,relation_embeddings,neighbord_node_list,neighbord_edge_list):
    #     Aentity = entity_embeddings[node]
    #     Arelation = relation_embeddings[rel_node]
    #     d1list = []
    #     d2list = []
    #     for node_neig in neighbord_node_list:
    #         d1 = 0
    #         Bentity = entity_embeddings[node_neig]
    #         for (a, b) in zip(Aentity, Bentity):
    #             d1 += math.sqrt(sum([(a - b) ** 2 ]))
    #         d1list.append(d1)
    #     for node_neig in neighbord_edge_list:
    #         d2 = 0
    #         Brelation = entity_embeddings[node_neig]
    #         for (a, b) in zip(Arelation, Brelation):
    #             d2 += math.sqrt(sum([(a - b) ** 2 ]))
    #         d2list.append(d2)
    #     dlist = np.sum([d1list, d2list], axis=0).tolist()
    #     select = dlist.index(min(dlist))
    #     return select,entity_embeddings[neighbord_node_list[select]]
    # def node2vecWalk(self, G, entity_embeddings, relation_embeddings, current_batch_2hop_indices):
    #     # walk = []
    #     edge_walk = []
    #     node_walk = []
    #     final_entity_embeddings = entity_embeddings
    #     final_relation_embeddings = relation_embeddings
    #     # batch_node2vec_entity = []
    #     # batch_node2vec_edge = []
    #     # nhop_head = current_batch_2hop_indices[:, 0].tolist()
    #     nhop_tail = current_batch_2hop_indices[:, 3].tolist()
    #     nhop_rela = current_batch_2hop_indices[:, 2].tolist()
    #     tail_walk = []
    #     i_nhop_rela = 0
    #
    #     for node in nhop_tail:
    #         tail_walk.append(node)
    #         edge_node = 0
    #         walk = [node]
    #         edge = []
    #         edge.append(nhop_rela[i_nhop_rela])
    #         walk_len = 0
    #         curr_walk = tail_walk[-1]
    #         # pre_rwalk = edge[-1]
    #         while walk_len <= 0:
    #             if curr_walk in G.keys():
    #                 pree_walk = curr_walk
    #                 # pre_rwalk = edge[-1]
    #                 neighbord_node_list = list(G[curr_walk].keys())
    #                 neighbord_edge_list = list(G[curr_walk].values())
    #                 next_walk,ent= self.euclidean_distance(curr_walk, nhop_rela[edge_node], entity_embeddings,
    #                                                     relation_embeddings, neighbord_node_list, neighbord_edge_list)
    #                 # next_walk = np.random.choice(len(neighbord_node_list))
    #                 walk.append(neighbord_node_list[next_walk])
    #                 edge.append(neighbord_edge_list[next_walk])
    #                 curr_walk = walk[-1]
    #                 # curr_rwalk = edge[-1]
    #             else:
    #                 pree_walk = curr_walk
    #                 pre_rwalk = edge[-1]
    #                 walk.append(np.random.choice(nhop_tail))
    #                 edge.append(random.randint(0, 10))
    #                 curr_walk = walk[-1]
    #                 # curr_rwalk = edge[-1]
    #             walk_len += 1
    #         i_nhop_rela += 1
    #         edge_node += 1
    #         node_walk.append(walk)
    #         edge_walk.append(edge)
    #         final_entity_embeddings[pree_walk] = final_entity_embeddings[pree_walk]+ent
    #         # final_relation_embeddings[pre_rwalk] = final_relation_embeddings[pre_rwalk]+rel
    #         pree_walk = curr_walk
    #         # pre_rwalk = curr_rwalk
    #     return final_entity_embeddings

        # G = self.G
        # for node in self.nhop_tail:
        #     walk = node
        #     walk_len=0
        #     if node in G.keys():
        #         while walk_len <= 0:
        #             curr_walk = walk[-1]
        #             neighbord_node_list = list(G[curr_walk].keys())
        #             next_walk = np.random.choice(neighbord_node_list)
        #             walk.append(next_walk)
        #             walk_len += 1
        #     else:
        #         walk.append(walk[-1])
        # return np.array(walk).astype(np.int32)
        # g = self.G
        # walk = [u]
        # nodes_info, edges_info = self.nodes_info, self.edges_info
        # while len(walk) < self.args.l:
        #     curr = walk[-1]
        #     v_curr = sorted(g.neighbors(curr))
        #     if len(v_curr) > 0:
        #         if len(walk) == 1:
        #             # print(adj_info_nodes[curr])
        #             # print(alias_draw(adj_info_nodes[curr][0], adj_info_nodes[curr][1]))
        #             walk.append(v_curr[self.alias_draw(nodes_info[curr][0], nodes_info[curr][1])])
        #         else:
        #             prev = walk[-2]
        #             ne = v_curr[self.alias_draw(edges_info[(prev, curr)][0], edges_info[(prev, curr)][1])]
        #             walk.append(ne)
        #     else:
        #         break
        #
        # return walk
