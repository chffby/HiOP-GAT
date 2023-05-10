import numpy as np
import sys
sys.path.append('../')
import numpy.random as npr
import random
import math


class Node2vec:
    def __init__(self, G,entity_embeddings, relation_embeddings,current_batch_2hop_indices):
        self.G = G
        self.nhop_head = current_batch_2hop_indices[:, 0].tolist()
        self.nhop_tail = current_batch_2hop_indices[:, 3].tolist()
        self.nhop_rela = current_batch_2hop_indices[:, 2].tolist()
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

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
        norm = sum(dlist)
        normalized_probs = [float(n) / norm for n in dlist]
        nodes_info[node] = self.alias_setup(normalized_probs)
        walk = self.alias_draw(nodes_info[node][0], nodes_info[node][1])
        return walk

    def alias_setup(self, probs):
        """
        :probs: v到所有x的概率
        :return: Alias数组与Prob数组
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
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
        edge_walk = []
        node_walk = []
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
            while walk_len <= 2:
                if curr_walk in G.keys():
                    neighbord_node_list = list(G[curr_walk].keys())
                    neighbord_edge_list = list(G[curr_walk].values())
                    next_walk = self.euclidean_distance(curr_walk, nhop_rela[edge_node], entity_embeddings,
                                                        relation_embeddings, neighbord_node_list, neighbord_edge_list)
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