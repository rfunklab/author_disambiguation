# Sampler from https://arxiv.org/abs/1702.02287

import numpy as np
import random
from author_disambiguation.utils import softmax


"""
(i, j) belongs positive sample set
(i, t) belongs negative sample set
notation details are in the paper
"""

class CoauthorGraphSampler():
    @staticmethod
    def generate_triplet_uniform(coauthor_graph):
        """
        sample negative instance uniformly
        """
        a_i = random.choice(list(coauthor_graph.nodes()))
        a_t = random.choice(list(coauthor_graph.nodes()))

        while True:
            neig_list = list(coauthor_graph.neighbors(a_i))
            if a_t not in neig_list and len(neig_list) > 0:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [coauthor_graph[a_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                a_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield a_i, a_j, a_t
                break

            else:
                a_i = random.choice(list(coauthor_graph.nodes()))
                a_t = random.choice(list(coauthor_graph.nodes()))

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        a_i = random.choice(dataset.C_Graph.nodes())
        neg_pair = random.sample(dataset.coauthor_list, 2)

        while True:
            neig_list = dataset.C_Graph.neighbors(a_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.C_Graph[a_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                a_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(a_i, neg_pair[0], "pp")
                sc2 = bpr_optimizer.predict_score(a_i, neg_pair[1], "pp")
                a_t = neg_pair[0] if sc1 >= sc2 else neg_pair[1]
                yield a_i, a_j, a_t
                break

            else:
                a_i = random.choice(dataset.C_Graph.nodes())
                neg_pair = random.sample(dataset.coauthor_list, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        a_i = random.choice(dataset.C_Graph.nodes())
        neg_list = list(set(dataset.coauthor_list) - \
                   set(dataset.C_Graph.neighbors(a_i)) - set([a_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.C_Graph.neighbors(a_i)
        weight_list = [dataset.C_Graph[a_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        a_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        norm_soft = softmax([bpr_optimizer.predict_score(a_i, ne, "pp")
                             for ne in neg_list])
        a_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
        yield a_i, a_j, a_t


class LinkedDocGraphSampler():
    @staticmethod
    def generate_triplet_uniform(work_graph):
        d_i = random.choice(list(work_graph.nodes()))
        d_t = random.choice(list(work_graph.nodes()))

        while True:
            neig_list = list(work_graph.neighbors(d_i))
            if d_t not in neig_list and len(neig_list) > 0:
                # given d_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [work_graph[d_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                d_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]
                yield d_i, d_j, d_t
                break

            else:
                d_i = random.choice(list(work_graph.nodes()))
                d_t = random.choice(list(work_graph.nodes()))

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        d_i = random.choice(dataset.D_Graph.nodes())
        neg_pair = random.sample(dataset.paper_list, 2)

        while True:
            neig_list = dataset.D_Graph.neighbors(d_i)
            if neg_pair[0] not in neig_list and neg_pair[1] not in neig_list:
                # given a_i, sample its neighbor based on its weight value
                # idea of edge sampling
                weight_list = [dataset.D_Graph[d_i][nbr]['weight']
                               for nbr in neig_list]
                norm_weight_list = [float(w) / sum(weight_list)
                                    for w in weight_list]
                d_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(d_i, neg_pair[0], "dd")
                sc2 = bpr_optimizer.predict_score(d_i, neg_pair[1], "dd")
                d_t = neg_pair[0] if sc1 >= sc2 else neg_pair[1]
                yield d_i, d_j, d_t
                break

            else:
                d_i = random.choice(dataset.D_Graph.nodes())
                neg_pair = random.sample(dataset.paper_list, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        d_i = random.choice(dataset.D_Graph.nodes())
        neg_list = list(set(dataset.paper_list) - \
                   set(dataset.D_Graph.neighbors(d_i)) - set([d_i]))

        # given a_i, sample its neighbor based on its weight value
        # idea of edge sampling
        neig_list = dataset.D_Graph.neighbors(d_i)
        weight_list = [dataset.D_Graph[d_i][nbr]['weight']
                       for nbr in neig_list]
        norm_weight_list = [float(w) / sum(weight_list)
                            for w in weight_list]
        d_j = np.random.choice(neig_list, 1, p=norm_weight_list)[0]

        # sample negative instance based on pre-defined exponential distribution
        norm_soft = softmax([bpr_optimizer.predict_score(d_i, ne, "dd")
                             for ne in neg_list])
        d_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
        yield d_i, d_j, d_t


class BipartiteGraphSampler():
    @staticmethod
    def generate_triplet_uniform(author_work_graph, coauthor_graph, work_graph):
        d_i = random.choice(list(work_graph.nodes()))
        a_t = random.choice(list(coauthor_graph.nodes()))

        while True:
            if not author_work_graph.has_edge(d_i, a_t):
                a_j = random.choice(list(author_work_graph.neighbors(d_i)))
                yield d_i, a_j, a_t
                break

            else:
                d_i = random.choice(list(work_graph.nodes()))
                a_t = random.choice(list(coauthor_graph.nodes()))

    @staticmethod
    def generate_triplet_reject(dataset, bpr_optimizer):
        """
        generate negative instance using ranking-aware rejection sampler
        consider linear case
        """
        d_i = random.choice(dataset.paper_list)
        neg_pair = random.sample(dataset.coauthor_list, 2)

        while True:
            if dataset.paper_authorlist_dict[d_i] != [] \
                and neg_pair[0] not in dataset.paper_authorlist_dict[d_i] \
                    and neg_pair[1] not in dataset.paper_authorlist_dict[d_i]:

                a_j = random.choice(dataset.paper_authorlist_dict[d_i])

                # sample negative instance using ranking-aware rejection sampler
                sc1 = bpr_optimizer.predict_score(d_i, neg_pair[0], "pd")
                sc2 = bpr_optimizer.predict_score(d_i, neg_pair[1], "pd")
                a_t = neg_pair[0] if sc1 >= sc2 else neg_pair[1]
                yield d_i, a_j, a_t
                break

            else:
                d_i = random.choice(dataset.paper_list)
                neg_pair = random.sample(dataset.coauthor_list, 2)

    @staticmethod
    def generate_triplet_adaptive(dataset, bpr_optimizer):
        """
        generate negative instance using adaptive sampling
        sample from a pre-defined exponential distribution
        """
        d_i = random.choice(dataset.paper_list)
        neg_list = list(set(dataset.coauthor_list) - \
                        set(dataset.paper_authorlist_dict[d_i]))

        while True:
            if dataset.paper_authorlist_dict[d_i] != []:
                a_j = random.choice(dataset.paper_authorlist_dict[d_i])

                # sample negative instance based on pre-defined exponential distribution
                norm_soft = softmax([bpr_optimizer.predict_score(d_i, ne, "pd")
                                     for ne in neg_list])
                a_t = np.random.choice(neg_list, 1, p = norm_soft)[0]
                yield d_i, a_j, a_t
                break

            else:
                d_i = random.choice(dataset.paper_list)
                neg_list = list(set(dataset.coauthor_list) - \
                                set(dataset.paper_authorlist_dict[d_i]))
