import networkx as nx
import itertools


class CoauthorGraph(nx.Graph):
    def __init__(self):
        super().__init__()

    def from_work_author_dict(self, work_author_dict):
        for key, value in work_author_dict.items():
            # author_pairs = itertools.product(value, repeat=2)
            for i in range(len(value)):
                # capture all authors as node in graph, even if no coauthor exists
                node1 = value[i]
                self.add_node(node1)
                for j in range(i+1, len(value)-1):
                    node2 = value[j]
                    if not self.has_edge(node1, node2):
                        self.add_edge(node1, node2, weight=1)
                    else:
                        self[node1][node2]['weight'] += 1

    @staticmethod
    def n_closure(work_author_dict, coauthor_graph, n=2):
        if n < 1:
            raise AttributeError('cannot have `n` < 1')
        if n == 1:
            return work_author_dict
        work_n_closure_dict = {}
        for work, authors in work_author_dict.items():
            n_auths = set()
            for author in authors:
                n_auths.add(author)
                if coauthor_graph.has_node(author):
                    if n == 2:
                        for neighbor in coauthor_graph.neighbors(author):
                            n_auths.add(neighbor)
                    else:
                        for neighbor, l in nx.single_source_shortest_path_length(coauthor_graph, author, cutoff=n).items():
                            if l > 1 and l <= n:
                                n_auths.add(neighbor)
            work_n_closure_dict[work] = list(n_auths)
        return work_n_closure_dict
