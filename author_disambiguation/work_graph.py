import networkx as nx
import progressbar

class WorkGraph(nx.Graph):
    def __init__(self):
        super().__init__()


    def from_n_closure_dict(self, n_closure_dict):
        works = list(n_closure_dict.keys())
        bar = progressbar.ProgressBar(maxval=len(works), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        print('Calculating WorkGraph Intersections...')
        bar.start()
        for i in range(len(works)):
            self.add_node(works[i])
            bar.update(i)
            for j in range(i+1, len(works)-1):
                self.add_node(works[j])
                s1 = set(n_closure_dict[works[i]])
                s2 = set(n_closure_dict[works[j]])

                w = len(s1.intersection(s2))
                if w != 0:
                    self.add_edge(works[i], works[j], weight=w)
        bar.finish()
