import networkx as nx


class AuthorWorkGraph(nx.Graph):
    def __init__(self):
        super().__init__()

    def from_work_author_dict(self, work_author_dict):
        for key, value in work_author_dict.items():
            work = key
            for author in value:
                if not self.has_edge(work, author):
                    self.add_edge(work, author, weight=1)
