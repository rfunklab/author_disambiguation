


class Work(object):
    '''Class object representing Works (papers, patents, etc.). Works identified
    by `id`.'''
    def __init__(self, id=None, title=None):
        self.id = id
        self.embedding = None
        self.title = title
        self.summary = None
        self.cites = []
        self.author_ids = []

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Work):
            return self.id == other.id

    def __repr__(self):
        if self.title is not None:
            return self.title
        else:
            return self.id

    def __str__(self):
        return self.title
