


class Work(object):
    '''Class object representing Works (papers, patents, etc.). Works identified
    by `id`.
    '''
    def __init__(self, id=None):
        self.id = id
        self.embedding = None
        self.title = None
        self.summary = None
        self.cites = []
        self.authors = []
