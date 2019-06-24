


class Author(object):
    '''Class object representing Authors. Author identified by `id`, but can be
    instantiated using only `name`.'''
    def __init__(self, name, lastname=None, firstname=None, id=None):
        self.name = name
        self.lastname = lastname
        self.firstname = firstname
        self.works = []
        self.id = id

    def __eq__(self, other):
        '''Checks for equality between author objects. Depends solely on the `id`
        variable. If `id` not present, returns false.'''
        if isinstance(other, Author):
            if self.id is not None and other.id is not None:
                return self.id == other.id
            else:
                return self.name == other.name
                # return False
        return False

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def add_work(self, work):
        '''Adds a work (paper, patent, etc.) to an author's `works` which is a
        list of work objects.'''
        if work not in self.works:
            self.works.append(work)
        else:
            raise AttributeError('Cannot add work: already in authors works.')
