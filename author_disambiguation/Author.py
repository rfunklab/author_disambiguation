


class Author(object):
    '''Class object representing Authors. Author
    '''
    def __init__(self, lastname, firstname=None):
        self.lastname = lastname
        self.firstname = firstname
        self.works = []
        self.id = None

    def __eq__(self, other):
    '''Checks for equality between author objects. Depends solely on the `id`
    variable for now.'''
        if isinstance(other, Author):
            if self.id is None:
                return False
            if other.id is None:
                return False
            else:
                return self.id == other.id
        return False

    def add_work(self, work_id):
        '''Adds a work (paper, patent, etc.) to an author's `works` which is a
        list of work ids
        '''
        if work_id not in self.works:
            self.works.append(work_id)
        else:
            raise AttributeError('Cannot add work; `work_id` already in authors works.')
