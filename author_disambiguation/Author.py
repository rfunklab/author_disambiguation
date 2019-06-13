


class Author(object):
    def __init__(self, lastname, firstname=None):
        self.lastname = lastname
        self.firstname = firstname
        self.works = []

    def add_work(self, work_id):
        if work_id not in self.works:
            self.works.append(work_id)
        else:
            raise AttributeError('Cannot add work; `work_id` already in authors works.')
