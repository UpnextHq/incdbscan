ObjectId = int


class Object:
    def __init__(self, id_):
        self.id: ObjectId = id_
        self.count = 1
        self.neighbors = set([self])

    @property
    def neighbor_count(self):
        return sum([neighbor.count for neighbor in self.neighbors])

    def __repr__(self):
        return f'{self.id_}_'