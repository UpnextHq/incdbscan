from typing import Dict

import numpy as np

from ._object import _Object, ObjectId
from ._utils import hash_


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


class _Objects:
    def __init__(self, eps, distance=euclidean_distance):
        self.eps = eps
        self.distance = distance
        self.objects: Dict[ObjectId, _Object] = dict()

    def get_object(self, value):
        id_ = hash_(value)
        return self.objects.get(id_)

    def insert_object(self, value):
        id_ = hash_(value)

        if id_ in self.objects:
            obj = self.objects[id_]
            obj.count += 1
            return obj

        new_object = _Object(value, id_)
        self._update_neighbors_during_insertion(new_object)
        self.objects[id_] = new_object
        return new_object

    def _update_neighbors_during_insertion(self, object_inserted):
        neighbors = self._get_neighbors(object_inserted)
        object_inserted.neighbors.update(neighbors)

        for obj in neighbors:
            obj.neighbors.add(object_inserted)

    def _get_neighbors(self, query_object):
        return [
            obj for obj in self.objects.values()
            if self.distance(query_object.value, obj.value) <= self.eps
        ]

    def delete_object(self, obj):
        obj.count -= 1
        if obj.count == 0:
            self._update_neighbors_during_deletion(obj)
            del self.objects[obj.id]

    def _update_neighbors_during_deletion(self, object_deleted):
        effective_neighbors = \
            object_deleted.neighbors.difference(set([object_deleted]))
        for neighbor in effective_neighbors:
            neighbor.neighbors.remove(object_deleted)

    @staticmethod
    def set_labels(objects, label):
        for obj in objects:
            obj.label = label

    def get_next_cluster_label(self):
        labels = [obj.label for obj in self.objects.values()]
        return max(labels) + 1

    def change_labels(self, change_from, change_to):
        for obj in self.objects.values():
            if obj.label == change_from:
                obj.label = change_to