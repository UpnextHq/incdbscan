from typing import Dict

from ._labels import LabelHandler
from ._neighbor_searcher import NeighborSearcher
from ._object import (
    Object,
    ObjectId
)
from ._utils import hash_


class Objects(LabelHandler):
    def __init__(self, eps, num_dims):
        super().__init__()
        self.objects: Dict[ObjectId, Object] = {}
        self.neighbor_searcher = \
            NeighborSearcher(radius=eps, num_dims=num_dims)

    def get_object(self, value):
        id_ = hash_(value)
        return self.objects.get(id_)

    def get_object_by_id(self, id_):
        return self.objects.get(id_)

    def insert_object(self, value, id_=None):
        id_ = id_ if id_ is not None else hash_(value)

        if id_ in self.objects:
            obj = self.objects[id_]
            obj.count += 1
            return obj

        new_object = Object(id_)
        self.objects[id_] = new_object
        self.set_label_of_inserted_object(new_object)

        self.neighbor_searcher.insert(value, id_)

        self._update_neighbors_during_insertion(new_object, value)
            
        return new_object

    def _update_neighbors_during_insertion(self, object_inserted, new_value):
        neighbors = self._get_neighbors(new_value)

        for obj in neighbors:
            obj.neighbors.add(object_inserted)
            object_inserted.neighbors.add(obj)

    def _get_neighbors(self, query_value):
        neighbor_ids = self.neighbor_searcher.query_neighbors(query_value)

        for id_ in neighbor_ids:
            yield self.objects[id_]

    def delete_object(self, obj):
        obj.count -= 1
        if obj.count == 0:
            del self.objects[obj.id]
            self.neighbor_searcher.delete(obj.id)
            self._update_neighbors_during_deletion(obj)
            self.delete_label_of_deleted_object(obj)

    def delete_object_by_id(self, id_):
        obj = self.objects.get(id_)
        if obj:
            self.delete_object(obj)
        else:
            # You can choose to raise an error or simply return a message
            # depending on how you want your system to handle such cases.
            raise ValueError(f"No object found with ID {id_}")

    @staticmethod
    def _update_neighbors_during_deletion(object_deleted):
        effective_neighbors = \
            object_deleted.neighbors.difference({object_deleted})
        for neighbor in effective_neighbors:
            neighbor.neighbors.remove(object_deleted)
