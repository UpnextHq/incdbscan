import dataclasses
from collections import defaultdict
from typing import Set, Dict

import networkx as nx

from ._labels import (
    CLUSTER_LABEL_NOISE,
    CLUSTER_LABEL_UNCLASSIFIED
)


# frozen=True automatically adds eq and hash methods based on the values
# so we can compare and dedupe merge operations with equality
@dataclasses.dataclass(frozen=True)
class MergeOperation:
    source: int
    destination: int


@dataclasses.dataclass
class InsertionModifications:
    added: Set[int] = dataclasses.field(default_factory=set)
    merged: Set[MergeOperation] = dataclasses.field(default_factory=set)
    expanded: Dict[int, Set] = dataclasses.field(default_factory=lambda: defaultdict(lambda: set()))

    def __len__(self):
        return len(self.added) + len(self.merged) + len(self.expanded)

class Inserter:
    def __init__(self, eps, min_pts, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.objects = objects

    def insert(self, object_value, id_=None):
        operations = InsertionModifications()
        object_inserted = self.objects.insert_object(object_value, id_)

        new_core_neighbors, old_core_neighbors = \
            self._separate_core_neighbors_by_novelty(object_inserted)

        if not new_core_neighbors:
            # If there is no new core object, only the new object has to be
            # put in a cluster.

            if old_core_neighbors:
                # If there are already core objects near to the new object,
                # the new object is put in the most recent cluster. This is
                # similar to case "Absorption" in the paper but not defined
                # there.

                label_of_new_object = max([
                    self.objects.get_label(obj) for obj in old_core_neighbors
                ])

                operations.expanded[label_of_new_object].add(object_inserted.id)
            else:
                # If the new object does not have any core neighbors,
                # it becomes a noise. Called case "Noise" in the paper.

                label_of_new_object = CLUSTER_LABEL_NOISE

            self.objects.set_label(object_inserted, label_of_new_object)
            return operations

        update_seeds = self._get_update_seeds(new_core_neighbors)

        connected_components_in_update_seeds = \
            self._get_connected_components(update_seeds)

        for component in connected_components_in_update_seeds:
            effective_cluster_labels = \
                self._get_effective_cluster_labels_of_objects(component)

            if not effective_cluster_labels:
                # If in a connected component of update seeds there are only
                # previously unclassified and noise objects, a new cluster is
                # created. Corresponds to case "Creation" in the paper.

                next_cluster_label = self.objects.get_next_cluster_label()
                self.objects.set_labels(component, next_cluster_label)
                operations.added.add(next_cluster_label)

            else:
                # If in a connected component of update seeds there are
                # already clustered objects, all objects in the component
                # will be merged into the most recent cluster.
                # Corresponds to cases "Absorption" and "Merge" in the paper.

                max_label = max(effective_cluster_labels)

                for node in component:
                    node_label = self.objects.get_label(node)
                    if node_label != max_label:
                        # We want to mark the new entry (unclassified) and the noise labels as expansion
                        # nodes.  Node in the component list can also be part of existing clusters, but
                        # this should be handled in the merge operation below.
                        if node_label == CLUSTER_LABEL_NOISE or node_label == CLUSTER_LABEL_UNCLASSIFIED:
                            operations.expanded[max_label].add(node.id)

                self.objects.set_labels(component, max_label)

                for label in effective_cluster_labels:
                    if label == max_label:
                        continue

                    operations.merged.add(MergeOperation(source=label, destination=max_label))
                    self.objects.change_labels(label, max_label)

        # Finally all neighbors of each new core object inherits a label from
        # its new core neighbor, thereby affecting border and noise objects,
        # and the object being inserted.
        self._set_cluster_label_around_new_core_neighbors(new_core_neighbors)

        return operations

    def _separate_core_neighbors_by_novelty(self, object_inserted):
        new_cores = set()
        old_cores = set()

        for obj in object_inserted.neighbors:
            if obj.neighbor_count == self.min_pts:
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                old_cores.add(obj)

        # If the inserted object is core, it is a new core

        if object_inserted in old_cores:
            old_cores.remove(object_inserted)
            new_cores.add(object_inserted)

        return new_cores, old_cores

    def _get_update_seeds(self, new_core_neighbors):
        seeds = set()

        for new_core_neighbor in new_core_neighbors:
            core_neighbors = [obj for obj in new_core_neighbor.neighbors
                              if obj.neighbor_count >= self.min_pts]
            seeds.update(core_neighbors)

        return seeds

    @staticmethod
    def _get_connected_components(objects):
        if len(objects) == 1:
            return [objects]

        graph = nx.Graph()

        for obj in objects:
            for neighbor in obj.neighbors:
                if neighbor in objects:
                    graph.add_edge(obj, neighbor)

        return nx.connected_components(graph)

    def _get_effective_cluster_labels_of_objects(self, objects):
        non_effective_cluster_labels = {CLUSTER_LABEL_UNCLASSIFIED,
                                        CLUSTER_LABEL_NOISE}
        effective_cluster_labels = set()

        for obj in objects:
            label = self.objects.get_label(obj)
            if label not in non_effective_cluster_labels:
                effective_cluster_labels.add(label)

        return effective_cluster_labels


    def _set_cluster_label_around_new_core_neighbors(self, new_core_neighbors):
        for obj in new_core_neighbors:
            label = self.objects.get_label(obj)
            self.objects.set_labels(obj.neighbors, label)
