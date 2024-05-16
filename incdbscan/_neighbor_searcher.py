import faiss
import numpy as np
from faiss import IndexIDMap
from opentelemetry import trace
from sortedcontainers import SortedList

BRUTE_FORCE_CUTOFF = 5000

class NeighborSearcher:
    def __init__(self, radius, num_dims):
        self.radius = radius

        self.num_dims = num_dims

        self.values = np.array([])
        self.ids = SortedList()

        self.neighbor_searcher = None
        self.remake_index()

    def insert(self, new_value, new_id):
        self.ids.add(new_id)
        position = self.ids.index(new_id)

        self._insert_into_array(new_value, position)

        if len(self.values) % BRUTE_FORCE_CUTOFF:
            self.remake_index()

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span('incdbscan_insert_neighborhood_searcher_fit'):
            new_value = np.array([new_value], dtype=np.float32)
            faiss.normalize_L2(new_value)

            self.neighbor_searcher.add_with_ids(new_value, np.array([new_id], dtype='int64'))

    def _insert_into_array(self, new_value, position):
        extended = np.insert(self.values, position, new_value, axis=0)
        if not self.values.size:
            extended = extended.reshape(1, -1)
        self.values = extended

    def query_neighbors(self, query_value):
        query_value = np.array([query_value], dtype=np.float32)
        faiss.normalize_L2(query_value)

        _, _, neighbors = self.neighbor_searcher.range_search(query_value, self.radius)

        for n_id in neighbors:
            yield self.ids[self.ids.index(n_id)]

    def delete(self, id_):
        self.neighbor_searcher.remove_ids(np.array([id_], dtype=np.int64))

        position = self.ids.index(id_)
        del self.ids[position]
        self.values = np.delete(self.values, position, axis=0)

    def remake_index(self):
        if len(self.ids) < BRUTE_FORCE_CUTOFF == 0:
            new_index = faiss.IndexFlatIP(self.num_dims)
        else:
            num_clusters = 100
            quantizer = faiss.IndexFlatIP(self.num_dims)  # the quantizer for inner product
            new_index = faiss.IndexIVFFlat(quantizer, self.num_dims, num_clusters, faiss.METRIC_INNER_PRODUCT)
            new_index.nprobe = 20

            new_index.train(self.values)

        self.neighbor_searcher = IndexIDMap(new_index)
