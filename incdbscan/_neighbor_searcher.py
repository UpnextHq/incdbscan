import faiss
import numpy as np
from opentelemetry import trace

BRUTE_FORCE_CUTOFF = 1000

TOMBSTONE_ID = -1

tracer = trace.get_tracer(__name__)

class NeighborSearcher:
    def __init__(self, radius, num_dims):
        self.radius = radius

        self.num_dims = num_dims

        self.values = np.empty((0, num_dims), dtype=np.float32)
        self.ids = []

        self.neighbor_searcher = None
        self.remake_index()

    def insert(self, new_value, new_id):
        if len(self.values) % BRUTE_FORCE_CUTOFF == 0:
            with tracer.start_as_current_span('incdbscan_insert_neighborhood_searcher_insert_remake_index'):
                self.remake_index()

        self.ids.append(new_id)

        with tracer.start_as_current_span('incdbscan_insert_neighborhood_searcher_insert_add_to_values'):
            new_value = np.array(new_value, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(new_value)
            self._insert_into_array(new_value)

        with tracer.start_as_current_span('incdbscan_insert_neighborhood_searcher_insert_add'):
            self.neighbor_searcher.add(new_value)

    def _insert_into_array(self, new_value):
        self.values = np.vstack((self.values, new_value))

    def query_neighbors(self, query_value):
        query_value = np.array([query_value], dtype=np.float32)
        faiss.normalize_L2(query_value)

        _, _, neighbors = self.neighbor_searcher.range_search(query_value, 1 - self.radius)
        for n_idx in neighbors:
            if self.ids[n_idx] == TOMBSTONE_ID:
                continue

            yield self.ids[n_idx]

    def delete(self, id_):
        pos = self.ids.index(id_)
        self.ids[pos] = TOMBSTONE_ID

    def remake_index(self):
        if len(self.ids) < BRUTE_FORCE_CUTOFF:
            new_index = faiss.IndexFlatIP(self.num_dims)
        else:
            num_centroids = int(len(self.ids) / 39)
            quantizer = faiss.IndexFlatIP(self.num_dims)  # the quantizer for inner product
            new_index = faiss.IndexIVFFlat(quantizer, self.num_dims, num_centroids, faiss.METRIC_INNER_PRODUCT)
            new_index.nprobe = 20

            new_index.train(self.values)

        self.neighbor_searcher = new_index

        orig_ids = self.ids
        orig_values = self.values

        self.values = np.empty((0, self.num_dims), dtype=np.float32)  # Ensure values is reset with the correct shape
        self.ids = []

        for orig_id, orig_val in zip(orig_ids, orig_values):
            if orig_id == TOMBSTONE_ID:
                continue

            self._insert_into_array(orig_val)
            self.ids.append(orig_id)

        if self.values.size > 0:
            self.neighbor_searcher.add(self.values)
