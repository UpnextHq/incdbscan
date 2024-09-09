import logging

import faiss
import numpy as np

BRUTE_FORCE_CUTOFF = 1000
REMAKE_INDEX_INTERVAL = 5000

TOMBSTONE_ID = -1

logger = logging.getLogger(__name__)

class NeighborSearcher:
    def __init__(self, radius, num_dims):
        self.radius = radius

        self.num_dims = num_dims

        self.values_batch_size = 1000
        self.values = np.empty((self.values_batch_size, num_dims), dtype=np.float32)
        self.values_count = 0
        self.ids = []

        self.neighbor_searcher = None
        self.remake_index()

    def insert(self, new_value, new_id):
        remake_multiple = BRUTE_FORCE_CUTOFF if len(self.values) < 15000 else REMAKE_INDEX_INTERVAL
        if self.values_count % remake_multiple == 0:
            self.remake_index()

        self.ids.append(new_id)

        new_value = np.array(new_value, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(new_value)
        self._insert_into_array(new_value)

        self.neighbor_searcher.add(new_value)

    def _insert_into_array(self, new_value):
        if self.values_count == self.values.shape[0]:
            self.values = np.vstack((self.values, np.empty((self.values_batch_size, self.num_dims), dtype=np.float32)))

        self.values[self.values_count] = new_value
        self.values_count += 1

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
        logger.info("Remaking neighbor index")

        if self.values_count < BRUTE_FORCE_CUTOFF:
            new_index = faiss.IndexFlatIP(self.num_dims)
        else:
            num_centroids = int(self.values_count / 39)
            quantizer = faiss.IndexFlatIP(self.num_dims)  # the quantizer for inner product
            new_index = faiss.IndexIVFFlat(quantizer, self.num_dims, num_centroids, faiss.METRIC_INNER_PRODUCT)
            new_index.nprobe = 20

            new_index.train(self.values[:self.values_count])

        self.neighbor_searcher = new_index

        orig_ids = self.ids
        orig_values = self.values[:self.values_count]

        self.values = np.empty((self.values_batch_size, self.num_dims), dtype=np.float32)
        self.values_count = 0
        self.ids = []

        for orig_id, orig_val in zip(orig_ids, orig_values):
            if orig_id == TOMBSTONE_ID:
                continue

            self._insert_into_array(orig_val)
            self.ids.append(orig_id)

        logger.info("Adding values to existing index")
        if self.values_count > 0:
            self.neighbor_searcher.add(self.values[:self.values_count])
