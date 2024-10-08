import warnings
from typing import Iterable, List

import numpy as np

from ._deleter import Deleter
from ._inserter import Inserter
from ._object import ObjectId
from ._objects import Objects
from ._utils import input_check


class IncrementalDBSCAN:
    """The incremental version of DBSCAN, a density-based clustering algorithm
    that handles outliers.

    After an initial clustering of an initial object set (i.e., set of data
    points), the object set can at any time be updated by increments of any
    size. An increment can be either the insertion or the deletion of objects.

    After each update, the result of the clustering is the same as if the
    updated object set (i.e., the initial object set modified by all
    increments) was clustered by DBSCAN. However, this result is reached by
    using information from the previous state of the clustering, and without
    the need of applying DBSCAN to the whole updated object set.

    Parameters
    ----------
    eps : float, optional (default=0.5)
        The radius of neighborhood calculation. An object is the neighbor of
        another if the distance between them is no more than eps.

    min_pts : int, optional (default=1)
        The minimum number of neighbors that an object needs to have to be a
        core object of a cluster.

    metric : string or callable, optional (default='minkowski')
        The distance metric to use to calculate distance between data objects.
        Accepts metrics that are accepted by scikit-learn's NearestNeighbors
        class, excluding 'precomputed'. The default is 'minkowski', which is
        equivalent to the Euclidean distance if p=2.

    p : float or int, optional (default=2)
        Parameter for Minkowski distance if metric='minkowski'.

    References
    ----------
    Ester et al. 1998. Incremental Clustering for Mining in a Data Warehousing
    Environment. In: Proceedings of the 24th International Conference on Very
    Large Data Bases (VLDB 1998).

    """

    def __init__(self, num_dims, eps=1, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.num_dims = num_dims

        self._objects = Objects(self.eps, self.num_dims)
        self._inserter = Inserter(self.eps, self.min_pts, self._objects)
        self._deleter = Deleter(self.eps, self.min_pts, self._objects)

    def insert(self, X, ids=None):
        """Insert objects into the object set, then update clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to be inserted into the object set.
        ids : array-like of shape (n_samples,), optional
            The custom IDs for the data objects. If not provided, IDs will
            be generated based on hashing.

        Returns
        -------
        self

        """
        X = input_check(X)
        if ids is not None and len(ids) != len(X):
            raise ValueError("The length of IDs must match the length of X")

        modifications = []
        for ix, value in enumerate(X):
            id_ = ids[ix] if ids is not None else None

            mods = self._inserter.insert(value, id_)
            modifications.append(mods)

        return modifications

    def delete(self, X):
        """Delete objects from object set, then update clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to be deleted from the object set.

        Returns
        -------
        self

        """
        X = input_check(X)

        for ix, value in enumerate(X):
            obj = self._objects.get_object(value)

            if obj:
                self._deleter.delete(obj)

            else:
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'Object at position {ix} was not deleted because '
                        'there is no such object in the object set.'
                    )
                )

        return self

    def delete_by_ids(self, ids):
        """Delete objects from the object set by their IDs, then update clustering.

        Parameters
        ----------
        ids : array-like
            The IDs of the data objects to be deleted from the object set.

        Returns
        -------
        self

        """
        if not isinstance(ids, (list, np.ndarray)):
            raise ValueError("IDs should be a list or numpy array")

        for id_ in ids:
            try:
                obj = self._objects.get_object_by_id(id_)
                self._deleter.delete(obj)
            except ValueError as e:
                warnings.warn(
                    IncrementalDBSCANWarning(str(e))
                )

        return self

    def delete_by_cluster_label(self, cluster_label):
        self._deleter.delete_cluster(cluster_label)

    def get_cluster_labels(self, X):
        """Get cluster labels of objects.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to get labels for.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
                 Cluster labels. Effective labels start from 0. -1 means the
                 object is noise. numpy.nan means the object was not in the
                 object set.

        """
        X = input_check(X)

        labels = np.zeros(len(X))

        for ix, value in enumerate(X):
            obj = self._objects.get_object(value)

            if obj:
                label = self._objects.get_label(obj)

            else:
                label = np.nan
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'No label was retrieved for object at position {ix} '
                        'because there is no such object in the object set.'
                    )
                )

            labels[ix] = label

        return labels

    def get_cluster_labels_by_ids(self, ids: List[ObjectId]):
        """Get cluster labels of objects.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to get labels for.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
                 Cluster labels. Effective labels start from 0. -1 means the
                 object is noise. numpy.nan means the object was not in the
                 object set.

        """
        labels = np.zeros(len(ids))

        for ix, value in enumerate(ids):
            obj = self._objects.get_object_by_id(value)

            if obj:
                label = self._objects.get_label(obj)

            else:
                label = np.nan
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'No label was retrieved for object at position {ix} '
                        'because there is no such object in the object set.'
                    )
                )

            labels[ix] = label

        return labels

    def get_object_ids_for_label(self, label: int) -> List[ObjectId]:
        res = self._objects.get_objects_for_label(label)
        return list(map(lambda o: o.id, res))

    # Returns IDs within the specified radius
    def query_neighbors(self, query_value, radius: float):
        return self._objects.neighbor_searcher.query_neighbors(query_value, radius)

class IncrementalDBSCANWarning(Warning):
    pass
