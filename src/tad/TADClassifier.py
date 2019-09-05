"""
Topological Anomaly Detection per Gartley and Basener (2009):

    http://www.cis.rit.edu/~mxgpci/pubs/gartley-7334-1.pdf
    
Author: David Marx
Date: 9/5/2019
Version: 0.2
License: BSD-3
"""

from sklearn.neighbors import BallTree, radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin, ClusterMixin


class TADClassifier(BaseEstimator, OutlierMixin, ClusterMixin):
    # Deprecated the rq parameter in favor of not calculating all pairwise distances.
    # Constructing a BallTree instead, should facilitate exploring a range of radius values
        
    def fit(self, X, *args, **kwargs):
        self.X_ = X
        self.X_tree_ = BallTree(X, *args, **kwargs)
        return self
    
    def predict(self, X=None, radius=None, q=.1, return_scores=False, *args, **kwargs):
        if X is None or X is self.X_:
            X = self.X_tree_
        else:
            self.fit(X)
            X = self.X_tree_

        # These 4 lines comprise the entire algorithm. Everything else is aesthetic.
        self.g_ = radius_neighbors_graph(X, radius=radius, *args, **kwargs)
        _, cc = connected_components(self.g_)
        perc_obs_cc = np.bincount(cc) / len(cc)
        anomalous_components = np.where(perc_obs_cc<q)[0]

        # There's probably a cleaner way to do this via numpy/scipy.
        self.anomalous_clusters_ = [np.where(cc == comp_idx)[0] for comp_idx in anomalous_components]
        
        # Use the .labels_ attribute to group outliers. Per sklearn convention, inliers are assigned the label "1"
        inliers = np.logical_not(np.isin(cc, anomalous_components))
        self.labels_ = -cc
        self.labels_[inliers] = 1
        
        return self.labels_
    
    def fit_predict(self, X, y=None, *args, **kwargs):
        return self.fit(X).predict(*args, **kwargs)
    
    @property
    def outliers_(self):
        return np.where(self.labels_ < 1)[0]
    
    @property
    def inliers_(self):
        return np.where(self.labels_ > 0)[0]
    
    def score(self, X=None, y=None):
        """X and y ignored."""
        scores = [self._outlier_scores(cluster) for cluster in self.anomalous_clusters_]
        scores_flat = self._flatten_scores(scores)
        sp_scores = self._dict_to_vect(scores_flat, len(self.X_))
        return sp_scores
    
    def _outlier_scores(self, cluster):
        """
        Calculates the distance from each outlier to its nearest inlier.
        Results are returned in the same cluster groups as the input.
        """
        outliers = self.X_.values[cluster,:]
        d,i = self.X_tree_.query(outliers,k=len(outliers)+1)
        
        # Need to handle cases that have multiple inlier neighbors
        d[np.isin(i, cluster)] = d.max()*2
        scores = d.min(axis=1)
        return scores
    
    def _flatten_scores(self, scores_in):
        """
        Given a collection of outlier scores grouped into their respective clusters, 
        pairs each score with the index of its associated observation. 
        Returns a dict of {observationIndex:outlierScore}.
        """
        scores = {}
        for idx, score in zip(self.anomalous_clusters_, scores_in):
            scores.update(dict(zip(idx,score)))
        return scores
    
    def _dict_to_vect(self, d, n):
        """
        Given a dict of integer keys and float values, returns an N x 1 sparse vector (COO)
        whose indexes are the dict keys and whose values match the dict values.
        """
        row = list(d.keys())
        col = np.zeros(len(row))
        data = list(d.values())
        sp_scores = coo_matrix((data, (row, col)), shape=(n, 1))
        return sp_scores
