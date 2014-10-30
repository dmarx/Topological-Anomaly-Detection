# Hierarchical clustering without invoking networkx
# 
# Overarching goal: design intelligent classes tailored for agglomerative clustering.

import itertools
from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import combinations, chain
from scipy.misc import comb

# Classes for building and merging clusters. 

class Clusters(object):
    """
    Parent class. Upon initializaiton, takes data as input, assigns each datum 
    (by index) to a unique cluster. Clusters can be merged efficiently. Merging clusters updates
    the cluster object assigned to a given pointer. The .clusters attribute is 
    a list of active cluster ids pointing to cluster objects such that 
    len(Clusters.clusters) = len(Clusters.size). The datamap attribute maps data indices to 
    the clusters to which they are assigned.
    """
    def __init__(self, n):
        self.n = n # number of terminal leaves
        self.datamap = {}
        self.clusters = {}
        for ix in range(n):
            cluster = Cluster([ix], ix)
            self.datamap[ix] = cluster
            self.clusters[ix] = cluster
    def merge_clusters(self,a,b):
        """
        Takes two cluster objects as input. If they aren't pointing to the 
        same cluster, merges the two clusters and updates the one of the 
        pointers so it's no longer pointing to a deprecated cluster.
        
        Returns a boolean to indicate whether or not a merge action was taken or not
        """
        if a.id == b.id: # Would it be faster to just do a==b or something like that?
            return False
        a.merge(b)
        for ix in b.values:
            self.datamap[ix]=a
        self.clusters.pop(b.id)
        return True
    def merge_clusters_by_id(self, ix1, ix2):
        """
        Takes to data indices as input, merges the clusters they are currently
        assigned to (if they are not currently assigned to the same cluster)
        """
        a = self.datamap[ix1]
        b = self.datamap[ix2]
        return self.merge_clusters(a,b)
    @property
    def size(self):
        """
        Returns a list whose items are the sizes of each individual cluster        
        """
        return [c.size for c in self.clusters.itervalues()]
        
class Cluster(object):
    """
    Implement a single "cluster" which can be merged with other clusters    
    """
    __slots__ = ['values','id','size']
    def __init__(self, values, id):
        """
        Values should be a list of indexes to be assigned to the cluster. 
        Normally, values will be a list of length one (i.e. containing 'id').
        """
        self.id = id
        self.values = values
        self.size = len(self.values)
    def merge(self, cluster):
        # if self.id > cluster.id: self.id = cluster.id ## Not really necessary
        self.values.extend(cluster.values)
        self.size = self.size + cluster.size

        
def flag_outliers(clusters, perc):    
    n = clusters.n
    threshhold = np.floor(n*perc)
    clust_size = Counter(clusters.size)
    outliers_count = 0
    outliers = False
    for size, count in clust_size.iteritems():
        n_obs = size*count
        outliers_count += n_obs
        if outliers_count <= threshhold:
            outliers = True
        else: 
            break
    outlier_observations = []
    if outliers:
        for id, clust in clusters.clusters.iteritems():
            if clust.size < size:
                outlier_observations.extend(clust.values)
    return outlier_observations
        
def score_outliers(outliers, dx):
    mat = squareform(dx)
    m = mat.shape[0]
    inliers = np.setdiff1d( range(m), outliers)
    s1 = mat[inliers,:]
    return s1[:,outliers].min(axis=0) # axis: 0=columns, 1=rows ... This seems backwards
        
def comb_index(n, k):
    """
    via http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    """
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)
        
def combs_nd(a, r, axis=0):
    a = np.asarray(a)
    if axis < 0:
        axis += a.ndim
    indices = np.arange(a.shape[axis])
    dt = np.dtype([('', np.intp)]*r)
    indices = np.fromiter(combinations(indices, r), dt)
    indices = indices.view(np.intp).reshape(-1, r)
    return np.take(a, indices, axis=axis)
        
def hclust_tad(X, method='euclidean', perc=.05, score=True):
    """
    Performs hierarchical clustering on the input data to identify 
    outlier observations.
    """
    n = X.shape[0]
    dx = pdist(X, method)    
    #ix = np.array([ij for ij in itertools.combinations(range(n),2)])
    #ix = cartesian((np.arange(n), np.arange(n))) # incorrect dimensions, doesn't respect ordering
    ix = comb_index(n,2)
    #ix = combs_nd(np.arange(n),2)
    d_ij = np.hstack((dx[:,None], ix)) # append edgelist
    d_ij = d_ij[dx.argsort(),:] # order by distance
        
    clusters = Clusters(n)
    
    last_d = 0
    r = 0 # graph resolution
    merged = False
    for dij in d_ij:  
        d,i,j = dij 
        if last_d != d:
            r = d
            if merged: # test if number of clusters has changed since last modification to graph resolution
                merged = False # reset for new graph resolution
                outliers = flag_outliers(clusters, perc)
                if outliers:
                    break
                    
        # Add an edge to the graph (i.e. merge clusters as necessary)
        merged_this_iter = clusters.merge_clusters_by_id(i,j)
        merged = merged or merged_this_iter    
    
    scores = None
    if score:
        scores = score_outliers(outliers, dx)
    
    return {'outliers':outliers, 'scores':scores, 'clusters':clusters, 'graph_resolution':r}
    
if __name__ == '__main__':
    from sklearn import datasets
    import time
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    X = datasets.load_iris().data
    start = time.time()
    test = hclust_tad(X)
    print "Elapsed: {t}".format(t=time.time()-start)
    
    
    if 1==0:
        X_pca = PCA().fit_transform(X)
        colors = []
        for obs in range(X.shape[0]):
            if obs in test['outliers']:
                colors.append('r')
            else:
                colors.append('b')
        plt.scatter(X_pca[:,0],X_pca[:,1], color=colors)
        plt.show()