# Hierarchical clustering without invoking networkx
# 
# Overarching goal: design intelligent classes tailored for agglomerative clustering.

import itertools

# Classes for building and merging clusters. 

class Clusters(object):
    """
    Parent class. Upon initializaiton, takes data as input, assigns each datum 
    (by index) to a unique cluster. Clusters can be merged efficiently. datamap 
    attribute takes and index as input and returns a ClusterPointer to the 
    appropriate cluster, so Clusters().datamap[ix].cluster gives the actual
    cluster object that datum is currently assigned to. Merging clusters updates
    the cluster object assigned to a given pointer. The .clusters 
    """
    def __init__(self, data):
        self.id_sequence = itertools.count()
        self.data = data
        self.datamap = {}
        self.clusters = {}
        for ix in range(data.shape[0]):
            pointer = ClusterPointer([ix],self)
            self.datamap[ix] = pointer
            self.clusters[ix] = pointer.cluster
    def merge_clusters(self,a,b):
        """
        Takes two clusters as input. If they aren't pointing to the 
        same cluster, merges the two clusters and updates the one of the 
        pointers so it's no longer pointing to a deprecated cluster.
        """
        if a.cluster.id == b.cluster.id: # Would it be faster to just do a.cluster==b.cluster or something like that?
            return
        a.cluster.merge(b.cluster)
        self.clusters.pop(b.cluster.id)
        b.cluster = a.cluster
    def merge_clusters_by_id(self, ix1, ix2):
        """
        Takes to data indices as input, merges the clusters they are currently
        assigned to (if they are not currently assigned to the same cluster)
        """
        a = self.datamap[ix1]
        b = self.datamap[ix2]
        self.merge_clusters(a,b)
    @property
    def size(self):
        """
        Returns a list whose items are the sizes of each individual cluster        
        """
        return [len(c) for c in self.clusters.itervalues()]
     
class ClusterPointer(object):
    """
    Points to a cluster object
    """
    def __init__(self, values, parent):
        self.cluster = Cluster(values, parent)
        
class Cluster(object):
    """
    Implement a single "cluster" which can be merged with other clusters    
    """
    def __init__(self, values, parent):
        """
        Parent should be an instance of class Clusters.
        Values should be a list of indexes to be assigned to the cluster. 
        Normally, values will be a list of length one.
        """
        self.parent = parent
        self.id = parent.id_sequence.next()
        self.values = values
        self._len = len(self.values)
    def merge(self, cluster):
        # if self.id > cluster.id: self.id = cluster.id ## Not really necessary
        self.values.extend(cluster.values)
        self._len = self._len + len(cluster)
    def __len__(self):
        return self._len # only update this when values list is grown
        