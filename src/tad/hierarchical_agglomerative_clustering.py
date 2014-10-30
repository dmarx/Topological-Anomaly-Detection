"""
Agglomerative Hierarchical Clustering for Anomaly Detection, an 
extension/simplification of Topological Anomaly Detection

NB: This is not unheard of. See: http://www.dcc.fc.up.pt/~ltorgo/Papers/ODCM.pdf
--> This paper does not implement the unsupervised parameter estimation technique I implement below.
    ... Is it possible I've actually stumbled on something new?

"""
from collections import defaultdict
import copy
from itertools import combinations
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd

# I'm building this up from scratch, but I could probably just use 
#   scipy.cluster.hierarchy.linkage
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

# stripped from TADClassifier.py
def construct_graph(edges, n):
    """
    Constructs a networkx.Graph from a condensed distance matrix (as an
    adjacency matrix).
    """
    g = nx.Graph()
    for z, ij in enumerate(combinations(range(n),2)):
        d = edges[z]
        if d:
            i,j = ij
            g.add_edge(i,j, weight=d)
    return g

def assign_observations(clusters):
    """
    Converts a list of lists (of the kind returned by 
    networkx.connected_components) into a flat vector whose indices correspond
    to the unique nested values (nodes) in the 'clusters' input, and whose values 
    correspond to a component id (the index of the "cluster's" list in the 
    input list after sorting).
    """
    clusters.sort()
    assignments = []
    for c, clust in enumerate(clusters):
        for obs in clust:
            assignments.append([obs, c])
    assignments = np.array(assignments)
    ix = assignments[:,0].argsort()
    return assignments[ix, 1]
    
def count_outliers(clusters, cutoff=0):
    """
    Counts the number of observations that would get flagged as outliers given
    clustering assignments of observations.
    
    clusters: A list of lists of the kind produced by 
        networkx.connected_components, where observations in the same cluster 
        are unioned into the same list. 
    """
    # Basically just wrapping this in a function to clean up the hclust_outliers code
    # which was getting messy with nested conditionals and loops.
    outlier_clusters = None
    clust_size = defaultdict(list)
    
    for i, c in enumerate(clusters):
        clust_size[len(c)].append(i)
        
    unq_clust_size = list(set(clust_size.keys()))
    unq_clust_size.sort()            
    count_outlier_obs = 0
    this_outlier_clusters = []
    
    retval=0
    count_n0 = 0
    for i, s in enumerate(unq_clust_size):
        this_outlier_clusters.extend(clust_size[s])
        nobs = len(clust_size[s])*s
        if i==0:
            count_n0 = nobs
        if count_outlier_obs + nobs > cutoff:
            break
        count_outlier_obs += nobs
        outlier_clusters = copy.deepcopy(this_outlier_clusters)
    
    if count_outlier_obs:            
        retval = outlier_clusters
    
    return outlier_clusters, count_n0
    
# At present, this implementation is actually really stupid slow.
# .560, somehow better than the other version I've got
def calculate_anomaly_scores_SLOW(outliers, adj, n):
    """
    The "anomalous-ness" of an anomaly is the distance between that 
    observation and the nearest background component, i.e. the distance to the
    nearest non-anomaly observation. Scores are returned as a pandas.Series
    """
    scores = {}
    for a in outliers:
        scores[a] = 0
        for z, ij in enumerate(combinations(range(n),2)):
            i,j = ij
            if (i == a or j == a) and (
                i not in outliers or
                j not in outliers):
                d = adj[z]
                if scores[a]:
                    scores[a] = np.min([scores[a], d])
                else:
                    scores[a] = d
    return pd.Series(scores)
    
# Faster anomaly scoring

import numpy as np
from itertools import combinations, chain
from scipy.misc import comb

def comb_index(n, k):
    """
    Via http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    """
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)

    #1.381 seconds
def calculate_anomaly_scores_SLOW2(outliers, adj, n):
    """
    The "anomalous-ness" of an anomaly is the distance between that 
    observation and the nearest background component, i.e. the distance to the
    nearest non-anomaly observation. Scores are returned as a pandas.Series
    """
    scores = {}
    for z, ij in enumerate(combinations(range(n),2)):
        i,j = ij
        i_outl = False
        if i in outliers:
            a = i
            i_outl = True
        if j in outliers:
            if i_outl:
                continue
            a = j
        d = adj[z]
        if scores.has_key(a):
            scores[a] = np.min([scores[a], d])
        else:
            scores[a] = d
    return pd.Series(scores)
    
### The bottle neck here is the np.min call ###
# Alternate scoring method:
#   1. convert the distance matrix into an array of [i,j,d] rows.
#   2. drop all records where d < graph_resolution
#   3. sort on d
#   4. iterate through records, setting the score for each outlier to the first value
#      in the array where that outlier is compared to a non-outlier


### Alternative 2 ###
# Vectorize that shit!
# 1. squareform dx into a matrix
# 2. Dump columns that correspond to outliers
# 3. select out outlier rows
# 4. calculate the min along each row.
# 5. Profit.
# .010 seconds
#def calculate_anomaly_scores_VECTORIZED(outliers, adj, n):
def calculate_anomaly_scores(outliers, adj, n):
    mat = squareform(adj)
    #inliers = adj.index.difference(outliers)
    m = mat.shape[0]
    inliers = np.setdiff1d( range(m), outliers)
    s1 = mat[inliers,:]
    return s1[:,outliers].min(axis=0) # axis: 0=columns, 1=rows ... This seems backwards
    
def hclust_outliers(X, percentile=.05, method='euclidean', track_stats=True, track_assignments=False, score=True):
    """
    Agglomerative hierarchical clustering for outlier analysis. Constructs the
    the hierarchy incrementally. At each break, tests to see how many 
    observations would be labeled as outliers if we stopped at that break. If
    the number is below the threshold, the algorithm can optionally stop before
    constructing the full hierarchical tree.
    
    Inputs:
        X: Input data
        percentile: Upper bound on percent of observations to be flagged as outliers. 
            if percentile is None, returns the full hierarchy and doesn't flag outliers.
        method: method for calculating distances (passed to pdist). In 
            scipy.cluster.hierarchy.linkage language, this function currently only supports
            "single" linkage (Nearest Point Algorithm) for agglomerating clusters.
    """
        
    # initialize an unconnected graph
    n=X.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    
    dx = pdist(X, method)    
    ix = np.array([ij for ij in combinations(range(n),2)])
    d_ij = np.hstack((dx[:,None], ix)) # append edgelist
    d_ij = d_ij[dx.argsort(),:] # order by distance
    
    k=0 # counter for the number of break points
    
    if track_assignments:
        assignments = np.empty((n,n)) #max number of breaks=n
        assignments[k,:] = range(n)
    else:
        if percentile is None:
            raise Exception("Either specify a target percentile, or enable assignment tracking.")
        assignments = None
    
    
    count_n0_vs_r = {0:n} # {k:v}-> r:count of obs in V
    
    # Incrementally add edges to the graph to determine clustering
    r=0 # current graph resolution
    last_d = 0 # prior observed 'd'
    nclust = n # number of components on initialization
    r_nclust = [] # list of tuples, giving the graph resolution and number of clusters at each split
    r_nclust.append([r,nclust])
    if percentile:
        cutoff = np.floor(n*percentile) # target number of points we want to characterize as outliers
    for dij in d_ij:
        d,i,j = dij
        if d != last_d: # test that we have gone through all observations for a particular graph resolution
            r = d
            clust  = nx.connected_components(g)
            nclust = len(clust)
            if r_nclust[-1][1] > nclust: # test that the number of clusters has changed as we update graph resolution
                r_nclust.append([r, nclust])
                k+=1
                assign_k = assign_observations(clust)
                if track_assignments:
                    assignments[k,:] = assign_k
                
                if percentile: 
                    outlier_clusters, count_n0 = count_outliers(clust, cutoff)
                    if track_stats:
                        count_n0_vs_r[k] = count_n0
                    if outlier_clusters:
                        break      
        g.add_edge(i,j)
        last_d = d
    if track_assignments:
        assignments = assignments[:k+1, :] # Trim out unused rows
    
    # flag outliers
    outliers=None
    if percentile:
        #last_assign = assignments[k,:]
        last_assign = assign_k
        # There's probably a more vectorized way to do this
        outliers = [i for i,c in enumerate(last_assign) if c in outlier_clusters] 
        
    if score:
        scores = calculate_anomaly_scores(outliers, dx, n)
    else:
        scores=None
    
    return {'assignments':assignments, 'distances':dx, 'outliers':outliers, 'graph':g, 'count_n0_vs_r':count_n0_vs_r, 'scores':scores}
    
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    test = hclust_outliers(X)
    
    if 1==0:
        # generate a plot for k vs. count(n0) to demonstrate that
        # when we grow the AGNES tree, count(n0) is non-increasing.
        # But... is it? Maybe it isn't always,, just here
        stats = pd.Series(test['count_n0_vs_r'])
        stats.plot()
        plt.show()
        
        # 173.266 sec