"""
Agglomerative Hierarchical Clustering for Anomaly Detection, an 
extension/simplification of Topological Anomaly Detection

NB: This is not unheard of. See: http://www.dcc.fc.up.pt/~ltorgo/Papers/ODCM.pdf
--> This paper does not implement the unsupervised parameter estimation technique I implement below.
    ... Is it possible I've actually stumbled on something new?

"""
from collections import defaultdict
import copy
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd



# I'm building this up from scratch, but I could probably just use 
#   scipy.cluster.hierarchy.linkage
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

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
    
def row_col_from_condensed_index(n,ix):
    b = 1 -2*n
    x = np.floor((-b - np.sqrt(b**2 - 8*ix))/2)
    y = ix + x*(b + x + 2)/2 + 1
    return (x,y)  

def condensed_index_from_row_col(n,i,j):
    # i is always the smaller index... or doesn't it make a difference? I think it doesn't make a difference.
    ix1 = i>j
    #ix2 = i>j
    ix2 = np.logical_not(ix1)
    i1, j1 = i[ix1], j[ix1]
    i2, j2 = j[ix2], i[ix2]
    i = np.hstack([i1,i2])
    j = np.hstack([j1,j2])
    return n*j - j*(j+1)/2 + i - 1 - j
    
def score_outliers__index_method(outliers, dx, n):
    inliers = np.setdiff1d( range(n), outliers)
    #m = len(inliers)
    m = n - len(outliers)
    scores = []
    for outl in outliers:
        ix = condensed_index_from_row_col(n, np.repeat(outl, m), inliers)
        scores.append(dx[ix].min())
    return scores

def hclust_outliers(X, percentile=.05, method='euclidean', track_stats=True, track_assignments=False, score=True, distances = False):
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
        safe_scoreing: Scoring can be performed quickly by performing a vectorized operation on 
                        the squareform distance matrix. If memory allocation is potentially 
                        going to be an issue, construction of the squareform matrix can be avoided
                        entirely, but the scoring operation will be much, much slower. Way slower.
                        Like, maybe-not-even-worth-it slower.
    """
    
    # initialize an unconnected graph
    n=X.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    
    dx = pdist(X, method)    
    
    unq_dx = np.unique(dx)
    unq_dx.sort()
    
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
    for d in unq_dx :
        r = d
        #print d
        for i,j in zip(*row_col_from_condensed_index(n, np.where(dx==d)[0])):
        #print block
        #for i,j in zip(*block): # there may be a more efficient way of doing this
            if i==j:
                continue
            g.add_edge(i,j)
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
        scores = score_outliers__index_method(outliers, dx, n)
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