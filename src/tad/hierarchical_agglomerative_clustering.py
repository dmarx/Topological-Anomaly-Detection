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
from scipy.sparse import csr_matrix
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
    
def calculate_anomaly_scores(outliers, dx, n):
    inliers = np.setdiff1d( range(n), outliers)
    #m = len(inliers)
    m = n - len(outliers)
    scores = []
    for outl in outliers:
        ix = condensed_index_from_row_col(n, np.repeat(outl, m), inliers)
        scores.append(dx[ix].min())
    return scores
    
def build_full_graph_from_condensed_distance_matrix(dx, n, early_stop=True, check_every_perc = .05):    
    if early_stop:
        k = dx.nonzero()[dx.argsort()]
        check = np.floor(n*check_every_perc)
    else:
        k = dx.nonzero()
    x,y = row_col_from_condensed_index(n,k)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for iter, ij in enumerate(zip(x,y)):
        i,j = ij
        g.add_edge(i,j)
        if early_stop:
            if iter % check == 0:
                if len(nx.connected_components) <= 2:
                    break
    return g

def hclust_outliers(X, percentile=.05, method='euclidean', divisive=True, maximal_clustering=False, score=True, distances = False, early_stop=True, track_stats=False, track_assignments=False):
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
    dx = pdist(X, method)        
    
    if divisive:
        k = np.floor(n*(n-1)*percentile) # minimum number of unnecessary distance calculations, 
                                         # assuming the only frivolous calculations are from the 
                                         # outliers to all observations that are not their nearest 
                                         # inlier observations
        #dx[dx.argsort()][-k:] = 0 # set unnecessary calculations to zero for a 10% (?) reduction in edges
        ix = dx.argsort()
        dx2 = dx[ix] # fancy indexing returns a copy :(
        maxd = dx2[-k]
        del dx2
        #print "Starting resource"
        #with dx[ix] as dx2: # will this work better than 'del'?
        #    maxd = dx2[-k]
        #    print "Done with resource"
        dx[dx>maxd] =0
        
        unq_dx = np.unique(dx)
        unq_dx.sort()
        unq_dx = unq_dx[::-1] # reverse. Not sure if this is best way... http://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        #g = nx.from_numpy_matrix(squareform(dx)) # would be nice if we could avoid building the squareform
        g = build_full_graph_from_condensed_distance_matrix(dx, n)
        # Not sure if this is better or worse...
        print "Hold on to your ass..."
        #g = nx.from_scipy_sparse_matrix(csr_matrix(squareform(dx)))
        print "Full graph built"
    else:
        unq_dx = np.unique(dx)
        unq_dx.sort()
        g = nx.Graph()
        g.add_nodes_from(range(n))
    
    k=0 # counter for the number of break points
    
    if track_assignments:
        assignments = np.empty((n,n)) #max number of breaks=n
        assignments[k,:] = range(n)
    else:
        if percentile is None:
            raise Exception("Either specify a target percentile, or enable assignment tracking.")
        assignments = None
    
    if early_stop and maximal_clustering:
        raise Exception("""
Maximizing outlier clustering requires iterating through all graph possible resolutions. 
This is mutually exclusive with early stopping. 
Set 
    early_stop=False
to achieve maximal clustering, or set
    maximal_clustering=False
to allow early stopping.""")
    
    count_n0_vs_r = {0:n} # {k:v}-> r:count of obs in V
    
    # Incrementally add edges to the graph to determine clustering
    r=0 # current graph resolution
    last_d = 0 # prior observed 'd'
    last_clust=n
    r_nclust = [] # list of tuples, giving the graph resolution and number of clusters at each split
    if track_stats:
        nclust = n # number of components on initialization
        r_nclust.append([r,nclust])
    outlier_objs = [] # {resolution, outliers, scores}
    if percentile:
        cutoff = np.floor(n*percentile) # target number of points we want to characterize as outliers
    outlier_count = 0
    for d in unq_dx :
        r = d
        for i,j in zip(*row_col_from_condensed_index(n, np.where(dx==d)[0])):
            if i==j:
                continue
            if divisive:
                g.remove_edge(i,j)
            else:
                g.add_edge(i,j)
        clust  = [c for c in nx.connected_components(g)]
        nclust = len(clust)
        if last_clust != nclust: # test that the number of clusters has changed as we update graph resolution
            #r_nclust.append([r, nclust])
            last_clust = nclust
            k+=1
            assign_k = assign_observations(clust)
            if track_assignments:
                assignments[k,:] = assign_k
            if track_stats:
                r_nclust.append([r, nclust])
            
            if percentile: 
                outlier_clusters, count_n0 = count_outliers(clust, cutoff)
                if track_stats:
                    count_n0_vs_r[k] = count_n0
                if outlier_clusters:
                    outliers=None
                    scores=None
                    if percentile:
                        last_assign = assign_k
                        outliers = [i for i,c in enumerate(last_assign) if c in outlier_clusters] 
                        last_outlier_count = outlier_count
                        outlier_count = len(outliers)
                    if divisive and outlier_count < last_outlier_count:
                            break
                    if score and not maximal_clustering:        
                        scores = calculate_anomaly_scores(outliers, dx, n)
                    outlier_objs.append({'resolution':r,'scores':scores,'outliers':outliers})
                    if early_stop and not divisive:
                        break
                        
    if track_assignments:
        assignments = assignments[:k+1, :] # Trim out unused rows
    
    if score:        
        scores = calculate_anomaly_scores(outliers, dx, n)
    else:
        scores=None
        
    maximal_assignment = outlier_objs[-1]
    if maximal_clustering and not divisive:
        outlier_objs.reverse()
        m=0 # number of outliers
        for ix, obj in enumerate(outlier_objs):
            last_m = m
            m = len(obj['outliers'])
            if last_m > m:
                ix = ix-1
                break
        maximal_assignment = outlier_objs[ix]
        if score:
            maximal_assignment['scores'] = calculate_anomaly_scores(maximal_assignment['outliers'] , dx, n)
        outlier_objs.reverse()
    
    return {'assignments':assignments, 'distances':dx, 'outliers':outlier_objs, 'graph':g, 'count_n0_vs_r':count_n0_vs_r, 'r_nclust':r_nclust, 'maximal_assignment':maximal_assignment}
    



if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from pandas.tools.plotting import scatter_matrix
    from sklearn.decomposition import PCA

    iris = datasets.load_iris()
    X = iris.data
    df = pd.DataFrame(X)
    #res = hclust_outliers(X)
    res = hclust_outliers(X, divisive=False, maximal_clustering=True, early_stop=False)

    #print res['scores']
    print res['maximal_assignment']['outliers']

    df['anomaly']=0
    df.anomaly.ix[res['maximal_assignment']['outliers']] = 1
    scatter_matrix(df.ix[:,:4], c=df.anomaly, s=(25 + 50*df.anomaly), alpha=.8)
    plt.show()

    print 'Anomalies:', res['maximal_assignment']['outliers']
    g = res['graph']
    X_pca = PCA().fit_transform(df)
    pos = dict((i,(X_pca[i,0], X_pca[i,1])) for i in range(X_pca.shape[0]))
    colors = []
    for obs in range(X.shape[0]):
        if obs in res['maximal_assignment']['outliers']:
            colors.append('r')
        else:
            colors.append('b')
    labels = {}
    for node in g.nodes():
        if node in res['maximal_assignment']['outliers']:
            labels[node] = node
        else:
            labels[node] = ''
    nx.draw(g, pos=pos, node_color = colors, labels=labels)
    plt.show()

    if 1==0:
        # generate a plot for k vs. count(n0) to demonstrate that
        # when we grow the AGNES tree, count(n0) is non-increasing.
        # But... is it? Maybe it isn't always,, just here
        stats = pd.Series(test['count_n0_vs_r'])
        stats.plot()
        plt.show()
        
        # 173.266 sec