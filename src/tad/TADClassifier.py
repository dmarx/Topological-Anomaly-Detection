"""
Topological Anomaly Detection per Gartley and Basener (2009):

    http://www.cis.rit.edu/~mxgpci/pubs/gartley-7334-1.pdf
    
Author: David Marx
Date: 8/3/2014
Version: 0.1
License: BSD-3
"""

from __future__ import division
import numpy as np
from scipy.spatial.distance import pdist
from itertools import combinations
import networkx as nx
import pandas as pd

def trim_adjacency_matrix(adj, r=None, rq=.1):
    """
    Given a condensed distance matrix (i.e. of the kind outputted by pdist), 
    returns a copy of the distance matrix where all entries greater than 'r'
    are set to zero. If 'r' is not provided, evaluates the 'rq' quantile of the
    input distances and uses that as a heuristic for 'r'. Default behavior is
    to use the 10th percentile of distances as 'r'.
    """
    if r is None:
        r = np.percentile(adj, 100*rq)
    print "r:", r
    adj2 = adj.copy()
    adj2[adj>r] = 0 
    return adj2, r

def construct_graph(edges, n):
    """
    Constructs a networkx.Graph from a condensed distance matrix (as an
    adjacency matrix).
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for z, ij in enumerate(combinations(range(n),2)):
        d = edges[z]
        if d:
            i,j = ij
            g.add_edge(i,j, weight=d) # I don't think we need the weight for anything...
    return g
    

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
    
def construct_constrained_graph(adj, r, n):
    """
    given an adjacency matrix adj in the form of a condensed distance matrix
    (of the kind returned by pdist) for n observations, returns the similarity
    graph for all distances less than or equal to r.
    """
    ij = row_col_from_condensed_index(n, np.where(adj<=r)[0])
    g = nx.from_edgelist(zip(*ij))
    g.add_nodes_from(range(n))
    return g
    
def flag_anomalies(g, min_pts_bgnd, node_colors={'anomalies':'r', 'background':'b'}):
    """
    Given an input graph, extracts the connected components of the graph and 
    flags all observations in components with fewer than a threshold number of 
    nodes (min_pts_bgnd) as anomalies. A "class" attribute is added to the 
    graph to indicate if an observation is background or anomalous, and a dict
    is also returned whose keys are the class labels "anomalies" and 
    "background" and whose values are lists of observation indices. 
    
    Inputs:
        g: a graph
        min_pts_bgnd: min number of points for a component to be considered 
                      background
     
    Returns:
        res: a dict keyed to two lists, one for anomalies and one for 
             background.
        g:   The input graph with a "class" attribute added to nodes to indicate
             whether or not they are anomalous
     """
    print "min_pts_bgnd:", min_pts_bgnd
    outliers = []
    for c in nx.connected_components(g):
        if len(c) < min_pts_bgnd:
            #outliers.extend(c)
            outliers.append(c) # maintain clustering
    return outliers

def calculate_anomaly_scores(outliers, adj, n):
    outliers_flat = [outl for cluster in outliers for outl in cluster]
    inliers = np.setdiff1d( range(n), outliers_flat)    
    #m = n - len(outliers) # is this faster than len(inliers) ?
    m = len(inliers)
    scores = {}
    for outl in outliers_flat:
            ix = condensed_index_from_row_col(n, np.repeat(outl, m), inliers)        
            #scores.append(adj[ix.astype(int)].min())
            scores[outl] = adj[ix.astype(int)].min()
    return scores

def tad_classify(X, method='euclidean', r=None, rq=.1, p=.1, distances=None):
    """
    Performs TAD classification over the input data X
    
    Inputs:
        X: A 2d numpy array or pandas DataFrame with observations on the rows 
            and features on the columns.
        method: The method to be used by scipy.spatial.distance.pdist to 
            calculate inter-observation distances. See the pdist documentation
            for supported values. Default's to 'euclidean'.
        r: The "graph resolution." The paper gives no advice here, so I'm just
            going to use the 10% percentile of observed distances
        rq: The a percentile to be used as a heuristic to determine 'r' if 'r' 
            is not specified. Defaults to .1
        p: Min percentage of points necessary for a component to be considered 
            "background" (default=10%).
        distances: An input (condensed) distance matrix. If none is provided, 
            a distance matrix will be calculated in accordance with the 
            provided 'method' parameter.
    
    Returns a dict with the following keys:
        classed: A dict keyed to two lists of observations: "anomalies" and 
            "background".
        g: The graph of the data used by the algorithm, with observations
            flagged as "anomaly" or "background" in the node "class" attribute.
        scores: A pd.Series giving the "anomaly score" for each anomalous
            observation, calculated as the distance from the observation to the
            nearest "background" observation.
        r: The graph resolution. If this was not provided as an input, it is 
           calculated from the 'rq' parameter as described above.
        min_pts_bgnd: The minimum number of points required for a connected 
            component to be considered as "background." Calculated as n*p
            where 'n' is the total number of observations.
        distances: A condensed distance matrix giving all inter-observation
            distances (unfiltered by trim_adjacency_matrix). 
    """
    # Maintain original matrix for calculating anomaly score
    if not distances:
        adj = pdist(X, method)
    if r is None:
        r = np.percentile(adj, 100*rq)
    n = X.shape[0]
    g = construct_constrained_graph(adj, r, n)
    outliers =  flag_anomalies(g, n*p)
    print "outliers"
    print outliers
    scores = calculate_anomaly_scores(outliers, adj, n)
    return {'outliers':outliers, 'g':g, 'scores':scores, 'r':r, 'min_pts_bgnd':n*p, 'distances':adj}