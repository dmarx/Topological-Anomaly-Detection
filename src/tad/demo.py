import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from TADClassifier import tad_classify

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
res = tad_classify(df.values)

plot = False
if plot:
    df['anomaly']=0
    df.anomaly.ix[res['outliers']] = 1
    scatter_matrix(df.ix[:,:4], c=df.anomaly, s=(25 + 50*df.anomaly), alpha=.8)
    plt.show()

    print 'Anomalies:', res['outliers']
    g = res['g']
    X_pca = PCA().fit_transform(df)
    pos = dict((i,(X_pca[i,0], X_pca[i,1])) for i in range(X_pca.shape[0]))
    colors = []
    labels = {}
    for node in g.nodes():
        if node in res['outliers']:
            labels[node] = node
            colors.append('r')
        else:
            labels[node] = ''
            colors.append('b')
    nx.draw(g, pos=pos, node_color = colors)#, labels=labels)
    nx.draw_networkx_labels(g,pos,labels)
    plt.show()