import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from TADClassifier import tad_classify
import time

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)

start = time.time()
res = tad_classify(df.values)
print "Elapsed: {t}".format(t=time.time()-start)

print res['scores']

plot = False
if plot:
    df['anomaly']=0
    df.anomaly.ix[res['outliers']] = 1
    scatter_matrix(df.ix[:,:4], c=df.anomaly, s=(25 + 50*df.anomaly), alpha=.8)
    plt.show()

    print 'Anomalies:', res['classed']['anomalies']
    g = res['g']
    X_pca = PCA().fit_transform(df)
    pos = dict((i,(X_pca[i,0], X_pca[i,1])) for i in range(X_pca.shape[0]))
    colors = [node[1]['color'] for node in g.nodes(data=True)]
    labels = {}
    print len(g.nodes())
    for node in g.nodes():
        if node in res['classed']['anomalies']:
            labels[node] = node
        else:
            labels[node] = ''
    nx.draw(g, pos=pos, node_color = colors, labels=labels)
    nx.draw_networkx_labels(g, pos=pos, node_color = colors, labels=labels)
    plt.show()

