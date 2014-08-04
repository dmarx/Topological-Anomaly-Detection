import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn import datasets
from sklearn.decomposition import PCA
from TADClassifier import tad_classify

iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
res = tad_classify(df)

df['anomaly']=0
df.anomaly.ix[res['classed']['anomalies']] = 1
scatter_matrix(df.ix[:,:4], c=df.anomaly, s=(25 + 50*df.anomaly), alpha=.8)
plt.show()

print 'Anomalies:', res['classed']['anomalies']
g = res['g']
X_pca = PCA().fit_transform(df)
pos = dict((i,(X_pca[i,0], X_pca[i,1])) for i in range(X_pca.shape[0]))
colors = [node[1]['color'] for node in g.nodes(data=True)]
labels = {}
for node in g.nodes():
    if node in res['classed']['anomalies']:
        labels[node] = node
    else:
        labels[node] = ''
nx.draw(g, pos=pos, node_color = colors, labels=labels)
plt.show()