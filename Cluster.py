import os
import csv
from time import time
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from sklearn import (manifold, decomposition, ensemble, random_projection)
from scipy.cluster.vq import kmeans,vq


def getColumnFromFile(file_path,columnIndx):
    with open(file_path,"r") as input_file:
        reader = csv.reader(input_file,delimiter="\t")
        data = list(reader)
        column = [row[columnIndx] for row in data]
        return column

#columns : ['Date', 'Day', 'View', 'Like', 'Dislike', 'Comments']
def getFeatureVectorsFromFiles(folder):
    nameToIndxMap = {}
    indx = 0
    featureVectors = []
    for fn in os.listdir(folder):
        if fn.endswith(".pattern"):
            #save file name to indx mapping
            nameToIndxMap[fn] = indx
            indx += 1
            # get View Column for current file/video as Feature Vector
            file_path = os.path.join(folder,fn)
            column = getColumnFromFile(file_path,2)
            if len(column) == 101:
                featureVector = [float(val) for val in column[1:]]
                featureVectors.append(featureVector)
    return (nameToIndxMap,featureVectors)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], "o",
                 color=plt.cm.Set1(idx[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

folder = 'prased_txt/'
(nameToIndxMap,featureVectors) = getFeatureVectorsFromFiles(folder)

data = np.array(featureVectors)

#computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,4)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
n_samples, n_features = data.shape

#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(data)
plot_embedding(X_projected, "Random Projection of the view counts history")

#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(data)
plot_embedding(X_pca,
               "Principal Components projection of the view counts history (time %.2fs)" %
               (time() - t0))
plt.show()



