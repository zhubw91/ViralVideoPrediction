import os
import csv
from time import time
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib.patches as mpatches
import numpy as np
from sklearn import (manifold, decomposition, ensemble, random_projection)
from scipy.cluster.vq import kmeans,vq
from operator import add
from SH_Model import performRegression


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
            # get View Column for current file/video as Feature Vector
            file_path = os.path.join(folder,fn)
            column = getColumnFromFile(file_path,2)
            if len(column) == 101:
                featureVector = [float(val) for val in column[1:]]
                featureVectors.append(featureVector)
                #save file name to indx mapping
                nameToIndxMap[fn] = indx
                indx += 1
    return (nameToIndxMap,featureVectors)

def get_cluster_file_list(folder,nameToIndxMap,clusterIndx,idx):
    file_list = []
    for fn in os.listdir(folder):
        if fn.endswith(".pattern") and fn in nameToIndxMap and idx[nameToIndxMap[fn]] == clusterIndx:
            file_path = os.path.join(folder,fn)
            file_list.append(file_path)
    return file_list

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], "o",label="Cluster "+str(idx[i]),
                 color=plt.cm.Set1(idx[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    #plt.legend()
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

folder = 'prased_txt/'
(nameToIndxMap,featureVectors) = getFeatureVectorsFromFiles(folder)

# select first 20 days
for i in range(len(featureVectors)):
    featureVectors[i] = featureVectors[i][:20]

data = np.array(featureVectors)

#computing K-Means with K = 3 (2 clusters)
k = 5
centroids,_ = kmeans(data,k)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
n_samples, n_features = data.shape

y = np.bincount(idx)
ii = np.nonzero(y)[0]
clusterSizes = zip(ii,y[ii])
print clusterSizes

clusters = [[] for i in range(k)]
clusterPeakIndx = [[] for q in range(k)]
for n in range(n_samples):
    clusters[idx[n]].append(featureVectors[n])
    clusterPeakIndx[idx[n]].append(featureVectors[n].index(max(featureVectors[n])))

for indx in range(k):
    if len(clusterPeakIndx[indx]) < 100:
        print indx
        print clusterPeakIndx[indx]
    print indx,': ',np.mean(clusterPeakIndx[indx])

file_list = get_cluster_file_list(folder,nameToIndxMap,np.argmax(y),idx)
print file_list
performRegression(file_list)
#print centroids of clusters
#print centroids.shape
plt.figure()
for indx in range(k):
    plt.plot([i for i in range(n_features)],centroids[indx],color=plt.cm.Set1(indx / 10.),label="Cluster "+str(indx))
plt.legend()
plt.title("Centroids of Clusters")
# #----------------------------------------------------------------------
# # Random 2D projection using a random unitary matrix
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(data)
# plot_embedding(X_projected, "Random Projection of the view counts history")

#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(data)
plot_embedding(X_pca,
               "Principal Components projection of the view counts history (time %.2fs)" %
               (time() - t0))
plt.show()



