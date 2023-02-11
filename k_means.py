import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import datasets

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs
# Create a dataset of 2D distributions
centers = 2
""" iris = datasets.load_iris()
X_train = iris.data[:,:]# take the features
true_labels = iris.target #take the label """

Alldata="/var/home/nhamad/k-means/dataset4.csv"


TRdata = pd.read_csv(Alldata, encoding='utf-8') 

X_train = pd.DataFrame(TRdata, columns = ['f1','f2'])
true_labels = pd.DataFrame(TRdata, columns = ['label'])

X_train = X_train.to_numpy()
true_labels=true_labels.to_numpy().flatten()


X_train = StandardScaler().fit_transform(X_train)
# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)
# View results

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
points = pipeline.fit_transform(X_train)
points = pd.DataFrame(points)
x = points[0]
y = points[1]
#print("centt")
#print(c)

c = pipeline.fit_transform(kmeans.centroids)
c = pd.DataFrame(c)
centroids_x=c[0]
centroids_y= c[1]


class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x,
                y,
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot(centroids_x,
         centroids_y,
         'k+',
         markersize=10,
         )                

plt.title("K-means Visualization")
plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.savefig('dataset4test1.png')
plt.show()