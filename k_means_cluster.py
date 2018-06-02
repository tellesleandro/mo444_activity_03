import logging
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
from pdb import set_trace as bp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

class KMeansCluster:

    def __init__(self, X):
        self.X = X;

    def fit_transform(self, n_clusters):
        self.kmeans = MiniBatchKMeans(n_clusters = n_clusters, init_size = 3 * n_clusters)
        self.kmeans.fit_transform(self.X)

    def fit_transform_range(self, n_clusters_begin, n_clusters_end, n_clusters_step):
        self.cost = {}
        self.distortions = {}
        # X_array = self.X.toarray()
        for n_clusters in range(n_clusters_begin, n_clusters_end, n_clusters_step):
            logging.info('Computing K-Means for ' + str(n_clusters) + ' clusters')
            self.fit_transform(n_clusters)
            self.cost[n_clusters] = self.inertia()
            # self.distortions[n_clusters] = sum(np.min(cdist(X_array, self.kmeans.cluster_centers_, 'euclidean'), axis=1)) / self.X.shape[0]

    def cluster(self, cluster_number):
        return [idx for idx, val in enumerate(self.labels()) if val == cluster_number]

    def labels(self):
        return self.kmeans.labels_

    def predict(self, Y):
        return self.kmeans.predict(Y)

    def cluster_centers(self):
        return self.kmeans.cluster_centers_

    def inertia(self):
        return self.kmeans.inertia_
