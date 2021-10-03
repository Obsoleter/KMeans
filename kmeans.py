# Libraries
import random
import numpy as np

# Types
from typing import Tuple


class KMeanIteration:
    def __init__(self, clusters) -> None:
        self.clusters = clusters

    def generate_centers(self, x_min, x_max, y_min, y_max) -> np.ndarray:
        centers = []
        for i in range(self.clusters):
            center = np.array((random.uniform(x_min, x_max), random.uniform(y_min, y_max)), dtype='float64')
            centers.append(center)

        return np.array(centers)

    def iterate(self, array) -> Tuple[int, np.ndarray, np.ndarray]:
        # Generate centers
        x_min = min(array[:, 0])
        y_min = min(array[:, 1])
        x_max = max(array[:, 0])
        y_max = max(array[:, 1])
        centers = self.generate_centers(x_min, x_max, y_min, y_max)

        # Recenter till the end
        clusters_points = None
        prev_centers = None
        while prev_centers is None or not(np.array_equal(centers, prev_centers)):
            # Save previous centers to compare & Reset clusters of points
            prev_centers = centers.copy()
            clusters_points = []

            # Cluster points and calculate new centers
            points_mean = np.zeros((self.clusters, 3))

            # Compare length of each point to centers and choose the closest one
            for point in array:
                minlen = None
                n_cluster = None
                for i in range(self.clusters):
                    # Find length between point and center
                    center = centers[i]
                    x = center[0] - point[0]
                    y = center[1] - point[1]
                    len = np.sqrt(x*x + y*y)

                    # Save length and cluster of better case
                    if  minlen == None or len < minlen:
                        minlen = len
                        n_cluster = i

                # Save point info to calculate new centers & Save cluster num
                points_mean[n_cluster][0] += point[0]
                points_mean[n_cluster][1] += point[1]
                points_mean[n_cluster][2] += 1
                clusters_points.append(n_cluster)

            # Recentre centers
            for i in range(self.clusters):
                if points_mean[i][2] == 0:
                    continue
                centers[i][0] = points_mean[i][0] / points_mean[i][2]
                centers[i][1] = points_mean[i][1] / points_mean[i][2]

        # Find SSE of this iteration
        sse = 0
        for point, n_cluster in zip(array, clusters_points):
            center = centers[n_cluster]
            x = center[0] - point[0]
            y = center[1] - point[1]
            sse += x*x + y*y

        return (sse, centers, np.array(clusters_points))


# KMean Exceptions
class NoInfoFeeded : Exception


class KMean:
    def __init__(self, clusters, iterations=100) -> None:
        self.clusters = clusters
        self.iterations = iterations

    def feed(self, array) -> Tuple[int, np.ndarray, np.ndarray]:
        # Info of the best iteration
        best_sse = None
        best_centers = None
        best_clusters = None

        # Iterate
        for i in range(self.iterations):
            iteration = KMeanIteration(self.clusters)
            result = iteration.iterate(array)

            # Save info, if iteration is better
            if  best_sse == None or result[0] < best_sse:
                best_sse = result[0]
                best_centers = result[1]
                best_clusters = result[2]

        # Save some info in class
        self.sse = best_sse
        self.centers = best_centers
        return (best_sse, best_centers, best_clusters)

    def clusterize(self, array) -> np.ndarray:
        # Raise exception if no info was feeded
        if self.centers is None:
            raise NoInfoFeeded('No info was Feeded!')

        # Collect points to clusters by closest length to centers
        clusters = []
        for point in array:
            minlen = None
            n_cluster = None

            # Choose the best cluster
            for i in range(self.clusters):
                x = self.centers[i][0] - point[0]
                y = self.centers[i][1] - point[1]
                len = np.sqrt(x*x + y*y)

                # Save cluster if is better
                if minlen == None or len < minlen:
                    minlen = len
                    n_cluster = i

            # Save the best cluster
            clusters.append(n_cluster)
        return np.array(clusters)

    def get_sse(self) -> int:
        return self.sse

    def get_centers(self) -> np.ndarray:
        return self.centers