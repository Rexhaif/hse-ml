#!/usr/bin/env python
# coding: utf-8

from typing import *
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

np.random.seed(42)


# # Task 1 - cluster problem generator


def scatter_clusters(
  centers: ArrayLike,
  spread: ArrayLike,
  n_points: int
) -> ArrayLike:
    result_points = []
    result_clusters = []
    per_cluster_size = n_points // len(centers)
    for i, center in enumerate(centers):
        x_vals = np.random.normal(loc=center[0], scale=spread[i], size=per_cluster_size)
        y_vals = np.random.normal(loc=center[1], scale=spread[i], size=per_cluster_size)
        points = np.vstack([x_vals, y_vals]).reshape((per_cluster_size, 2))
        result_points.append(points)
        result_clusters.append(np.full(per_cluster_size, i))
        
    return np.concatenate(result_points), np.concatenate(result_clusters)


# I use easy/hard as follows:

# Easy = spread < (distance_between_centers / 2)
# 
# Hard = spread >= (distance_between_centers / 2)


easy_points, easy_clusters = scatter_clusters(
    centers=[[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]],
    spread=[2.0, 2.0, 2.0],
    n_points=1000
)
hard_points, hard_clusters = scatter_clusters(
    centers=[[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]],
    spread=[6.0, 6.0, 6.0],
    n_points=1000
)


# Helper function to plot coloured clusters

def plot_clusters(points: np.ndarray, clusters: np.ndarray):
    cmap = {0:'r',1:'g',2:'b'}
    _ = plt.figure(figsize=(10.0, 10.0))
    for p, c in zip(points, clusters):
        plt.scatter(p[0], p[1], c=cmap[c])

def kmeans_cluster_assignment(
  k: int,
  points: ArrayLike,
  centers_guess: Optional[ArrayLike] = None,
  max_iterations: Optional[int] = None,
  tolerance: Optional[float] = None
) -> ArrayLike:
    if max_iterations is None:
        return kmeans_cluster_assignment(k, points, centers_guess, 9999, tolerance)
    if centers_guess is None:
        # make random center guess
        centers = []
        for i in range(k):
            x = np.random.normal(loc=np.mean(points[:, 0]), scale=np.std(points[:, 0]), size=1)
            y = np.random.normal(loc=np.mean(points[:, 1]), scale=np.std(points[:, 1]), size=1)
            centers.append([x, y])
        return kmeans_cluster_assignment(k, points, centers, max_iterations, tolerance)
    else:
        # do the iteration
        max_iterations -= 1
        assignments = []
        for point in points:
            distances = [euclidean(point, center) for center in centers_guess]
            cluster_idx = np.argmin(distances)
            assignments.append(cluster_idx)
            
        assignments = np.array(assignments)
        if max_iterations == 0:
            #print(f"== remaining iterations: {max_iterations}")
            return assignments
        
        new_centers = []
        for i in range(k):
            cluster_points = points[np.argwhere(assignments == i).ravel()]
            if len(cluster_points) != 0:
                # if any points assigned to cluster
                new_x = np.mean(cluster_points[:, 0])
                new_y = np.mean(cluster_points[:, 1])
            else:
                # if no points assigned to cluster = re-guess center
                new_x = np.random.normal(loc=np.mean(points[:, 0]), scale=np.std(points[:, 0]), size=1)
                new_y = np.random.normal(loc=np.mean(points[:, 1]), scale=np.std(points[:, 1]), size=1)
            new_centers.append([new_x, new_y])
            
        if tolerance is None:
            _tolerance = -1.0
        else:
            _tolerance = tolerance
            
        min_cluster_difference = min([cityblock(old, new) for old, new in zip(centers_guess, new_centers)])
        if min_cluster_difference < _tolerance:
            #print(f"== remaining iterations: {max_iterations}")
            return assignments
        else:
            return kmeans_cluster_assignment(k, points, new_centers, max_iterations, tolerance)


# ### Cluster plot for random init

random_guess = kmeans_cluster_assignment(3, easy_points, max_iterations=1, tolerance=0.1)
plot_clusters(easy_points, random_guess)
plt.savefig("./random-init.png")


# ### For 25% iterations(lets assume 100% == 20 iterations)

_25 = kmeans_cluster_assignment(3, easy_points, max_iterations=5, tolerance=0.1)
plot_clusters(easy_points, _25)
plt.savefig("./25-pct.png")


# ### For 50% iterations

_50 = kmeans_cluster_assignment(3, easy_points, max_iterations=10, tolerance=0.1)
plot_clusters(easy_points, _50)
plt.savefig("./50-pct.png")


# ### For 75% iterations

_75 = kmeans_cluster_assignment(3, easy_points, max_iterations=15, tolerance=0.1)
plot_clusters(easy_points, _75)
plt.savefig("./75-pct.png")


# ### For 100% iterations

_100 = kmeans_cluster_assignment(3, easy_points, max_iterations=20, tolerance=0.1)
plot_clusters(easy_points, _100)
plt.savefig("./100-pct.png")


# # Task 3 - comparing to scipy implementation

from memory_profiler import memory_usage
import timeit
from scipy.cluster.vq import kmeans


# - Memory

our_mem_usage = memory_usage((kmeans_cluster_assignment, (3, hard_points), {'tolerance': 10e-5, 'max_iterations': 20}))
scipy_mem_usage = memory_usage((kmeans, (hard_points, 3)))
print(f"Mean memory usage of our   implementations: {np.mean(our_mem_usage):.4f} MiB")
print(f"Mean memory usage of scipy implementations: {np.mean(scipy_mem_usage):.4f} MiB")


# - Speed

our_timing = timeit.repeat(
    "kmeans_cluster_assignment(3, hard_points, tolerance=10e-5, max_iterations=20)",
    globals=globals(),
    repeat=7,
    number=100
)
scipy_timing = timeit.repeat(
    "kmeans(hard_points, 3)",
    globals=globals(),
    repeat=7,
    number=100
)
print(f"Run time for our   implementation: {np.mean(our_timing):.2f}+-{np.std(our_timing):.2f} ms")
print(f"Run time for scipy implementation:  {np.mean(scipy_timing):.2f}+-{np.std(scipy_timing):.2f} ms")


# - Quality
our_cluster_assignments = kmeans_cluster_assignment(3, hard_points, tolerance=10e-5, max_iterations=20)

centroids, _ = kmeans(hard_points, 3)
# To measure quality for scipy implementation i am performing cluster assignments given resulting centroids
sp_assignments = []
for point in hard_points:
    distances = [euclidean(point, center) for center in centroids]
    cluster_idx = np.argmin(distances)
    sp_assignments.append(cluster_idx)
sp_assignments = np.array(sp_assignments)

print(f"Rand score for our own implementation: {rand_score(hard_clusters, our_cluster_assignments):.4f}")
print(f"Rand score for scipy kmeans: {rand_score(hard_clusters, sp_assignments):.4f}")


# ## Results:
# From memory prospective, both implementations seems equal, but it depends on a method of memory consumption recording
# 
# From speed - it is clear that scipy implementation is a lot faster than our own, probably because they implement it in C lang
# 
# From quality - both implementations performs equally

# # Task 4
# 4.1: Plot clustering performance against percant of completed iterations
# 
# 4.2: Split data 90/10 and repeat 4.1
# 
# I'm using rand_score from sklearn because it is essentially a percent of points assigned to a correct cluster

# 4.1
performance = []
for i in range(0, 11):
    assignment = kmeans_cluster_assignment(3, hard_points, tolerance=10e-5, max_iterations=i)
    score = rand_score(hard_clusters, assignment)
    performance.append(score)

plt.plot(performance)
plt.savefig("./4.1-plot.png")

# 4.2
from sklearn.model_selection import train_test_split
x_train, x_test, _, y_test = train_test_split(hard_points, hard_clusters, test_size=0.1, random_state=42)
performance_2 = []
for i in range(11):
    assignment = kmeans_cluster_assignment(3, x_train, max_iterations=i, tolerance=10e-5)
    centroids = []
    for j in [0, 1, 2]:
        cluster_points = x_train[np.argwhere(assignment == j).ravel()]
        centroids.append((
            np.mean(cluster_points[:, 0]),
            np.mean(cluster_points[:, 1])
        ))
        
    test_assignments = []
    for point in x_test:
        distances = [euclidean(point, center) for center in centroids]
        cluster_idx = np.argmin(distances)
        test_assignments.append(cluster_idx)
    test_assignments = np.array(test_assignments)
    score = rand_score(y_test, test_assignments)
    performance_2.append(score)

plt.plot(performance_2)
plt.savefig("./4.2-plot.png")


# ## Results:
# - for 4.2 we see that performance on the test test fluctuates just as int 4.1 plot, where we do not split into train/test

# # Task 5:
# Cross-Validation

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=10)
performances = {i:[] for i in range(11)}
for train_idx, test_idx in kf.split(hard_points, hard_clusters):
    x_train = hard_points[train_idx]
    x_test = hard_points[test_idx]
    y_test = hard_clusters[test_idx]
    for i in range(11):
        assignment = kmeans_cluster_assignment(3, x_train, max_iterations=i, tolerance=10e-5)
        centroids = []
        for j in [0, 1, 2]:
            cluster_points = x_train[np.argwhere(assignment == j).ravel()]
            centroids.append((
                np.mean(cluster_points[:, 0]),
                np.mean(cluster_points[:, 1])
            ))

        test_assignments = []
        for point in x_test:
            distances = [euclidean(point, center) for center in centroids]
            cluster_idx = np.argmin(distances)
            test_assignments.append(cluster_idx)
        test_assignments = np.array(test_assignments)
        score = rand_score(y_test, test_assignments)
        performances[i].append(score)

x = list(range(11))
y1 = []
y2 = []
for i in x:
    mean = np.mean(performances[i])
    std = np.std(performances[i])
    y1.append(mean-std)
    y2.append(mean+std)

plt.fill_between(x, y1, y2)
plt.savefig("./5-plot.png")
