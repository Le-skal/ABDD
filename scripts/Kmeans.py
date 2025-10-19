# kmeans_from_scratch_with_prints.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------
# 1️⃣ Define the points
# ------------------------------------------------------
points_data = [
    {"point": "A1", "X": 2, "Y": 10},
    {"point": "A2", "X": 2, "Y": 5},
    {"point": "A3", "X": 8, "Y": 4},
    {"point": "A4", "X": 5, "Y": 8},
    {"point": "A5", "X": 7, "Y": 5},
    {"point": "A6", "X": 6, "Y": 4},
    {"point": "A7", "X": 1, "Y": 2},
    {"point": "A8", "X": 4, "Y": 9},
]

df = pd.DataFrame(points_data)
X = df[["X", "Y"]].values
print("Initial points:\n", df, "\n")

# ------------------------------------------------------
# 2️⃣ Initialize K-Means parameters
# ------------------------------------------------------
k = 3  # number of clusters (you can change this)
random.seed(42)
initial_centroids = np.array([
    df[df["point"] == "A1"][["X", "Y"]].values[0],
    df[df["point"] == "A4"][["X", "Y"]].values[0],
    df[df["point"] == "A7"][["X", "Y"]].values[0],
])

centroids = np.array(initial_centroids)
print(f"Initial centroids (A1, A4, A7):\n{centroids}\n")
# ------------------------------------------------------
# 3️⃣ K-Means algorithm (manual loop)
# ------------------------------------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, c) for c in centroids]
        cluster_idx = np.argmin(distances)
        clusters.append(cluster_idx)
    return np.array(clusters)

def compute_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        points_in_cluster = X[clusters == i]
        if len(points_in_cluster) > 0:
            new_centroids.append(points_in_cluster.mean(axis=0))
        else:
            # empty cluster (rare case)
            new_centroids.append(random.choice(X))
    return np.array(new_centroids)

max_iters = 10
for iteration in range(max_iters):
    clusters = assign_clusters(X, centroids)
    new_centroids = compute_centroids(X, clusters, k)
    print(f"Iteration {iteration+1}:")
    print(" -> Cluster assignments:", clusters)
    print(" -> Centroids:\n", new_centroids, "\n")
    if np.allclose(centroids, new_centroids):
        print("Converged!\n")
        break
    centroids = new_centroids

df["Cluster"] = clusters
print("Final DataFrame with cluster assignments:\n", df, "\n")

# ------------------------------------------------------
# 4️⃣ Visualization
# ------------------------------------------------------
plt.figure(figsize=(7, 5))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

for i in range(k):
    subset = df[df["Cluster"] == i]
    plt.scatter(subset["X"], subset["Y"], color=colors[i % len(colors)], s=80, edgecolor="k", label=f"Cluster {i+1}")

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=200, marker="X", label="Centroids")

# Label each point
for _, row in df.iterrows():
    plt.text(row["X"] + 0.1, row["Y"] + 0.1, row["point"], fontsize=9)

plt.title(f"K-Means Clustering (k={k})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
