import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

points = {
    "M1": (2, 0),
    "M2": (0, 1),
    "M3": (0, 2),
    "M4": (3, 4),
    "M5": (5, 4),
}

df = pd.DataFrame(points).T
df.columns = ["X", "Y"]
X = df[["X", "Y"]].values
labels = df.index.tolist()
print("Initial points:\n", df, "\n")


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def ward_distance(cluster1, cluster2):
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    n1, n2 = len(cluster1), len(cluster2)
    return np.sqrt(
        (n1 * n2) / (n1 + n2) * euclidean_distance(centroid1, centroid2) ** 2
    )


n = X.shape[0]
clusters = {i: X[i : i + 1] for i in range(n)}
cluster_names = {i: labels[i] for i in range(n)}
next_id = n

linkage_matrix = []
print("Hierarchical clustering steps:")

while len(clusters) > 1:
    min_dist = float("inf")
    merge_pair = None

    # Find the two closest clusters
    cluster_ids = list(clusters.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            c1_id, c2_id = cluster_ids[i], cluster_ids[j]
            dist = ward_distance(clusters[c1_id], clusters[c2_id])
            if dist < min_dist:
                min_dist = dist
                merge_pair = (c1_id, c2_id)

    c1_id, c2_id = merge_pair
    merged = np.vstack([clusters[c1_id], clusters[c2_id]])

    print(
        f" Merge {cluster_names[c1_id]} + {cluster_names[c2_id]} -> G{next_id}: distance {min_dist:.3f}, centroid {np.mean(merged, axis=0)}"
    )

    linkage_matrix.append([c1_id, c2_id, min_dist, len(merged)])

    clusters[next_id] = merged
    cluster_names[next_id] = f"G{next_id}"

    del clusters[c1_id]
    del clusters[c2_id]
    next_id += 1

print(f"\nFinal centroid: {np.mean(X, axis=0)}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

positions = {i: i for i in range(n)}
cluster_height = {i: 0 for i in range(n)}
color_idx = 0

for c1, c2, dist, size in linkage_matrix:
    x1 = positions[c1]
    x2 = positions[c2]
    x_new = (x1 + x2) / 2

    y1 = cluster_height[c1]
    y2 = cluster_height[c2]

    color = colors[color_idx % len(colors)]

    # Vertical lines from each cluster up to the merge point
    ax.plot([x1, x1], [y1, dist], color=color, lw=2)
    ax.plot([x2, x2], [y2, dist], color=color, lw=2)
    ax.plot([x1, x2], [dist, dist], color=color, lw=2)

    new_cluster_id = n + color_idx
    positions[new_cluster_id] = x_new
    cluster_height[new_cluster_id] = dist

    color_idx += 1

ax.set_xticks(range(n))
ax.set_xticklabels(labels)
ax.set_ylabel("Ward distance")
ax.set_title("CAH (Ward) dendrogram - M1..M5")
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
plt.tight_layout()
plt.show()
