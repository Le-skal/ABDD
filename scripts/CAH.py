# cah_from_scratch_with_prints.py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Points from your PPTX
points = {
    "M1": (2, 0),
    "M2": (0, 1),
    "M3": (0, 2),
    "M4": (3, 4),
    "M5": (5, 4),
}

df = pd.DataFrame(points).T
df.columns = ['X','Y']
X = df[['X','Y']].values
labels = df.index.tolist()
print("Initial points:\n", df, "\n")

# Ward linkage
Z = linkage(X, method='ward')
Z_df = pd.DataFrame(Z, columns=['Cluster1','Cluster2','Distance','NumPoints'])
print("Linkage (Ward) matrix:\n", Z_df, "\n")

# Manually compute centroids at each merge to get G6,G7,G8,G9
# SciPy encodes original observations as 0..n-1
n = X.shape[0]
clusters = {i: [i] for i in range(n)}  # cluster id -> list of member indices
next_cluster_id = n

centroids = {i: X[i] for i in range(n)}
print("Manual merge steps and computed centroids:")
for row in Z:
    c1 = int(row[0]); c2 = int(row[1])
    members = clusters[c1] + clusters[c2]
    # compute centroid coordinates
    coords = np.mean([X[i] for i in members], axis=0)
    print(f" Merge {c1} + {c2} -> new cluster {next_cluster_id}: members indices {members}, centroid {coords}")
    # update
    clusters[next_cluster_id] = members
    centroids[next_cluster_id] = coords
    # remove old if you like (not strictly necessary)
    next_cluster_id += 1

# Final centroid (G9)
final_centroid = np.mean(X, axis=0)
print("\nFinal centroid G9:", final_centroid)

# Dendrogram
plt.figure(figsize=(7,5))
dendrogram(Z, labels=labels)
plt.title("CAH (Ward) dendrogram - M1..M5")
plt.ylabel("Ward distance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
