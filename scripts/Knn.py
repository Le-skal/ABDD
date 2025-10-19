# knn_from_scratch_with_prints.py
# -*- coding: utf-8 -*-
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1️⃣ Define the points
# ------------------------------------------------------
points_data = [
    {"point": 1, "category": "A", "X": 1, "Y": 3},
    {"point": 2, "category": "B", "X": 6, "Y": 5},
    {"point": 3, "category": "B", "X": 8, "Y": 3},
    {"point": 4, "category": "A", "X": 2, "Y": 4},
    {"point": 5, "category": None, "X": 6, "Y": 3},
    {"point": 6, "category": None, "X": 2, "Y": 2},
    {"point": 7, "category": None, "X": 8, "Y": 1},
    {"point": 8, "category": None, "X": 4, "Y": 3},
    {"point": 9, "category": None, "X": 4, "Y": 4},
    {"point": 10, "category": None, "X": 5, "Y": 2},
    {"point": 11, "category": None, "X": 7, "Y": 4},
    {"point": 12, "category": None, "X": 3, "Y": 1},
    {"point": 13, "category": None, "X": 8, "Y": 5},
    {"point": 14, "category": None, "X": 3, "Y": 2},
]

df = pd.DataFrame(points_data)
print("Initial points data:\n", df, "\n")

# ------------------------------------------------------
# 2️⃣ Example distance computation (point 1 vs point 5)
# ------------------------------------------------------
old, new = 1, 5
pointOld = df[df["point"] == old][["X", "Y"]].values[0]
pointNew = df[df["point"] == new][["X", "Y"]].values[0]
distance = math.sqrt((pointNew[0] - pointOld[0])**2 + (pointNew[1] - pointOld[1])**2)
print(f"Coordinates of Point {new}: {pointNew}")
print(f"Coordinates of Point {old}: {pointOld}")
print(f"Euclidean distance between Point {new} and Point {old}: {distance:.3f}\n")

# ------------------------------------------------------
# 3️⃣ KNN Helper functions
# ------------------------------------------------------
def calculate_distances(target_point, df):
    """Compute distances from target_point to all categorized (A/B) points."""
    target = df[df["point"] == target_point][["X", "Y"]].values[0]
    categorized = df[df["category"].notna()]
    results = []
    for _, row in categorized.iterrows():
        other_point = (row["X"], row["Y"])
        dist = math.sqrt((target[0] - other_point[0])**2 + (target[1] - other_point[1])**2)
        results.append((row["point"], row["category"], dist))
    return results

def best_three(distances_list):
    """Keep the 3 points with the smallest distances."""
    return sorted(distances_list, key=lambda x: x[2])[:3]

def final_decision(top3):
    """Assign a category (A/B) based on majority vote among the 3 nearest neighbors."""
    countA = sum(1 for _, c, _ in top3 if c == "A")
    countB = sum(1 for _, c, _ in top3 if c == "B")
    return "A" if countA > countB else "B"

# ------------------------------------------------------
# 4️⃣ Classify unlabeled points iteratively
# ------------------------------------------------------
unlabeled_points = df[df["category"].isna()]["point"].tolist()
print("Unlabeled points (in order):", unlabeled_points, "\n")

for point in unlabeled_points:
    print(f"\n----- Point {point} -----")
    distances_list = calculate_distances(point, df)
    print("All distances:", distances_list)
    top3 = best_three(distances_list)
    print("Nearest 3 neighbors:", top3)
    final = final_decision(top3)
    print("-> Assigned category:", final)
    df.loc[df["point"] == point, "category"] = final
    print("--------------------------------------------------")

# ------------------------------------------------------
# 5️⃣ Final categorized dataset
# ------------------------------------------------------
print("\nFinal categorized dataset:\n", df)

# ------------------------------------------------------
# 6️⃣ Visualization
# ------------------------------------------------------
plt.figure(figsize=(7, 5))
for category, color in zip(["A", "B"], ["tab:blue", "tab:orange"]):
    subset = df[df["category"] == category]
    plt.scatter(subset["X"], subset["Y"], label=f"Class {category}", color=color, s=80, edgecolor="k")

# Label each point with its number
for _, row in df.iterrows():
    plt.text(row["X"] + 0.1, row["Y"] + 0.1, str(row["point"]), fontsize=9)

plt.title("KNN Classification of Points (A vs B)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
