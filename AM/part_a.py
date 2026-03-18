# ============================================================
#  Dependencies: numpy, matplotlib
#  Run: python3 assignment3.py
# ============================================================
 
import numpy as np
import time
import matplotlib.pyplot as plt
 
np.random.seed(42)
 
print("=" * 65)
print("  Python Assignment — NumPy Operations")
print("=" * 65)
 
# ============================================================
# PART A — Concept Application
# ============================================================
print("\n" + "=" * 65)
print("PART A — Concept Application")
print("=" * 65)
 
# A1 — Arrays, Indexing, Slicing
print("\n--- A1: Array Creation, Indexing & Slicing ---")
 
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"\n1D array  : {arr_1d}")
print(f"  Index [0]         : {arr_1d[0]}")
print(f"  Index [-1]        : {arr_1d[-1]}")
print(f"  Slice [2:6]       : {arr_1d[2:6]}")
print(f"  Every other [::2] : {arr_1d[::2]}")
print(f"  Reversed  [::-1]  : {arr_1d[::-1]}")
 
arr_2d = np.array([[ 1, 2, 3, 4, 5],
                   [ 6, 7, 8, 9,10],
                   [11,12,13,14,15],
                   [16,17,18,19,20]])
print(f"\n2D array (4x5):\n{arr_2d}")
print(f"  Element [1,3]         : {arr_2d[1,3]}")
print(f"  Row 2                 : {arr_2d[2]}")
print(f"  Column 3              : {arr_2d[:,3]}")
print(f"  Rows 1-2, Cols 1-3    :\n{arr_2d[1:3,1:4]}")
print(f"  Last two rows         :\n{arr_2d[-2:]}")
 
arr_3d = np.arange(1,25).reshape(2,3,4)
print(f"\n3D array (2x3x4):\n{arr_3d}")
print(f"  Block 0             :\n{arr_3d[0]}")
print(f"  Block 1, Row 2      : {arr_3d[1,2]}")
print(f"  Element [1,1,2]     : {arr_3d[1,1,2]}")
print(f"  All blocks, Row 0, Col 1-3 : {arr_3d[:,0,1:4]}")
 
# A2 — Operations & Stats
print("\n--- A2: Element-wise Operations & Stats (no loops) ---")
a = np.array([3,7,2,9,4,6,1,8,5,10], dtype=float)
b = np.array([2,4,6,1,3,8,5,7,9,10], dtype=float)
print(f"\na = {a}\nb = {b}")
print(f"\nAddition     (a + b) : {a + b}")
print(f"Subtraction  (a - b) : {a - b}")
print(f"Multiply     (a * b) : {a * b}")
print(f"Division     (a / b) : {np.round(a/b,3)}")
print(f"\nMean     : {np.mean(a):.4f}")
print(f"Variance : {np.var(a):.4f}")
print(f"Std Dev  : {np.std(a):.4f}")
 
# A3 — Broadcasting
print("\n--- A3: Broadcasting ---")
matrix  = np.array([[1,2,3],[4,5,6],[7,8,9]])
row_vec = np.array([10,20,30])
col_vec = np.array([[1],[2],[3]])
print(f"\n(i) matrix + row_vec:\n{matrix + row_vec}")
print(f"(ii) matrix * 5:\n{matrix * 5}")
print(f"(iii) matrix * col_vec:\n{matrix * col_vec}")
 
# A4 — Vectorised operations
print("\n--- A4: Vectorised Operations ---")
data = np.array([-5,-3,0,2,4,-1,7,-8,6,3], dtype=float)
print(f"\nOriginal : {data}")
print(f"Squares  : {data**2}")
print(f"Cubes    : {data**3}")
print(f"Clipped  : {np.where(data<0,0,data)}")
normed = (data - data.min()) / (data.max() - data.min())
print(f"Normalized: {np.round(normed,4)}")
print(f"  min={normed.min():.4f}  max={normed.max():.4f}")
 
# A5 — Dataset analytics
print("\n--- A5: Dataset Analytics ---")
dataset = np.random.randint(1, 200, size=(6,8))
print(f"\nDataset (6x8):\n{dataset}")
top5 = np.sort(dataset.flatten())[::-1][:5]
print(f"\nTop 5 values     : {top5}")
print(f"Row-wise sums    : {dataset.sum(axis=1)}")
print(f"Column-wise sums : {dataset.sum(axis=0)}")
threshold = 150
indices = np.argwhere(dataset > threshold)
print(f"\nValues > {threshold}:")
for idx in indices:
    print(f"  dataset[{idx[0]},{idx[1]}] = {dataset[idx[0],idx[1]]}")
