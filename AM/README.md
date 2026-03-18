# 🔢 Python Assignment — NumPy Operations
### Array Operations · Broadcasting · Indexing · Vectorisation

---

## 📁 File Structure

```
assignment3/
├── assignment3.py           # All Python code — Parts A, B, C
├── python_assignment3.docx  # Full submission document
├── part_b3_performance.png  # Bar chart: Loop vs NumPy speed (B3)
└── README.md                # This file
```

---

## 🗂️ Assignment Structure

### Part A — Concept Application (40%)

| Task | NumPy Feature Used |
|------|--------------------|
| Create 1D, 2D, 3D arrays | `np.array()`, `np.arange().reshape()` |
| Indexing & slicing | `arr[i]`, `arr[i:j]`, `arr[::2]`, `arr[::-1]` |
| Extract rows, columns, subarrays | `arr[row, col]`, `arr[:, col]`, `arr[r1:r2, c1:c2]` |
| Element-wise add, sub, mul, div | `+`, `-`, `*`, `/` operators |
| Mean, variance, std dev | `np.mean()`, `np.var()`, `np.std()` |
| Add 1D to 2D (broadcasting) | Shape `(3,)` broadcast over `(3,3)` |
| Multiply matrix by scalar & vector | Broadcasting: scalar → `()`, col_vec → `(3,1)` |
| Squares & cubes (vectorised) | `arr ** 2`, `arr ** 3` |
| Replace negatives with 0 | `np.where(arr < 0, 0, arr)` |
| Normalize to [0, 1] | `(arr - min) / (max - min)` |
| Top 5 maximum values | `np.sort().flatten()[::-1][:5]` |
| Row-wise & column-wise sums | `arr.sum(axis=1)`, `arr.sum(axis=0)` |
| Indices of values > threshold | `np.argwhere(arr > threshold)` |

### Part B — Stretch Problem (30%)

| Task | NumPy API |
|------|-----------|
| Matrix multiplication | `A @ B` or `np.matmul(A, B)` |
| Transpose | `A.T` |
| Determinant | `np.linalg.det(A)` |
| Solve linear equations | `np.linalg.solve(coef, rhs)` |
| Performance comparison | `time.perf_counter()` + Python loop vs `arr.sum()` |

### Part C — Interview Ready (20%)

| Question | Topic |
|----------|-------|
| Q1 | NumPy broadcasting — rules and usefulness |
| Q2 | `normalize(X)` — scale array to [0, 1] using NumPy |
| Q3 | Vectorisation vs loops — why NumPy is faster |

### Part D — AI-Augmented Task (10%)

- Prompt documented with full AI output
- AI-generated examples tested and verified in Python
- Evaluation table: correctness, efficiency, runnable, improvements

---

## ▶️ How to Run

### Install dependencies
```bash
pip install numpy matplotlib
```

### Run the assignment
```bash
python3 assignment3.py
```

Requires **Python 3.7+**. One `.png` chart will be generated in the same directory.

---

## 📈 Charts Generated

| File | Contents |
|------|----------|
| `part_b3_performance.png` | Bar chart comparing Python loop vs NumPy sum time |

---

## ✅ Sample Output

```
--- A1: Indexing & Slicing ---
1D slice [2:6]       : [30 40 50 60]
2D row 2             : [11 12 13 14 15]
2D column 3          : [ 4  9 14 19]
3D element [1,1,2]   : 19

--- A2: Statistics ---
Mean     : 5.5000
Variance : 8.2500
Std Dev  : 2.8723

--- A3: Broadcasting ---
matrix (3×3) + row_vec (3,) → each row gets [+10, +20, +30]

--- A4: Vectorised ---
Clipped (negatives → 0) : [0. 0. 0. 2. 4. 0. 7. 0. 6. 3.]
Normalized [0,1]         : [0.2 0.333 ... 1.0]

--- B3: Performance ---
Python loop : 0.7213s
NumPy sum   : 0.0103s
NumPy is ≈ 70× faster
```

---

## 📌 Key Concepts

**Broadcasting** — NumPy's mechanism for applying operations on arrays of different but compatible shapes without copying data. Dimensions are aligned from the right; size-1 dimensions are stretched to match.

**Vectorisation** — Replacing explicit Python `for` loops with NumPy array operations that run in compiled C/FORTRAN code, using SIMD CPU instructions for bulk computation.

**Why NumPy is faster than loops:**
- Python loops carry per-iteration interpreter overhead
- NumPy stores data in contiguous memory blocks (no object boxing)
- Operations dispatch to optimised BLAS/LAPACK routines
- CPU SIMD processes multiple array elements per clock cycle

**Normalisation formula:**  
`Z = (X − X_min) / (X_max − X_min)`  → scales all values to the range [0, 1]

**Broadcasting rules (trailing-edge alignment):**
1. Pad shape with 1s on the left until both arrays have the same rank
2. Any dimension of size 1 is stretched to match the other
3. If sizes differ and neither is 1 → `ValueError`

---

> 💡 No explicit Python `for` loops were used in Parts A and C. All array operations are fully vectorised using NumPy.
