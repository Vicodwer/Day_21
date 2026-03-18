# ============================================================
# PART B — Stretch Problem
# ============================================================
print("\n" + "=" * 65)
print("PART B — Stretch Problem")
print("=" * 65)
 
# B1 — Matrix ops
print("\n--- B1: Matrix Operations ---")
A = np.array([[2,3,1],[4,1,5],[3,2,4]], dtype=float)
B = np.array([[1,0,2],[3,4,1],[2,1,0]], dtype=float)
print(f"\nA @ B:\n{A @ B}")
print(f"\nA.T:\n{A.T}")
print(f"\ndet(A) = {np.linalg.det(A):.4f}")
 
# B2 — Linear equations
print("\n--- B2: Solve Linear Equations ---")
coef = np.array([[2,3],[1,-2]], dtype=float)
rhs  = np.array([8,-3], dtype=float)
sol  = np.linalg.solve(coef, rhs)
print(f"\n2x + 3y = 8  |  x - 2y = -3")
print(f"Solution: x = {sol[0]:.4f},  y = {sol[1]:.4f}")
print(f"Verify: {coef @ sol}")
 
# B3 — Performance
print("\n--- B3: Performance Comparison ---")
SIZE  = 10_000_000
large = np.random.rand(SIZE)
py_list = large.tolist()
t0 = time.perf_counter()
total = 0.0
for x in py_list: total += x
t_loop = time.perf_counter() - t0
t1 = time.perf_counter()
total = large.sum()
t_numpy = time.perf_counter() - t1
print(f"\nArray size : {SIZE:,}")
print(f"Loop  : {t_loop:.4f}s")
print(f"NumPy : {t_numpy:.4f}s")
print(f"NumPy is ~{t_loop/t_numpy:.1f}x faster")
 
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(["Python Loop","NumPy"], [t_loop,t_numpy],
              color=["#E57373","#42A5F5"], edgecolor="white", width=0.4)
ax.set_ylabel("Time (seconds)")
ax.set_title(f"B3 — Loop vs NumPy: Sum of {SIZE:,} elements", fontweight="bold")
for bar, val in zip(bars, [t_loop, t_numpy]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + t_loop*0.02,
            f"{val:.4f}s", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("part_b3_performance.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved -> part_b3_performance.png")
