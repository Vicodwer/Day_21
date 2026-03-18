# ============================================================
# PART C — Interview Ready
# ============================================================
print("\n" + "=" * 65)
print("PART C — Interview Ready")
print("=" * 65)
 
def normalize(X):
    X  = np.asarray(X, dtype=float)
    mn, mx = X.min(), X.max()
    return np.zeros_like(X) if mx == mn else (X - mn) / (mx - mn)
 
test_arr = np.array([10,25,5,40,15,30,20])
normed   = normalize(test_arr)
print(f"\nQ2 — normalize(X):")
print(f"  Input     : {test_arr}")
print(f"  Normalized: {np.round(normed,4)}")
print(f"  Min={normed.min():.4f}  Max={normed.max():.4f}")
 
print("\n✅  All parts executed successfully.")
