# ============================================================
# PART C — Interview Ready
# ============================================================
print("=" * 65)
print("PART C — Interview Ready")
print("=" * 65)
 
# Q1 — Conceptual (see docx for full table)
print("""
Q1 — Regression vs Classification (real-world examples)
  Regression     : Predicting house price, stock value, temperature.
  Classification : Spam detection, cancer diagnosis, image labelling.
  Key difference : Regression → continuous output.
                   Classification → discrete class label.
""")
 
# Q2 — calculate_mse
print("--- Q2: calculate_mse(y_true, y_pred) ---")
 
def calculate_mse(y_true, y_pred):
    """Return Mean Squared Error without using numpy.mean."""
    n = len(y_true)
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        total += (yt - yp) ** 2
    return total / n
 
y_true_q2 = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_q2 = np.array([2.5,  0.0, 2.0, 8.0])
mse_q2    = calculate_mse(y_true_q2, y_pred_q2)
 
print(f"  y_true : {y_true_q2}")
print(f"  y_pred : {y_pred_q2}")
print(f"  MSE    : {mse_q2:.4f}")
 
# Q3 — Conceptual (see docx)
print("""
Q3 — Bias–Variance Tradeoff
  Underfitting (high bias, low variance):
    → Model too simple to capture the pattern.
    → High training AND test error.
    → Fix: increase model complexity, add features.
 
  Overfitting (low bias, high variance):
    → Model memorises training data including noise.
    → Low training error, high test error.
    → Fix: regularisation, more data, reduce complexity.
 
  Goal: find the sweet spot where total generalisation error
        (Bias² + Variance) is minimised.
""")
 
print("✅  All parts executed successfully.")
