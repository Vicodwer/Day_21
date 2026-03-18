# ============================================================
#  Dependencies: numpy, matplotlib
#  Run: python3 assignment4.py
# ============================================================
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
np.random.seed(42)
 
print("=" * 65)
print("  Python Assignment — Regression, Classification &")
print("  Bias–Variance Tradeoff")
print("=" * 65)
 
 
# ============================================================
# PART A — Concept Application
# ============================================================
print("\n" + "=" * 65)
print("PART A — Concept Application")
print("=" * 65)
 
 
# ──────────────────────────────────────────────────────────────
# A1 + A2 — Synthetic datasets, model training, visualisation
# ──────────────────────────────────────────────────────────────
print("\n--- A1 & A2: Synthetic Datasets + Model Training ---")
 
# ── Regression dataset: y = 3x + 5 + noise ──────────────────
X_reg = np.linspace(0, 10, 100)
y_reg = 3 * X_reg + 5 + np.random.normal(0, 3, 100)
 
# Manual linear regression via Ordinary Least Squares
def linear_regression_fit(X, y):
    """Return slope (m) and intercept (b) via OLS formulas."""
    n   = len(X)
    m   = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / \
          (n * np.sum(X ** 2) - np.sum(X) ** 2)
    b   = (np.sum(y) - m * np.sum(X)) / n
    return m, b
 
m_reg, b_reg = linear_regression_fit(X_reg, y_reg)
y_pred_reg   = m_reg * X_reg + b_reg
 
print(f"\n[Regression]")
print(f"  True equation : y = 3x + 5")
print(f"  Fitted        : y = {m_reg:.4f}x + {b_reg:.4f}")
 
# MSE
mse_reg = np.mean((y_reg - y_pred_reg) ** 2)
print(f"  MSE           : {mse_reg:.4f}")
 
# ── Classification dataset: threshold on noisy linear data ───
X_cls  = np.linspace(0, 10, 100)
scores = 2 * X_cls + np.random.normal(0, 4, 100)
y_cls  = (scores > 10).astype(int)            # binary label
 
# Logistic-style decision boundary: classify by threshold on X
threshold_cls = 5.0
y_pred_cls    = (X_cls >= threshold_cls).astype(int)
accuracy      = np.mean(y_pred_cls == y_cls) * 100
 
print(f"\n[Classification]")
print(f"  Decision threshold on X : {threshold_cls}")
print(f"  Accuracy                : {accuracy:.2f}%")
 
# ── Plot A1/A2 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
# Regression plot
axes[0].scatter(X_reg, y_reg, alpha=0.5, color="steelblue", label="Data points")
axes[0].plot(X_reg, y_pred_reg, color="crimson", linewidth=2,
             label=f"Fit: y={m_reg:.2f}x+{b_reg:.2f}")
axes[0].set_title("A1/A2 — Linear Regression", fontweight="bold")
axes[0].set_xlabel("X"); axes[0].set_ylabel("y")
axes[0].legend()
 
# Classification plot
colors = np.where(y_cls == 1, "#E57373", "#42A5F5")
axes[1].scatter(X_cls, scores, c=colors, alpha=0.7,
                label="Class 0 / Class 1")
axes[1].axvline(threshold_cls, color="black", linestyle="--", linewidth=2,
                label=f"Threshold X={threshold_cls}")
axes[1].set_title("A1/A2 — Binary Classification", fontweight="bold")
axes[1].set_xlabel("X"); axes[1].set_ylabel("Score")
axes[1].legend()
 
plt.tight_layout()
plt.savefig("part_a_reg_cls.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n  Saved → part_a_reg_cls.png")
 
 
# ──────────────────────────────────────────────────────────────
# A3 — Identify regression vs classification
# ──────────────────────────────────────────────────────────────
print("\n--- A3: Identify Problem Type from Target Variable ---")
 
datasets_info = [
    ("House prices ($)",      [215000, 340000, 125000, 480000], "Regression",
     "Target is continuous numeric — predict exact dollar value"),
    ("Spam or Not Spam",      [0, 1, 0, 1, 1, 0],               "Classification",
     "Target is binary label — assign to one of two categories"),
    ("Student grade (0–100)", [72, 85, 91, 60, 77],             "Regression",
     "Target is a continuous score — predict numeric grade"),
    ("Disease present (Y/N)", [1, 0, 1, 0, 0, 1],               "Classification",
     "Target is a categorical label — assign to a class"),
]
 
for name, sample, problem_type, reason in datasets_info:
    print(f"\n  Dataset : {name}")
    print(f"  Sample  : {sample}")
    print(f"  Type    : {problem_type}")
    print(f"  Reason  : {reason}")
 
 
# ──────────────────────────────────────────────────────────────
# A4 — Manual regression + MSE
# ──────────────────────────────────────────────────────────────
print("\n--- A4: Manual Linear Regression + MSE ---")
 
X_a4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y_a4 = np.array([2.1, 4.0, 5.9, 7.8, 10.2, 11.8, 14.1, 16.0, 18.2, 20.1])
 
m_a4, b_a4    = linear_regression_fit(X_a4, y_a4)
y_pred_a4     = m_a4 * X_a4 + b_a4
 
def calculate_mse(y_true, y_pred):
    """Mean Squared Error: average of squared differences."""
    n = len(y_true)
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        total += (yt - yp) ** 2
    return total / n
 
mse_a4 = calculate_mse(y_a4, y_pred_a4)
 
print(f"\n  X        : {X_a4}")
print(f"  y_true   : {y_a4}")
print(f"  Fitted   : y = {m_a4:.4f}x + {b_a4:.4f}")
print(f"  y_pred   : {np.round(y_pred_a4, 2)}")
print(f"  MSE      : {mse_a4:.4f}")
 
 
# ──────────────────────────────────────────────────────────────
# A5 — Manual classification logic + accuracy
# ──────────────────────────────────────────────────────────────
print("\n--- A5: Manual Classification Logic + Accuracy ---")
 
scores_a5  = np.array([35, 72, 58, 90, 45, 83, 61, 29, 77, 55,
                        42, 88, 67, 38, 95, 50, 71, 33, 84, 62])
threshold_a5 = 60
y_true_a5    = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                          0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
 
# Threshold classifier
y_pred_a5 = (scores_a5 >= threshold_a5).astype(int)
 
def calculate_accuracy(y_true, y_pred):
    """Accuracy = correct predictions / total predictions."""
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    return correct / len(y_true) * 100
 
accuracy_a5 = calculate_accuracy(y_true_a5, y_pred_a5)
 
print(f"\n  Scores     : {scores_a5}")
print(f"  Threshold  : {threshold_a5}")
print(f"  y_true     : {y_true_a5}")
print(f"  y_pred     : {y_pred_a5}")
print(f"  Accuracy   : {accuracy_a5:.2f}%")
 
 
# ──────────────────────────────────────────────────────────────
# A6 — Compare regression vs classification outputs
# ──────────────────────────────────────────────────────────────
print("\n--- A6: Regression vs Classification Comparison ---")
print("""
  ┌─────────────────┬───────────────────────────┬────────────────────────────┐
  │ Aspect          │ Regression                │ Classification             │
  ├─────────────────┼───────────────────────────┼────────────────────────────┤
  │ Output type     │ Continuous numeric value  │ Discrete class label       │
  │ Example output  │ 245,000 (house price)     │ 0 or 1 (spam / not spam)   │
  │ Use cases       │ Price, temperature, score │ Spam, disease, image class │
  │ Loss function   │ MSE, MAE, RMSE            │ Accuracy, F1, AUC-ROC      │
  │ Model examples  │ Linear, Ridge, SVR        │ Logistic, SVM, Decision    │
  └─────────────────┴───────────────────────────┴────────────────────────────┘
""")
