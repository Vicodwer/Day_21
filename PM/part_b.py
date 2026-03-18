# ============================================================
# PART B — Stretch Problem: Bias–Variance Tradeoff
# ============================================================
print("\n" + "=" * 65)
print("PART B — Bias–Variance Tradeoff")
print("=" * 65)
 
print("\n--- B1 & B2: Polynomial Fitting — Underfitting to Overfitting ---")
 
# True function: y = sin(2πx)
X_bv     = np.linspace(0, 1, 30)
y_noise  = np.sin(2 * np.pi * X_bv) + np.random.normal(0, 0.3, 30)
X_plot   = np.linspace(0, 1, 300)
y_true_plot = np.sin(2 * np.pi * X_plot)
 
degrees = [1, 2, 5, 9, 15]
 
def poly_fit_predict(X_train, y_train, X_test, deg):
    """Fit a polynomial of given degree and return predictions on X_test."""
    coeffs = np.polyfit(X_train, y_train, deg)
    return np.polyval(coeffs, X_test), np.polyval(coeffs, X_train)
 
fig, axes = plt.subplots(1, len(degrees), figsize=(17, 4), sharey=True)
train_errors = []
 
for i, deg in enumerate(degrees):
    y_plot_pred, y_train_pred = poly_fit_predict(X_bv, y_noise, X_plot, deg)
    y_train_fit, _            = poly_fit_predict(X_bv, y_noise, X_bv, deg)
    train_mse = np.mean((y_noise - y_train_fit) ** 2)
    train_errors.append(train_mse)
 
    axes[i].scatter(X_bv, y_noise, s=20, color="steelblue", alpha=0.7, label="Data")
    axes[i].plot(X_plot, y_true_plot, "g--", linewidth=1.5, label="True f(x)")
    axes[i].plot(X_plot, y_plot_pred, "crimson", linewidth=2, label=f"Deg {deg}")
    axes[i].set_ylim(-2.5, 2.5)
    axes[i].set_title(
        f"Degree {deg}\n"
        f"{'Underfit' if deg <= 1 else ('Overfit' if deg >= 9 else 'Good fit')}",
        fontweight="bold"
    )
    axes[i].set_xlabel("X")
    if i == 0:
        axes[i].set_ylabel("y")
 
plt.suptitle("B1 — Bias–Variance: Polynomial Degree vs Fit Quality",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("part_b_bias_variance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved → part_b_bias_variance.png")
 
# Training error vs complexity plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(degrees, train_errors, "o-", color="crimson", linewidth=2, markersize=8)
for d, e in zip(degrees, train_errors):
    ax.annotate(f"{e:.3f}", (d, e), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)
ax.axvspan(0, 2, alpha=0.1, color="blue", label="Underfitting zone")
ax.axvspan(8, 16, alpha=0.1, color="red", label="Overfitting zone")
ax.axvspan(2, 8, alpha=0.1, color="green", label="Sweet spot")
ax.set_title("B2 — Training Error vs Model Complexity", fontweight="bold")
ax.set_xlabel("Polynomial Degree (Complexity)")
ax.set_ylabel("Training MSE")
ax.legend()
plt.tight_layout()
plt.savefig("part_b_error_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved → part_b_error_curve.png")
 
print(f"\n  Degree | Train MSE | Assessment")
print(f"  {'─'*40}")
labels = ["Underfitting (high bias)", "Slight underfit", "Good fit",
          "Overfitting (high variance)", "Severe overfit"]
for d, e, lbl in zip(degrees, train_errors, labels):
    print(f"  {d:6d} | {e:9.4f} | {lbl}")
 
print("""
--- B3: Bias, Variance & Optimal Model ---
 
  Bias
    → Error from wrong assumptions in the model.
    → High bias = model too simple → underfitting.
    → Example: fitting a straight line to curved data.
 
  Variance
    → Error from sensitivity to small fluctuations in training data.
    → High variance = model too complex → overfitting.
    → Example: a degree-15 polynomial memorising every noise point.
 
  Optimal model (Bias–Variance tradeoff sweet spot)
    → Minimises TOTAL error = Bias² + Variance + Irreducible noise.
    → Use cross-validation to find the right complexity.
    → For this dataset, degree 2–5 gives the best generalisation.
""")
