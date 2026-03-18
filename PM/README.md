# 🤖 Python Assignment — Regression, Classification & Bias–Variance Tradeoff
### Linear Regression · Binary Classification · MSE · Accuracy · Underfitting · Overfitting

---

## 📁 File Structure

```
assignment4/
├── assignment4.py              # All Python code — Parts A, B, C
├── python_assignment4.docx     # Full submission document
├── part_a_reg_cls.png          # Scatter + regression line & classification boundary (A1/A2)
├── part_b_bias_variance.png    # Polynomial fits: degree 1 → 15 (B1)
├── part_b_error_curve.png      # Training error vs model complexity (B2)
└── README.md                   # This file
```

---

## 🗂️ Assignment Structure

### Part A — Concept Application (40%)

| Task | Method / Formula |
|------|-----------------|
| Synthetic regression dataset | `y = 3x + 5 + noise`, 100 samples |
| Synthetic classification dataset | Binary labels via score threshold |
| Manual linear regression fit | OLS: `m = (nΣxy − ΣxΣy) / (nΣx² − (Σx)²)` |
| Plot regression line | `matplotlib` scatter + line |
| Classification boundary | Threshold on X; `y = (X ≥ threshold)` |
| Identify problem type (A3) | Continuous target → Regression; discrete → Classification |
| MSE calculation (A4) | `MSE = (1/n) Σ(y_true − y_pred)²` — manual loop |
| Threshold classifier + accuracy (A5) | `accuracy = correct / total × 100` — manual loop |
| Regression vs classification comparison (A6) | Output type, use cases, metrics, model types |

### Part B — Stretch Problem (30%)

| Task | Detail |
|------|--------|
| Polynomial fits (degrees 1, 2, 5, 9, 15) | `np.polyfit()` + `np.polyval()` |
| Visualise underfit → good fit → overfit | 5-panel plot with shaded zones |
| Training error vs complexity curve | MSE plotted against polynomial degree |
| Bias definition | Error from oversimplified assumptions |
| Variance definition | Sensitivity to fluctuations in training data |
| Optimal model | Cross-validate; minimise Bias² + Variance |

### Part C — Interview Ready (20%)

| Question | Topic |
|----------|-------|
| Q1 | Regression vs classification with real-world examples |
| Q2 | `calculate_mse(y_true, y_pred)` — manual loop implementation |
| Q3 | Bias–variance tradeoff: underfitting, overfitting, sweet spot |

### Part D — AI-Augmented Task (10%)

- Prompt documented with full AI explanation and code
- AI output tested for correctness and runnability
- Evaluation table: 6 criteria including gaps identified

---

## ▶️ How to Run

### Install dependencies
```bash
pip install numpy matplotlib
```

### Run the assignment
```bash
python3 assignment4.py
```

Requires **Python 3.7+**. Three `.png` chart files are generated automatically.

---

## 📈 Charts Generated

| File | Contents |
|------|----------|
| `part_a_reg_cls.png` | Left: regression scatter + fitted line · Right: classification with decision boundary |
| `part_b_bias_variance.png` | 5-panel comparison: polynomial degrees 1, 2, 5, 9, 15 |
| `part_b_error_curve.png` | Training MSE vs polynomial degree with underfitting/overfitting zones |

---

## ✅ Sample Output

```
[Regression]
  True equation : y = 3x + 5
  Fitted        : y = 3.0414x + 4.4816
  MSE           : 7.3341

[Classification]
  Decision threshold : 5.0
  Accuracy           : 86.00%

--- A4: Manual Linear Regression ---
  Fitted   : y = 2.0145x + -0.0600
  MSE      : 0.0179

--- A5: Threshold Classifier ---
  Accuracy : 100.00%

--- B1: Polynomial Degrees ---
  Degree 1  → MSE: 0.3949  (Underfitting)
  Degree 5  → MSE: 0.1010  (Good fit)
  Degree 15 → MSE: 0.0571  (Severe overfit)
```

---

## 📌 Key Concepts

**Regression** — Predicts a *continuous* numeric value.
- Formula: `ŷ = mx + b`
- Loss: `MSE = (1/n) Σ(y − ŷ)²`
- Examples: house price, temperature, stock return

**Classification** — Assigns input to a *discrete class*.
- Logic: `class = 1 if score ≥ threshold else 0`
- Metric: `Accuracy = correct_predictions / total × 100`
- Examples: spam/not-spam, disease detection, digit recognition

**Bias** — Error from a model that is too simple to capture the true pattern (underfitting). High bias → high training *and* test error.

**Variance** — Error from a model that is too sensitive to training data noise (overfitting). High variance → low training error, high test error.

**Bias–Variance Tradeoff:**
```
Total Error = Bias² + Variance + Irreducible Noise
```
- Increase complexity → bias ↓, variance ↑
- Decrease complexity → bias ↑, variance ↓
- Optimal model minimises the sum using cross-validation

---

> 💡 All core calculations (OLS regression, MSE, accuracy) are implemented manually using loops — no `sklearn` or `scipy` used.
