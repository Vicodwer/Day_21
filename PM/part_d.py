D1 — Prompt Given to AI
Prompt: "Explain regression vs classification and bias-variance tradeoff with Python examples and visualizations."

D2 — AI Output (Documented)
AI Explanation Summary
•	Regression: Predicts a continuous value. Example: house price prediction using linear regression. Loss = MSE.
•	Classification: Predicts a class label. Example: spam detection using logistic regression. Metric = Accuracy.
•	Bias: Error from underfitting — model cannot capture data pattern. High bias → wrong on training data too.
•	Variance: Error from overfitting — model fits noise. Low training error, high test error.
•	Optimal model: Cross-validate; the best complexity minimises Bias² + Variance.

AI-Generated Python Example
import numpy as np
import matplotlib.pyplot as plt

# Regression
X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,5,4,5], dtype=float)
m = np.polyfit(X, y, 1)          # [slope, intercept]
y_pred = np.polyval(m, X)
mse = np.mean((y - y_pred)**2)
print(f'Regression MSE: {mse:.4f}')

# Classification
scores    = np.array([30,70,50,90,40,80])
y_true    = np.array([0, 1, 0, 1, 0, 1])
y_pred_cls = (scores >= 60).astype(int)
acc = np.mean(y_pred_cls == y_true) * 100
print(f'Classification Accuracy: {acc:.1f}%')

# Bias-variance: polynomial degrees
X_bv = np.linspace(0, 1, 20)
y_bv = np.sin(2*np.pi*X_bv) + np.random.normal(0,0.2,20)
for d in [1, 5, 15]:
    coeffs = np.polyfit(X_bv, y_bv, d)
    err    = np.mean((y_bv - np.polyval(coeffs, X_bv))**2)
    print(f'Degree {d}: MSE={err:.4f}')

D3 — Evaluation of AI Output
Criterion	Result	Notes
Is regression explanation correct?	Yes ✓	Linear model and MSE correctly described and applied.
Is classification explanation correct?	Yes ✓	Threshold logic and accuracy metric are correct.
Is bias-variance explanation correct?	Yes ✓	Bias/variance definitions accurate; polynomial example is appropriate.
Is the code runnable?	Yes ✓	Tested in Python 3.10 + NumPy 1.26 — runs without errors.
Do visualisations show underfit/overfit?	Partial ✗	AI code does not include plots — we added the full 5-panel chart and error curve in Part B.
Manual implementations included?	No ✗	AI used np.polyfit and np.mean for MSE. Our code implements OLS and MSE with explicit loops as required.
