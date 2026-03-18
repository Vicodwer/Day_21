D1 — Prompt Given to AI
Prompt: "Explain NumPy broadcasting and vectorisation with practical Python examples."

D2 — AI Output (Documented)
AI Explanation Summary
•	Broadcasting: Arrays with different shapes can be operated on together as long as their shapes are compatible. NumPy aligns shapes from the right and expands size-1 dims automatically.
•	Vectorisation: Replace Python loops with NumPy operations. This pushes computation into optimised C code and avoids Python's per-element overhead.

AI-Generated Code Example
import numpy as np

# Broadcasting example
A = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)
b = np.array([10, 20, 30])       # shape (3,)
print(A + b)
# [[11 22 33]
#  [14 25 36]]

# Vectorisation example
x = np.array([1, 4, 9, 16, 25])
print(np.sqrt(x))               # [1. 2. 3. 4. 5.]

# Normalize
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

data = np.array([10, 20, 30, 40, 50])
print(normalize(data))
# [0.   0.25 0.5  0.75 1.  ]

D3 — Evaluation of AI Output
Criterion	Result	Notes
Is the broadcasting explanation correct?	Yes ✓	Shape alignment rules accurately described; (2,3) + (3,) example is correct.
Is the vectorisation explanation correct?	Yes ✓	Correctly contrasts loops with NumPy; np.sqrt() example is idiomatic.
Is the code runnable?	Yes ✓	Tested in Python 3.10 + NumPy 1.26 — runs without errors.
Are outputs verified?	Yes ✓	[[11 22 33] [14 25 36]] and [1. 2. 3. 4. 5.] confirmed correct.
Performance comparison included?	No ✗	AI did not show time benchmarking — added in our B3 section.
3D broadcasting covered?	No ✗	AI only showed 2D. We extended to 3D (H,W,C) image-bias pattern.
