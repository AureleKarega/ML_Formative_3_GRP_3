"""
Part 4: Gradient Descent in Code
---------------------------------
This converts the manual (Part 3) calculations into working Python code.


import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt

# ---- Data (same as Part 3) ----
X = np.array([[1, 3],
              [4, 10]], dtype=float)
y = np.array([5, 6], dtype=float)

m = np.array([-1.0, 2.0])   # initial m
b = np.array([1.0, 1.0])    # initial b

alpha = 0.01                 # learning rate used in Part 3
n_iterations = 4              # team of 4, one update per member


def predict(m, b, X):
    """y_hat = X @ m + b, matrix form (no scalar treatment)."""
    return X @ m + b


def cost_function(params, X, y):
    """
    The 'equation' referred to in the assignment: Mean Squared Error.
    params packs m and b into one vector so SciPy can differentiate
    with respect to all four numbers at once.
    """
    m = params[:2]
    b = params[2:]
    y_hat = X @ m + b
    error = y_hat - y
    return np.mean(error ** 2)


def compute_gradient_scipy(m, b, X, y, epsilon=1e-6):
    """
    Required by Part 4: 'Using SciPy, implement a function that accepts
    an equation and computes its derivative.'
    approx_fprime numerically differentiates cost_function w.r.t. every
    entry of params (i.e. w.r.t. m and b).
    """
    params = np.concatenate([m, b])
    grad = approx_fprime(params, cost_function, epsilon, X, y)
    return grad[:2], grad[2:]   # grad_m, grad_b


# ---- storage for the two required plots ----
history_m, history_b, history_error = [], [], []

print("=" * 60)
print("INITIAL STATE (Iteration 0)")
print("=" * 60)
y_hat = predict(m, b, X)
error = y_hat - y
mse = np.mean(error ** 2)
print(f"m = {m}")
print(f"b = {b}")
print(f"y_hat = {y_hat}")
print(f"error (y_hat - y) = {error}")
print(f"MSE = {mse:.4f}")

history_m.append(m.copy())
history_b.append(b.copy())
history_error.append(mse)

for i in range(1, n_iterations + 1):
    print("\n" + "=" * 60)
    print(f"ITERATION {i}")
    print("=" * 60)

    # --- derivative computed by SciPy (required tool for this part) ---
    grad_m_scipy, grad_b_scipy = compute_gradient_scipy(m, b, X, y)
    print(f"SciPy gradient wrt m : {grad_m_scipy}")
    print(f"SciPy gradient wrt b : {grad_b_scipy}")

    # --- same gradient, written out analytically (matches Part 3 by hand) ---
    y_hat = predict(m, b, X)
    error = y_hat - y
    n = len(y)
    grad_m = (2 / n) * (X.T @ error)
    grad_b = (2 / n) * error
    print(f"Analytic gradient wrt m: {grad_m}  (should match SciPy line above)")
    print(f"Analytic gradient wrt b: {grad_b}  (should match SciPy line above)")

    # --- explicit gradient descent update, matrix form, every step visible ---
    m = m - alpha * grad_m
    b = b - alpha * grad_b

    y_hat = predict(m, b, X)
    error = y_hat - y
    mse = np.mean(error ** 2)

    print(f"Updated m = {m}")
    print(f"Updated b = {b}")
    print(f"New y_hat = {y_hat}")
    print(f"New error = {error}")
    print(f"New MSE   = {mse:.4f}")

    history_m.append(m.copy())
    history_b.append(b.copy())
    history_error.append(mse)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(f"Final m = {m}")
print(f"Final b = {b}")
print(f"Final predictions y_hat = {predict(m, b, X)}")
print(f"Target y                = {y}")

# ---- Plot 1: how m and b change over iterations ----
history_m = np.array(history_m)
history_b = np.array(history_b)

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(history_m[:, 0], marker='o', label='m1')
ax1.plot(history_m[:, 1], marker='o', label='m2')
ax1.plot(history_b[:, 0], marker='s', label='b1')
ax1.plot(history_b[:, 1], marker='s', label='b2')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Parameter value')
ax1.set_title('m and b over iterations')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.savefig('/home/claude/params_over_iterations.png', dpi=150, bbox_inches='tight')

# ---- Plot 2: how error changes over iterations ----
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(history_error, marker='o', color='red')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE')
ax2.set_title('Error (MSE) over iterations')
ax2.grid(True, alpha=0.3)
plt.savefig('/home/claude/error_over_iterations.png', dpi=150, bbox_inches='tight')

print("\nPlots saved: params_over_iterations.png, error_over_iterations.png")