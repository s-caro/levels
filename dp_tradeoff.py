import numpy as np
import matplotlib.pyplot as plt

def H(x):
    """Binary entropy function H(x) = -(x log2 x + (1 - x) log2(1 - x))"""
    if x == 0 or x == 1:
        return 0
    return - (x * np.log2(x) + (1 - x) * np.log2(1 - x))

def H_inverse(s, precision=1e-6):
    """Find H^(-1)(s) using ternary search."""
    lo, hi = 0, 0.5
    while hi - lo > precision:
        mid1 = lo + (hi - lo) / 3
        mid2 = hi - (hi - lo) / 3
        if H(mid1) < s:
            lo = mid1
        else:
            hi = mid1
        if H(mid2) < s:
            lo = mid2
        else:
            hi = mid2
    return (lo + hi) / 2

def compute_P(r, s, alpha1, i, alpha_i, T):
    """Compute P[r, s, alpha1, i, alpha_i] using dynamic programming."""
    if i == 1:
        return 0  # Base case: known from precomputation
    
    if i == 2:
        return 0.5 * H(alpha1 / alpha_i) + (alpha_i - alpha1) * T[r-1].get(s / (alpha_i - alpha1), np.inf)
    
    alpha_prev_values = np.linspace(alpha1, alpha_i, num=100)
    P_values = [0.5 * H(alpha_prev / alpha_i) + (alpha_i - alpha_prev) * T[r-1].get(s / (alpha_i - alpha_prev), np.inf) for alpha_prev in alpha_prev_values]
    
    return min(P_values)

def compute_T(r_max, s_max, k):
    """Compute the optimal running time T[r, s] using dynamic programming."""
    T = [{} for _ in range(r_max + 1)]
    
    # Base case
    for s in np.linspace(0, s_max, 101):
        T[0][s] = 1  # O(2^n) algorithm
    
    for r in range(1, r_max + 1):
        for s in np.linspace(0, s_max, 101):
            alpha1_opt = H_inverse(s)  # Compute H^(-1)(s)
            
            # Compute T[r, s] using minimization over alpha1
            T_values = [H(alpha1_opt) + 0.5 + compute_P(r, s, alpha1_opt, k + 1, 0.5, T)]
            T[r][s] = min(T_values)
    
    return T

# Parameters
r_max = 10  # Maximum recursion depth
s_max = 1.0  # Max memory fraction
k = 6  # Number of levels

# Compute T values
T = compute_T(r_max, s_max, k)

# Save results to a file
with open("T_values.txt", "w") as f:
    for r in range(r_max + 1):
        for s, value in T[r].items():
            f.write(f"{r} {s:.6f} {value:.6f}\n")

# Plot results
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, r_max + 1))

for r in range(r_max + 1):
    s_values = list(T[r].keys())
    T_values = list(T[r].values())
    plt.scatter(s_values, T_values, color=colors[r], label=f"r={r}", alpha=0.7)

plt.xlabel("s values")
plt.ylabel("T values")
plt.title("Scatter plot of T values for different r")
plt.legend()
plt.grid()
plt.savefig("T_plot.png")
plt.show()
