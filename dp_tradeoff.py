import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import os
import time

R = 7
K = 3

# Modify these parameters
s_steps = 1  # Now steps of 1 instead of 0.01
rounding = 0  # No decimal places needed for integers
starting_point = 0

# Initialize s_values with integers
s_values = np.arange(starting_point, 101, s_steps)  # 0 to 100 inclusive

precision = 4
atol = 10**(-precision)

# Initialize dictionaries to store results

P = {}
T = {}

# Add normalization functions
def normalize(x):
    """Convert integer 0-100 to float 0-1"""
    return x / 100.0

def denormalize(x):
    """Convert float 0-1 to integer 0-100"""
    return int(np.floor(x * 100))

# Binary entropy function H(x)
def H(x):
    if x == 0 or x == 1:
        return 0
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


# Modify H_inverse to work with normalized values
@lru_cache(maxsize=None)
def H_inverse(s, atol):
    s_normalized = normalize(s)
    left, right = 0, 0.5
    while right - left > atol:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        H_m1 = H(m1)
        H_m2 = H(m2)

        if np.isclose(H_m1, s_normalized, atol=atol):
            return m1
        if np.isclose(H_m2, s_normalized, atol=atol):
            return m2

        if H_m1 < s_normalized:
            left = m1
        else:
            right = m2
    return (left + right) / 2


def plot_results():
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, R + 1))

    for r in range(R + 1):
        s_values = [s for (r_key, s) in T.keys() if r_key == r]
        T_values = [T[(r, s)] for s in s_values]
        plt.scatter(s_values, T_values, color=colors[r], label=f"r={r}", alpha=0.7)

    plt.xlabel("s values")
    plt.ylabel("T values")
    plt.title("Scatter plot of T values for different r")
    plt.legend()
    plt.grid()

    # Create the directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), f"{K}\\dp\\")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"T_values_k_{K}.txt"), 'w') as f:
        for r in range(R + 1):
            for s in s_values:
                f.write(f"T[{r}, {s}] = {T[(r, s)]}\n")

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"Running_time_s_{s_steps}_precision_{atol}_T.pdf"))
    plt.show()


def main():
    # Initialize T[0,s] = 1 for all s
    for s in s_values:
        T[(0, s)] = 1
        h_inverse = H_inverse(s, atol)
        num_steps = int(denormalize(h_inverse)) + 1
        alpha_1_values = np.arange(0, num_steps + 1)
        for r in range(R + 1):
            for alpha_1 in alpha_1_values:
                P[(r, s, alpha_1, 1, alpha_1)] = 0

    for r in range(1, R + 1):
        for i in range(1, K + 2):
            start_time = time.time()  # Start timing
            for s in s_values:
                h_inverse = H_inverse(s, atol)
                num_steps = int(denormalize(h_inverse)) + 1
                alpha_1_values = np.arange(0, num_steps + 1)
                for alpha_1 in alpha_1_values:
                    if i == 2:
                        alpha_2_values = np.arange(alpha_1, 51)  # 50 is half of 100
                        for alpha_2 in alpha_2_values:
                            if alpha_1 == alpha_2:
                                P[(r, s, alpha_1, 2, alpha_2)] = 1
                            else:
                                norm_alpha_1 = normalize(alpha_1)
                                norm_alpha_2 = normalize(alpha_2)
                                P[(r, s, alpha_1, 2, alpha_2)] = 1 / 2 * H(norm_alpha_1 / norm_alpha_2) + \
                                    (norm_alpha_2 - norm_alpha_1) * T[(r - 1, min(s // (alpha_2 - alpha_1), 100))]
                    elif i >= 3:
                        alpha_i_values = np.arange(alpha_1, 51)
                        for alpha_i in alpha_i_values:
                            alpha_i_1_values = np.arange(alpha_1, alpha_i + 1)
                            for alpha_i_1 in alpha_i_1_values:
                                if len(alpha_i_1_values) == 0:
                                    P[(r, s, alpha_1, i, alpha_i)] = 1
                                else:
                                    P_i_possible_values = []
                                    for alpha_i_1 in alpha_i_1_values:
                                        if alpha_i_1 == alpha_i:
                                            P_i_possible_values.append(1)
                                        else:
                                            norm_alpha_i_1 = normalize(alpha_i_1)
                                            norm_alpha_i = normalize(alpha_i)
                                            P_i_possible_values.append(
                                                1 / 2 * H(norm_alpha_i_1 / norm_alpha_i) + 
                                                max(P[(r, s, alpha_1, i - 1, alpha_i_1)],
                                                    (norm_alpha_i - norm_alpha_i_1) * 
                                                    T[(r - 1, min(s // (alpha_i - alpha_i_1), 100))])
                                            )
                                    P[(r, s, alpha_1, i, alpha_i)] = min(P_i_possible_values)
                
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            endAll_t = time.localtime()
            print(f"now: {time.strftime('%d/%b, %H:%M:%S', endAll_t)} - r : {r} - i : {i} - in time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

        for s in s_values:
            if s == 0:
                T[(r, s)] = 1
            else:
                h_inverse = H_inverse(s, atol)
                num_steps = int(denormalize(h_inverse)) + 1
                alpha_1_values = np.arange(0, num_steps + 1)
                T_possible_values = []
                for alpha_1 in alpha_1_values:
                    T_possible_values.append(max(H(normalize(alpha_1)), P[(r, s, alpha_1, K + 1, 50)]))
                T[(r, s)] = min(T_possible_values)


if __name__ == "__main__":
    main()
    plot_results()