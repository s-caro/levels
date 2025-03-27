import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import os
import time

def truncate_float(value, step):
    """
    Truncates a float value to the precision of the step.

    Args:
        value (float): The float value to truncate.
        step (float): The step value, determining the precision.

    Returns:
        float: The truncated float value.
    """
    precision = len(str(step).split('.')[1])
    return round(value, precision)

def generate_float_range(start, end, step):
    """
    Generates a sorted list of float values from start to end (inclusive)
    with the given step, avoiding floating-point approximation issues and rounding.

    Args:
        start (float): The starting value.
        end (float): The ending value.
        step (float): The step value.

    Returns:
        list[float]: A sorted list of float values.
    """
    if step == 0:
        if start == end:
            return [start]
        else:
            return []

    if (start > end and step > 0) or (start < end and step < 0):
        return []

    result = []
    current = start
    while (step > 0 and current <= end) or (step < 0 and current >= end):
        result.append(truncate_float(current, step))
        current += step

    # Ensure the end value is included, even if the loop misses it slightly
    if truncate_float(current - step, step) != truncate_float(end, step):
        result.append(truncate_float(end, step))

    # Remove duplicates from rounding errors
    result = sorted(list(set(result)))

    return result



R = 7
K = 1

# Modify these parameters
s_steps = 1/100
starting_point = 0

# Maximum relative amount of memory that we are allowed to use
s_values = generate_float_range(starting_point, 1, s_steps)  # 0 to 100 inclusive

precision = 4
atol = 10**(-precision)

# Initialize dictionaries to store results

P = {}
T = {}



# Since we are working with integers we need to add normalization functions
def normalize(x):
    """Convert integer 0-100 to float 0-1"""
    return x / s_steps

def denormalize(x):
    """Convert float 0-1 to integer 0-100"""
    return int(np.floor(x * s_steps))

# Binary entropy function H(x)
def H(x):
    if x == 0 or x == 1:
        return 0
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


# H_inverse to compute the value alpha such that alpha = H(s)
@lru_cache(maxsize=None)
def H_inverse(s, atol):
    #s_normalized = normalize(s)
    left, right = 0, 0.5
    while right - left > atol:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        H_m1 = H(m1)
        H_m2 = H(m2)

        if np.isclose(H_m1, s, atol=atol):
            return m1
        if np.isclose(H_m2, s, atol=atol):
            return m2

        if H_m1 < s:
            left = m1
        else:
            right = m2
    return (left + right) / 2


# Plotting function
def plot_results():
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, R + 1))

    for r in range(R + 1):
        s_values = [s for (r_key, s) in T.keys() if r_key == r]
        T_values = [T[(r, s)] for s in s_values]
        plt.scatter(s_values, T_values, color=colors[r], label=f"r={r}", alpha=0.5)

    plt.xlabel("s values")
    plt.ylabel("T values")
    plt.title("Scatter plot of T values for different r")
    plt.legend()
    plt.grid()

    # Create the directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), f"{K}\\dp\\")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"T_values_k_{K}_R_{R}_{s_steps}.txt"), 'w') as f:
        for r in range(R + 1):
            for s in s_values:
                f.write(f"T[{r}, {s}] = {T[(r, s)]}\n")

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"Running_time_s_{s_steps}_k_{K}_R_{R}_{s_steps}.pdf"))
    plt.show()


def main():
    # Initialize T[0,s] = 1 for all s
    for s in s_values:
        T[(0, s)] = 1
        # Compute alpha = H(s)
        h_inverse = H_inverse(s, atol)
        # Values of allowed memory: alpha_1
        alpha_1_values = generate_float_range(0, h_inverse, s_steps)
        # Initialize P[r,s, alpha_1,1,alpha_1] = 1 for all alpha_1 < s as the required value is known from the precomputation
        for r in range(1, R + 1):
            for alpha_1 in alpha_1_values:
                P[(r, s, alpha_1, 1, alpha_1)] = 0

    # End initialization step


    # R loop over all possible allowed recursive calls
    for r in range(1, R + 1):
        # i loop over all possible levels
        for i in range(1, K + 2):
            start_time = time.time()
            # For each maximum relative amount of memory compute the intermediary values
            for s in s_values:
                # Compute alpha = H(s)
                h_inverse = H_inverse(s, atol)
                # Values of allowed memory: alpha_1
                alpha_1_values = generate_float_range(0, h_inverse, s_steps)
                for alpha_1 in alpha_1_values:
                    # case for i = 2
                    if i == 2:
                        # Values of alpha_2 that ranges from alpha_1 to 1/2
                        alpha_2_values = generate_float_range(alpha_1, 1/2, s_steps)
                        # Save all possible values of alpha_2
                        for alpha_2 in alpha_2_values:
                            # The partial complexity is 0 is alpha_1 equals alpha_2 (the entropy is 0 and the (alpha_2 - alpha_1) coefficient is 0) or if alpha_2 is 0 (then also alpha_1 is 0 and we have P[(r, s, 0, 2, 0)])
                            if alpha_2 == 0 or alpha_2 == alpha_1:
                                P[(r, s, alpha_1, 2, alpha_2)] = 0
                            else:
                                P[(r, s, alpha_1, 2, alpha_2)] = 0.5 * H(alpha_1 / alpha_2) + (alpha_2 - alpha_1) * T[(r - 1, min(truncate_float(s / (alpha_2 - alpha_1),s_steps), 1))]
                    # general case for i>=3
                    elif i >= 3:
                        # Values of alpha_i that ranges from alpha_1 to 1/2
                        alpha_i_values = generate_float_range(alpha_1, 1/2, s_steps)
                        for alpha_i in alpha_i_values:
                            # Values of alpha_i-1 that ranges from alpha_1 to alpha_i
                            alpha_i_1_values = generate_float_range(alpha_1, alpha_i, s_steps)
                            # Since we need to find the alpha_i-1 value that minimizes P[(r, s, alpha_1, i, alpha_i)] we initiate P[(r, s, alpha_1, i, alpha_i)] to 1
                            P[(r, s, alpha_1, i, alpha_i)] = 1
                            # We use a temp value to store the current value for P[(r, s, alpha_1, i, alpha_i)]
                            P_temp = 1
                            for alpha_i_1 in alpha_i_1_values:
                                # The partial complexity is 0 if alpha_i is 0 (then also alpha_1 is 0 and we have P[(r, s, 0, 2, 0)])
                                if alpha_i==0:
                                    P_temp = 0
                                # If alpha_i and alpha_i-1 have the same value then P[(r, s, alpha_1, i, alpha_i_1)] is equal to P[(r, s, alpha_1, i - 1, alpha_i_1)] since the H(alpha_i_1 / alpha_i) is 0 and the (alpha_i - alpha_i_1) coefficient is 0
                                elif alpha_i == alpha_i_1:
                                    P_temp = (
                                        P[(r, s, alpha_1, i - 1, alpha_i_1)]
                                    )
                                else:
                                    P_temp = (
                                        0.5 * H(alpha_i_1 / alpha_i) + 
                                        max(P[(r, s, alpha_1, i - 1, alpha_i_1)],
                                            (alpha_i - alpha_i_1) * 
                                            T[(r - 1, min(truncate_float(s / (alpha_i - alpha_i_1),s_steps), 1))])
                                    )
                                # We update P[(r, s, alpha_1, i, alpha_i)] value if the new P_temp is better
                                if P_temp < P[(r, s, alpha_1, i, alpha_i)]:
                                    P[(r, s, alpha_1, i, alpha_i)] = P_temp
                
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time
            endAll_t = time.localtime()
            print(f"now: {time.strftime('%d/%b, %H:%M:%S', endAll_t)} - r : {r} - i : {i} - in time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

        # At the end of the computation, given a specific r, of P[(r, s, alpha_1, i, alpha_i)] (for and all i and s), we can compute the values T[(r,s)]
        for s in s_values:
            h_inverse = H_inverse(s, atol)
            # Values of allowed memory: alpha_1
            alpha_1_values =generate_float_range(0, h_inverse, s_steps)
            # Since we need to find the alpha_1 value that minimizes T[r,s] we initiate T[r,s] to 1                            
            T[(r, s)] = 1
            # We use a temp value to store the current value for T[r,s]
            T_temp = 1
            for alpha_1 in alpha_1_values:
                T_temp = (max(H(alpha_1), 0.5 + P[(r, s, alpha_1, K + 1, 1/2)]))
                # We update T[r,s] value if the new T_temp is better
                if T_temp < T[(r, s)]:
                    T[(r, s)] = T_temp
        print(T)

if __name__ == "__main__":
    main()
    plot_results()