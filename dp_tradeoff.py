import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import os
import time



@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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




R = 5
K = 5

# Modify these parameters
step = 1/500
starting_point = 0

# Maximum relative amount of memory that we are allowed to use
s_values = generate_float_range(starting_point, 1, step)  # 0 to 100 inclusive
precision = 8
atol = 10**(-precision)

# Initialize dictionaries to store results

P = {}
T = {}

@lru_cache(maxsize=None)
def find_closest_value(target_value):
    """
    Finds the closest value in a sorted list to a given target value.

    Args:
        sorted_list: A list of sorted float values (ascending order).
        target_value: The target float value.

    Returns:
        The closest float value in the list to the target value.
    """
    if not s_values:
        return None  # Handle empty list

    if target_value <= s_values[0]:
        return s_values[0]
    if target_value >= s_values[-1]:
        return s_values[-1]

    left, right = 0, len(s_values) - 1
    while left < right - 1:
        mid = (left + right) // 2
        if s_values[mid] == target_value:
            return target_value
        elif s_values[mid] < target_value:
            left = mid
        else:
            right = mid
    return s_values[left] if abs(s_values[left] - target_value) <= abs(s_values[right] - target_value) else s_values[right]

@lru_cache(maxsize=None)
def extract_range_from_closest(start, end):
    """
    Extracts a range from a sorted list, based on the closest values to start and end.

    Args:
        sorted_list: A list of sorted float values (ascending order).
        start: The starting float value of the desired range.
        end: The ending float value of the desired range.

    Returns:
        A list containing the extracted range.
    """
    if not s_values:
        return []

    closest_start = find_closest_value(start)
    closest_end = find_closest_value(end)

    start_index = s_values.index(closest_start)
    end_index = s_values.index(closest_end)

    if start_index > end_index:
        return []  # Return empty list if start is after end

    return s_values[start_index : end_index + 1]

# Binary entropy function H(x)
def H(x):
    if x == 0 or x == 1:
        return 0
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


# H_inverse to compute the value alpha such that alpha = H(s)
@lru_cache(maxsize=None)
def H_inverse(s, atol):
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
    # Create first figure with two subplots side by side
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, R + 1))

    # First plot (original scale)
    for r in range(R + 1):
        s_values = [s for (r_key, s) in T.keys() if r_key == r]
        T_values = [T[(r, s)] for s in s_values]
        ax1.scatter(s_values, T_values, color=colors[r], label=f"r={r}", alpha=0.5)

    ax1.set_xlabel("s values")
    ax1.set_ylabel("T values")
    ax1.set_title("Scatter plot of T values for different r (auto scale)")
    ax1.legend()
    ax1.grid()

    # Second plot (fixed y-axis scale)
    for r in range(R + 1):
        s_values = [s for (r_key, s) in T.keys() if r_key == r]
        T_values = [T[(r, s)] for s in s_values]
        ax2.scatter(s_values, T_values, color=colors[r], label=f"r={r}", alpha=0.5)

    ax2.set_xlabel("s values")
    ax2.set_ylabel("T values")
    ax2.set_title("Scatter plot of T values for different r")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid()

    # Adjust layout for first figure
    plt.tight_layout()

    # Create second figure for comparison plot
    fig2, ax3 = plt.subplots(figsize=(10, 6))

    # Third plot (comparison)
    k_values = np.arange(0, 0.5 + 0.01, 0.01)
    time_values = (1.12717 ** k_values) * (1.82653 ** 1)
    space_values = (0.79703 ** k_values) * (1.82653 ** 1)
    x_values = [2**s for (r_key, s) in T.keys() if r_key == R]
    y_values = [2**T[(R, s)] for s in s_values]

    ax3.scatter(x_values, y_values, color="b", label="2^(T[7,s]) (scatter)")
    ax3.plot(x_values, y_values, color="blue", linestyle="--", label="2^(T[7,s]) (line)")
    ax3.plot(space_values, time_values, label='Time-Space Complexity (line)', linestyle="--", color='purple')
    ax3.scatter(space_values, time_values, color="purple", label="Time-Space Complexity (scatter)")
    ax3.set_title(f'Comparison: Samples {1/step} - {K} alpha values')
    ax3.set_xlabel("Space Complexity")
    ax3.set_ylabel("Time Complexity")
    ax3.grid(True)
    ax3.legend()

    # Create the directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), f"{K}\\dp\\")
    os.makedirs(output_dir, exist_ok=True)

    # Save data to file
    with open(os.path.join(output_dir, f"T_values_k_{K}_R_{R}_{step}.txt"), 'w') as f:
        for r in range(R + 1):
            for s in s_values:
                f.write(f"T[{r}, {s}] = {T[(r, s)]}\n")

    # Save both plots
    fig1.savefig(os.path.join(output_dir, f"T_values_s_{step}_k_{K}_R_{R}.pdf"))
    fig2.savefig(os.path.join(output_dir, f"Comparison_s_{step}_k_{K}_R_{R}.pdf"))
    
    # Show both plots
    #plt.show()
    
    # Close both figures to free memory
    plt.close(fig1)
    plt.close(fig2)


def main():

    # Initialize base cases: when r=0, T[0,s] = 1 for all memory values s
    for s in s_values:
        T[(0, s)] = 1
        h_inverse = H_inverse(s, atol)
        alpha_1_values = extract_range_from_closest(0, find_closest_value(h_inverse))
        for r in range(1, R + 1):
            for alpha_1 in alpha_1_values:
                P[(r, s, alpha_1, 1, alpha_1)] = 0

    # Main dynamic programming loops
    # Outer loop: number of allowed recursive calls (r)
    for r in range(1, R + 1):
        # Middle loop: number of levels in the data structure (i)
        for i in range(2, K + 2):
            start_time = time.time()
            # Inner loop: available memory (s)
            for s in s_values:
                h_inverse = H_inverse(s, atol)
                alpha_1_values = extract_range_from_closest(0, find_closest_value(h_inverse))
                
                # For each starting memory allocation alpha_1
                for alpha_1 in alpha_1_values:
                    # Special case: two-level data structure
                    if i == 2:
                        # Generate possible memory allocations for second level
                        alpha_2_values = extract_range_from_closest(find_closest_value(alpha_1), 1/2)
                        for alpha_2 in alpha_2_values:
                            if alpha_2 == 0 or alpha_2 == alpha_1:
                                # No complexity when levels have same memory or no memory
                                P[(r, s, alpha_1, 2, alpha_2)] = 0
                            else:
                                # Compute complexity: entropy term + recursive cost
                                P[(r, s, alpha_1, 2, alpha_2)] = (0.5 *alpha_2* H(alpha_1 / alpha_2) + 
                                    (alpha_2 - alpha_1) * T[(r - 1, min(find_closest_value(s / (alpha_2 - alpha_1)), 1))])
                    
                    # General case: i-level data structure (i ≥ 3)
                    elif i >= 3:
                        # Generate possible memory allocations for level i
                        alpha_i_values = extract_range_from_closest(find_closest_value(alpha_1), 1/2)
                        for alpha_i in alpha_i_values:
                            # Generate possible memory allocations for level i-1
                            alpha_i_1_values = extract_range_from_closest(find_closest_value(alpha_1), find_closest_value(alpha_i))
                            # Initialize with worst case complexity
                            P[(r, s, alpha_1, i, alpha_i)] = 1
                            P_temp = 1
                            
                            # Find optimal memory allocation for level i-1
                            for alpha_i_1 in alpha_i_1_values:
                                if alpha_i == 0:
                                    P_temp = 0  # No complexity for zero memory
                                elif alpha_i == alpha_i_1:
                                    # When consecutive levels have same memory,
                                    # complexity comes only from previous level
                                    P_temp = P[(r, s, alpha_1, i - 1, alpha_i_1)]
                                else:
                                    # Compute complexity: entropy term + max(previous level, recursive cost)
                                    P_temp = (
                                        0.5 * alpha_i * H(alpha_i_1 / alpha_i) + 
                                        max(P[(r, s, alpha_1, i - 1, alpha_i_1)],
                                            (alpha_i - alpha_i_1) * 
                                            T[(r - 1, min(find_closest_value(s / (alpha_i - alpha_i_1)), 1))])
                                    )
                                    
                                # Update if better complexity found
                                P[(r, s, alpha_1, i, alpha_i)] = min(P[(r, s, alpha_1, i, alpha_i)], P_temp)
            
            # Log progress with timing information
            end_time = time.time()
            elapsed_time = end_time - start_time
            endAll_t = time.localtime()
            print(f"now: {time.strftime('%d/%b, %H:%M:%S', endAll_t)} - step : {step} - r : {r} - i : {i} - in time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

        # After computing all P values for current r, update T values
        for s in s_values:
            h_inverse = H_inverse(s, atol)
            alpha_1_values =  extract_range_from_closest(0, find_closest_value(h_inverse))
            # Initialize with worst case
            T[(r, s)] = 1
            T_temp = 1
            # Find optimal alpha_1 that minimizes complexity
            for alpha_1 in alpha_1_values:
                T_temp = max(H(alpha_1), 0.5 + P[(r, s, alpha_1, K + 1, 1/2)])
                T[(r, s)] = min(T[(r, s)], T_temp)
            

            
            # Call plot_results with specific K and step values
    plot_results()

if __name__ == "__main__":
    main()