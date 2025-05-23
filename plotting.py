import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import os
import time
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors
import math
import json

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


# Define path (using raw string)

R = 5
K = 6

# Modify these parameters
step = 1/1000
starting_point = 0

# Maximum relative amount of memory that we are allowed to use
s_values = generate_float_range(starting_point, 1, step)  # 0 to 100 inclusive
precision = 8
allowed_error = 10**(-precision)

def create_dictionary(k):


    T = {}
    pattern = re.compile(r'T\[(\d+),\s*([\d.]+)\]\s*=\s*([\d.]+)')
    input_dir = os.path.join(os.getcwd(), f"levels\\plots\\actual\\")
    filename = os.path.join(input_dir, f'T_values_k_{k+1}_R_5_0,001_partial_5_par.txt')

    with open(filename, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                a = int(match.group(1))
                b = float(match.group(2))
                c = float(match.group(3))
                T[(a, b)] = c

    # Sort and create ordered dictionary
    sorted_T = OrderedDict(sorted(T.items(), key=lambda x: (x[0][0], x[0][1])))
    return sorted_T

# Plotting function
def plot_results():
    # Create the directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), f"levels\\plots\\actual\\")
    os.makedirs(output_dir, exist_ok=True)
    # Create first figure with two subplots side by side
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = [i for i in mcolors.TABLEAU_COLORS.keys()]
    
    for k in range(K):
        T = create_dictionary(k)
        print(f'dict created for {k}')
        x_values = [s for (r, s) in T.keys() if r==5]
        y_values = [2**T[(5, s)] for s in s_values]
        ax1.plot(x_values, y_values, color=colors[k], label=f"k={k+1}", linewidth=1.5)
            
        
    x_s_values = [s for s in s_values]
    y_s_values = [2**s for s in s_values]
    ax1.plot(x_s_values, y_s_values, color=colors[K+2], label=f"S=T")

     # Third plot (comparison)
    k_values = np.arange(0, 0.5 + 0.01, 0.01)
    time_values = (1.12717 ** k_values) * (1.82653 ** 1)
    space_values = [math.log2((0.79703 ** k) * (1.82653 ** 1)) for k in k_values]
    #space_values = math.log2((0.79703 ** k_values) * (1.82653 ** 1))
    ax1.plot(space_values, time_values, color=colors[K+1], label='time-space tradeoff', linewidth=1.5)
    

    ax1.set_xlabel("S")
    ax1.set_ylabel("T")
    #ax1.set_title("")
    ax1.legend()
    ax1.grid(axis='y', linestyle='dashed', color='grey', alpha=0.5)

    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)






    # Save both plots
    fig1.savefig(os.path.join(output_dir, f"Results_{step}_.pdf"))
    
    # Show both plots
    plt.show()
    
    # Close both figures to free memory
    plt.close(fig1)

def main():
    
    # Plot the results
    plot_results()

if __name__ == "__main__":
    main()