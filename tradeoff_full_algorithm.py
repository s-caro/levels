import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
import glob
import math
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from collections import defaultdict
import os.path
from pathlib import Path

from operator import itemgetter

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json

import multiprocessing as mp  


from functools import partial
from multiprocessing.pool import ThreadPool  
from functools import lru_cache




# the setrecursionlimit function is
# used to modify the default recursion
# limit set by python. Using this,
# we can increase the recursion limit
# to satisfy our needs

sys.setrecursionlimit(10**6)

# Binary entropy function H(x)
def H(x):
    if x == 0 or x == 1:
        return 0
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))



# Function f(a, b) with small epsilon to handle very small values of a
def f(a, b):
    return H(b / a) * a



# Optimized function to find alpha_c using ternary search with caching
# Optimized function to find alpha_c using ternary search with caching
def find_alpha_c(c, s, atol):
    # Check if the (s, c) pair is already in the cache
    #print(f'c: {c}, s: {s}, atol: {atol}')
    if (s, c) in alpha_c_cache:
        return alpha_c_cache[(s, c)]

    # Perform the calculation if not cached
    left, right = 0, 0.5
    while True:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        f_m1 = f(c, m1 * c)
        f_m2 = f(c, m2 * c)

        if np.isclose(f_m1, s, atol=atol):
            alpha_c_cache[(s, c)] = m1  # Cache the result
            return m1
        if np.isclose(f_m2, s, atol=atol):
            alpha_c_cache[(s, c)] = m2  # Cache the result
            return m2

        if abs(f_m1 - s) < abs(f_m2 - s):
            right = m2
        else:
            left = m1

        if abs(f(c, (left + right) / 2 * c) - s) < atol:
            result = (left + right) / 2
            alpha_c_cache[(s, c)] = result  # Cache the result
            return result

# Parameters for s and atol
alpha_steps = 1/75
precision = 4
atol = 10**(-precision)

# Initialize dictionaries to store results

s_values = np.round(np.arange(0, (1+alpha_steps), alpha_steps), precision)
F_s = {}  # Memoization array for each fixed s

# Initialize a dictionary to store cached alpha_c values
max_recursion = 7
alpha_c_cache = {}

# Define a global or local dictionary for memoization
term2_memo = {}

# Initialize caches
term2_cache = {}
t_rec_cache = {}

levels = 5
parallel = True
inside_parallel = False
max_rec = False

# Recursive function for Ts(c)
def T_rec_1(c, s):

    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []

    #c = np.round(c, precision)

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append(1)

        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)

        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            term1 = f(c, c / 2) / 2
            term2 = f(c / 2, alpha_1_opt * c) / 2

            term3 = T_rec_1(c / 2 - alpha_1_opt * c, s)
            min_time_alpha_1.append(max((term1 + term2 + term3),alpha_1))

    opt_time =  min(min_time_alpha_1)


    return opt_time


# Recursive function for Ts(c) with maximum recursion limit
def T_s_fixed_rec_2_levels(c, s, max_recursion, current_recursion=0):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > max_recursion:
        return None

    alpha_1_values = np.round(np.arange(0, (s + alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)

            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:
                    term1 = f(c, c * alpha_1_opt)
                    term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_rec_2((alpha_2_opt - alpha_1_opt) * c, s, max_recursion, current_recursion + 1)
                    term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_rec_2((0.5 - alpha_2_opt) * c, s, max_recursion, current_recursion + 1)

                    # If any recursive call returns None, propagate it
                    if None in (term2, term3):
                        min_time_alpha_2.append(None)
                    else:
                        min_time_alpha_2.append(max((term1, term2, term3)))

            # If any recursive call returns None, propagate it
            if None in min_time_alpha_2:
                min_time_alpha_1.append(None)
            else:
                min_time_alpha_1.append(min(min_time_alpha_2))

    # If any recursive call returns None, propagate it
    if None in min_time_alpha_1:
        return None
    else:
        return min(min_time_alpha_1)


# Recursive function for Ts(c)
def T_rec_2(c, s):

    
    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    #c = np.round(c, precision)
    #print(f'c: {c}, s: {s}, alpha_1_values: {alpha_1_values}')
    for alpha_1 in alpha_1_values:
        # print(f'alpha_1: {alpha_1}, c: {c}')
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
            #print('here 1')
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c,alpha_1,atol)
            
            alpha_2_values = np.round(np.arange((alpha_1_opt+alpha_steps), c/2,alpha_steps), precision)
            min_time_alpha_2 = []
            # print(f'alpha_1: {alpha_1}, alpha_1_opt: {alpha_1_opt}, c: {c}, alpha_2_values: {alpha_2_values}')
            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:
                    term1 = f(c,c*alpha_1_opt)
                    #print(f'alpha_2_opt - alpha_1_opt: {alpha_2_opt} - {alpha_1_opt} =  {alpha_2_opt - alpha_1_opt}')
                    term2 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + T_rec_2((alpha_2_opt - alpha_1_opt)*c,s)

                    term3 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + T_rec_2((0.5 - alpha_2_opt)*c,s)
                    min_time_alpha_2.append(max((term1,term2,term3)))
            min_time_alpha_1.append(min(min_time_alpha_2))

    opt_time =  min(min_time_alpha_1)


    return opt_time



# Recursive function for Ts(c)
def T_rec_3(c, s):

    
    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    #c = np.round(c, precision)
    #print(f'c: {c}, s: {s}, alpha_1_values: {alpha_1_values}')
    for alpha_1 in alpha_1_values:
        # print(f'alpha_1: {alpha_1}, c: {c}')
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
            #print('here 1')
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c,alpha_1,atol)
            
            alpha_2_values = np.round(np.arange((alpha_1_opt+alpha_steps), c/2,alpha_steps), precision)
            min_time_alpha_2 = []
            # print(f'alpha_1: {alpha_1}, alpha_1_opt: {alpha_1_opt}, c: {c}, alpha_2_values: {alpha_2_values}')
            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:

                    alpha_3_values = np.round(np.arange((alpha_2_opt+alpha_steps),c/2,alpha_steps), precision)
                    min_time_alpha_3 = []
                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append(c)
                    else:
                        for alpha_3_opt in alpha_3_values:
                            term1 = f(c,c*alpha_1_opt)
                            term2 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + T_rec_3((alpha_2_opt - alpha_1_opt)*c,s)

                            term3 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + T_rec_3((0.5 - alpha_2_opt)*c,s)
                            term4 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + f(c*alpha_3_opt,c*alpha_2_opt)/2 + T_rec_3((alpha_3_opt-alpha_2_opt)*c,s)
                            min_time_alpha_3.append(max((term1,term2,term3,term4)))
                    min_time_alpha_2.append(min(min_time_alpha_3))
            min_time_alpha_1.append(min(min_time_alpha_2))

    opt_time =  min(min_time_alpha_1)


    return opt_time

import numpy as np

# Recursive function for Ts(c) with maximum recursion limit
def T_s_fixed_rec_3_levels(c, s, max_recursion, current_recursion=0):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > max_recursion:
        return None

    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    
    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            
            alpha_2_values = np.round(np.arange((alpha_1_opt+alpha_steps), c/2, alpha_steps), precision)
            min_time_alpha_2 = []
            
            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:
                    alpha_3_values = np.round(np.arange((alpha_2_opt+alpha_steps), c/2, alpha_steps), precision)
                    min_time_alpha_3 = []
                    
                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append(c)
                    else:
                        for alpha_3_opt in alpha_3_values:
                            term1 = f(c, c*alpha_1_opt)
                            term2 = f(c, c*0.5)/2 + f(c*0.5, c*alpha_2_opt)/2 + f(c*alpha_2_opt, c*alpha_1_opt)/2 + T_s_fixed_rec_3_levels((alpha_2_opt - alpha_1_opt)*c, s, max_recursion, current_recursion + 1)
                            term3 = f(c, c*0.5)/2 + f(c*0.5, c*alpha_2_opt)/2 + T_s_fixed_rec_3_levels((0.5 - alpha_2_opt)*c, s, max_recursion, current_recursion + 1)
                            term4 = f(c, c*0.5)/2 + f(c*0.5, c*alpha_2_opt)/2 + f(c*alpha_2_opt, c*alpha_1_opt)/2 + f(c*alpha_3_opt, c*alpha_2_opt)/2 + T_s_fixed_rec_3_levels((alpha_3_opt - alpha_2_opt)*c, s, max_recursion, current_recursion + 1)
                            
                            # If any recursive call returns None, propagate it
                            if None in (term2, term3, term4):
                                min_time_alpha_3.append(None)
                            else:
                                min_time_alpha_3.append(max((term1, term2, term3, term4)))
                    
                    # If any recursive call returns None, propagate it
                    if None in min_time_alpha_3:
                        min_time_alpha_2.append(None)
                    else:
                        min_time_alpha_2.append(min(min_time_alpha_3))
            
            # If any recursive call returns None, propagate it
            if None in min_time_alpha_2:
                min_time_alpha_1.append(None)
            else:
                min_time_alpha_1.append(min(min_time_alpha_2))

    # If any recursive call returns None, propagate it
    if None in min_time_alpha_1:
        return None
    else:
        return min(min_time_alpha_1)


def T_rec_4(c, s):

    
    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    #c = np.round(c, precision)
    #print(f'c: {c}, s: {s}, alpha_1_values: {alpha_1_values}')
    for alpha_1 in alpha_1_values:
        # print(f'alpha_1: {alpha_1}, c: {c}')
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
            #print('here 1')
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c,alpha_1,atol)
            
            alpha_2_values = np.round(np.arange((alpha_1_opt+alpha_steps), c/2,alpha_steps), precision)
            min_time_alpha_2 = []
            # print(f'alpha_1: {alpha_1}, alpha_1_opt: {alpha_1_opt}, c: {c}, alpha_2_values: {alpha_2_values}')
            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:

                    alpha_3_values = np.round(np.arange((alpha_2_opt+alpha_steps),c/2,alpha_steps), precision)
                    min_time_alpha_3 = []
                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append(c)
                    else:
                        for alpha_3_opt in alpha_3_values:
                            alpha_4_values = np.round(np.arange((alpha_3_opt+alpha_steps),c/2,alpha_steps), precision)
                            min_time_alpha_4 = []
                            if len(alpha_4_values) == 0:
                                min_time_alpha_4.append(c)
                            else:
                                for alpha_4_opt in alpha_4_values:
                                    term1 = f(c,c*alpha_1_opt)
                                    term3 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + T_rec_4((0.5 - alpha_2_opt)*c,s)
                                    term2 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + T_rec_4((alpha_2_opt - alpha_1_opt)*c,s)

                                    
                                    term4 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + f(c*alpha_3_opt,c*alpha_2_opt)/2 + T_rec_4((alpha_3_opt-alpha_2_opt)*c,s)
                                    term5 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + f(c*alpha_3_opt,c*alpha_2_opt)/2 + f(c*alpha_4_opt,c*alpha_3_opt)/2 + T_rec_4((alpha_4_opt-alpha_3_opt)*c,s)
                                    min_time_alpha_4.append(max((term1,term2,term3,term4,term5)))
                            min_time_alpha_3.append(min(min_time_alpha_4))
                    min_time_alpha_2.append(min(min_time_alpha_3))
            min_time_alpha_1.append(min(min_time_alpha_2))

    opt_time =  min(min_time_alpha_1)


    return opt_time

import numpy as np

# Recursive function for Ts(c) with maximum recursion limit
def T_s_fixed_rec_4_levels(c, s, max_recursion, current_recursion=0):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > max_recursion:
        return None

    alpha_1_values = np.round(np.arange(0, (s + alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)

            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:
                    alpha_3_values = np.round(np.arange((alpha_2_opt + alpha_steps), c / 2, alpha_steps), precision)
                    min_time_alpha_3 = []

                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append(c)
                    else:
                        for alpha_3_opt in alpha_3_values:
                            alpha_4_values = np.round(np.arange((alpha_3_opt + alpha_steps), c / 2, alpha_steps), precision)
                            min_time_alpha_4 = []

                            if len(alpha_4_values) == 0:
                                min_time_alpha_4.append(c)
                            else:
                                for alpha_4_opt in alpha_4_values:
                                    term1 = f(c, c * alpha_1_opt)
                                    term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_rec_4_serial((0.5 - alpha_2_opt) * c, s, max_recursion, current_recursion + 1)
                                    term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_s_fixed_rec_4_levels((alpha_2_opt - alpha_1_opt) * c, s, max_recursion, current_recursion + 1)
                                    term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + T_s_fixed_rec_4_levels((alpha_3_opt - alpha_2_opt) * c, s, max_recursion, current_recursion + 1)
                                    term5 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + T_s_fixed_rec_4_levels((alpha_4_opt - alpha_3_opt) * c, s, max_recursion, current_recursion + 1)

                                    # If any recursive call returns None, propagate it
                                    if None in (term2, term3, term4, term5):
                                        min_time_alpha_4.append(None)
                                    else:
                                        min_time_alpha_4.append(max((term1, term2, term3, term4, term5)))

                            # If any recursive call returns None, propagate it
                            if None in min_time_alpha_4:
                                min_time_alpha_3.append(None)
                            else:
                                min_time_alpha_3.append(min(min_time_alpha_4))

                    # If any recursive call returns None, propagate it
                    if None in min_time_alpha_3:
                        min_time_alpha_2.append(None)
                    else:
                        min_time_alpha_2.append(min(min_time_alpha_3))

            # If any recursive call returns None, propagate it
            if None in min_time_alpha_2:
                min_time_alpha_1.append(None)
            else:
                min_time_alpha_1.append(min(min_time_alpha_2))

    # If any recursive call returns None, propagate it
    if None in min_time_alpha_1:
        return None
    else:
        return min(min_time_alpha_1)


@lru_cache(maxsize=None)
def T_rec_5(c, s):

    
    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    #c = np.round(c, precision)
    #print(f'c: {c}, s: {s}, alpha_1_values: {alpha_1_values}')
    for alpha_1 in alpha_1_values:
        # print(f'alpha_1: {alpha_1}, c: {c}')
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
            #print('here 1')
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c,alpha_1,atol)
            
            alpha_2_values = np.round(np.arange((alpha_1_opt+alpha_steps), c/2,alpha_steps), precision)
            min_time_alpha_2 = []
            # print(f'alpha_1: {alpha_1}, alpha_1_opt: {alpha_1_opt}, c: {c}, alpha_2_values: {alpha_2_values}')
            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:

                    alpha_3_values = np.round(np.arange((alpha_2_opt+alpha_steps),c/2,alpha_steps), precision)
                    min_time_alpha_3 = []
                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append(c)
                    else:
                        for alpha_3_opt in alpha_3_values:
                            alpha_4_values = np.round(np.arange((alpha_3_opt+alpha_steps),c/2,alpha_steps), precision)
                            min_time_alpha_4 = []
                            if len(alpha_4_values) == 0:
                                min_time_alpha_4.append(c)
                            else:
                                for alpha_4_opt in alpha_4_values:
                                    alpha_5_values = np.round(np.arange((alpha_4_opt+alpha_steps),c/2,alpha_steps), precision)
                                    min_time_alpha_5 = []
                                    if len(alpha_5_values) == 0:
                                        min_time_alpha_5.append(c)
                                    else:
                                        for alpha_5_opt in alpha_5_values:
                                            term1 = f(c,c*alpha_1_opt)
                                            term3 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + T_rec_5((0.5 - alpha_2_opt)*c,s)
                                            term2 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + T_rec_5((alpha_2_opt - alpha_1_opt)*c,s)

                                            
                                            term4 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + f(c*alpha_3_opt,c*alpha_2_opt)/2 + T_rec_5((alpha_3_opt-alpha_2_opt)*c,s)
                                            term5 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + f(c*alpha_3_opt,c*alpha_2_opt)/2 + f(c*alpha_4_opt,c*alpha_3_opt)/2 + T_rec_5((alpha_4_opt-alpha_3_opt)*c,s)
                                            term6 = f(c,c*0.5)/2 + f(c*0.5,c*alpha_2_opt)/2 + f(c*alpha_2_opt,c*alpha_1_opt)/2 + f(c*alpha_3_opt,c*alpha_2_opt)/2 + f(c*alpha_4_opt,c*alpha_3_opt)/2 + f(c*alpha_5_opt,c*alpha_4_opt)/2 + T_rec_5((alpha_5_opt-alpha_4_opt)*c,s)
                                            min_time_alpha_5.append(max((term1,term2,term3,term4,term5,term6)))
                                    min_time_alpha_4.append(min(min_time_alpha_5))
                            min_time_alpha_3.append(min(min_time_alpha_4))
                    min_time_alpha_2.append(min(min_time_alpha_3))
            min_time_alpha_1.append(min(min_time_alpha_2))

    opt_time =  min(min_time_alpha_1)


    return opt_time


import numpy as np

# Recursive function for Ts(c) with maximum recursion limit
def T_s_fixed_rec_5_levels(c, s, max_recursion, current_recursion=0):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > max_recursion:
        return None

    alpha_1_values = np.round(np.arange(0, (s + alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append(1)
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append(c)
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)

            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append(c)
            else:
                for alpha_2_opt in alpha_2_values:
                    alpha_3_values = np.round(np.arange((alpha_2_opt + alpha_steps), c / 2, alpha_steps), precision)
                    min_time_alpha_3 = []

                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append(c)
                    else:
                        for alpha_3_opt in alpha_3_values:
                            alpha_4_values = np.round(np.arange((alpha_3_opt + alpha_steps), c / 2, alpha_steps), precision)
                            min_time_alpha_4 = []

                            if len(alpha_4_values) == 0:
                                min_time_alpha_4.append(c)
                            else:
                                for alpha_4_opt in alpha_4_values:
                                    alpha_5_values = np.round(np.arange((alpha_4_opt + alpha_steps), c / 2, alpha_steps), precision)
                                    min_time_alpha_5 = []

                                    if len(alpha_5_values) == 0:
                                        min_time_alpha_5.append(c)
                                    else:
                                        for alpha_5_opt in alpha_5_values:
                                            term1 = f(c, c * alpha_1_opt)
                                            term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_s_fixed_rec_5_levels((0.5 - alpha_2_opt) * c, s, max_recursion, current_recursion + 1)
                                            term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_s_fixed_rec_5_levels((alpha_2_opt - alpha_1_opt) * c, s, max_recursion, current_recursion + 1)
                                            term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + T_s_fixed_rec_5_levels((alpha_3_opt - alpha_2_opt) * c, s, max_recursion, current_recursion + 1)
                                            term5 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + T_s_fixed_rec_5_levels((alpha_4_opt - alpha_3_opt) * c, s, max_recursion, current_recursion + 1)
                                            term6 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + f(c * alpha_5_opt, c * alpha_4_opt) / 2 + T_s_fixed_rec_5_levels((alpha_5_opt - alpha_4_opt) * c, s, max_recursion, current_recursion + 1)

                                            # If any recursive call returns None, propagate it
                                            if None in (term2, term3, term4, term5, term6):
                                                min_time_alpha_5.append(None)
                                            else:
                                                min_time_alpha_5.append(max((term1, term2, term3, term4, term5, term6)))

                                    # If any recursive call returns None, propagate it
                                    if None in min_time_alpha_5:
                                        min_time_alpha_4.append(None)
                                    else:
                                        min_time_alpha_4.append(min(min_time_alpha_5))

                            # If any recursive call returns None, propagate it
                            if None in min_time_alpha_4:
                                min_time_alpha_3.append(None)
                            else:
                                min_time_alpha_3.append(min(min_time_alpha_4))

                    # If any recursive call returns None, propagate it
                    if None in min_time_alpha_3:
                        min_time_alpha_2.append(None)
                    else:
                        min_time_alpha_2.append(min(min_time_alpha_3))

            # If any recursive call returns None, propagate it
            if None in min_time_alpha_2:
                min_time_alpha_1.append(None)
            else:
                min_time_alpha_1.append(min(min_time_alpha_2))

    # If any recursive call returns None, propagate it
    if None in min_time_alpha_1:
        return None
    else:
        return min(min_time_alpha_1)



def plot_function(F_s):
    # Extract the keys (s values) and corresponding F_s[s][1] values
    x_values = sorted(F_s.keys())  # Sort the s values for proper plotting
    y_values = [max(F_s[s],s) for s in x_values]  # Extract F_s[s][1] for each s
    y_values_T_s = [F_s[s] for s in x_values]
    y_values_s = [s for s in x_values]


        # Scatter and line plot of running times
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values_s, color="purple", label="s (line)")
    plt.plot(x_values, y_values_T_s, color="green", label="T_s(1) (line)")
    plt.scatter(x_values, y_values, color="b", label="max(T_s(1),s) (scatter)")
    #plt.plot(x_values, y_values, color="blue", linestyle="--", label="max(T_s(1),s) (line)")
    plt.xlabel("s")
    plt.ylabel(f"T")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Running_time_s_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}.pdf")
    plt.close()


def plot_function_recursive(F_s):
    # Extract the keys (s values) and corresponding F_s[s][1] values
    x_values = sorted(F_s.keys())  # Sort the s values for proper plotting
    y_values = [max(F_s[s],s) for s in x_values]  # Extract F_s[s][1] for each s
    y_values_T_s = [F_s[s] for s in x_values]
    y_values_s = [s for s in x_values]


        # Scatter and line plot of running times
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values_s, color="purple", label="s (line)")
    plt.plot(x_values, y_values_T_s, color="green", label="T_s(1) (line)")
    plt.scatter(x_values, y_values, color="b", label="max(T_s(1),s) (scatter)")
    #plt.plot(x_values, y_values, color="blue", linestyle="--", label="max(T_s(1),s) (line)")
    plt.xlabel("s")
    plt.ylabel(f"T")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Running_time_s_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}.pdf")
    plt.close()

def comparison_plot(F_s_par):
    
# Parameters

    k_values = np.arange(0, 0.5 + 0.01, 0.01)  # k from 0 to 0.5 with precision 0.01

    # Functions (with n = 1)

    time_values = (1.12717 ** k_values) * (1.82653 ** 1)

    space_values = (0.79703 ** k_values) * (1.82653 ** 1)
    x_values = [2 ** s for s in sorted(F_s_par.keys())]  # Sort the s values for proper plotting
    y_values = [2 ** max(F_s_par[s],s) for s in sorted(F_s_par.keys())]  # Extract F_s[s][1] for each s


        # Scatter and line plot of running times
    plt.figure(figsize=(10, 6))

    plt.scatter(x_values, y_values, color="b", label="max(T_s(1),s) (scatter)")
    plt.plot(x_values, y_values, color="blue", linestyle="--", label="max(T_s(1),s) (line)")



    plt.plot(space_values, time_values, label='Time-Space Complexity (line)',linestyle="--", color='purple')

    plt.scatter(space_values, time_values, color="purple", label="Time-Space Complexity (scatter)")


    plt.xlabel("Space Complexity")

    plt.ylabel("Time Complexity")
    plt.grid(True)

    plt.legend()

    plt.savefig(f"Running_time_s_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}_comparison.pdf")
    plt.close()


# Wrapper function to allow parallel processing of T_s_fixed_rec
def process_s(s):
    # get the start time
    # st = time.time()
    # start_t = time.localtime()
    
    # print(f'\n=========================\n')
    # print(f'OUTER s: {s} - Start: {time.strftime("%H:%M:%S", start_t)}\n')
    if levels == 1:
        result = T_rec_1(1,s)
    if levels==2:
        result = T_rec_2(1,s)
    if levels==3:
        result = T_rec_3(1,s)
    if levels==4:
        result = T_rec_4(1,s)
    if levels==5:
        result = T_rec_5(1,s)
    # get the end time
    # et = time.time()
    # elapsed_time = et - st
    # end_t = time.localtime()
    
    # print(f'OUTER s: {s} - End: {time.strftime("%H:%M:%S", end_t)} - In time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')
    # print(f'\n=========================\n')
    
    return s, result

# Main parallelized loop
def parallelized_recursion(s_values):
    max_cores = multiprocessing.cpu_count()  # Number of available CPU cores

    # Parallel processing for each s in s_values
    
    with ProcessPoolExecutor() as executor:
        # Map arguments for each s value
        tasks = [executor.submit(process_s, s) for s in s_values]
        total_tasks = len(tasks)
        completed_tasks = 0
        
        # Retrieve results as they complete
        for future in as_completed(tasks):
            s, result = future.result()
            F_s[s] = result
            completed_tasks += 1
            print(f"Task completed {s}: {completed_tasks}/{total_tasks}")
    

    

    return F_s

def serial_recursion(s_values):

    stAll = time.time()
    startAll_t = time.localtime()
    #print(f'BEGIN - Start: {time.strftime("%H:%M:%S", startAll_t)}\n')
    for s in s_values:
        # get the start time
        st = time.time()
        start_t = time.localtime()

        print(f'\n=========================\n')

        print(f'OUTER s: {s} - Start: {time.strftime("%H:%M:%S", start_t)}\n')
        if levels == 1:


            F_s[s] = T_rec_1(1,s)
        if levels==2:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_[i][s] = T_s_fixed_rec_2_levels(1,s, i, 1)
            else:
                F_s[s] = T_rec_2(1,s)
        if levels==3:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_[i][s] = T_s_fixed_rec_3_levels(1,s, i, 1)
            else:
                F_s[s] = T_rec_3(1,s)
        if levels==4:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_[i][s] = T_s_fixed_rec_4_levels(1,s, i, 1)
            else:
                F_s[s] = T_rec_4(1,s)
        if levels==5:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_[i][s] = T_s_fixed_rec_5_levels(1,s, i, 1)
            else:
                F_s[s] = T_rec_5(1,s)
        print(F_s[s])
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st
        end_t = time.localtime()


        #print(f'x_values: {x_values}\n')
        #print(f'y_values: {y_values}\n')

        print(f'OUTER s: {s} - End: {time.strftime("%H:%M:%S", end_t)} - In time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n')
        print(f'\n=========================\n')

    etAll = time.time()

        # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    #print(f'FINISH - End: {time.strftime("%H:%M:%S", endAll_t)} - In time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_timeAll))}\n')


    return F_s


def main():


    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'START - Start: {time.strftime("%H:%M:%S", startAll_t)}\n')

    if parallel:
        F_s = parallelized_recursion(s_values)
    else:
        F_s = serial_recursion(s_values)

    etAll = time.time()

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'FINISH - End: {time.strftime("%H:%M:%S", endAll_t)} - In time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')
    #print(F_s)
    plot_function(F_s)
    comparison_plot(F_s)
    with open(f'F_s_values_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}.txt', 'w') as file:
        file.write(json.dumps(F_s)) # use `json.loads` to do the reverse
    print("values saved")

if __name__ == "__main__":
    main()