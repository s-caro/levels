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
@lru_cache(maxsize=None)
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
alpha_steps = 1/100
precision = 6
atol = 10**(-precision)

# Initialize dictionaries to store results

s_values = np.round(np.arange(0, (1+alpha_steps), alpha_steps), precision)


# Initialize a dictionary to store cached alpha_c values
max_recursion = 7
alpha_c_cache = {}

# Define a global or local dictionary for memoization
term2_memo = {}

# Initialize caches
term2_cache = {}
t_rec_cache = {}
 
F_s = {}  # Memoization array for each fixed s
F_s_rec = {rec : {} for rec in range(max_recursion+1)}

levels = 3
parallel = True
inside_parallel = False
max_rec = False

@lru_cache(maxsize=None)
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
@lru_cache(maxsize=None)
def T_s_fixed_rec_1_levels(c, s, max_recursion, current_recursion=0):

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
            term1 = f(c, c / 2) / 2
            term2 = f(c / 2, alpha_1_opt * c) / 2

            # Recursive call with incremented recursion depth
            term3 = T_s_fixed_rec_1_levels(c / 2 - alpha_1_opt * c, s, max_recursion, current_recursion + 1)

            # If the recursive call returns None, propagate it
            if term3 is None:
                min_time_alpha_1.append(None)
            else:
                min_time_alpha_1.append(max((term1 + term2 + term3), alpha_1))

    # If any recursive call returned None, propagate it
    if None in min_time_alpha_1:
        return None
    else:
        return min(min_time_alpha_1)


@lru_cache(maxsize=None)
def T_s_fixed_rec_2_levels(c, s, allowed_recu, current_recursion=1):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > allowed_recu:
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
                    rec_term_2 = T_s_fixed_rec_2_levels((alpha_2_opt - alpha_1_opt) * c, s, allowed_recu, current_recursion + 1)
                    rec_term_3 =  T_s_fixed_rec_2_levels((0.5 - alpha_2_opt) * c, s, allowed_recu, current_recursion + 1)
                    

                    # If any recursive call returns None, propagate it
                    if None in (rec_term_2, rec_term_3):
                        min_time_alpha_2.append(None)
                    else:
                        term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + rec_term_2
                        term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + rec_term_3
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


@lru_cache(maxsize=None)
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



@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
def T_s_fixed_rec_3_levels(c, s, allowed_recu, current_recursion=1):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > allowed_recu:
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
                            rec_term_2 = T_s_fixed_rec_3_levels((alpha_2_opt - alpha_1_opt)*c, s, allowed_recu, current_recursion + 1)
                            rec_term_3 =T_s_fixed_rec_3_levels((0.5 - alpha_2_opt)*c, s, allowed_recu, current_recursion + 1)
                            rec_term_4 = T_s_fixed_rec_3_levels((alpha_3_opt - alpha_2_opt)*c, s, allowed_recu, current_recursion + 1)
                            
                            
                            # If any recursive call returns None, propagate it
                            if None in (rec_term_2, rec_term_3, rec_term_4):
                                min_time_alpha_3.append(None)
                            else:
                                term2 = f(c, c*0.5)/2 + f(c*0.5, c*alpha_2_opt)/2 + f(c*alpha_2_opt, c*alpha_1_opt)/2 + rec_term_2
                                term3 = f(c, c*0.5)/2 + f(c*0.5, c*alpha_2_opt)/2 + rec_term_3
                                term4 = f(c, c*0.5)/2 + f(c*0.5, c*alpha_2_opt)/2 + f(c*alpha_2_opt, c*alpha_1_opt)/2 + f(c*alpha_3_opt, c*alpha_2_opt)/2 + rec_term_4
                                 
                                
                                 
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
def T_s_fixed_rec_4_levels(c, s, allowed_recu, current_recursion=1):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > allowed_recu:
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
                                    rec_term_2 = T_s_fixed_rec_4_levels((0.5 - alpha_2_opt) * c, s, allowed_recu, current_recursion + 1)
                                    rec_term_3 = T_s_fixed_rec_4_levels((alpha_2_opt - alpha_1_opt) * c, s, allowed_recu, current_recursion + 1)
                                    rec_term_4 =T_s_fixed_rec_4_levels((alpha_3_opt - alpha_2_opt) * c, s, allowed_recu, current_recursion + 1)
                                    rec_term_5 =T_s_fixed_rec_4_levels((alpha_4_opt - alpha_3_opt) * c, s, allowed_recu, current_recursion + 1)
                                    
                                    # If any recursive call returns None, propagate it
                                    if None in (rec_term_2, rec_term_3, rec_term_4, rec_term_5):
                                        min_time_alpha_4.append(None)
                                    else:
                                        term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + rec_term_2
                                        term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + rec_term_3
                                        term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + rec_term_4
                                        term5 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + rec_term_5

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
def T_s_fixed_rec_5_levels(c, s, allowed_recu, current_recursion=1):

    # Check if the current recursion depth exceeds the maximum allowed
    if current_recursion > allowed_recu:
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
                                            term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_s_fixed_rec_5_levels((0.5 - alpha_2_opt) * c, s, allowed_recu, current_recursion + 1)
                                            term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_s_fixed_rec_5_levels((alpha_2_opt - alpha_1_opt) * c, s, allowed_recu, current_recursion + 1)
                                            term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + T_s_fixed_rec_5_levels((alpha_3_opt - alpha_2_opt) * c, s, allowed_recu, current_recursion + 1)
                                            term5 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + T_s_fixed_rec_5_levels((alpha_4_opt - alpha_3_opt) * c, s, allowed_recu, current_recursion + 1)
                                            term6 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + f(c * alpha_5_opt, c * alpha_4_opt) / 2 + T_s_fixed_rec_5_levels((alpha_5_opt - alpha_4_opt) * c, s, allowed_recu, current_recursion + 1)

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
    plt.savefig(f"{levels}\\free recursion\\Running_time_s_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}.pdf")
    plt.close()


def plot_function_recursive(F_s_rec):
    # Extract the keys (s values) and corresponding F_s[s][1] values
    x_values = sorted(s_values)
    colormap = plt.get_cmap("viridis", (max_recursion+1))
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, x_values, color = "purple", alpha=0.3, label="s (line)")
    for rec in range(1,max_recursion+1):
         # Sort the s values for proper plotting
        y_values = [max(F_s_rec[rec][s],s) if F_s[r][s] is not None else None for s in x_values]  # Extract F_s[s][1] for each s
        y_values_T_s = [F_s_rec[rec][s] for s in x_values]
        y_values_s = [s for s in x_values]


            # Scatter and line plot of running times
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values_s, color="purple", label="s (line)")
        plt.plot(x_values, y_values_T_s, color="green", label="T_s(1) (line)")
        plt.scatter(x_values, y_values, color="b", label="max(T_s(1),s) (scatter)")
        plt.scatter(x_values, y_values, s=20, color=colormap(r), alpha=((1/(max_recursion+1)) * (rec+1)),label=f"lev: {rec}")
        plt.plot(x_values, y_values, color=colormap(rec), linestyle="--", alpha=((1/(max_recursion+1)) * (rec+1)), label=f"lev: {rec}")

          # Sort the s values for proper plotting
          # Extract F_s[s][1] for each s


            # Scatter and line plot of running times
        

        
    #plt.plot(x_values, y_values, color="blue", linestyle="--", label="max(T_s(1),s) (line)")
    plt.xlabel("s")
    plt.ylabel(f"T")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{levels}\\max recursion\\Running_time_s_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}_max_rec_{max_rec}_{max_recursion}.pdf")
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

    plt.savefig(f"{levels}\\free recursion\\Running_time_s_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}_comparison.pdf")
    plt.close()


# Wrapper function to allow parallel processing of T_s_fixed_rec
def process_s(s):

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

    
    return s, result

# Main parallelized loop
def parallelized_recursion(s_values):
    max_cores = multiprocessing.cpu_count()  # Number of available CPU cores

    # Parallel processing for each s in s_values
    
    with ProcessPoolExecutor(20) as executor:
        # Map arguments for each s value
        tasks = [executor.submit(process_s, s) for s in s_values]
        total_tasks = len(tasks)
        completed_tasks = 0
        
        # Retrieve results as they complete
        for future in as_completed(tasks):
            s, result = future.result()
            F_s[s] = result
            completed_tasks += 1
            
            print(f"Task completed {s}: {completed_tasks}/{total_tasks} - Time: {time.strftime('%d %m, %H:%M:%S', time.localtime())}")
    

    

    return F_s

def serial_recursion(s_values):

    stAll = time.time()
    startAll_t = time.localtime()
    #print(f'BEGIN - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')
    for s in s_values:
        # get the start time
        st = time.time()
        start_t = time.localtime()

        print(f'\n=========================\n')

        print(f'OUTER s: {s} - Start: {time.strftime("%d day, %H:%M:%S", start_t)}\n')
        if levels == 1:
            if max_rec:
                for i in range(1,max_recursion+1):
                    F_s_rec[i][s] = T_s_fixed_rec_1_levels(1,s,i,1)
                    print(f"done rec: {i}\n")
            else:

                F_s[s] = T_rec_1(1,s)
        if levels==2:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_s_rec[i][s] = T_s_fixed_rec_2_levels(1,s, i, 1)
                    print(f"done rec: {i}\n")
            else:
                F_s[s] = T_rec_2(1,s)
        if levels==3:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_s_rec[i][s] = T_s_fixed_rec_3_levels(1,s, i, 1)
                    print(f"done rec: {i}\n")
            else:
                F_s[s] = T_rec_3(1,s)
        if levels==4:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_s_rec[i][s] = T_s_fixed_rec_4_levels(1,s, i, 1)
                    print(f"done rec: {i}\n")
            else:
                F_s[s] = T_rec_4(1,s)
        if levels==5:
            if max_rec:
                for i in range(1, max_recursion+1):
                    F_s_rec[i][s] = T_s_fixed_rec_5_levels(1,s, i, 1)
                    print(f"done rec: {i}\n")
            else:
                F_s[s] = T_rec_5(1,s)
        #print(F_s[s])
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st
        end_t = time.localtime()


        #print(f'x_values: {x_values}\n')
        #print(f'y_values: {y_values}\n')

        print(f'OUTER s: {s} - End: {time.strftime("%d day, %H:%M:%S", end_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_time))}\n')
        print(f'\n=========================\n')

    etAll = time.time()

        # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    #print(f'FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')


    return F_s

# Wrapper function to allow parallel processing of T_s_fixed_rec
def process_s_max_rec(s, rec):

    if levels == 1:
        result = T_s_fixed_rec_1_levels(1,s,rec,1)
    if levels==2:
        result = T_s_fixed_rec_2_levels(1,s, rec, 1)
    if levels==3:
        result = T_s_fixed_rec_3_levels(1,s, rec, 1)
    if levels==4:
        result = T_s_fixed_rec_4_levels(1,s, rec, 1)
    if levels==5:
        result = T_s_fixed_rec_5_levels(1,s, rec, 1)

    
    return s, result

# Main parallelized loop
def parallelized_recursion_max_rec(s_values, rec):
    max_cores = multiprocessing.cpu_count()  # Number of available CPU cores

    # Parallel processing for each s in s_values
    
    with ProcessPoolExecutor() as executor:
        # Map arguments for each s value
        tasks = [executor.submit(process_s_max_rec, s, rec) for s in s_values]
        total_tasks = len(tasks)
        completed_tasks = 0
        
        # Retrieve results as they complete
        for future in as_completed(tasks):
            s, result = future.result()
            F_s_rec[rec][s] = result
            completed_tasks += 1
            print(f"Rec: {rec} - Task completed {s}: {completed_tasks}/{total_tasks} - Time: {time.strftime('%d %m, %H:%M:%S', time.localtime())}")
    

    

    return F_s



def main():



    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'level: {levels} - START - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')
    if parallel and max_rec:
        for i in range(1, max_recursion+1):
            parallelized_recursion_max_rec(s_values, i)
           
    if parallel and not max_rec:
        F_s = parallelized_recursion(s_values)
    elif not parallel:
        F_s = serial_recursion(s_values)

    etAll = time.time()

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'level: {levels} - FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')
    #print(F_s)
    if max_rec:
        plot_function_recursive(F_s_rec)
        with open(f'{levels}\\max recursion\\F_s_rec_values_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}_max_rec_{max_rec}_{max_recursion}.txt', 'w') as file:
            file.write(json.dumps(F_s_rec)) # use `json.loads` to do the reverse
        print("values saved")
    else:
        plot_function(F_s)
        comparison_plot(F_s)
        with open(f'{levels}\\free recursion\\F_s_values_{alpha_steps}_precision_{atol}_parallel_{parallel}_inside_{inside_parallel}_{levels}.txt', 'w') as file:
            file.write(json.dumps(F_s)) # use `json.loads` to do the reverse
        print("values saved")

if __name__ == "__main__":
    main()