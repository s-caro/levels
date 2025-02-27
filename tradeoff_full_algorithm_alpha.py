
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

# Binary entropy function H(x)
def H(x):
    if x == 0 or x == 1:
        return 0
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))

# Function f(a, b) with small epsilon to handle very small values of a
def f(a, b):
    return H(b / a) * a

# Optimized function to find alpha_c using ternary search with caching
@lru_cache(maxsize=None)
def find_alpha_c(c, s, atol):
    if (s, c) in alpha_c_cache:
        return alpha_c_cache[(s, c)]
    left, right = 0, 0.5
    while True:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        f_m1 = f(c, m1 * c)
        f_m2 = f(c, m2 * c)
        if np.isclose(f_m1, s, atol=atol):
            alpha_c_cache[(s, c)] = m1
            return m1
        if np.isclose(f_m2, s, atol=atol):
            alpha_c_cache[(s, c)] = m2
            return m2
        if abs(f_m1 - s) < abs(f_m2 - s):
            right = m2
        else:
            left = m1
        if abs(f(c, (left + right) / 2 * c) - s) < atol:
            result = (left + right) / 2
            alpha_c_cache[(s, c)] = result
            return result

alpha_steps = 1/100
precision = 4
atol = 10**(-precision)
alpha_c_cache = {}

F_s = {}



""" @lru_cache(maxsize=None)
def T_rec_1(c, s):
    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    opt_alphas_1 = {}

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append((1, {'alpha_1_opt': alpha_1}))
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append((c, {'alpha_1_opt': alpha_1}))
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            term1 = f(c, c / 2) / 2
            term2 = f(c / 2, alpha_1_opt * c) / 2
            term3, opt_alphas_term3 = T_rec_1(c / 2 - alpha_1_opt * c, s)
            min_time_alpha_1.append((max((term1 + term2 + term3), alpha_1), {'alpha_1_opt': alpha_1_opt} | opt_alphas_term3))

    opt_time, opt_alphas_1 = min(min_time_alpha_1, key=lambda x: x[0])
    return opt_time, opt_alphas_1

@lru_cache(maxsize=None)
def T_rec_2(c, s):
    alpha_1_values = np.round(np.arange(0, (s+alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    opt_alphas_2 = {}

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append((1, {'alpha_1_opt': alpha_1}))
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append((c, {'alpha_1_opt': alpha_1}))
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append((c, {'alpha_1_opt': alpha_1_opt}))
            else:
                for alpha_2_opt in alpha_2_values:
                    term1 = f(c, c * alpha_1_opt)
                    term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_rec_2((alpha_2_opt - alpha_1_opt) * c, s)[0]
                    term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_rec_2((0.5 - alpha_2_opt) * c, s)[0]
                    min_time_alpha_2.append((max((term1, term2, term3)), {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt}))

            min_time_alpha_1.append(min(min_time_alpha_2, key=lambda x: x[0]))

    opt_time, opt_alphas_2 = min(min_time_alpha_1, key=lambda x: x[0])
    return opt_time, opt_alphas_2

@lru_cache(maxsize=None)
def T_rec_3(c, s):
    alpha_1_values = np.round(np.arange(0, (s + alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    opt_alphas_3 = {}

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append((1, {'alpha_1_opt': alpha_1}))
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append((c, {'alpha_1_opt': alpha_1}))
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append((c, {'alpha_1_opt': alpha_1_opt}))
            else:
                for alpha_2_opt in alpha_2_values:
                    alpha_3_values = np.round(np.arange((alpha_2_opt + alpha_steps), c / 2, alpha_steps), precision)
                    min_time_alpha_3 = []

                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append((c, {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt}))
                    else:
                        for alpha_3_opt in alpha_3_values:
                            term1 = f(c, c * alpha_1_opt)
                            term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_rec_2((alpha_2_opt - alpha_1_opt) * c, s)[0]
                            term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_rec_2((0.5 - alpha_2_opt) * c, s)[0]
                            term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + T_rec_3((alpha_3_opt - alpha_2_opt) * c, s)[0]
                            min_time_alpha_3.append((max((term1, term2, term3, term4)), {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt, 'alpha_3_opt': alpha_3_opt}))

                    min_time_alpha_2.append(min(min_time_alpha_3, key=lambda x: x[0]))
            min_time_alpha_1.append(min(min_time_alpha_2, key=lambda x: x[0]))

    opt_time, opt_alphas_3 = min(min_time_alpha_1, key=lambda x: x[0])
    return opt_time, opt_alphas_3

@lru_cache(maxsize=None)
def T_rec_4(c, s):
    alpha_1_values = np.round(np.arange(0, (s + alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    opt_alphas_4 = {}

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append((1, {'alpha_1_opt': alpha_1}))
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append((c, {'alpha_1_opt': alpha_1}))
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append((c, {'alpha_1_opt': alpha_1_opt}))
            else:
                for alpha_2_opt in alpha_2_values:
                    alpha_3_values = np.round(np.arange((alpha_2_opt + alpha_steps), c / 2, alpha_steps), precision)
                    min_time_alpha_3 = []

                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append((c, {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt}))
                    else:
                        for alpha_3_opt in alpha_3_values:
                            alpha_4_values = np.round(np.arange((alpha_3_opt + alpha_steps), c / 2, alpha_steps), precision)
                            min_time_alpha_4 = []

                            if len(alpha_4_values) == 0:
                                min_time_alpha_4.append((c, {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt, 'alpha_3_opt': alpha_3_opt}))
                            else:
                                for alpha_4_opt in alpha_4_values:
                                    term1 = f(c, c * alpha_1_opt)
                                    term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_rec_3((alpha_2_opt - alpha_1_opt) * c, s)[0]
                                    term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_rec_3((0.5 - alpha_2_opt) * c, s)[0]
                                    term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + T_rec_4((alpha_3_opt - alpha_2_opt) * c, s)[0]
                                    term5 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + T_rec_4((alpha_4_opt - alpha_3_opt) * c, s)[0]
                                    min_time_alpha_4.append((max((term1, term2, term3, term4, term5)), {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt, 'alpha_3_opt': alpha_3_opt, 'alpha_4_opt': alpha_4_opt}))

                            min_time_alpha_3.append(min(min_time_alpha_4, key=lambda x: x[0]))
                    min_time_alpha_2.append(min(min_time_alpha_3, key=lambda x: x[0]))
            min_time_alpha_1.append(min(min_time_alpha_2, key=lambda x: x[0]))

    opt_time, opt_alphas_4 = min(min_time_alpha_1, key=lambda x: x[0])
    return opt_time, opt_alphas_4


@lru_cache(maxsize=None)
def T_rec_5(c, s):
    alpha_1_values = np.round(np.arange(0, (s + alpha_steps), alpha_steps), precision)
    min_time_alpha_1 = []
    opt_alphas_5 = {}

    for alpha_1 in alpha_1_values:
        if alpha_1 == 0:
            min_time_alpha_1.append((1, {'alpha_1_opt': alpha_1}))
        elif c <= alpha_1 and alpha_1 != 0:
            min_time_alpha_1.append((c, {'alpha_1_opt': alpha_1}))
        elif c > alpha_1 and alpha_1 != 0:
            alpha_1_opt = find_alpha_c(c, alpha_1, atol)
            alpha_2_values = np.round(np.arange((alpha_1_opt + alpha_steps), c / 2, alpha_steps), precision)
            min_time_alpha_2 = []

            if len(alpha_2_values) == 0:
                min_time_alpha_2.append((c, {'alpha_1_opt': alpha_1_opt}))
            else:
                for alpha_2_opt in alpha_2_values:
                    alpha_3_values = np.round(np.arange((alpha_2_opt + alpha_steps), c / 2, alpha_steps), precision)
                    min_time_alpha_3 = []

                    if len(alpha_3_values) == 0:
                        min_time_alpha_3.append((c, {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt}))
                    else:
                        for alpha_3_opt in alpha_3_values:
                            alpha_4_values = np.round(np.arange((alpha_3_opt + alpha_steps), c / 2, alpha_steps), precision)
                            min_time_alpha_4 = []

                            if len(alpha_4_values) == 0:
                                min_time_alpha_4.append((c, {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt, 'alpha_3_opt': alpha_3_opt}))
                            else:
                                for alpha_4_opt in alpha_4_values:
                                    alpha_5_values = np.round(np.arange((alpha_4_opt + alpha_steps), c / 2, alpha_steps), precision)
                                    min_time_alpha_5 = []

                                    if len(alpha_5_values) == 0:
                                        min_time_alpha_5.append((c, {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt, 'alpha_3_opt': alpha_3_opt, 'alpha_4_opt': alpha_4_opt}))
                                    else:
                                        for alpha_5_opt in alpha_5_values:
                                            term1 = f(c, c * alpha_1_opt)
                                            term2 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + T_rec_4((alpha_2_opt - alpha_1_opt) * c, s)[0]
                                            term3 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + T_rec_4((0.5 - alpha_2_opt) * c, s)[0]
                                            term4 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + T_rec_5((alpha_3_opt - alpha_2_opt) * c, s)[0]
                                            term5 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + T_rec_5((alpha_4_opt - alpha_3_opt) * c, s)[0]
                                            term6 = f(c, c * 0.5) / 2 + f(c * 0.5, c * alpha_2_opt) / 2 + f(c * alpha_2_opt, c * alpha_1_opt) / 2 + f(c * alpha_3_opt, c * alpha_2_opt) / 2 + f(c * alpha_4_opt, c * alpha_3_opt) / 2 + f(c * alpha_5_opt, c * alpha_4_opt) / 2 + T_rec_5((alpha_5_opt - alpha_4_opt) * c, s)[0]
                                            min_time_alpha_5.append((max((term1, term2, term3, term4, term5, term6)), {'alpha_1_opt': alpha_1_opt, 'alpha_2_opt': alpha_2_opt, 'alpha_3_opt': alpha_3_opt, 'alpha_4_opt': alpha_4_opt, 'alpha_5_opt': alpha_5_opt}))

                                    min_time_alpha_4.append(min(min_time_alpha_5, key=lambda x: x[0]))
                            min_time_alpha_3.append(min(min_time_alpha_4, key=lambda x: x[0]))
                    min_time_alpha_2.append(min(min_time_alpha_3, key=lambda x: x[0]))
            min_time_alpha_1.append(min(min_time_alpha_2, key=lambda x: x[0]))

    opt_time, opt_alphas_5 = min(min_time_alpha_1, key=lambda x: x[0])
    return opt_time, opt_alphas_5 """



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


def main():



    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'level: 1 - START - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')

    F_s[1] = T_rec_1(1,1)

    with open(f'Opt_values_and_alpha_values_1_memory.txt', 'a') as file:
        file.write(json.dumps(F_s[1])) # use `json.loads` to do the reverse
    print("values saved")

    etAll = time.time()

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'level: 1 - FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')

    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'level: 2 - START - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')

    F_s[2] = T_rec_2(1,1)

    with open(f'Opt_values_and_alpha_values_1_memory.txt', 'a') as file:
        file.write(json.dumps(F_s[2])) # use `json.loads` to do the reverse
    print("values saved")

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'level: 2 - FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')

    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'level: 3 - START - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')

    F_s[3] = T_rec_3(1,1)

    with open(f'Opt_values_and_alpha_values_1_memory.txt', 'a') as file:
        file.write(json.dumps(F_s[3])) # use `json.loads` to do the reverse
    print("values saved")

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'level: 3 - FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')

    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'level: 4 - START - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')

    F_s[4] = T_rec_4(1,1)

    with open(f'Opt_values_and_alpha_values_1_memory.txt', 'a') as file:
        file.write(json.dumps(F_s[4])) # use `json.loads` to do the reverse
    print("values saved")

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'level: 4 - FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')

    stAll = time.time()
    startAll_t = time.localtime()
    # Run the parallelized version
    print(f'\n=========================\n')
    print(f'level: 5 - START - Start: {time.strftime("%d day, %H:%M:%S", startAll_t)}\n')

    F_s[5] = T_rec_5(1,1)

    with open(f'Opt_values_and_alpha_values_1_memory.txt', 'a') as file:
        file.write(json.dumps(F_s[5])) # use `json.loads` to do the reverse
    print("values saved")

    # get the execution time
    elapsed_timeAll = etAll - stAll
    endAll_t = time.localtime()
    print(f'level: 5 - FINISH - End: {time.strftime("%d day, %H:%M:%S", endAll_t)} - In time: {time.strftime("%d day, %H:%M:%S", time.gmtime(elapsed_timeAll))}\n')
    print(f'\n=========================\n')
    
    #print(F_s)
    with open(f'Opt_values_and_alpha_values_1_memory_general.txt', 'a') as file:
        file.write(json.dumps(F_s)) # use `json.loads` to do the reverse
    print("values saved")



if __name__ == "__main__":
    main()