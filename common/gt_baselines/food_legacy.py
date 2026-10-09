"""
Survival Rate Analysis for Food Chain System
=============================================
Calculate survival rate as a function of bifurcation parameter K.
This complements the std vs K analysis by quantifying the probability of system collapse.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
from tqdm import tqdm
import time

# System parameters
xc = 0.4
yc = 2.009
xp = 0.08
yp = 2.876
r0 = 0.16129
c0 = 0.5

def food_chain_eq(t, x, k):
    """Food chain dynamical system"""
    dxdt = np.zeros(3)
    dxdt[0] = x[0] * (1 - x[0]/k) - xc * yc * x[1] * x[0] / (x[0] + r0)
    dxdt[1] = xc * x[1] * (yc * x[0] / (x[0] + r0) - 1) - xp * yp * x[2] * x[1] / (x[1] + c0)
    dxdt[2] = xp * x[2] * (yp * x[1] / (x[1] + c0) - 1)
    return dxdt






