import numpy as np
import scipy
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import time
from american_put import finite_difference_american_put
from european_put import crank_nicolson_european_put

def i_j_analysis(N, r_0, constants, bond_expiry_date, bond_expiry_value, option_expiry_date, strike_price, rmax, option_type):

    
    T = bond_expiry_date
    F = bond_expiry_value
    X = strike_price
    T_1 = option_expiry_date
    
    i_j = np.array([100* 2**x for x in range(1,N)])

    B0_old = 0
    V0_old = 0
    diff_old = 1
    for imax in i_j:
        start_time = time.time()
        # r, BNew = crank_nicolson_european_put(constants, imax, 100, r_max, boundary = 'Neumann')
        r,vNew, v_T1, BNew = finite_difference_american_put(constants, X, T, F, T_1, imax, 100, rmax, option_type, boundary_condition = 'Neumann', use_penalty_method = True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # B0_new = np.interp(r_0,r,BNew)
        V0_new =np.interp(r_0, r, vNew)
        diff = V0_new - V0_old
        # diff = B0_new - B0_old
        # print("imax: ", imax, "B(r_0=0,T) :=", B0_new, "Time: ", elapsed_time, "diff: ", diff, "diff ratio: ", diff_old/diff)
        print("imax: ", imax, "V(r_0=0,T) :=", V0_new, "Time: ", elapsed_time, "diff: ", diff, "diff ratio: ", diff_old/diff)
        # B0_old = B0_new
        V0_old = V0_new
        diff_old = diff
    
    B0_old = 0
    V0_old = 0
    diff_old = 1
    for jmax in i_j:
        start_time = time.time()
        # r, BNew = crank_nicolson_european_put(constants, 100, jmax, r_max, boundary = 'Neumann')
        r,vNew, v_T1, BNew = finite_difference_american_put(constants, X, T, F, T_1, 100, jmax, rmax, option_type, boundary_condition = 'Neumann', use_penalty_method = True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # B0_new = np.interp(r_0,r,BNew)
        V0_new = np.interp(r_0, r, vNew)
        diff = V0_new - V0_old
        # diff = B0_new - B0_old
        # print("jmax: ", jmax, "B(r_0=0,T) :=", B0_new, "Time: ", elapsed_time, "diff: ", diff, "diff ratio: ", diff_old/diff)
        print("jmax: ", jmax, "V(r_0=0,T) :=", V0_new, "Time: ", elapsed_time, "diff: ", diff, "diff ratio: ", diff_old/diff)
        # B0_old = B0_new
        V0_old = V0_new
        diff_old = diff

    B0_old = 0
    V0_old = 0
    diff_old = 1
    for max in i_j:
        start_time = time.time()
        # r, BNew = crank_nicolson_european_put(constants, max, max, r_max, boundary = 'Neumann')
        r,vNew, v_T1, BNew = finite_difference_american_put(constants, X, T, F, T_1, max, max, rmax, option_type, boundary_condition = 'Neumann', use_penalty_method = True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # B0_new = np.interp(r_0,r,BNew)
        V0_new = np.interp(r_0, r, vNew)
        diff = V0_new - V0_old
        # diff = B0_new - B0_old
        # print("max: ", max, "B(r_0=0,T) :=", B0_new, "Time: ", elapsed_time, "diff: ", diff, "diff ratio: ", diff_old/diff)
        print("max: ", max, "V(r_0=0,T) :=", V0_new, "Time: ", elapsed_time, "diff: ", diff, "diff ratio: ", diff_old/diff)
        # B0_old = B0_new
        V0_old = V0_new
        diff_old = diff


def r_max_analysis(N, constants, bond_expiry_date, bond_expiry_value, r_0):
    T = bond_expiry_date
    F = bond_expiry_value
    
    r_max_vals = np.array([2**i for i in range(N)])
    
    for r_max in r_max_vals:
        j_max = 100*r_max
        i_max = 100
        start_time = time.time()
        r, BNew = crank_nicolson_european_put(constants,T,F,i_max, j_max, r_max, boundary_condition = 'Neumann')
        end_time = time.time()
        elapsed_time = end_time - start_time
        B0_new = np.interp(r_0,r,BNew)
        print("r_max: ", r_max, "j_max: ", j_max, "Time: ", elapsed_time, "B(r_0=0,T) :=", B0_new)
