import numpy as np
import scipy
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import time
from matrices import construct_bond_matrix
from matrices import construct_option_matrix
from american_put import finite_difference_american_put
from european_put import crank_nicolson_european_put
from analysis import i_j_analysis
from analysis import r_max_analysis

BOND_MATURITY_TIME = 3
OPTION_MATURITY_TIME = 0.9839
OPTION_STRIKE_PRICE = 49.6
BOND_FACE_VALUE = 51
INITIAL_INTEREST_RATE = 0.0275
TIME_STEPS = 1000
INTEREST_RATE_STEPS = 1000
MAX_INTEREST_RATE = 1.0

OPTION_TYPE = "put" # "call" or "put"

#sigma, kappa, theta, mu, alpha, beta, C
PARAMETERS = (0.251, 0.08116, 0.0409, -0.0222, 0.01, 0.653, 1.07)

if __name__ == "__main__":
    



    r, BNew_n = crank_nicolson_european_put(PARAMETERS, BOND_MATURITY_TIME, BOND_FACE_VALUE, TIME_STEPS, INTEREST_RATE_STEPS, MAX_INTEREST_RATE, boundary_condition = 'Neumann')
    r, BNew_d = crank_nicolson_european_put(PARAMETERS, BOND_MATURITY_TIME, BOND_FACE_VALUE, TIME_STEPS, INTEREST_RATE_STEPS, MAX_INTEREST_RATE, boundary_condition = 'Dirichlet')
    print("B(r_0=,0,T) :=", np.interp(INITIAL_INTEREST_RATE, r, BNew_d))

    plt.plot(r,BNew_n, label = 'Neumann Boundary Condition')
    plt.plot(r,BNew_d, label = 'Dirichlet Boundary Condition')
    plt.xlabel('r')
    plt.ylabel('B(r,0,T)')
    plt.legend()
    plt.grid()
    plt.show()



    i_j_analysis(4, INITIAL_INTEREST_RATE, PARAMETERS, BOND_MATURITY_TIME, BOND_FACE_VALUE, OPTION_MATURITY_TIME, OPTION_STRIKE_PRICE, MAX_INTEREST_RATE, OPTION_TYPE)
    r_max_analysis(4, PARAMETERS, BOND_MATURITY_TIME, BOND_FACE_VALUE, INITIAL_INTEREST_RATE)



    r,vNew, v_T1, BNew = finite_difference_american_put(PARAMETERS, OPTION_STRIKE_PRICE, BOND_MATURITY_TIME, BOND_FACE_VALUE, OPTION_MATURITY_TIME, TIME_STEPS, INTEREST_RATE_STEPS, MAX_INTEREST_RATE, OPTION_TYPE, boundary_condition = 'Neumann', use_penalty_method = True)
    print("V(r_0,0;T_1,T) :=", np.interp(INITIAL_INTEREST_RATE, r, vNew))

    # Filter data for 0.04 < r < 0.1
    mask = (r > 0.04) & (r < 0.1)
    r_filtered = r[mask]
    V_filtered = v_T1[mask]




    coefficients = np.polyfit(r_filtered, V_filtered, 1)
    m, c = coefficients

    # Calculate r-intercept
    r_intercept = -c / m
    print("r-intercept:", r_intercept)
    print("m:", m)
    print("c:", c)



    plt.plot(r,vNew, label = 'V(r,0;T_1,T)')
    plt.plot(r,v_T1, label = 'V(r,T_1;T_1,T)')
    plt.xlim(0,0.1)
    plt.ylim(-0.4,8)
    plt.xlabel('r')
    plt.ylabel('Option Value V')
    plt.legend()
    plt.grid()
    plt.show()