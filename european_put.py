import numpy as np
import scipy
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import time
from matrices import construct_option_matrix
from matrices import construct_bond_matrix

def crank_nicolson_european_put(parameters, bond_maturity_time, bond_face_value, time_steps, rate_steps, max_interest_rate, boundary_condition='Dirichlet'):
    """
    Calculate the value of a European put option on a bond using the Crank-Nicolson finite difference method.
    
    Args:
        parameters (tuple): Model parameters (sigma, kappa, theta, mu, alpha, beta, C).
        bond_maturity_time (float): Time to bond maturity (T).
        bond_face_value (float): Face value of the bond (F).
        time_steps (int): Number of time intervals (imax).
        rate_steps (int): Number of interest rate intervals (jmax).
        max_interest_rate (float): Maximum interest rate in the grid (rmax).
        boundary_condition (str): Type of boundary condition ('Dirichlet' or 'Neumann').
        
    Returns:
        np.ndarray: Array of interest rates.
        np.ndarray: Array of bond values at t=0.
    """
    
    sigma, kappa, theta, mu, alpha, beta, C = parameters
    T = bond_maturity_time
    F = bond_face_value

    # Calculate the step sizes for interest rate (dr) and time (dt)
    dr = max_interest_rate / rate_steps
    dt = T / time_steps

    # Initialize the grid for interest rates and bond values
    interest_rates = np.array([j * dr for j in range(rate_steps + 1)])
    time_grid = np.array([i * dt for i in range(time_steps + 1)])
    bond_value_new, bond_value_old = [np.full(rate_steps + 1, F)] * 2      
        
    # Backward iteration over time steps
    for i in range(time_steps - 1, -1, -1):
        A_banded, d, l_and_u = construct_bond_matrix(bond_value_old, i, dr, dt, parameters, time_steps, rate_steps, max_interest_rate, boundary_condition)
        bond_value_new = solve_banded(l_and_u, A_banded, d)
        bond_value_old = np.copy(bond_value_new)
        
    return interest_rates, bond_value_new