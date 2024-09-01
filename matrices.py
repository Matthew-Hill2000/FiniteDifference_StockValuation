import numpy as np
import scipy
from scipy.linalg import solve_banded

def construct_bond_matrix(bond_values_old, time_index, interest_rate_step, time_step, params, time_steps, rate_steps, max_interest_rate, boundary_condition='Dirichlet'):
    """
    Construct the tridiagonal matrix for the bond value calculation using the Crank-Nicolson method.
    
    Args:
        bond_values_old (np.ndarray): Bond values at the previous time step.
        time_index (int): Current time step index.
        interest_rate_step (float): Step size in the interest rate grid (dr).
        time_step (float): Step size in the time grid (dt).
        params (tuple): Model parameters (sigma, kappa, theta, mu, alpha, beta, C).
        time_steps (int): Total number of time intervals (imax).
        rate_steps (int): Total number of interest rate intervals (jmax).
        max_interest_rate (float): Maximum interest rate in the grid (rmax).
        boundary_condition (str): Type of boundary condition ('Dirichlet' or 'Neumann').
        
    Returns:
        np.ndarray: Tridiagonal matrix for bond value calculation (A_banded).
        np.ndarray: Right-hand side vector (d).
        tuple: Lower and upper band indices for the banded matrix solver (l_and_u).
    """
    
    sigma, kappa, theta, mu, alpha, beta, C = params

    A_banded = np.zeros(shape=(3, rate_steps + 1))
    l_and_u = (1, 1)

    # Boundary condition at the first node
    A_banded[1][0] = 1 / time_step + (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step)  # b_0
    A_banded[0][1] = -1 / interest_rate_step * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step)  # c_0
    
    # Populate the middle rows
    for j in range(1, rate_steps):
        A_banded[2][j - 1] = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                                     - (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                                     + kappa * j)  # a_j
        A_banded[1][j] = -((1 / time_step) + 0.5 * sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                           + j * interest_rate_step / 2)  # b_j
        A_banded[0][j + 1] = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                                     + (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                                     - kappa * j)  # c_j

    # Boundary condition at the last node
    if boundary_condition == 'Neumann':
        A_banded[2][rate_steps - 1] = -1 / interest_rate_step  # a_jmax
        A_banded[1][rate_steps] = 1 / interest_rate_step  # b_jmax
    else:
        A_banded[2][rate_steps - 1] = 0.0  # a_jmax 
        A_banded[1][rate_steps] = 1.0  # b_jmax

    # Create the right-hand side vector
    d = np.zeros(rate_steps + 1)
    d[0] = 1 / time_step * bond_values_old[0] + C * np.exp(-alpha * (time_index + 0.5) * time_step)  # d_0
    for j in range(1, rate_steps):
        a_j = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                      - (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                      + kappa * j)  # a_j
        b_j = ((1 / time_step) - 0.5 * sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
               - j * interest_rate_step / 2)  # b_j
        c_j = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                      + (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                      - kappa * j)  # c_j
        d[j] = -a_j * bond_values_old[j - 1] - b_j * bond_values_old[j] - c_j * bond_values_old[j + 1] - C * np.exp(-alpha * (time_index + 0.5) * time_step)  # d_j
    d[rate_steps] = 0.0

    return A_banded, d, l_and_u


def construct_option_matrix(option_values_old, bond_values_old, strike_price, time_index, interest_rate_step, time_step, params, time_steps, rate_steps, max_interest_rate, option_type, boundary_condition='Dirichlet'):
    """
    Construct the tridiagonal matrix for the option value calculation using the Crank-Nicolson method.
    
    Args:
        option_values_old (np.ndarray): Option values at the previous time step.
        bond_values_old (np.ndarray): Bond values at the previous time step.
        strike_price (float): Strike price of the option (X).
        time_index (int): Current time step index.
        interest_rate_step (float): Step size in the interest rate grid (dr).
        time_step (float): Step size in the time grid (dt).
        params (tuple): Model parameters (sigma, kappa, theta, mu, alpha, beta, C).
        time_steps (int): Total number of time intervals (imax).
        rate_steps (int): Total number of interest rate intervals (jmax).
        max_interest_rate (float): Maximum interest rate in the grid (rmax).
        option_type (str): Type of option ('call' or 'put').
        boundary_condition (str): Type of boundary condition ('Dirichlet' or 'Neumann').
        
    Returns:
        np.ndarray: Tridiagonal matrix for option value calculation (A_banded).
        np.ndarray: Right-hand side vector (d).
        tuple: Lower and upper band indices for the banded matrix solver (l_and_u).
    """
    
    sigma, kappa, theta, mu, alpha, beta, C = params
    X = strike_price

    A_banded = np.zeros(shape=(3, rate_steps + 1))
    l_and_u = (1, 1)
    
    # Boundary condition at the first node
    if option_type == "call":
        A_banded[1][0] = 1  # b_0
        A_banded[0][1] = 0  # c_0
    elif option_type == "put":
        A_banded[1][0] = 1 / time_step + (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step)  # b_0
        A_banded[0][1] = -1 / interest_rate_step * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step)  # c_0

    # Populate the middle rows
    for j in range(1, rate_steps):
        A_banded[2][j - 1] = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                                     - (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                                     + kappa * j)  # a_j
        A_banded[1][j] = -((1 / time_step) + 0.5 * sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                           + j * interest_rate_step / 2)  # b_j
        A_banded[0][j + 1] = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                                     + (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                                     - kappa * j)  # c_j

    # Boundary condition at the last node
    if boundary_condition == 'Neumann':
        A_banded[2][rate_steps - 1] = -1 / interest_rate_step  # a_jmax
        A_banded[1][rate_steps] = 1 / interest_rate_step  # b_jmax
    else:
        A_banded[2][rate_steps - 1] = 0.0  # a_jmax 
        A_banded[1][rate_steps] = 1.0  # b_jmax

    # Create the right-hand side vector
    d = np.zeros(rate_steps + 1)
    
    if option_type == "call":
        d[0] = bond_values_old[0] - X  # d_0
    elif option_type == "put":
        d[0] = 1 / time_step * option_values_old[0]  # d_0
    
    for j in range(1, rate_steps):
        a_j = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                      - (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                      + kappa * j)  # a_j
        b_j = ((1 / time_step) - 0.5 * sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
               - j * interest_rate_step / 2)  # b_j
        c_j = 0.25 * (sigma**2 * pow(j, 2 * beta) * pow(interest_rate_step, 2 * (beta - 1)) 
                      + (1 / interest_rate_step) * kappa * theta * np.exp(mu * (time_index + 0.5) * time_step) 
                      - kappa * j)  # c_j
        d[j] = -a_j * option_values_old[j - 1] - b_j * option_values_old[j] - c_j * option_values_old[j + 1]  # d_j
    
    if option_type == "call":
        d[rate_steps] = 0.0
    elif option_type == "put":
        d[rate_steps] = X - bond_values_old[rate_steps]

    return A_banded, d, l_and_u
