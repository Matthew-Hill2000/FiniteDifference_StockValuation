import numpy as np
import scipy
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import time
from matrices import construct_bond_matrix
from matrices import construct_option_matrix

def shift_time_grid(time_grid, time_step, option_maturity_time):
    """
    Adjust the time grid so that the option maturity time aligns with the nearest grid point.
    
    Args:
        time_grid (np.ndarray): Array representing the time grid.
        time_step (float): The time step size (dt).
        option_maturity_time (float): The maturity time of the option (T1).
        
    Returns:
        np.ndarray: Adjusted time grid.
        float: Upper time step for the new grid.
        float: Lower time step for the new grid.
        int: Index of the nearest grid point to the option maturity time.
    """
    nearest_index = np.abs(time_grid - option_maturity_time).argmin()
    closest_grid_value = time_grid[nearest_index]

    if closest_grid_value < option_maturity_time:
        time_grid[1:-1] += np.abs(closest_grid_value - option_maturity_time)
        dt_upper = time_step - np.abs(closest_grid_value - option_maturity_time)
        dt_lower = time_step + np.abs(closest_grid_value - option_maturity_time)

    elif closest_grid_value > option_maturity_time:
        time_grid[1:-1] -= np.abs(closest_grid_value - option_maturity_time)
        dt_upper = time_step + np.abs(closest_grid_value - option_maturity_time)
        dt_lower = time_step - np.abs(closest_grid_value - option_maturity_time)

    time_grid[nearest_index] = option_maturity_time   

    return time_grid, dt_upper, dt_lower, nearest_index


def finite_difference_american_put(params, strike_price, bond_maturity_time, bond_face_value, option_maturity_time, time_steps, rate_steps, max_interest_rate, option_type, boundary_condition='Neumann', use_penalty_method=False):
    """
    Calculate the value of an American put option on a bond using the finite difference method.
    
    Args:
        params (tuple): Model parameters (sigma, kappa, theta, mu, alpha, beta, C).
        strike_price (float): Strike price of the option (X).
        bond_maturity_time (float): Maturity time of the bond (T).
        bond_face_value (float): Face value of the bond (F).
        option_maturity_time (float): Maturity time of the option (T1).
        time_steps (int): Number of time intervals (imax).
        rate_steps (int): Number of interest rate intervals (jmax).
        max_interest_rate (float): Maximum interest rate in the grid (rmax).
        option_type (str): Type of option ('call' or 'put').
        boundary_condition (str): Type of boundary condition ('Dirichlet' or 'Neumann').
        use_penalty_method (bool): Whether to use the penalty method for early exercise.
        
    Returns:
        np.ndarray: Array of interest rates.
        np.ndarray: Array of option values at t=0.
        np.ndarray: Array of option values at t=T1.
        np.ndarray: Array of bond values at t=0.
    """
    
    sigma, kappa, theta, mu, alpha, beta, C = params
    X = strike_price
    T = bond_maturity_time
    T_1 = option_maturity_time
    F = bond_face_value

    # Calculate step sizes for interest rate (dr) and time (dt)
    dr = max_interest_rate / rate_steps
    dt = T / time_steps

    option_value_new, option_value_old = [np.zeros(rate_steps+1)]*2
    bond_value_new, bond_value_old = [np.full(rate_steps+1, F)]*2

    # Set up the grid for interest rates and time
    interest_rates = np.array([j*dr for j in range(rate_steps+1)])
    time_grid = np.array([i*dt for i in range(time_steps+1)])

    # Adjust the time grid so that T1 aligns with a grid point
    time_grid, dt_upper, dt_lower, T1_nearest_index = shift_time_grid(time_grid, dt, T_1)

    # Backward iteration from T to T1 to calculate bond value at t=T1
    for i in range(time_steps-1, T1_nearest_index-1, -1):
        dt = T / time_steps if i != time_steps-1 else dt_upper

        A_banded, d, l_and_u = construct_bond_matrix(bond_value_old, i, dr, dt, params, time_steps, rate_steps, max_interest_rate, boundary_condition)
        bond_value_new = solve_banded(l_and_u, A_banded, d)       
        bond_value_old = np.copy(bond_value_new)

    # Set the boundary condition for the option value at T1
    if option_type == "call":
        option_value_old = np.maximum(-X + bond_value_new, 0) 
        option_value_new = np.maximum(-X + bond_value_new, 0) 
    elif option_type == "put":
        option_value_old = np.maximum(X - bond_value_new, 0) 
        option_value_new = np.maximum(X - bond_value_new, 0) 
    
    # Save the option value at T1
    option_value_at_T1 = option_value_old
        
    # Backward iteration from T1 to 0 to calculate bond and option values
    for i in range(T1_nearest_index-1, -1, -1):
        dt = T / time_steps if i != 0 else dt_lower
        
        # Calculate bond value at the next time step
        bond_matrix, d, l_and_u = construct_bond_matrix(bond_value_old, i, dr, dt, params, time_steps, rate_steps, max_interest_rate, boundary_condition)
        bond_value_new = solve_banded(l_and_u, bond_matrix, d)
        
        # Calculate option value at the next time step
        option_matrix_banded, d, l_and_u = construct_option_matrix(option_value_old, bond_value_old, X, i, dr, dt, params, time_steps, rate_steps, max_interest_rate, option_type, "Dirichlet")
        option_value_new = solve_banded(l_and_u, option_matrix_banded, d)
        
        if use_penalty_method:
            rho = 1.e8
            tol = 1e-8
            maxiter = 50
            option_value_new = apply_penalty_method(option_value_new, rate_steps, rho, tol, maxiter, l_and_u, option_matrix_banded, d, bond_value_new, i, option_type, strike_price)

        elif option_type == "call":
            option_value_new = np.maximum(option_value_new, bond_value_new-X)
        elif option_type == "put":
            option_value_new = np.maximum(option_value_new, X-bond_value_new)

        option_value_old = np.copy(option_value_new)
        bond_value_old = np.copy(bond_value_new)
    
    return interest_rates, option_value_new, option_value_at_T1, bond_value_new

def apply_penalty_method(option_value_new, rate_steps, penalty_factor, tolerance, max_iterations, l_and_u, option_matrix_banded, d, bond_value_new, time_step, option_type, strike_price):
    """
    Apply the penalty method to ensure the option value is always greater than the intrinsic value, especially for early exercise scenarios.
    
    Args:
        option_value_new (np.ndarray): Current option values.
        rate_steps (int): Number of interest rate intervals (jmax).
        penalty_factor (float): Penalty factor (rho).
        tolerance (float): Convergence tolerance (tol).
        max_iterations (int): Maximum number of iterations (maxiter).
        l_and_u (tuple): Lower and upper band indices for the banded solver.
        option_matrix_banded (np.ndarray): Banded matrix for the option.
        d (np.ndarray): Right-hand side vector.
        bond_value_new (np.ndarray): Current bond values.
        time_step (int): Current time step index.
        option_type (str): Type of option ('call' or 'put').
        strike_price (float): Strike price of the option (X).
        
    Returns:
        np.ndarray: Updated option values after applying the penalty method.
    """
    
    X = strike_price
    
    option_matrix_adjusted = np.copy(option_matrix_banded)
    d_adjusted = np.copy(d)
    
    sign = -1 if option_type == "call" else 1

    # Apply penalty method to ensure option value is greater than intrinsic value
    for q in range(max_iterations):
        for j in range(1, rate_steps):
            if option_value_new[j] < (sign*(X - bond_value_new[j])):
                option_matrix_adjusted[1][j] = option_matrix_banded[1][j] - penalty_factor
                d_adjusted[j] = d[j] - penalty_factor*(sign*(X - bond_value_new[j]))
            else:
                option_matrix_adjusted[1][j] = option_matrix_banded[1][j]
                d_adjusted[j] = d[j]

        # Solve the adjusted linear system
        y = scipy.linalg.solve_banded(l_and_u, option_matrix_adjusted, d_adjusted)
        
        # Calculate the change from the previous iteration
        error = y - option_value_new
        
        # Update the option values
        option_value_new = y
        
        # Check for convergence
        if np.linalg.norm(error, 1) < tolerance:
            break
        
    return option_value_new
