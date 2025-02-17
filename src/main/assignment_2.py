import numpy as np

def nevilles_interpolation(x_vals, y_vals, x_target):
    num_points = len(x_vals)
    interp_matrix = [[0 for _ in range(num_points)] for _ in range(num_points)]

    for i in range(num_points):
        interp_matrix[i][i] = y_vals[i]

    for j in range(1, num_points):
        for i in range(num_points - j):
            interp_matrix[i][i + j] = (
                (x_target - x_vals[i + j]) * interp_matrix[i][i + j - 1] -
                (x_target - x_vals[i]) * interp_matrix[i + 1][i + j]
            ) / (x_vals[i] - x_vals[i + j])
    
    return interp_matrix[0][num_points - 1]

def newtons_forward_method(x_vals, y_vals):
    num_points = len(x_vals)
    diff_table = np.zeros((num_points, num_points))
    diff_table[:, 0] = y_vals

    for i in range(1, num_points):
        for j in range(num_points - i):
            diff_table[j, i] = (
                (diff_table[j + 1, i - 1] - diff_table[j, i - 1]) /
                (x_vals[j + i] - x_vals[j])
            )
    
    return diff_table

def newtons_forward_interpolation(x_vals, coeffs, x_target):
    num_coeffs = len(coeffs)
    result = coeffs[0]
    term = 1

    for i in range(1, num_coeffs):
        term *= (x_target - x_vals[i - 1])
        result += coeffs[i] * term
    
    return result

def hermite_interpolation(x_vals, y_vals, dy_vals):
    n = len(x_vals)
    m = 2 * n
    z = [x for x in x_vals for _ in range(2)]
    Q = [[0.0] * m for _ in range(m)]
    
    for i in range(n):
        Q[2*i][0]   = y_vals[i]
        Q[2*i+1][0] = y_vals[i]
        Q[2*i+1][1] = dy_vals[i]
        if i > 0:
            Q[2*i][1] = (Q[2*i][0] - Q[2*i-1][0]) / (z[2*i] - z[2*i-1])
    
    for j in range(2, m):
        for i in range(j, m):
            Q[i][j] = (Q[i][j-1] - Q[i-1][j-1]) / (z[i] - z[i-j])
    
    return z, Q


def cubic_spline_interpolation(x_vals, y_vals):
    num_intervals = len(x_vals) - 1
    step_sizes = np.diff(x_vals)
    coef_matrix = np.zeros((num_intervals + 1, num_intervals + 1))
    rhs_vector = np.zeros(num_intervals + 1)

    coef_matrix[0, 0] = coef_matrix[num_intervals, num_intervals] = 1

    for i in range(1, num_intervals):
        coef_matrix[i, i - 1] = step_sizes[i - 1]
        coef_matrix[i, i] = 2 * (step_sizes[i - 1] + step_sizes[i])
        coef_matrix[i, i + 1] = step_sizes[i]
        rhs_vector[i] = (
            (3 / step_sizes[i]) * (y_vals[i + 1] - y_vals[i]) -
            (3 / step_sizes[i - 1]) * (y_vals[i] - y_vals[i - 1]))
    
    spline_coeffs = np.linalg.solve(coef_matrix, rhs_vector)
    return coef_matrix, rhs_vector, spline_coeffs


