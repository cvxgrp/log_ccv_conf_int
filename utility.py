import autograd.numpy as np
from autograd import grad


def lower_bound_L(ell_p_plue_1_ell_p, x_p_plus_1, x_p, tol=1e-8):#rhs_of_eq8
    ell_p_plus_1 = ell_p_plue_1_ell_p[0]
    ell_p = ell_p_plue_1_ell_p[1]
    if (np.abs(ell_p_plus_1 - ell_p) < tol):
        return (x_p_plus_1 - x_p) * np.exp(ell_p)
    else:
        return (x_p_plus_1 - x_p) * np.exp(ell_p) * (np.exp(ell_p_plus_1 - ell_p) - 1) / (ell_p_plus_1 - ell_p)


def upper_bound_V(ell_p_p_minus_1, x_p_plus_1, x_p, x_p_minus_1, tol=1e-8): #rhs_of_eq7
    ell_p = ell_p_p_minus_1[0]
    ell_p_minus_1 = ell_p_p_minus_1[1]
    if (np.abs(ell_p - ell_p_minus_1) < tol):
        return np.exp(ell_p) * (x_p_plus_1 - x_p)
    else:
        return np.exp(ell_p) * (np.exp((ell_p - ell_p_minus_1) / (x_p - x_p_minus_1) * (x_p_plus_1 - x_p)) - 1.0) \
               / (ell_p - ell_p_minus_1) * (x_p - x_p_minus_1)


def upper_bound_U(ell_p_plus_2_p_plus_1, x_p_plus_2, x_p_plus_1, x_p, tol=1e-8): #rhs_of_eq7_new
    ell_p_plus_2 = ell_p_plus_2_p_plus_1[0]
    ell_p_plus_1 = ell_p_plus_2_p_plus_1[1]
    if (np.abs(ell_p_plus_2 - ell_p_plus_1) < tol):
        return np.exp(ell_p_plus_1) * (x_p_plus_1 - x_p)
    else:
        return np.exp(ell_p_plus_1) * (1.0 - np.exp((ell_p_plus_2 - ell_p_plus_1) / (x_p_plus_2 - x_p_plus_1) * (x_p - x_p_plus_1))) \
               / (ell_p_plus_2 - ell_p_plus_1) * (x_p_plus_2 - x_p_plus_1)


grad_upper_bound_V = grad(upper_bound_V)
grad_upper_bound_U = grad(upper_bound_U)
grad_lower_bound_L = grad(lower_bound_L)