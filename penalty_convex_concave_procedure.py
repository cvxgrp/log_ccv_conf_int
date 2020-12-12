from scipy.stats import gaussian_kde
import cvxpy as cp
import autograd.numpy as np
import time
from utility import *

def generate_ccp_subp(M, m, J_card, X, idxes_of_design_pts, max_iters):
    ell = cp.Variable(m)
    ell_prev = cp.Parameter(m)
    tau = cp.Parameter()

    constrs_log_concavity = [
        ell[i + 1] <= ell[i] + (ell[i] - ell[i - 1]) / (X[idxes_of_design_pts[i]] - X[idxes_of_design_pts[i - 1]])
        * (X[idxes_of_design_pts[i + 1]] - X[idxes_of_design_pts[i]])
        for i in range(1, m - 1)]

    A_V = cp.Parameter((J_card, m))
    b_V = cp.Parameter(J_card)
    s_V = cp.Variable(J_card, nonneg=True)
    constrs_V = [-A_V @ (ell - ell_prev) - b_V <= s_V]

    A_U = cp.Parameter((J_card, m))
    b_U = cp.Parameter(J_card)
    s_U = cp.Variable(J_card, nonneg=True)
    constrs_U = [-A_U @ (ell - ell_prev) - b_U <= s_U]

    A_L = cp.Parameter((J_card * max_iters, m))
    b_L = cp.Parameter(J_card * max_iters)
    constrs_L = [A_L @ ell + b_L <= 0]

    constrs_numeric_lower_bound = [ell >= -M]

    constrs = constrs_log_concavity + constrs_V + constrs_U + constrs_L + constrs_numeric_lower_bound

    weights = cp.Parameter(m)
    objf = weights.T @ ell + tau * cp.sum(s_V) + tau * cp.sum(s_U)
    prob = cp.Problem(cp.Minimize(objf), constrs)

    return prob, ell, s_V, s_U, ell_prev, tau, A_V, A_U, A_L, b_V, b_U, b_L, weights

def ccp_iterations(prob, ell, s_V, s_U, ell_prev, tau, A_V, A_U, A_L, b_V, b_U, b_L, weights,
                   m, J_card, X, idxes_of_design_pts, p, ell_g_idxes_to_sum_over, cs, ds,
                   want_max, want_verbose, solver_choice,
                   tau_max, tau_init, opt_tol, s_tol, mu, max_iters, min_iters):
    if (want_max == False):
        weights.value = [0 for i in range(0, p)] + [1] + [0 for i in range(p + 1, m)]
    else:
        weights.value = [0 for i in range(0, p)] + [-1] + [0 for i in range(p + 1, m)]

    solver_failed = False
    objf_val_prev = np.inf
    objf_val = np.inf
    tau.value = tau_init
    g_kde = gaussian_kde(X)
    ell_prev.value = np.log(g_kde.evaluate(X[idxes_of_design_pts]))
    s_sum = np.inf

    A_L_value = np.zeros((J_card * max_iters, m))
    b_L_value = -np.array(ds * max_iters)

    for iter_idx in range(max_iters):
        start = time.time()
        # ineq 7 V
        A_V_value = np.zeros((J_card, m))
        b_V_value = -np.array(cs)
        for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
            for i in idxes:
                if (i == 0):
                    func_value = upper_bound_U(
                        np.concatenate((np.array([ell_prev[2].value]), np.array([ell_prev[1].value]))),
                        X[idxes_of_design_pts[2]], X[idxes_of_design_pts[1]], X[idxes_of_design_pts[0]])
                    grad_value = grad_upper_bound_U(
                        np.concatenate((np.array([ell_prev[2].value]), np.array([ell_prev[1].value]))),
                        X[idxes_of_design_pts[2]], X[idxes_of_design_pts[1]], X[idxes_of_design_pts[0]])
                    A_V_value[idxes_idx, 2] += grad_value[0]
                    A_V_value[idxes_idx, 1] += grad_value[1]
                    b_V_value[idxes_idx] += func_value
                else:
                    func_value = upper_bound_V(
                        np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                        X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                    grad_value = grad_upper_bound_V(
                        np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                        X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                    A_V_value[idxes_idx, i] += grad_value[0]
                    A_V_value[idxes_idx, i - 1] += grad_value[1]
                    b_V_value[idxes_idx] += func_value
        # ineq 7 new U
        A_U_value = np.zeros((J_card, m))
        b_U_value = -np.array(cs)
        for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
            for i in idxes:
                if (i == m - 2):
                    func_value = upper_bound_V(
                        np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                        X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                    grad_value = grad_upper_bound_V(
                        np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                        X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                    A_U_value[idxes_idx, i] += grad_value[0]
                    A_U_value[idxes_idx, i - 1] += grad_value[1]
                    b_U_value[idxes_idx] += func_value
                else:
                    func_value = upper_bound_U(
                        np.concatenate((np.array([ell_prev[i + 2].value]), np.array([ell_prev[i + 1].value]))),
                        X[idxes_of_design_pts[i + 2]], X[idxes_of_design_pts[i + 1]],
                        X[idxes_of_design_pts[i]])
                    grad_value = grad_upper_bound_U(
                        np.concatenate((np.array([ell_prev[i + 2].value]), np.array([ell_prev[i + 1].value]))),
                        X[idxes_of_design_pts[i + 2]], X[idxes_of_design_pts[i + 1]],
                        X[idxes_of_design_pts[i]])
                    A_U_value[idxes_idx, i + 2] += grad_value[0]
                    A_U_value[idxes_idx, i + 1] += grad_value[1]
                    b_U_value[idxes_idx] += func_value

        # ineq 8 L
        for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
            for i in idxes:
                func_value = lower_bound_L(
                    np.concatenate((np.array([ell_prev[i + 1].value]), np.array([ell_prev[i].value]))),
                    X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]])
                grad_value = grad_lower_bound_L(
                    np.concatenate((np.array([ell_prev[i + 1].value]), np.array([ell_prev[i].value]))),
                    X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]])
                A_L_value[iter_idx * J_card + idxes_idx, i + 1] += grad_value[0]
                A_L_value[iter_idx * J_card + idxes_idx, i] += grad_value[1]
                b_L_value[iter_idx * J_card + idxes_idx] += func_value
            b_L_value[iter_idx * J_card + idxes_idx] += - ell_prev.value.dot(A_L_value[iter_idx * J_card + idxes_idx, :])

        try:
            A_V.value = A_V_value
            b_V.value = b_V_value
            A_U.value = A_U_value
            b_U.value = b_U_value
            A_L.value = A_L_value
            b_L.value = b_L_value
            if (solver_choice == "ECOS"):
                prob.solve(verbose=False, solver=solver_choice, warm_start=True, abstol=1e-9, reltol=1e-9)
            elif (solver_choice == "MOSEK"):
                prob.solve(verbose=False, solver=solver_choice, warm_start=True)
            elif (solver_choice == "SCS"):
                prob.solve(verbose=False, solver=solver_choice, warm_start=True, eps=1e-9, max_iters=10000)
            else:
                print("ERROR: unsupported solver")
                break
            s_sum = np.sum(s_V.value) + np.sum(s_U.value)
            objf_val = ell[p].value
            end = time.time()
            if (want_verbose):
                print(
                    "  objf (i.e., log prob)=%f, prob=%f, slack sum=%f, tau=%f, prob.status=%s, after iter #%d, time elapsed=%f" %
                    (objf_val, np.exp(objf_val), s_sum, tau.value, prob.status, iter_idx + 1, (end - start)))
        except Exception as e:
            solver_failed = True
            if (want_verbose):
                print("  WARNING: solver failed w/ prob.status=%s + error message=%s; quitting early ..." %
                      (prob.status, e))
            break

        if ((np.abs(objf_val_prev - objf_val) <= opt_tol * np.abs(objf_val) \
             and (s_sum < s_tol) and (iter_idx >= min_iters)) \
                or (solver_failed == True)):
            if (want_verbose):
                print("  quitting early ...")
            break
        ell_prev.value = ell.value
        tau.value = min(mu * tau.value, tau_max)
        objf_val_prev = objf_val

    # print result
    if want_verbose:
        if not want_max:
            print("minimizing over ell[", p, "]:",
                  "result: objf (i.e., log prob)=", objf_val,
                  "slack sum=", s_sum, "tau=", tau.value, "prob.status=", prob.status,
                  "num of iters=", iter_idx)
        else:
            print("maximizing over ell[", p, "]:",
                  "result: objf (i.e., log prob)=", objf_val,
                  "slack sum=", s_sum, "tau=", tau.value, "prob.status=", prob.status,
                  "num of iters=", iter_idx)
    return solver_failed, s_sum