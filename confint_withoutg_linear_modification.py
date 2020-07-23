from autograd import grad
import autograd.numpy as np

from scipy.stats import beta
import time
from scipy.stats import gaussian_kde

import cvxpy as cp

from multiprocessing import Array, Pipe, Process

from numpy.random import choice

def rhs_of_eq8(ell_p_plue_1_ell_p, x_p_plus_1, x_p, tol=1e-8):
    ell_p_plus_1 = ell_p_plue_1_ell_p[0]
    ell_p = ell_p_plue_1_ell_p[1]
    if (np.abs(ell_p_plus_1 - ell_p) < tol):
        return (x_p_plus_1 - x_p) * np.exp(ell_p)
    else:
        return (x_p_plus_1 - x_p) * np.exp(ell_p) * (np.exp(ell_p_plus_1 - ell_p) - 1) / (ell_p_plus_1 - ell_p)

def rhs_of_eq7(ell_p_p_minus_1, x_p_plus_1, x_p, x_p_minus_1, tol=1e-8):
    ell_p = ell_p_p_minus_1[0]
    ell_p_minus_1 = ell_p_p_minus_1[1]
    if (np.abs(ell_p - ell_p_minus_1) < tol):
        return np.exp(ell_p) * (x_p_plus_1 - x_p)
    else:
        return np.exp(ell_p) * (np.exp((ell_p - ell_p_minus_1) / (x_p - x_p_minus_1) * (x_p_plus_1 - x_p)) - 1.0) \
               / (ell_p - ell_p_minus_1) * (x_p - x_p_minus_1)

def rhs_of_eq7_new(ell_p_plus_2_p_plus_1, x_p_plus_2, x_p_plus_1, x_p, tol=1e-8):
    ell_p_plus_2 = ell_p_plus_2_p_plus_1[0]
    ell_p_plus_1 = ell_p_plus_2_p_plus_1[1]
    if (np.abs(ell_p_plus_2 - ell_p_plus_1) < tol):
        return np.exp(ell_p_plus_1) * (x_p_plus_1 - x_p)
    else:
        return np.exp(ell_p_plus_1) * (1.0 - np.exp((ell_p_plus_2 - ell_p_plus_1) / (x_p_plus_2 - x_p_plus_1) * (x_p - x_p_plus_1))) \
               / (ell_p_plus_2 - ell_p_plus_1) * (x_p_plus_2 - x_p_plus_1)

grad_rhs_of_eq7 = grad(rhs_of_eq7)
grad_rhs_of_eq7_new = grad(rhs_of_eq7_new)
grad_rhs_of_eq8 = grad(rhs_of_eq8)

class confint():

    def __init__(self, n, X, alpha, opt_pts_ratio=1):

        self.X = X
        J, cs, ds, idxes_of_design_pts = self.generate_cs_ds(n, alpha)  # indices in J are among 1 to m

        self.ds = ds
        self.cs = cs
        m = len(idxes_of_design_pts)
        self.m = m
        J_card = len(J)
        self.J_card = J_card
        self.idxes_of_design_pts = idxes_of_design_pts

        self.ell_g_idxes_to_sum_over = self.compute_idxes_among_1_to_m_sum_over(J)

        # subsample design points to optimize over
        self.idxes_of_design_pts_to_opt = self.sampling_pts_to_opt_over(opt_pts_ratio)
        self.num_design_pts_to_opt = len(self.idxes_of_design_pts_to_opt)
        print("num design points to optimize", self.num_design_pts_to_opt)
        list_idxes_of_design_pts = list(self.idxes_of_design_pts)
        self.idxes_opt_pts_in_design_pts = [list_idxes_of_design_pts.index(self.idxes_of_design_pts_to_opt[i])
                                            for i in range(self.num_design_pts_to_opt)]
        self.lo_opt_pts = np.zeros(self.num_design_pts_to_opt)
        self.hi_opt_pts = np.zeros(self.num_design_pts_to_opt)

        # shared memory
        self.lo = Array('d', m)
        self.hi = Array('d', m)
        self.lo_slack = Array('d', m)
        self.hi_slack = Array('d', m)

        assert(J_card == len(ds))
        assert(J_card == len(cs))
        assert(J_card == len(self.ell_g_idxes_to_sum_over))

    def sampling_pts_to_opt_over(self, sampling_ratio=1.0, tail_ratio=0.5):
        if sampling_ratio == 1.0:
            return self.idxes_of_design_pts
        sorted_idxes_of_design_pts = sorted(self.idxes_of_design_pts)
        bin_width = min([self.X[sorted_idxes_of_design_pts[i+1]] - self.X[sorted_idxes_of_design_pts[i]] for i in range(self.m - 1)])
        design_pts_pdf = np.zeros(self.m)

        for i in range(self.m):
            left = sorted_idxes_of_design_pts[max(0, i - 1)]
            right = sorted_idxes_of_design_pts[min(i + 1, self.m - 1)]
            for j in range(left, right + 1):
                dist = np.abs(self.X[j] - self.X[sorted_idxes_of_design_pts[i]])
                if dist <= bin_width * 0.5:
                    design_pts_pdf[i] += 1

        arg_pdf_sort = np.argsort(design_pts_pdf)
        K = int(self.m * sampling_ratio)

        result = [sorted_idxes_of_design_pts[arg_pdf_sort[i]] for i in range(int(K * tail_ratio))]

        other_idxes_of_design_pts = []
        other_design_pts_pdf = []
        for i in range(self.m):
            idx = sorted_idxes_of_design_pts[i]
            if idx not in result:
                other_idxes_of_design_pts.append(idx)
                other_design_pts_pdf.append(design_pts_pdf[i])
        other_design_pts_pdf = np.array(other_design_pts_pdf)
        sample_others = choice(other_idxes_of_design_pts, size = K - int(K * tail_ratio), replace=False,
                               p=1.0 / other_design_pts_pdf / np.sum(1.0 / other_design_pts_pdf))
        result = result + list(sample_others)

        arg_max_pdf = np.argmax(design_pts_pdf)

        if sorted_idxes_of_design_pts[0] not in result:
            result.append(sorted_idxes_of_design_pts[0])
        if sorted_idxes_of_design_pts[self.m - 1] not in result:
            result.append(sorted_idxes_of_design_pts[self.m - 1])
        if sorted_idxes_of_design_pts[arg_max_pdf] not in result:
            result.append(sorted_idxes_of_design_pts[arg_max_pdf])
        return np.sort(result)

    def generate_cs_ds(self, n_in, alpha):
        s_n = int(np.ceil(np.log2(np.log(n_in))))
        m = 1 + int(np.floor((n_in - 1) / 2.0 ** s_n))
        B_max = int(np.floor(np.log2(n_in / 4.0)) - s_n + 1.0)
        t_n = np.sum([1.0 / i for i in range(2, B_max + 1)])

        J = []
        cs = []
        ds = []

        for B in range(2, B_max + 1):
            n_B = int(np.floor((m - 1.0) / 2 ** (B - 2)))

            for i in range(1, n_B + 1):
                j = 1.0 + (i - 1.0) * 2 ** (B - 2)  # among 1 to m
                k = 1.0 + (i) * 2 ** (B - 2)
                J += [(int(j - 1), int(k - 1))]

                d = beta.ppf(1.0 - alpha / (2.0 * B * n_B * t_n), (k - j) * 2 ** s_n, n_in + 1 - (k - j) * 2 ** s_n)
                ds += [d]

                c = beta.ppf(alpha / (2.0 * B * n_B * t_n), (k - j) * 2 ** s_n, n_in + 1 - (k - j) * 2 ** s_n)
                cs += [c]

                assert (k >= j)
                assert (d >= c)

        idxes_of_design_pts = np.array([int((i - 1.0) * 2 ** s_n) for i in range(1, m + 1)])
        return J, cs, ds, idxes_of_design_pts

    def compute_idxes_among_1_to_m_sum_over(self, J):
        idxes_among_1_to_m_sum_over = []
        for interval in J:
            j = interval[0]
            k = interval[1]
            idxes_among_1_to_m_sum_over += [[idx for idx in range(j, k)]]
        return idxes_among_1_to_m_sum_over


    def generate_ccp_subp(self, M, m, J_card, ell_g_idxes_to_sum_over, X, idxes_of_design_pts, ds, modification, max_iters):
        ell = cp.Variable(m)
        ell_prev = cp.Parameter(m)
        tau = cp.Parameter()

        constrs_log_concavity = [
            ell[i + 1] <= ell[i] + (ell[i] - ell[i - 1]) / (X[idxes_of_design_pts[i]] - X[idxes_of_design_pts[i - 1]])
            * (X[idxes_of_design_pts[i + 1]] - X[idxes_of_design_pts[i]])
            for i in range(1, m - 1)]

        constrs_rhs_eq5 = []
        for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
            #rhs_eq_5_as_list = [
            #            cp.exp((ell[i + 1] + ell[i]) / 2.0) * (X[idxes_of_design_pts[i + 1]] - X[idxes_of_design_pts[i]])
            #            for i in idxes]
            rhs_eq_5_as_list = [
                (ell[i + 1] + ell[i]) / 2.0 * (X[idxes_of_design_pts[i + 1]] - X[idxes_of_design_pts[i]])
                for i in idxes]
            idxes_sort = np.sort(idxes)
            constrs_rhs_eq5 += [np.log(ds[idxes_idx] / (X[idxes_of_design_pts[idxes_sort[-1] + 1]] - X[idxes_of_design_pts[idxes_sort[0]]]))
                                * (X[idxes_of_design_pts[idxes_sort[-1] + 1]] - X[idxes_of_design_pts[idxes_sort[0]]])
                                >= cp.sum(rhs_eq_5_as_list)]

        A = cp.Parameter((J_card, m))
        b = cp.Parameter(J_card)
        s = cp.Variable(J_card, nonneg=True)
        constrs_rhs_eq7 = [-A * (ell - ell_prev) - b <= s]

        A_new = cp.Parameter((J_card, m))
        b_new = cp.Parameter(J_card)
        s_new = cp.Variable(J_card, nonneg=True)
        constrs_rhs_eq7_new = [-A_new * (ell - ell_prev) - b_new <= s_new]

        A_8 = cp.Parameter((J_card * max_iters, m))
        b_8 = cp.Parameter(J_card * max_iters)
        constrs_eq8 = [A_8 * (ell) + b_8 <= 0]

        constrs_norm = [ell >= -M]

        constrs = constrs_log_concavity + constrs_rhs_eq7 + constrs_norm

        if modification == 0 or modification == 1:
            constrs += constrs_rhs_eq5
        if modification == 1 or modification == 3:
            constrs += constrs_rhs_eq7_new
        if modification == 2 or modification == 3:
            constrs += constrs_eq8

        weights = cp.Parameter(m)
        objf = weights.T * ell + tau * cp.sum(s) + tau * cp.sum(s_new)
        prob = cp.Problem(cp.Minimize(objf), constrs)

        return prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights

    def ccp_iterations(self, prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights,
                       m, J_card, X, idxes_of_design_pts, p, ell_g_idxes_to_sum_over, cs, ds, modification,
                       want_max=False,
                       want_verbose=False, solver_choice="MOSEK",
                       tau_max=1e3, tau_init=1e-3, opt_tol=1e-4, s_tol=1e-4, mu=8.0, max_iters=50, min_iters=10):

        if (want_max == False):
            weights.value = [0 for i in range(0, p)] + [1] + [0 for i in range(p + 1, m)]
        else:
            weights.value = [0 for i in range(0, p)] + [-1] + [0 for i in range(p + 1, m)]

        solver_failed = False
        objf_val_prev = np.inf
        objf_val = np.inf
        tau.value = tau_init
        g_kde = gaussian_kde(self.X)
        ell_prev.value = np.log(g_kde.evaluate(self.X[self.idxes_of_design_pts]))
        s_sum = np.inf

        A_value_8 = np.zeros((J_card * max_iters, m))
        b_value_8 = -np.array(ds * max_iters)

        for iter_idx in range(max_iters):
            start = time.time()
            #ineq 7
            A_value = np.zeros((J_card, m))
            b_value = -np.array(cs)
            for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
                for i in idxes:
                    if (i == 0):
                        rhs_of_eq7_value = rhs_of_eq7_new(
                            np.concatenate((np.array([ell_prev[2].value]), np.array([ell_prev[1].value]))),
                            X[idxes_of_design_pts[2]], X[idxes_of_design_pts[1]], X[idxes_of_design_pts[0]])
                        rhs_of_eq7_grad = grad_rhs_of_eq7_new(
                            np.concatenate((np.array([ell_prev[2].value]), np.array([ell_prev[1].value]))),
                            X[idxes_of_design_pts[2]], X[idxes_of_design_pts[1]], X[idxes_of_design_pts[0]])
                        A_value[idxes_idx, 2] += rhs_of_eq7_grad[0]
                        A_value[idxes_idx, 1] += rhs_of_eq7_grad[1]
                        b_value[idxes_idx] += rhs_of_eq7_value
                    else:
                        rhs_of_eq7_value = rhs_of_eq7(
                                np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                                X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                        rhs_of_eq7_grad = grad_rhs_of_eq7(
                                np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                                X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                        A_value[idxes_idx, i] += rhs_of_eq7_grad[0]
                        A_value[idxes_idx, i - 1] += rhs_of_eq7_grad[1]
                        b_value[idxes_idx] += rhs_of_eq7_value
            # ineq 7 new
            if modification == 1 or modification == 3:
                A_value_new = np.zeros((J_card, m))
                b_value_new = -np.array(cs)
                for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
                    for i in idxes:
                        if (i == m-2):
                            rhs_of_eq7_value = rhs_of_eq7(
                                np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                                X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                            rhs_of_eq7_grad = grad_rhs_of_eq7(
                                    np.concatenate((np.array([ell_prev[i].value]), np.array([ell_prev[i - 1].value]))),
                                    X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]], X[idxes_of_design_pts[i - 1]])
                            A_value_new[idxes_idx, i] += rhs_of_eq7_grad[0]
                            A_value_new[idxes_idx, i - 1] += rhs_of_eq7_grad[1]
                            b_value_new[idxes_idx] += rhs_of_eq7_value
                        else:
                            rhs_of_eq7_value = rhs_of_eq7_new(
                                np.concatenate((np.array([ell_prev[i + 2].value]), np.array([ell_prev[i + 1].value]))),
                                X[idxes_of_design_pts[i + 2]], X[idxes_of_design_pts[i + 1]],
                                X[idxes_of_design_pts[i]])
                            rhs_of_eq7_grad = grad_rhs_of_eq7_new(
                                np.concatenate((np.array([ell_prev[i + 2].value]), np.array([ell_prev[i + 1].value]))),
                                X[idxes_of_design_pts[i + 2]], X[idxes_of_design_pts[i + 1]],
                                X[idxes_of_design_pts[i]])
                            A_value_new[idxes_idx, i + 2] += rhs_of_eq7_grad[0]
                            A_value_new[idxes_idx, i + 1] += rhs_of_eq7_grad[1]
                            b_value_new[idxes_idx] += rhs_of_eq7_value

            # ineq 8
            if modification == 2 or modification == 3:
                for idxes_idx, idxes in enumerate(ell_g_idxes_to_sum_over):
                    for i in idxes:
                        lhs_of_eq8_value = rhs_of_eq8(
                            np.concatenate((np.array([ell_prev[i + 1].value]), np.array([ell_prev[i].value]))),
                            X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]])
                        lhs_of_eq8_grad = grad_rhs_of_eq8(
                            np.concatenate((np.array([ell_prev[i + 1].value]), np.array([ell_prev[i].value]))),
                            X[idxes_of_design_pts[i + 1]], X[idxes_of_design_pts[i]])
                        A_value_8[iter_idx * J_card + idxes_idx, i + 1] += lhs_of_eq8_grad[0]
                        A_value_8[iter_idx * J_card + idxes_idx, i] += lhs_of_eq8_grad[1]
                        b_value_8[iter_idx * J_card + idxes_idx] += lhs_of_eq8_value
                    b_value_8[iter_idx * J_card + idxes_idx] += - ell_prev.value.dot(A_value_8[iter_idx * J_card + idxes_idx, :])

            try:
                A.value = A_value
                b.value = b_value
                if modification == 1 or modification == 3:
                    A_new.value = A_value_new
                    b_new.value = b_value_new
                if modification == 2 or modification == 3:
                    A_8.value = A_value_8
                    b_8.value = b_value_8
                if (solver_choice == "ECOS"):
                    prob.solve(verbose=want_verbose, solver=solver_choice, warm_start=True, abstol=1e-9, reltol=1e-9)  # , abstol_inacc=1e-9, reltol=1e-9, reltol_inacc=1e-9, feastol=1e-9
                elif (solver_choice == "MOSEK"):
                    prob.solve(verbose=want_verbose, solver=solver_choice, warm_start=True)
                elif (solver_choice == "SCS"):
                    prob.solve(verbose=want_verbose, solver=solver_choice, warm_start=True, eps=1e-9, max_iters=10000)
                else:
                    print("ERROR: unsupported solver")
                    break
                s_sum = np.sum(s.value) + np.sum(s_new.value)
                objf_val = ell[p].value
                end = time.time()
                print("  objf (i.e., log prob)=%f, prob=%f, slack sum=%f, tau=%f, prob.status=%s, after iter #%d, time elapsed=%f" %
                    (objf_val, np.exp(objf_val), s_sum, tau.value, prob.status, iter_idx + 1, (end - start)))
            except Exception as e:
                solver_failed = True
                print("  WARNING: solver failed w/ prob.status=%s + error message=%s; quitting early ..." %
                      (prob.status, e))
                break

            if((np.abs(objf_val_prev - objf_val) <= opt_tol * np.abs(objf_val) \
                 and (s_sum < s_tol) and (iter_idx >= min_iters)) \
                    or (solver_failed == True) ):
                print("  quitting early ...")
                break
            ell_prev.value = ell.value
            tau.value = min(mu * tau.value, tau_max)
            objf_val_prev = objf_val

        # print result
        if not want_max:
            print("minimizing over ell[",p,"]:",
            "result: objf (i.e., log prob)=", objf_val,
            "slack sum=", s_sum, "tau=", tau.value, "prob.status=", prob.status,
            "num of iters", iter_idx)
        else:
            print("maximizing over ell[",p,"]:",
            "result: objf (i.e., log prob)=", objf_val,
            "slack sum=", s_sum, "tau=", tau.value, "prob.status=", prob.status,
            "num of iters", iter_idx)

        return solver_failed, s_sum

    def worker(self, X, m, J_card, ell_g_idxes_to_sum_over, idxes_of_design_pts, ds, cs, modification,
               M, tau_max, mu, max_iters, solver_choice,
               pipe):

        prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights \
            = self.generate_ccp_subp(M, m, J_card, ell_g_idxes_to_sum_over, X, idxes_of_design_pts, ds, modification, max_iters)

        while True:
            pipe_receive = pipe.recv()
            p = pipe_receive
            if p >= m:
                print ("invalid test point index", p)
                pipe.send(0)
                break
            print("worker received", p, "begin ccp iterations")

            solver_failed, s_sum = self.ccp_iterations(prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights,
                                                       m, J_card, X, idxes_of_design_pts, p, ell_g_idxes_to_sum_over, cs, ds,
                                                       modification,
                                                       solver_choice=solver_choice,
                                                       tau_max=tau_max, mu=mu, max_iters=max_iters)


            if (solver_failed == False):
                self.lo[p] = np.exp(ell[p].value)
                self.lo_slack[p] = s_sum
            else:
                self.lo[p] = np.nan
                self.lo_slack[p] = np.inf

            solver_failed, s_sum = self.ccp_iterations(prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights,
                                                       m, J_card, X, idxes_of_design_pts, p, ell_g_idxes_to_sum_over, cs, ds,
                                                       modification,
                                                       solver_choice=solver_choice,
                                                       tau_max=tau_max, mu=mu, max_iters=max_iters,
                                                       want_max=True)
            if (solver_failed == False):
                self.hi[p] = np.exp(ell[p].value)
                self.hi_slack[p] = s_sum
            else:
                self.hi[p] = np.nan
                self.hi_slack[p] = np.inf

            pipe.send(0)

    def compute_pw_conf_ints(self, thread_num, M, tau_max, mu, max_iters, modification=0, solver_choice="MOSEK"):
        # Setup the workers
        pipes = []
        procs = []

        print("setup workers")

        t0 = time.time()

        for i in range(thread_num):
            local, remote = Pipe()
            pipes += [local]

            procs += [Process(target=self.worker, args=(self.X, self.m, self.J_card, self.ell_g_idxes_to_sum_over,
                                                        self.idxes_of_design_pts, self.ds, self.cs, modification,
                                                        M, tau_max, mu, max_iters, solver_choice,
                                                        remote))]
            procs[-1].start()

        outer_loop_num = self.num_design_pts_to_opt // thread_num + 1
        idxes_opt_pts_in_design_pts_ext = [self.idxes_opt_pts_in_design_pts[i] for i in range(self.num_design_pts_to_opt)] \
                                            + [np.inf for i in range(self.num_design_pts_to_opt, outer_loop_num * thread_num)]

        t1 = time.time()

        for j in range(outer_loop_num):
            [pipes[i].send(idxes_opt_pts_in_design_pts_ext[j * thread_num + i]) for i in range(thread_num)]
            [pipe.recv() for pipe in pipes]
            print("outer loop", j, self.lo[:], self.hi[:])

        t2 =time.time()

        [proc.terminate() for proc in procs]
        print("Done. Data:", self.X[self.idxes_of_design_pts])
        print("lo =", self.lo[:])
        print("hi = ", self.hi[:])
        print("lo slack =", self.lo_slack[:])
        print("number of lo slack >= tol:", np.sum(1 * (np.array(self.lo_slack[:]) >= 1e-4)))
        print("high slack = ", self.hi_slack[:])
        print("number of hi slack >= tol:", np.sum(1 * (np.array(self.hi_slack[:]) >= 1e-4)))
        print("time cost: setup workers: ", t1 - t0, "algorithm:", t2 - t1)

        self.lo_opt_pts = np.array([self.lo[self.idxes_opt_pts_in_design_pts[i]] for i in range(self.num_design_pts_to_opt)])
        self.hi_opt_pts = np.array([self.hi[self.idxes_opt_pts_in_design_pts[i]] for i in range(self.num_design_pts_to_opt)])