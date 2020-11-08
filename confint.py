import autograd.numpy as np
from scipy.stats import beta
import time
from numpy.random import choice
from scipy.spatial import ConvexHull

from multiprocessing import Array, Pipe, Process

from penalty_convex_concave_procedure import generate_ccp_subp, ccp_iterations

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
        self.lo_opt_pts = np.zeros(self.num_design_pts_to_opt)
        self.hi_opt_pts = np.zeros(self.num_design_pts_to_opt)
        print("num design points to optimize", self.num_design_pts_to_opt)

        list_idxes_of_design_pts = list(self.idxes_of_design_pts)
        self.idxes_opt_pts_in_design_pts = [list_idxes_of_design_pts.index(self.idxes_of_design_pts_to_opt[i])
                                            for i in range(self.num_design_pts_to_opt)]
        self.improved_lo_opt_pts = None
        self.improved_hi_opt_pts = None

        # shared memory
        self.lo = Array('d', m)
        self.hi = Array('d', m)
        self.lo_slack = Array('d', m)
        self.hi_slack = Array('d', m)
        
        self.num_nans = 0

        assert(J_card == len(ds))
        assert(J_card == len(cs))
        assert(J_card == len(self.ell_g_idxes_to_sum_over))

    def sampling_pts_to_opt_over(self, sampling_ratio, tail_ratio=0.5):
        if sampling_ratio == 1.0:
            return self.idxes_of_design_pts
        sorted_idxes_of_design_pts = sorted(self.idxes_of_design_pts)
        bin_width = min([self.X[sorted_idxes_of_design_pts[i+1]] - self.X[sorted_idxes_of_design_pts[i]]
                         for i in range(self.m - 1)])
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
        sample_others = choice(other_idxes_of_design_pts, size=K - int(K * tail_ratio), replace=False,
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

    def worker(self, X, m, J_card, ell_g_idxes_to_sum_over, idxes_of_design_pts, ds, cs, M,
               want_verbose, solver_choice,
               tau_max, tau_init, opt_tol, s_tol, mu, max_iters, min_iters,
               pipe):

        prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights \
            = generate_ccp_subp(M, m, J_card, X, idxes_of_design_pts, max_iters)

        while True:
            pipe_receive = pipe.recv()
            p = pipe_receive
            if p >= m:
                if (want_verbose):
                    print("invalid test point index", p)
                pipe.send(0)
                break
            if (want_verbose):
                print("worker received", p, "begin ccp iterations")

            solver_failed, s_sum = ccp_iterations(prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights,
                                                  m, J_card, X, idxes_of_design_pts, p, ell_g_idxes_to_sum_over, cs, ds,
                                                  False,
                                                  want_verbose, solver_choice,
                                                  tau_max, tau_init, opt_tol, s_tol, mu, max_iters, min_iters)

            if (solver_failed == False):
                self.lo[p] = np.exp(ell[p].value)
                self.lo_slack[p] = s_sum
            else:
                self.lo[p] = np.nan
                self.lo_slack[p] = np.inf

            solver_failed, s_sum = ccp_iterations(prob, ell, s, s_new, ell_prev, tau, A, A_new, A_8, b, b_new, b_8, weights,
                                                  m, J_card, X, idxes_of_design_pts, p, ell_g_idxes_to_sum_over, cs, ds,
                                                  True,
                                                  want_verbose, solver_choice,
                                                  tau_max, tau_init, opt_tol, s_tol, mu, max_iters, min_iters)
            if (solver_failed == False):
                self.hi[p] = np.exp(ell[p].value)
                self.hi_slack[p] = s_sum
            else:
                self.hi[p] = np.nan
                self.hi_slack[p] = np.inf

            pipe.send(0)

    def compute_pw_conf_ints(self, thread_num=2,
                             M=8.0, tau_max=1e3, mu=8.0, max_iters=50, min_iters=20,
                             tau_init=1e-5, opt_tol=1e-4, s_tol=1e-4,
                             solver_choice="MOSEK", want_verbose=False):
        pipes = []
        procs = []

        t0 = time.time()

        for i in range(thread_num):
            local, remote = Pipe()
            pipes += [local]
            procs += [Process(target=self.worker, args=(self.X, self.m, self.J_card, self.ell_g_idxes_to_sum_over,
                                                        self.idxes_of_design_pts, self.ds, self.cs, M,
                                                        want_verbose, solver_choice,
                                                        tau_max, tau_init, opt_tol, s_tol, mu, max_iters, min_iters,
                                                        remote))]
            procs[-1].start()

        outer_loop_num = self.num_design_pts_to_opt // thread_num + 1
        idxes_opt_pts_in_design_pts_ext = [self.idxes_opt_pts_in_design_pts[i] for i in range(self.num_design_pts_to_opt)] \
                                            + [np.inf for i in range(self.num_design_pts_to_opt, outer_loop_num * thread_num)]

        t1 = time.time()

        for j in range(outer_loop_num):
            [pipes[i].send(idxes_opt_pts_in_design_pts_ext[j * thread_num + i]) for i in range(thread_num)]
            [pipe.recv() for pipe in pipes]
            if (want_verbose):
                print("outer loop ", j, " finished")
                print(self.lo[:], self.hi[:])

        t2 =time.time()

        [proc.terminate() for proc in procs]
        if (want_verbose):
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

        self.remove_nan()
        self.improve_bounds_new()
        
    def remove_nan(self):
        lo_nans = np.where(np.isnan(self.lo_opt_pts))
        hi_nans = np.where(np.isnan(self.hi_opt_pts))
        nans = np.union1d(lo_nans,hi_nans)
        self.failure_design_pts = self.X[self.idxes_of_design_pts_to_opt[nans]]
        self.num_nans = len(nans)
        self.lo_opt_pts = np.delete(self.lo_opt_pts, nans)
        self.hi_opt_pts = np.delete(self.hi_opt_pts, nans)
        self.idxes_of_design_pts_to_opt = np.delete(self.idxes_of_design_pts_to_opt, nans)
        self.num_design_pts_to_opt = len(self.idxes_of_design_pts_to_opt)
        
    def improve_bounds_new(self):
        '''
        improved bounds in 1d
        '''
        lo = np.log(self.lo_opt_pts)
        hi = np.log(self.hi_opt_pts)
        x = self.X[self.idxes_of_design_pts_to_opt]        
        m = self.num_design_pts_to_opt
        if m <= 1:
            self.improved_lo_opt_pts = None
            self.extended_x = None
            self.improved_hi_extended_pts = None
            self.improved_lo_extended_pts = None
            return

        truepoints = [np.array([x[i], lo[i]]) for i in range(m)]
        lo_min = np.min(lo)
        shadowpoints = [np.array([x[i], lo_min - 1.0]) for i in range(m)]
        allpoints = np.array(truepoints + shadowpoints)
        hull = ConvexHull(allpoints)
        result_idx = []
        for simplex in hull.simplices:
            if all(simplex < m):
                result_idx.extend([i for i in simplex])
        result_idx_sort = sorted(list(set(result_idx)))

        self.improved_lo_opt_pts = np.zeros(m)
        for i in range(len(result_idx_sort) - 1):
            self.improved_lo_opt_pts[result_idx_sort[i]] = np.exp(lo[result_idx_sort[i]])
            slop = (lo[result_idx_sort[i + 1]] - lo[result_idx_sort[i]]) / (x[result_idx_sort[i + 1]] - x[result_idx_sort[i]])
            for j in range(result_idx_sort[i], result_idx_sort[i + 1]):
                self.improved_lo_opt_pts[j] = np.exp(slop * (x[j] - x[result_idx_sort[i]]) + lo[result_idx_sort[i]])
        self.improved_lo_opt_pts[m-1] = np.exp(lo[m-1])
        
        ######
        lo = np.log(self.improved_lo_opt_pts)
        Left = np.zeros(m)
        Right = np.zeros(m)
        for k in range(1, m):
            Left[k] = min([(hi[k] - lo[j])/(x[k] - x[j]) for j in range(k)])
        for k in range(0, m-1):
            Right[k] = max([(lo[j] - hi[k])/(x[j] - x[k]) for j in range(k+1, m)])
        kink_points = np.zeros(m)
        for i in range(1, m-2):
            kink_points[i] = (hi[i+1] - hi[i] + Left[i] * x[i] - Right[i+1] * x[i+1]) / (Left[i] - Right[i+1])
        ###
        extended_x = [x[0]]
        extended_hi = [hi[0]]
        extended_lo = [lo[0]]
        for i in range(1, m-2):
            extended_x.append(x[i])
            extended_x.append(kink_points[i])
            extended_hi.append(hi[i])
            extended_hi.append(hi[i] + Left[i] * (kink_points[i] - x[i]))
            extended_lo.append(lo[i])
            slop = (lo[i] - lo[i+1]) / (x[i] - x[i+1])
            extended_lo.append(lo[i+1] + slop * (kink_points[i] - x[i+1]))
        extended_x.append(x[m-2])
        extended_hi.append(hi[m-2])
        extended_lo.append(lo[m-2])
                          
        extended_x.append(x[m-1])
        extended_hi.append(hi[m-1])
        extended_lo.append(lo[m-1])

        self.extended_x = np.array(extended_x)
        self.improved_hi_extended_pts = np.exp(np.array(extended_hi))
        self.improved_lo_extended_pts = np.exp(np.array(extended_lo))