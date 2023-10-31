import numpy as np
import cvxpy as cp
import math

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

class opt_solver:
    def __init__(self, C, R):
        self.C = C
        self.R = R

        self.Di = np.load('data/Di.npy')
        self.bi = np.load('data/bi.npy')
        self.lb = np.load('data/lb.npy')
        self.Ai = np.load('data/Ai.npy')
        self.ub = np.load('data/ub.npy')

        self.T = self.C.shape[1]
        self.N = self.C.shape[0]
        self.d = self.C.shape[2]
        self.m = self.Di.shape[0]

    def gi(self, xi):
        return self.Di@xi - self.bi

    def get_opt_val(self, t):
        var_x = cp.Variable((self.N, self.d))

        obj = 0.
        for i in range(self.N):
            obj = obj + self.C[i, t] @ var_x[i] + (self.R[i, t]/2) * cp.norm(var_x[i])**2

        cons1 = [self.Ai @ var_x[i] >= self.lb for i in range(self.N)]
        cons2 = [self.Ai @ var_x[i] <= self.ub for i in range(self.N)]
        coupling_cons = np.zeros(self.m)
        for i in range(self.N):
            coupling_cons = coupling_cons + self.gi(var_x[i])
        cons3 = [coupling_cons <= np.zeros(self.m)]
        cons = cons1 + cons2 + cons3

        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.MOSEK)
        return prob.value

    def get_OptVal_list(self):
        opt_val_list = []
        sum_opt_val = 0.
        logging.info('Solving the optimal values ...')
        for t in range(self.T):
            opt_val = self.get_opt_val(t)
            sum_opt_val += opt_val
            opt_val_list.append(sum_opt_val)
        return opt_val_list

class DUST:
    def __init__(self, network, C, R, opt_val_list):
        self.W_subG = network
        self.B = self.W_subG.shape[0]
        self.C = C
        self.R = R
        self.opt_val_list = opt_val_list

        self.Di = np.load('data/Di.npy')
        self.bi = np.load('data/bi.npy')
        self.lb = np.load('data/lb.npy')
        self.Ai = np.load('data/Ai.npy')
        self.ub = np.load('data/ub.npy')

        self.T = self.C.shape[1]
        self.N = self.W_subG.shape[1]
        self.d = self.C.shape[2]
        self.m = self.Di.shape[0]

        # algorithm init: t=0
        self.x = np.zeros((self.N, self.d))
        for i in range(self.N):
            self.x[i] = np.concatenate((3.87*np.ones(5), -4.11*np.ones(6), 3.87*np.ones(6), -4.11*np.ones(3), 3.87*np.ones(3), 3.8*np.ones(1)), axis=None)
        self.y = np.zeros((self.N, self.m))
        for i in range(self.N):
            self.y[i] = self.gi(self.x[i])
        self.c = np.ones(self.N)
        self.mu = np.zeros((self.N, self.m))

        # construct the opt prob
        self.var_xi = cp.Variable(self.d)
        var_diff = cp.Variable(self.d)
        var_gi = cp.Variable(self.m)

        self.param_Vt_grad = cp.Parameter(self.d)
        self.param_lambda = cp.Parameter(self.m)
        self.param_eta = cp.Parameter(nonneg=True)
        self.param_xit = cp.Parameter(self.d)

        obj = self.param_Vt_grad @ var_diff + self.param_lambda @ var_gi + self.param_eta * cp.quad_form(var_diff, np.identity(self.d))

        cons = [
        var_diff == self.var_xi - self.param_xit,
        var_gi == self.Di@self.var_xi - self.bi,
        self.Ai @ self.var_xi >= self.lb,
        self.Ai @ self.var_xi <= self.ub
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dcp(dpp=True)

    def gi(self, xi):
        return self.Di@xi - self.bi

    def compute_metrics(self):
        '''
        This method returns two lists that store the metrics at each step: reg_log & cons_vio_log
        '''

        tot_value = 0.
        reg_log = []
        tot_vio = np.zeros(self.m)
        cons_vio_log = []

        x_tp1 = np.zeros((self.N, self.d))
        y_tp1 = np.zeros((self.N, self.m))
        c_tp1 = np.zeros(self.N)
        mu_tp1 = np.zeros((self.N, self.m))

        logging.info('DUST begins ...')

        for t in range(self.T):
            subG_idx = t % self.B
            round_value = 0.
            round_vio = np.zeros(self.m)
            for i in range(self.N):
                hat_mu_i = self.W_subG[subG_idx, i] @ self.mu
                hat_y_i = self.W_subG[subG_idx, i] @ self.y
                c_i_tp1 = self.W_subG[subG_idx, i] @ self.c
                lambda_i_tp1 = (1/c_i_tp1) * hat_mu_i

                self.param_Vt_grad.value = math.sqrt(t+1) * (self.C[i, t] + self.R[i, t]*self.x[i])
                self.param_lambda.value = lambda_i_tp1
                self.param_eta.value = t+1
                self.param_xit.value = self.x[i]
                try:
                    self.prob.solve(solver=cp.ECOS)
                    x_i_tp1 = self.var_xi.value
                except cp.error.SolverError as e:
                    logging.info(e)
                    x_i_tp1 = self.x[i]

                y_i_tp1 = hat_y_i + self.gi(x_i_tp1) - self.gi(self.x[i])
                mu_i_tp1 = np.maximum(hat_mu_i + y_i_tp1, np.zeros(self.m))

                # update x, y, c, mu
                x_tp1[i] = x_i_tp1
                y_tp1[i] = y_i_tp1
                c_tp1[i] = c_i_tp1
                mu_tp1[i] = mu_i_tp1

                round_value += self.C[i, t] @ self.x[i] + (self.R[i, t]/2) * np.linalg.norm(self.x[i])**2
                round_vio += self.gi(self.x[i])

            self.x = x_tp1.copy()
            self.y = y_tp1.copy()
            self.c = c_tp1.copy()
            self.mu = mu_tp1.copy()

            tot_value += round_value
            tot_vio += round_vio

            opt_val = self.opt_val_list[t]
            reg = tot_value - opt_val
            reg_log.append(reg/(t+1))

            cons_vio = np.linalg.norm(np.maximum(tot_vio, np.zeros(self.m)))
            cons_vio_log.append(cons_vio/(t+1))
            #logging.info('T = %s, Reg(T)/T = %s, cons_vio/T = %s', t+1, reg/(t+1), cons_vio/(t+1))

        return reg_log, cons_vio_log

class DOPP:
    def __init__(self, network, C, R, opt_val_list):
        self.W_subG = network
        self.B = self.W_subG.shape[0]
        self.C = C
        self.R = R
        self.opt_val_list = opt_val_list

        self.Di = np.load('data/Di.npy')
        self.bi = np.load('data/bi.npy')
        self.lb = np.load('data/lb.npy')
        self.Ai = np.load('data/Ai.npy')
        self.ub = np.load('data/ub.npy')

        self.T = self.C.shape[1]
        self.N = self.W_subG.shape[1]
        self.d = self.C.shape[2]
        self.m = self.Di.shape[0]

        # algorithm init: t=0
        self.c = np.ones(self.N)
        self.x = np.zeros((self.N, self.d))
        for i in range(self.N):
            self.x[i] = np.concatenate((3.87*np.ones(5), -4.11*np.ones(6), 3.87*np.ones(6), -4.11*np.ones(3), 3.87*np.ones(3), 3.8*np.ones(1)), axis=None)
        self.mu = np.zeros((self.N, self.m))
        self.y = np.zeros((self.N, self.m))
        for i in range(self.N):
            self.y[i] = self.gi(self.x[i])
        self.kappa=0.2

        # construct the opt prob
        self.var_xi = cp.Variable(self.d)
        self.param_diff = cp.Parameter(self.d)

        obj = cp.norm(self.param_diff - self.var_xi)

        cons = [
        self.Ai @ self.var_xi >= self.lb,
        self.Ai @ self.var_xi <= self.ub
        ]

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dcp(dpp=True)

    def gi(self, xi):
        return self.Di@xi - self.bi

    def compute_metrics(self):
        '''
        This method returns two lists that store the metrics at each step: reg_log & cons_vio_log
        '''

        tot_value = 0.
        reg_log = []
        tot_vio = np.zeros(self.m)
        cons_vio_log = []

        c_tp1 = np.zeros(self.N)
        x_tp1 = np.zeros((self.N, self.d))
        mu_tp1 = np.zeros((self.N, self.m))
        y_tp1 = np.zeros((self.N, self.m))

        logging.info('DOPP begins ...')
        for t in range(self.T):
            subG_idx = t % self.B
            round_value = 0.
            round_vio = np.zeros(self.m)
            if t == 0:
                alpha = 1.
                beta = 1.
            else:
                alpha = 1/math.sqrt(t)
                beta = 1/t**self.kappa

            for i in range(self.N):
                c_i_tp1 = self.W_subG[subG_idx, i] @ self.c
                hat_mu_i = self.W_subG[subG_idx, i] @ self.mu
                hat_y_i = self.W_subG[subG_idx, i] @ self.y
                s_i_tp1 = self.C[i, t] + self.R[i, t]*self.x[i] + self.Di.T @ ((1/c_i_tp1) * hat_mu_i)

                self.param_diff.value = self.x[i] - alpha*s_i_tp1
                try:
                    self.prob.solve(solver=cp.ECOS)
                    x_i_tp1 = self.var_xi.value
                except cp.error.SolverError as e:
                    logging.info(e)
                    x_i_tp1 = self.x[i]

                mu_i_tp1 = np.maximum(hat_mu_i+alpha*(hat_y_i/c_i_tp1 - beta*hat_mu_i), np.zeros(self.m))
                y_i_tp1 = hat_y_i + self.gi(x_i_tp1) - self.gi(self.x[i])

                # update x, y, c, mu
                c_tp1[i] = c_i_tp1
                x_tp1[i] = x_i_tp1
                mu_tp1[i] = mu_i_tp1
                y_tp1[i] = y_i_tp1

                round_value += self.C[i, t] @ self.x[i] + (self.R[i, t]/2) * np.linalg.norm(self.x[i])**2
                round_vio += self.gi(self.x[i])

            self.x = x_tp1.copy()
            self.y = y_tp1.copy()
            self.c = c_tp1.copy()
            self.mu = mu_tp1.copy()

            tot_value += round_value
            tot_vio += round_vio

            opt_val = self.opt_val_list[t]
            reg = tot_value - opt_val
            reg_log.append(reg/(t+1))

            cons_vio = np.linalg.norm(np.maximum(tot_vio, np.zeros(self.m)))
            cons_vio_log.append(cons_vio/(t+1))
            #logging.info('T = %s, Reg(T)/T = %s, cons_vio/T = %s', t+1, reg/(t+1), cons_vio/(t+1))

        return reg_log, cons_vio_log
    
class dual_subgradient:
    def __init__(self, network, C, R, opt_val_list):
        self.W_subG = network
        self.B = self.W_subG.shape[0]
        self.C = C
        self.R = R
        self.opt_val_list = opt_val_list

        self.Di = np.load('data/Di.npy')
        self.bi = np.load('data/bi.npy')
        self.lb = np.load('data/lb.npy')
        self.Ai = np.load('data/Ai.npy')
        self.ub = np.load('data/ub.npy')

        self.T = self.C.shape[1]
        self.N = self.W_subG.shape[1]
        self.d = self.C.shape[2]
        self.m = self.Di.shape[0]

        # algorithm init: t=1
        self.x = np.zeros((self.N, self.d))
        for i in range(self.N):
            self.x[i] = np.concatenate((3.87*np.ones(5), -4.11*np.ones(6), 3.87*np.ones(6), -4.11*np.ones(3), 3.87*np.ones(3), 3.8*np.ones(1)), axis=None)
        self.mu = np.zeros(self.m)

    def gi(self, xi):
        return self.Di@xi - self.bi
    
    def sum_gi(self, x):
        sum_gi = np.zeros(self.m)
        for i in range(self.N):
            sum_gi = sum_gi + self.gi(x[i])
        return sum_gi
    
    def sum_fi(self, x, t):
        sum_fi = 0.
        for i in range(self.N):
            sum_fi = sum_fi + self.C[i, t] @ x[i] + (self.R[i, t]/2) * np.linalg.norm(x[i])**2
        return sum_fi
    
    def _solve_argmin_prob(self, t, mu_t):
        var_x = cp.Variable((self.N, self.d))

        f_t = 0.
        g = np.zeros(self.m)
        for i in range(self.N):
            f_t = f_t + self.C[i, t] @ var_x[i] + (self.R[i, t]/2) * cp.norm(var_x[i])**2
            g = g + self.gi(var_x[i])

        obj = f_t + mu_t @ g
        cons1 = [self.Ai @ var_x[i] >= self.lb for i in range(self.N)]
        cons2 = [self.Ai @ var_x[i] <= self.ub for i in range(self.N)]
        cons = cons1 + cons2
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.MOSEK)
        return var_x.value
    
    def compute_metrics(self):
        '''
        This method returns two lists that store the metrics at each step: reg_log & cons_vio_log
        '''

        step_size = 0.108
        tot_value = 0.
        reg_log = []
        tot_vio = np.zeros(self.m)
        cons_vio_log = []

        logging.info('Dual subgradient begins ...')
        for t in range(self.T-1):
            x_tp1 = self._solve_argmin_prob(t+1, self.mu)
            mu_tp1 = np.maximum(self.mu + step_size * self.sum_gi(x_tp1), np.zeros(self.m))
            round_value = self.sum_fi(self.x, t)
            round_vio = self.sum_gi(self.x)

            self.x = x_tp1.copy()
            self.mu = mu_tp1.copy()

            tot_value += round_value
            tot_vio += round_vio

            opt_val = self.opt_val_list[t]
            reg = tot_value - opt_val
            reg_log.append(reg/(t+1))

            cons_vio = np.linalg.norm(np.maximum(tot_vio, np.zeros(self.m)))
            cons_vio_log.append(cons_vio/(t+1))

        round_value = self.sum_fi(self.x, self.T-1)
        round_vio = self.sum_gi(self.x)
        tot_value += round_value
        tot_vio += round_vio
        reg = tot_value - self.opt_val_list[self.T-1]
        reg_log.append(reg/self.T)
        cons_vio = np.linalg.norm(np.maximum(tot_vio, np.zeros(self.m)))
        cons_vio_log.append(cons_vio/self.T)

        return reg_log, cons_vio_log