import numpy as np
from numpy import *
from sympy import simplify, expand, symbols, Matrix, lambdify, MatrixSymbol, sympify, parse_expr
from scipy.optimize import minimize
from contextlib import contextmanager
import threading
import _thread
import time
import ot
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial





class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**' + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq: eq = eq.replace(c, 'C')
    return simplify_eq(eq)

dim_sys = 3
def path_func(t, param=2*np.ones(dim_sys)):

    return np.array([param[0]*np.exp(t), param[1]*np.exp(2*t)]) # -np.log(np.exp(-param) - t)

# path_func = np.vectorize(path_func)

TimeIdx = 3*np.arange(5) 
n = 50 # number of samples

data = np.empty((dim_sys, n, len(TimeIdx)))
df = pd.read_table("scRNA2.txt", delimiter=",", header=None)
nb_time = 5


for dim in range(dim_sys):
    for idx, time_idx in enumerate(TimeIdx):
        data[dim, :, idx] = df.iloc[idx*400+350: 100+300+ (idx)*400, dim] # + np.random.randn(n)


def reward(eq_loc):

    # Find numerical values in the equation
    eq_ = []
    num_values = []
    nb_plus = []
    for d in range(dim_sys):
        eq_.append(simplify_eq(eq_loc[d]))
        num_values.append(re.findall(r'\b\d+(?:\.\d+)?\b', eq_[d]))
        nb_plus.append(eq_[d].count("+"))
        num_values[d] = np.array([float(num) for num in num_values[d]]) # Discarding trivial equations
        if ((np.all(np.abs(num_values[d])< 0.1)) and (len(num_values[d]) > nb_plus[d]) ) or (eq_[d] == '0'):
            return 10000000000000

    def eval_eq(t, var):
        res = []
        for i in range(1, dim_sys+1):
            globals()["x"+str(i)] = var[i-1]
        for i in range(dim_sys):
            res.append(eval(eq_loc[i]))
        return np.array(res)


    estim = np.zeros((dim_sys, n, len(TimeIdx)))
    r = 0
    estim[:, :, 0] = data[:, :, 0]
    for idx in range(0, len(TimeIdx)-1):

        t = TimeIdx[idx + 1]
        for sample in range(n):
            
            nxt_samp = integrate.RK45(eval_eq, 0, data[:, sample, 0], TimeIdx[-1]+1, max_step=0.1)
            space_trj = {}
            while nxt_samp.t < t and nxt_samp.status != "failed":
                nxt_samp.step()
                space_trj[nxt_samp.t] = nxt_samp.y
            estim[:, sample, idx+1] = nxt_samp.y



        for i in range(dim_sys):

            M = ot.dist(data[i, :, idx].reshape((n, 1)), estim[i, :, idx].reshape((n, 1)))
            M /= M.max()
                
            
            hist1, _ = np.histogram(data[i, :, idx], bins=n)
            hist2, _ = np.histogram(estim[i, :, idx], bins=n)

            r += ot.emd2(hist1/n , hist2/n , M)

    return r

def score_with_est( eq_comp, t_limit=10000.0):
    """
    Calculate reward score for a complete parse tree
    If placeholder C is in the equation, also excute estimation for C
    Reward = 1 / (1 + MSE) * Penalty ** num_term

    Parameters
    ----------
    eq_comp : Str object.
        the discovered equation (with placeholders for coefficients).
    t_limit : Float object.
        time limit (seconds) for ssingle evaluation, default 1 second.

    Returns
    -------
    score: Float
        discovered equations.
    eq: Str
        discovered equations with estimated numerical values.
    """

    ## count number of numerical values in eq
    dim = len(eq_comp)
    c_count = []
    for i in range(dim):
        c_count.append(eq_comp[i].count('C'))
    # start_time = time.time()
    with time_limit(t_limit, 'sleep'):
        try:
            if c_count == [0 for _ in range(dim)]:  ## no numerical values
                f_pred = reward(eq_comp)
            elif np.sum(c_count) >= 20:  ## discourage over complicated numerical estimations
                return 0, eq_comp
            else:  ## with numerical values: coefficient estimation with Powell method

                c_lst = ['c' + str(i) for i in range(np.sum(c_count))]
                
                for d in range(dim):
                    for c in range(c_count[d]):
                        eq_comp[d] = eq_comp[d].replace('C', c_lst[int(c + np.sum(c_count[:d]))], 1)

                def eq_test(c):
                    nonlocal eq_comp
                    eq_loc = eq_comp.copy()
                    for d in range(dim):
                        for i in range(c_count[d]):
                            idx = int(i + np.sum(c_count[:d]))
                            globals()['c' + str(idx)] = c[idx]
                            eq_loc[d] = eq_loc[d].replace('c' + str(idx), str(c[idx]))

                    return reward(eq_comp) 

                x0 =  [0.0] * len(c_lst)
                opt = {'maxiter':20, 'disp':True}
                c_lst = minimize(eq_test, x0, method='Powell', tol=1e-2, options=opt).x.tolist()

                c_lst = [np.round(x, 4) for x in c_lst]
                eq_est = eq_comp.copy()
                
                for d in range(dim):
                    for i in range(c_count[d]):
                        idx = int(i + np.sum(c_count[:d]))
                        eq_est[d] = eq_est[d].replace('c' + str(idx), str(c_lst[idx]), 1)
                    eq_comp[d] = eq_est[d].replace('+-', '-')
                f_pred = reward(eq_comp)
                

        except:
             return 0, eq_comp

    r = 1 / (1 + f_pred )
    return r, eq_comp
