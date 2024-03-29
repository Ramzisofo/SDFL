import numpy as np
import pandas as pd
import time
from score import simplify_eq, score_with_est
# from spl_base import SplBase
from uni_dim import SdflBase
from sdfl_task_utils import *
from itertools import permutations
import re

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

dim = 3

def run_sdfl( task, num_run, transplant_step, data_dir='data/', max_len = 20, eta = 0.9999,
            max_module_init = 10, num_aug = 5, exp_rate = 1/np.sqrt(2), num_transplant = 20, 
            norm_threshold=1e-5, count_success = True):
    """
    Executes the main training loop of Symbolic Physics Learner.
    
    Parameters
    ----------
    task : String object.
        benchmark task name. 
    num_run : Int object.
        number of runs performed.
    transplant_step : Int object.
        number of iterations simulated for training between two transplantations. 
    data_dir : String object.
        directory of training data samples. 
    max_len : Int object.
        maximum allowed length (number of production rules ) of discovered equations.
    eta : Int object.
        penalty factor for rewarding. 
    max_module_init : Int object.
        initial maximum length for module transplantation candidates. 
    num_aug : Int object.
        number of trees for module transplantation. 
    exp_rate : Int object.
        initial exploration rate. 
    num_transplant : Int object.
        number of transplantation candidate update performed throughout traning. 
    norm_threshold : Float object.
        numerical error tolerance for norm calculation, a very small value. 
    count_success : Boolean object. 
        if success rate is recorded. 
        
    Returns
    -------
    all_eqs: List<Str>
        discovered equations. 
    success_rate: Float
        success rate of all runs performed. 
    all_times: List<Float>
        runtimes for successful runs. 
    """
    
    ## define production rules and non-terminal nodes. 
    grammars = rule_map[task]
    nt_nodes = ntn_map[task]


    all_times = []
    all_eqs = []

    sdim = 3

    module_grow_step = (max_len - max_module_init) / num_transplant

    for i_test in range(num_run):

        best_solution = ('nothing', 0)

        exploration_rate = exp_rate
        max_module = max_module_init
        reward_his = []
        best_modules = []
        aug_grammars = []

        start_time = time.time()

        success = 0

        for i_itr in range(num_transplant):

            spl_model = SdflBase(
                                base_grammars = grammars, 
                                aug_grammars = aug_grammars, 
                                nt_nodes = nt_nodes, 
                                max_len = max_len, 
                                max_module = max_module,
                                aug_grammars_allowed = num_aug,
                                func_score = score_with_est, 
                                exploration_rate = exploration_rate,
                                eta = eta)

            _, current_solution, good_modules = spl_model.run(transplant_step,
                                                              num_play=10, 
                                                              print_flag=True)

            end_time = time.time() - start_time
            all_times.append(end_time)

            if not best_modules:
                best_modules = good_modules
            else:
                best_modules = sorted(list(set(best_modules + good_modules)), key = lambda x: x[1])
            aug_grammars = [x[0] for x in best_modules[-num_aug:]]
            

            reward_his.append(best_solution[1])

            if current_solution[1] > best_solution[1]:
                best_solution = current_solution

            max_module += module_grow_step
            exploration_rate *= 5

            all_eqs = [best_solution[0] for _ in range(sdim)]
             


    return all_eqs, 1, all_times
