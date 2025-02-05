# CODE ADAPTED FROM https://github.com/andyfangzheng 

import sys
import numpy as np
sys.path.append(r'../')

from sdfl_train import run_sdfl
import time
from score import  score_with_est
from uni_dim import SdflBase
from sdfl_task_utils import *
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

dim_sys = 3

start_time = time.time()


e = []
for i in range(1, dim_sys+1):
    e.append('+ ' +'sin(x'+str(1)+'-x'+str(i)+')')
    for j in range(2, dim_sys+1):
        if j != i:
            e[i-1] += ' + ' + 'sin(x'+str(j)+'-x'+str(i)+')'


print( score_with_est(['-0.3362*cos(x2)*x1 + 0.1679*cos(x3)*x1', '-0.0905*cos(x1)*x2 + 0.0013*cos(x3)*x2', '0.029*cos(x2)*x3 + 0.0023*cos(x1)*x3']))

print( score_with_est( ['-0.8768*x1 + -1.2341*x2 + 0.008*x3', '-0.0004*x1 + -0.0247*x2 + -0.1453*x3', '0.0041*x1 + -0.0005*x2 + 0.0001*x3']) )


end_time = time.time()
execution_time = end_time - start_time
print(" time = ", execution_time )


# To run SDFL, uncomment the code below and add your system data file name in the score.py file

# start_time = time.time()
# all_eqs, _, _ = run_sdfl('elem_symb',
#                         num_run=1 ,
#                         max_len=20,
#                         eta=1-1e-3,
#                         max_module_init=20,
#                         num_transplant=1,
#                         num_aug=0,
#                         transplant_step=500,
#                         count_success=False)

# end_time = time.time()
# execution_time = end_time - start_time

# print(" time = ", execution_time )

# print("all eq ", all_eqs)



