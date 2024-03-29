
dim = 3
## production rules 
rule_map = {
    'nguyen-1': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'nguyen-2': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],  

    'nguyen-3': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'nguyen-4': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->x**2', 'A->x**4', 
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'], 

    'nguyen-5': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-6': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-7': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                 'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)', 
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-8': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                 'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 
                 'A->log(A)', 'A->sqrt(A)'],

    'nguyen-9': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                 'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B', 
                 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                 'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-10': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B', 
                  'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-11': ['A->x', 'A->y', 'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                  'A->exp(A)', 'A->log(B)', 'A->sqrt(B)', 'A->cos(B)', 'A->sin(B)', 
                  'B->B+B', 'B->B-B', 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-12': ['A->(A+A)', 'A->(A-A)', 'A->A*A', 'A->A/A',
                  'A->x', 'A->x**2', 'A->x**4', 'A->y', 'A->y**2', 'A->y**4', 
                  'A->1', 'A->2', 'A->exp(A)', 
                  'A->cos(x)', 'A->sin(x)', 'A->cos(y)', 'A->sin(y)'], 

    'nguyen-1c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4', 
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'], 

    'nguyen-2c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4', 
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'], 

    'nguyen-5c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                  'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 'A->A*C',
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'nguyen-7c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 'A->A*C',
                  'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)', 
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'nguyen-8c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
                  'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C',
                  'A->log(A)', 'A->sqrt(A)'], 

    'nguyen-9c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C', 
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'A->exp(B)', 
                  'B->B*C', 'B->1', 'B->B+B', 'B->B-B',
                  'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y', 
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    **dict.fromkeys(['dp_f1', 'dp_f2'],
                    ['A->C*wdot*cos(x1-x2)', 'A->A+A', 'A->A*A', 'A->C*A', 
                     'A->W', 'W->w1', 'W->w2', 'W->wdot', 'W->W*W', 
                     'A->cos(T)', 'A->sin(T)', 'T->x1', 'T->x2', 'T->T+T', 'T->T-T',
                     'A->sign(S)', 'S->w1', 'S->w2', 'S->wdot', 'A->S+S', 'B->S-S']), 

    **dict.fromkeys(['lorenz_x', 'lorenz_y', 'lorenz_z'], 
                    ['A->A+A', 'A->A-A', 'A->A*A']  + ['A->x'+str(i)+' ' for i in range(1, 2+1)] + ['A->sin(A)', 'A->cos(A)', 'A->exp(A)']  )
                   #  + ['A->x'+str(j) +'-'+ 'x'+str(i) for j in range(1, dim+1) for i in range(1, dim+1)])
}

