import numpy as np 
from math import factorial
try:
    from thewalrus import perm
except:
    print("You don't have 'thewalrus' installed, so you can't use it's fast permanent function,",
        "so you have to use a slower one instead")
    def perm(M):
        """
        calculate matrix permanents, only requiring numpy.
        taken from: https://github.com/scipy/scipy/issues/7151
        it uses Glynn's formula apparently
        """
        n = M.shape[0]
        d = np.ones(n)
        j =  0
        s = 1
        f = np.arange(n)
        v = M.sum(axis=0)
        p = np.prod(v)
        while (j < n-1):
            v -= 2*d[j]*M[j]
            d[j] = -d[j]
            s = -s
            prod = np.prod(v)
            p += s*prod
            f[0] = 0
            f[j] = f[j+1]
            f[j+1] = j+1
            j = f[0]
        return p/2**(n-1) 

fact_list = [factorial(n) for n in range(30)]

def create_get_amp_from_out_modes(U, out_modes):
       
    if len(out_modes) > 1:    
        def get_amp(in_modes):
            sub_matr = U[np.ix_(out_modes, in_modes)]
            return perm(sub_matr)
            
    elif len(out_modes) == 1:
        def get_amp(in_modes):
            return U[out_modes[0], in_modes[0]] 

    else: #if len(in_modes) == 0
        def get_amp(in_modes):
            return 1.

    return get_amp

def create_get_amp_from_in_modes(U, in_modes):
       
    if len(in_modes) > 1:    
        def get_amp(out_modes):
            sub_matr = U[np.ix_(out_modes, in_modes)]
            return perm(sub_matr)
            
    elif len(in_modes) == 1:
        def get_amp(out_modes):
            return U[out_modes[0], in_modes[0]] 

    else: #if len(in_modes) == 0
        def get_amp(out_modes):
            return 1.

    return get_amp