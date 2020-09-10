import numpy as np 
import numba

parallel=True

@numba.jit(nopython=True)
def numba_combinations(pool, r):
    """
    a numba compatible version of 
    itertools combinations function
    """
    n = len(pool)
    indices = list(range(r))
    empty = not(n and (0 < r <= n))

    if not empty:
        result = [pool[i] for i in indices]
        yield result

    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1

            result = [pool[i] for i in indices]
            yield result

@numba.jit(nopython=True, parallel=parallel)
def multi_perm(M, D):
    N = M.shape[0]
    rangeN = np.arange(N)

    combs = []
    for i in numba.prange(1,N+1):
        combs.append(list(numba_combinations(rangeN, i)))

    W = 0j * np.zeros((N,N,N)) 
    for k in range(N):
        for l in range(N):
            for j in range(N):
                W[k,l,j] = M[j,k] * M[j,l].conjugate() * D[l,k]
    P = 0.
    for R_size in numba.prange(1,N+1):
        for S_size in numba.prange(R_size,N+1):
            a = (2 - int(R_size == S_size)) * (-1) ** (S_size + R_size)
            for R in combs[R_size-1]:
                for S in combs[S_size-1]:
                    b = a
                    for j in range(N):
                        c = 0.
                        for r in R:
                            for s in S:
                                c += W[s,r,j]
                        b *= c
                    P += b.real 
    return P