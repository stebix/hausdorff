import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int squared_distance(int[:] vector_a, int[:] vector_b):
    """Fast squared Euclidean distance for two vectors from R^3"""
    cdef int d = 0
    cdef Py_ssize_t k = 0
    for k in range(3):
        d += (vector_a[k] - vector_b[k]) ** 2



@cython.boundscheck(False)
@cython.wraparound(False)
def directed_hausdorff_distances(double[:, ::1] X, double[:, ::1] Y):
    cdef double dmax = 0
    cdef double dmin = 0
    cdef double d = 0
    cdef Py_ssize_t card_X = X.shape[0]
    cdef Py_ssize_t card_Y = Y.shape[0]
    cdef int vec_dims = X.shape[1]
    cdef Py_ssize_t i, j, k

    # cdef np.ndarray[np.float64_t, ndim=1, mode='c'] cache
    cache = np.zeros(X.shape[0], dtype=np.float)
    cdef double[:] cacheview = cache

    dmax = 0
    for i in range(card_X):
        cmin = np.inf
        for j in range(card_Y):
            d = 0
            for k in range(vec_dims):
                d += (X[i, k] - Y[j, k]) ** 2
            if d < dmax:
                break
            
            if d < dmin:
                dmin = d
        
        cacheview[i] = dmin
    
    return cache



@cython.boundscheck(False)
@cython.wraparound(False)
def directed_hausdorff_distances_integer(long[:, ::1] X, long[:, ::1] Y):
    cdef long dmax = 0
    cdef long dmin = 0
    cdef long d = 0
    cdef Py_ssize_t card_X = X.shape[0]
    cdef Py_ssize_t card_Y = Y.shape[0]
    cdef long vec_dims = X.shape[1]
    cdef Py_ssize_t i, j, k

    # cdef np.ndarray[np.float64_t, ndim=1, mode='c'] cache
    cache = np.zeros(X.shape[0], dtype=np.int32)
    cdef long[:] cacheview = cache

    dmax = 0
    for i in range(card_X):
        dmin = 2000000000
        for j in range(card_Y):
            d = 0
            for k in range(vec_dims):
                d += (X[i, k] - Y[j, k]) ** 2
            if d < dmax:
                break
            
            if d < dmin:
                dmin = d
        
        cacheview[i] = dmin
    
    return cache






@cython.boundscheck(False)
def mysum(double[:, ::1] array_A, double[:, ::1] array_B):
    cdef double value = 0
    cdef Py_ssize_t card_A = array_A.shape[0]
    cdef Py_ssize_t card_B = array_B.shape[0]

    for i in range(card_A):
        value += array_A[i,1]

    return value

