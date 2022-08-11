import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
def mysum(double[:, ::1] array_A, double[:, ::1] array_B):
    cdef double value = 0
    cdef Py_ssize_t card_A = array_A.shape[0]
    cdef Py_ssize_t card_B = array_B.shape[0]

    for i in range(card_A):
        value += array_A[i,1]

    return value

