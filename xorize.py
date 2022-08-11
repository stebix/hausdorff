"""
Provide tools to perform XOR operation before the Hausdorff metric
computation to reduce computational load.
"""
import numpy as np
import cupy as cp

def xorize(array_a: np.ndarray, array_b: np.ndarray) -> dict:
    # Determine desired numerical array framework directly 
    # from the arguments
    xp = cp.get_array_module(array_a, array_b)
    xor = xp.logical_xor(array_a, array_b)
    congruencecount = xp.sum(xp.logical_not(xor))
    
