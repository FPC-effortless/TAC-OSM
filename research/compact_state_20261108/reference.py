"""Partial value/adjoint reference only, not a whole-model/allocator proof.

No Torch or TAC imports. Storage copy C is identity on values, and its adjoint
is identity on cotangents. Physical ownership/allocator guarantees need direct
runtime evidence; these NumPy equations do not establish them for PyTorch.
"""
import numpy as np


def copy_values(values):
    return [np.array(value, order='C', copy=True) for value in values]


def copy_pullback(cotangents):
    return copy_values(cotangents)
