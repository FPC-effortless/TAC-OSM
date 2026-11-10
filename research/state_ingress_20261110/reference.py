"""Independent scalar/array domain reference; NOT whole model or IEEE proof."""
import numpy as np


def admissible(numerator, mass, epsilon, dtype):
    n=np.asarray(numerator);d=np.asarray(mass)
    eps=float(epsilon)
    return (n.shape==d.shape and n.dtype==d.dtype==np.dtype(dtype) and
            np.isfinite(eps) and np.finfo(dtype).tiny<=eps<=1e-3 and
            np.isfinite(n).all() and np.isfinite(d).all() and
            (d>=0).all() and (np.abs(n)<=d).all())
