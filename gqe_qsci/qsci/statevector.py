import numpy as np

class SCIVector(np.ndarray):
    def __array_finalize__(self, obj):
        self._strs = getattr(obj, "_strs", None)


def as_scivector(coeffs, strs) -> SCIVector:
    scivec = np.asarray(coeffs).view(SCIVector)
    scivec._strs = np.asarray(strs, dtype=np.uint64)
    return scivec