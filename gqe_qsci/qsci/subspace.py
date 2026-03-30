from typing import Sequence

import numpy as np
from cudaq import SampleResult
  
from gqe_qsci.qsci.determinant import Determinant


class DeterminantSubspace:
    """Determinant subspace (a sequence of determinants).

    - determinants[i] represent one determinant.
    """
    def __init__(self, determinants: Sequence[Determinant]):
        self.determinants = determinants

    @property
    def ndet(self) -> int:
        return len(self.determinants)
    
    @classmethod
    def from_cudaq_sample_result(cls, sample_result: SampleResult) -> "DeterminantSubspace":
        sorted_by_count = sorted(sample_result.items(), key=lambda x: x[1], reverse=True)
        determinants = []
        for det, _ in sorted_by_count:
            determinants.append(Determinant.from_interleaved_bitstring(det, little_endian=True))
        return cls(determinants)
    
    @classmethod
    def from_count_array(cls, count_array: np.ndarray, norb: int, nelec: tuple[int, int]) -> "DeterminantSubspace":
        sampled = np.flatnonzero(count_array)
        sorted_by_count = np.argsort(count_array[sampled])[::-1]
        determinants = [Determinant.from_fullci_index(idx, norb, nelec) for idx in sampled[sorted_by_count]]
        return cls(determinants)
    
    def enlarge(self, max_dim: int, method: str = "symmetry_completion") -> "DeterminantSubspace":
        if method == "symmetry_completion":
            return self._symmetry_completion(max_dim)
        elif method == "None":
            return DeterminantSubspace(self.determinants[:max_dim])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def post_select_by_nelec(self, nelec: tuple[int, int]) -> "DeterminantSubspace":
        return DeterminantSubspace([det for det in self.determinants if det.is_nelec_preserving(nelec)])
    
    def _symmetry_completion(self, max_dim: int) -> "DeterminantSubspace":
        enlarged: list[Determinant] = []
        seen: set[tuple[int, int]] = set()

        for det in self.determinants:
            group = np.asarray(det.generate_group_for_totalspin_symmetry(), dtype=np.uint64)
            if group.ndim == 1:
                group = group.reshape(1, 2)
            elif group.ndim == 2:
                if group.shape[1] != 2:
                    raise ValueError(f"Invalid group shape: expected (N,2), got {group.shape}")

            for a, b in group:
                key = (int(a), int(b))
                if key in seen:
                    continue
                seen.add(key)
                enlarged.append(Determinant([a, b]))
                if len(enlarged) >= int(max_dim):
                    return DeterminantSubspace(enlarged)

        return DeterminantSubspace(enlarged)
