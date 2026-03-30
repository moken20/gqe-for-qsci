import numpy as np

from gqe_qsci.qsci.refine.gevp import GEVPSolver
from gqe_qsci.qsci.subspace import DeterminantSubspace, Determinant

class RefinePipeline:
    def __init__(self, hamiltonian, nelec, norb):
        self.hamiltonian = hamiltonian
        self.nelec = nelec
        self.norb = int(norb)
        self.gevp = GEVPSolver(hamiltonian=hamiltonian, nelec=nelec, norb=self.norb)

    def process(
        self,
        sci_states,
        max_dim: int,
        s_rcond: float = 1e-10,
    ):
        _, _, strs_union, A = self.gevp.solve(
            sci_states,
            n_roots=1,
            s_rcond=s_rcond,
            return_merged_amplitudes=True,
        )
        # A is merged amplitude on union basis (N,)
        weights = np.abs(A) ** 2

        k = min(int(max_dim), int(weights.size))
        if k <= 0:
            return DeterminantSubspace([])

        # O(N) top-k selection
        idx = np.argpartition(weights, -k)[-k:]
        idx = idx[np.argsort(weights[idx])[::-1]]

        dets = [Determinant([int(strs_union[i, 0]), int(strs_union[i, 1])]) for i in idx]
        return DeterminantSubspace(dets)