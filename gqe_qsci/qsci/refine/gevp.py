from typing import Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eigh as scipy_eigh
import pyci

from gqe_qsci.qsci.statevector import SCIVector


class GEVPSolver:
    """
    Solve H c = E S c on the span of given SCIStates.
    Also provides merged determinant amplitudes A = C @ c for the lowest root.
    """
    def __init__(self, hamiltonian, nelec: Tuple[int, int], norb: int) -> None:
        self.hamiltonian = hamiltonian
        self.nelec = nelec
        self.norb = int(norb)
        self._pyci_h2 = np.asarray(self.hamiltonian.h2.transpose(0, 2, 1, 3), order="C")
        self._pyci_ham = pyci.hamiltonian(
            float(self.hamiltonian.e_core),
            np.asarray(self.hamiltonian.h1, order="C"),
            self._pyci_h2,
        )

    def solve(
        self,
        sci_states: Sequence["SCIVector"],
        n_roots: int = 1,
        s_rcond: float = 1e-10,
        return_merged_amplitudes: bool = False,
    ):
        strs_union, C = self._embed_states_on_union_determinants_fast(sci_states)

        # Overlap
        S = C.conj().T @ C
        S = 0.5 * (S + S.T.conj())

        # Projected Hamiltonian
        H = self._build_projected_hamiltonian(strs_union, C, S)
        H = 0.5 * (H + H.T.conj())

        energies, coeffs = self._solve_gevp_stable(H, S, n_roots=n_roots, s_rcond=s_rcond)

        if not return_merged_amplitudes:
            return energies, coeffs

        # merged amplitudes on the union determinant basis for the lowest root
        c0 = coeffs[:, 0]
        A = C @ c0  # shape (N,)
        return energies, coeffs, strs_union, A

    def _embed_states_on_union_determinants_fast(
        self,
        sci_states: Sequence["SCIVector"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized union embedding without Python dict.

        Returns:
            strs_union: (N, 2) uint64
            C:         (N, K) complex128
        """
        strs_list = [np.asarray(s._strs, dtype=np.uint64) for s in sci_states]
        lens = np.array([x.shape[0] for x in strs_list], dtype=np.int64)
        all_pairs = np.concatenate(strs_list, axis=0)  # (sum lens, 2)

        # Use a structured view to unique rows efficiently
        view = all_pairs.view([("a", np.uint64), ("b", np.uint64)]).ravel()
        uniq_view, inv = np.unique(view, return_inverse=True)
        strs_union = uniq_view.view(np.uint64).reshape(-1, 2)

        N = strs_union.shape[0]
        K = len(sci_states)
        C = np.zeros((N, K), dtype=np.complex128)

        offset = 0
        for j, (s, L) in enumerate(zip(sci_states, lens)):
            idx = inv[offset : offset + L]
            C[idx, j] = np.asarray(s, dtype=np.complex128)
            offset += L

        return strs_union, C

    def _build_projected_hamiltonian(
        self,
        strs_union: np.ndarray,
        C: np.ndarray,
        S: np.ndarray,
    ) -> np.ndarray:
        """
        Build H_{ij} = <Psi_i|H|Psi_j> using pyci.sparse_op.
        """
        strs_union = np.asarray(strs_union, dtype=np.uint64, order="C")
        C = np.asarray(C, dtype=np.complex128, order="C")

        wfn = pyci.fullci_wfn(self._pyci_ham.nbasis, *self.nelec)
        for a, b in strs_union:
            wfn.add_det(np.asarray([a, b], dtype=np.uint64))

        op = pyci.sparse_op(self._pyci_ham, wfn)

        # Avoid extra copies if possible: op.data/indices/indptr are already CSR components in PyCI API.
        data = op.data()
        indices = op.indices()
        indptr = op.indptr()

        H_csr = csr_matrix((data, indices, indptr), shape=op.shape)

        HC = H_csr @ C  # (N, K)
        H = C.conj().T @ HC  # (K, K)

        # Constant shift: <Psi_i|ecore|Psi_j> = ecore * <Psi_i|Psi_j> = ecore * S_ij
        H += complex(op.ecore) * S
        return H

    def _solve_gevp_stable(
        self,
        H: np.ndarray,
        S: np.ndarray,
        n_roots: int,
        s_rcond: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stable GEVP solver with eigenvalue truncation of S.

        s_rcond is treated as a relative cutoff to max eigenvalue of S.
        """
        se, U = np.linalg.eigh(S)
        se = np.real(se)  # S should be Hermitian

        # Relative cutoff (scale-invariant)
        tol = float(s_rcond) * float(np.max(se))
        mask = se > tol

        if not np.any(mask):
            raise ValueError("Overlap matrix S is (numerically) singular: all eigenvalues <= tol.")

        Ured = U[:, mask]
        Sred = se[mask]  # positive

        # Reduce the problem to the well-conditioned subspace:
        # H_red c = E diag(Sred) c  (now B is SPD)
        H_red = Ured.conj().T @ H @ Ured
        B_red = np.diag(Sred)

        # Use LAPACK generalized Hermitian solver (fast/stable) on the reduced SPD problem.
        # Note: scipy.linalg.eigh assumes B is positive definite.
        w, v = scipy_eigh(
            H_red,
            B_red,
            subset_by_index=(0, min(n_roots, H_red.shape[0]) - 1),
            check_finite=False,
        )
        coeffs_full = Ured @ v  # back to original K-dim basis

        return w, coeffs_full