import math
import itertools
from typing import Generator

import numpy as np
from pyscf import fci


def bitstr_to_uint64(bits: str, little_endian: bool = False) -> np.uint64:
    if len(bits) > 64:
        raise ValueError(f"bit string too long for uint64: len={len(bits)}")
    v = int(bits, 2) if not little_endian else int(bits[::-1], 2)
    return np.uint64(v)
    
def uint64_to_bitstr(v: np.uint64, little_endian: bool = False) -> str:
    s = format(int(v), f"0{64}b")
    return s if not little_endian else s[::-1]

def iter_set_bits_u64(x: int) -> Generator[int, None, None]:
    while x:
        lsb = x & -x
        yield (lsb.bit_length() - 1)
        x ^= lsb


class Determinant(np.ndarray):
    def __new__(cls, determinant: np.ndarray | list[int]):
        arr = np.asarray(determinant, dtype=np.uint64).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"Determinant must contain exactly two uint64 values: [alpha, beta]. got size={arr.size}, det={determinant!r}")
        return arr.view(cls)
    
    def is_nelec_preserving(self, nelec: tuple[int, int]) -> bool:
        return self[0].bit_count() == nelec[0] and self[1].bit_count() == nelec[1]

    @classmethod
    def from_interleaved_bitstring(cls, bits: str, *, little_endian: bool = False) -> "Determinant":
        # Interleaving convention: [α0, β0, α1, β1, ...]
        # If `little_endian=True`, we interpret the leftmost char as orbital-0 / bit-0 (LSB).
        alpha_bit = bits[0::2]
        beta_bit = bits[1::2]

        alpha = bitstr_to_uint64(alpha_bit, little_endian)
        beta = bitstr_to_uint64(beta_bit, little_endian)
        return cls([alpha, beta])

    @classmethod
    def from_fullci_index(cls, idx: int, norb: int, nelec: tuple[int, int]) -> "Determinant":
        dim = math.comb(norb, nelec[0])
        idx_alpha = idx // dim
        idx_beta = idx % dim
        alpha = fci.cistring.addr2str(norb, nelec[0], idx_alpha)
        beta = fci.cistring.addr2str(norb, nelec[1], idx_beta)
        return cls([alpha, beta])
    
    def generate_group_for_totalspin_symmetry(self) -> np.ndarray:
        """For a single determinant, enumerate all spin assignments on open-shell orbitals (fixing #alpha(open))."""
        a = int(self[0])
        b = int(self[1])
        doub = a & b
        open_ = a ^ b
        if open_ == 0:
            # Return as (1,2) array
            return np.asarray(self, dtype=np.uint64).reshape(1, 2)

        open_orbs = list(iter_set_bits_u64(open_))
        n_alpha_open = (a & open_).bit_count()

        # Use tuple-set for uniqueness; Determinant (np.ndarray subclass) itself is not hashable.
        seen: set[tuple[int, int]] = set()
        rows: list[list[np.uint64]] = []
        for alpha_choice in itertools.combinations(open_orbs, n_alpha_open):
            a_new = doub
            b_new = doub
            alpha_set = set(alpha_choice)
            for orb in open_orbs:
                if orb in alpha_set:
                    a_new |= (1 << orb)
                else:
                    b_new |= (1 << orb)
            key = (a_new, b_new)
            if key in seen:
                continue
            seen.add(key)
            rows.append([np.uint64(a_new), np.uint64(b_new)])
        return np.asarray(rows, dtype=np.uint64)