from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QSCISampleResult:
    seq: tuple[int, ...] | None
    energy: float
    num_sampled_basis: int | None
    num_symmetry_preserving_basis: int | None
    subspace_dim: int
    cx_count: int | None
    total_gates: int | None


@dataclass(frozen=True, slots=True)
class QSCIResult:
    samples: tuple[QSCISampleResult, ...]
    local_refined: QSCISampleResult | None = None
    global_refined: QSCISampleResult | None = None

    @property
    def energies(self) -> list[float]:
        return [s.energy for s in self.samples]

    @property
    def num_sampled_basis(self) -> list[int]:
        return [s.num_sampled_basis for s in self.samples]

    @property
    def num_symmetry_preserving_basis(self) -> list[int]:
        return [s.num_symmetry_preserving_basis for s in self.samples]

    @property
    def subspace_dim(self) -> list[int]:
        return [s.subspace_dim for s in self.samples]

    @property
    def cx_counts(self) -> list[int]:
        return [s.cx_count for s in self.samples]

    @property
    def total_gates(self) -> list[int]:
        return [s.total_gates for s in self.samples]

    @property
    def best_sample(self) -> QSCISampleResult | None:
        if not self.samples:
            return None
        return min(self.samples, key=lambda s: s.energy)