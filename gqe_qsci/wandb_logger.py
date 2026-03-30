from typing import Any, Mapping

import numpy as np

from gqe_qsci.qsci.schema import QSCIResult, QSCISampleResult


class Logger:
    def __init__(self, reference_energies: Mapping[str, float] | None = None):
        self.reference_energies = reference_energies
    
    def log_result(
        self,
        pl_module,
        results: list[dict[str, Any]], 
    ) -> dict[str, float]:
        metrics = {}
        for r in results:
            metrics.update(extract_metrics(r["result"], prefix=r["prefix"], ref_energy=self.reference_energies))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return metrics
    

def _prefix_key(key: str, prefix: str) -> str:
    if prefix == "":
        return key
    return f"{prefix}/{key}"


@singledispatch
def extract_metrics(result: Any, *, prefix: str = "", ref_energy: dict[str, float] | None = None) -> dict[str, float]:
    """Extract logging metrics from an arbitrary object and return a flat dict.

    When you add a new result dataclass, register it here via `.register(Type)`.
    """
    # Default: do nothing (ignore unknown types)
    _ = (result, prefix, ref_energy)
    return {}


@extract_metrics.register
def _(result: QSCISampleResult, *, prefix: str = "", ref_energy: dict[str, float] | None = None) -> dict[str, float]:
    d: dict[str, float] = {}
    if ref_energy:
        for ref_key, ref_val in ref_energy.items():
            d[_prefix_key(f"energy - {ref_key}", prefix)] = result.energy - ref_val
    else:
        d[_prefix_key("energy", prefix)] = result.energy
    d[_prefix_key("subspace_dim", prefix)] = result.subspace_dim
    d[_prefix_key("num_sampled_basis", prefix)] = result.num_sampled_basis
    d[_prefix_key("num_symmetry_preserving_basis", prefix)] = result.num_symmetry_preserving_basis
    d[_prefix_key("cx_count", prefix)] = result.cx_count
    d[_prefix_key("total_gates", prefix)] = result.total_gates
    return d


@extract_metrics.register
def _(result: QSCIResult, *, prefix: str = "", ref_energy: dict[str, float] | None = None) -> dict[str, float]:
    if len(result.samples) == 0:
        return {}

    d: dict[str, float] = {}
    d[_prefix_key("energy/min", prefix)] = min(result.energies)
    d[_prefix_key("energy/mean", prefix)] = np.mean(result.energies)
    d[_prefix_key("energy/std", prefix)] = np.std(result.energies)

    d[_prefix_key("subspace_dim/max", prefix)] = min(result.subspace_dim)
    d[_prefix_key("cx_count/max", prefix)] = max(result.cx_counts)
    d[_prefix_key("total_gates/max", prefix)] = max(result.total_gates)
    return d