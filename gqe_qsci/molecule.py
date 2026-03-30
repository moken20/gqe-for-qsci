from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, cc, lib


@dataclass(frozen=True, slots=True)
class Hamiltonian:
  h1: np.ndarray
  h2: np.ndarray
  e_core: float | np.floating


class PySCFMolecule:
  def __init__(self, geometry, basis, nelecas, norbcas, spin, charge, num_threads: int | None = 1):
    lib.num_threads(num_threads)
    self.geometry = self._build_geometry(geometry)
    self.basis = basis

    nalpha = (nelecas + spin) // 2
    nbeta = (nelecas - spin) // 2
    self.nelec = (nalpha, nbeta)
    self.norb = norbcas
    self.spin = spin

    self.mol = gto.M(atom=self.geometry, basis=basis, charge=charge, spin=spin, unit='Angstrom')
    self.hf = scf.RHF(self.mol).run() if spin == 0 else scf.ROHF(self.mol).run()
    self.mc = mcscf.CASCI(self.hf, self.norb, self.nelec)

    self.active_indices = self._get_active_indices()
    self.cas_hamiltonian = self._build_cas_hamiltonian()
    self._ccsd_amplitude = None
    self._cache_key = self._build_cache_key(mol_key=self.mol.dumps(), nelecas=nelecas, norbcas=norbcas)
    self._cache_dir = Path(".cache") / "pyscf"
    self._cache_dir.mkdir(parents=True, exist_ok=True)

  def _build_cache_key(self, mol_key, nelecas, norbcas):
    payload = {"mol_key": mol_key, "nelecas": nelecas, "norbcas": norbcas}
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

  @property
  def ccsd_amplitude(self):
    if self._ccsd_amplitude is None:
      _ = self.compute_ccsd()
    return self._ccsd_amplitude
  
  def _build_geometry(self, geometry):
    if geometry.type == 'linear_chain':
      return make_linear_chain_geometry(geometry.atoms, geometry.bond_length)
    elif geometry.type == 'custom':
      return geometry.geometry
    else:
      raise ValueError(f"Invalid geometry type: {geometry.type}")

  def _get_active_indices(self):
    ncore = (self.mol.nelectron - sum(self.nelec)) // 2
    active_indices = list(range(ncore, ncore + self.norb))
    assert int(sum(self.hf.mo_occ[active_indices])) == sum(self.nelec), "Electron count in active indices does not match nelec."
    return active_indices

  def _build_cas_hamiltonian(self):
    mo = self.hf.mo_coeff
    h1_cas, e_core = self.mc.get_h1cas(mo)
    h2_cas = self.mc.get_h2cas(mo)
    h2_cas = ao2mo.restore(1, h2_cas, self.norb)
    return Hamiltonian(h1=h1_cas, h2=h2_cas, e_core=e_core)

  def compute_casci(self):
    cache_path = self._cache_dir / f"{self._cache_key}_casci.npz"
    if cache_path.exists():
      with np.load(cache_path, allow_pickle=False) as data:
        return data["energy"]

    e_fci, _ = self.mc.fcisolver.kernel(
      self.cas_hamiltonian.h1,
      self.cas_hamiltonian.h2,
      self.norb,
      self.nelec,
      ecore=self.cas_hamiltonian.e_core,
    )
    np.savez(cache_path, energy=e_fci)
    return e_fci

  def compute_ccsd(self):
    cache_path = self._cache_dir / f"{self._cache_key}_ccsd.npz"
    if cache_path.exists():
      with np.load(cache_path, allow_pickle=False) as data:
        self._ccsd_amplitude = {"t1": data["t1"], "t2": data["t2"]}
        return data["energy"]

    nmo = self.hf.mo_coeff.shape[1]
    active_set = set(self.active_indices)
    frozen_orbs = [i for i in range(nmo) if i not in active_set]
    mycc = cc.RCCSD(self.hf, frozen=frozen_orbs)
    mycc.verbose = 0
    mycc.kernel()
    e_tot = self.hf.e_tot + mycc.e_corr
    self._ccsd_amplitude = {"t1": mycc.t1, "t2": mycc.t2}
    np.savez(
      cache_path,
      energy=e_tot,
      t1=mycc.t1,
      t2=mycc.t2
    )
    return e_tot
  

## geometry construction helper
def make_linear_chain_geometry(
    atoms: list[str],
    bond_length: float,
) -> list[list[str | list[float]]]:
    geometry = [[atoms[0], [0., 0., 0.]]]
    dist = 0.0
    for sym in atoms[1:]:
        dist += bond_length
        geometry.append([sym, [0., 0., dist]])
    return geometry
