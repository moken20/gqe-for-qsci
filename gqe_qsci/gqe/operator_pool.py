from abc import ABC, abstractmethod
from collections import Counter
import cudaq
import numpy
import tequila as tq
from tequila.circuit import QCircuit
from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin, get_pauli_evolution_gate_count


class OperatorPool(ABC):
    def __init__(self, molecule: PySCFMolecule, params: list[float] | None, **kwargs):
        self.molecule = molecule
        self.n_qubits = molecule.norb * 2
        self.n_electrons = int(sum(molecule.nelec))
        self.params = params
        self.pool = self.build_operator_pool(**kwargs)

    def __len__(self):
        return len(self.pool)

    def __iter__(self):
        return iter(self.pool)

    def __getitem__(self, idx):
        return self.pool[idx]

    @abstractmethod
    def get_vocab_size(self):
        pass

    @abstractmethod
    def build_operator_pool(self):
        pass

    @abstractmethod
    def get_gate_count(self, seq: list[int]) -> Counter:
        pass

    def get_identity_operator(self):
        In = cudaq.spin.i(0)
        for q in range(1, self.n_qubits):
            In = In * cudaq.spin.i(q)
        return 1.0 * cudaq.SpinOperator(In)


class UCCSDBasedPool(OperatorPool, ABC):
    def __init__(self, molecule: PySCFMolecule, params: list[float] | None, threshold: float =1e-8, **kwargs):
        super().__init__(molecule, params, threshold=threshold, **kwargs)
    
    def get_vocab_size(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def build_operator_pool(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_gate_count(self, seq: list[int]) -> Counter:
        raise NotImplementedError("Subclasses must implement this method")

    def generate_excitations(self, threshold: float):
        ccsd_amplitudes = ClosedShellAmplitudes(tIjAb=self.molecule.ccsd_amplitude["t2"], tIA=self.molecule.ccsd_amplitude["t1"])
        amplitudes_all = ccsd_amplitudes.make_parameter_dictionary(threshold=0.0, screening=False)
        amplitudes = {
            k: v for k, v in amplitudes_all.items()
            if not numpy.isclose(v, 0.0, atol=threshold)
        }
        amplitudes = dict(sorted(amplitudes.items(), key=lambda x: numpy.fabs(x[1]), reverse=True))
        indices = {}
        for key, t in amplitudes.items():
            assert (len(key) % 2 == 0)
            if not numpy.isclose(t, 0.0, atol=threshold):
                if len(key) == 2:
                    angle = 2.0 * t
                    idx_a = (2 * key[0], 2 * key[1])
                    idx_b = (2 * key[0] + 1, 2 * key[1] + 1)
                    indices[idx_a] = angle
                    indices[idx_b] = angle
                else:
                    assert len(key) == 4
                    angle = 2.0 * t
                    idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                    indices[idx_abab] = angle
                    if key[0] != key[2] and key[1] != key[3]:
                        idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                        idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                        partner = tuple([key[2], key[1], key[0], key[3]])
                        partner_t = amplitudes_all.get(partner, 0.0)
                        anglex = 2.0 * (t - partner_t)
                        indices[idx_aaaa] = anglex
                        indices[idx_bbbb] = anglex
        return indices

    def make_uccsd_ansatz(self, threshold: float):
        screened_indices = self.generate_excitations(threshold=threshold)
        geometry_lines = [f"{atom_type} {coords[0]} {coords[1]} {coords[2]}" for atom_type, coords in self.molecule.geometry]
        geometry_str = "\n".join(geometry_lines)
        tq_molecule = tq.Molecule(
            geometry=geometry_str, basis_set=self.molecule.basis, active_orbitals=self.molecule.active_indices, transformation="jordan-wigner"
        )
        ansatz = QCircuit()
        for idx, angle in screened_indices.items():
            converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
            ansatz += tq_molecule.make_excitation_gate(indices=converted, angle=angle)
        return ansatz


class PauliEvolutionPool(UCCSDBasedPool):
    def __init__(
        self,
        molecule: PySCFMolecule,
        params: list[float] | None,
        threshold: float =1e-8,
        remove_z_ladder: bool = False,
        only_use_first_pauli: bool = False,
    ):
        super().__init__(molecule, params, threshold=threshold, remove_z_ladder=remove_z_ladder, only_use_first_pauli=only_use_first_pauli)
    
    def get_vocab_size(self):
        return len(self.pool)
    
    def build_operator_pool(self, threshold, remove_z_ladder=False, only_use_first_pauli=False):
        uccsd_ansatz = self.make_uccsd_ansatz(threshold=threshold)
        seen = set()
        operator_pool = [self.get_identity_operator()]
        for g in uccsd_ansatz.gates:
            coeff = g.parameter
            for p in g.generator.paulistrings:
                if remove_z_ladder:
                    p = {k: v for k, v in p.items() if v.lower() != 'z'}
                term = convert_pauli_to_cudaq_spin(p)
                if str(term) in seen:
                    continue
                seen.add(str(term))
                if self.params is None:
                    operator_pool.append(coeff * cudaq.SpinOperator(term))
                else:
                    for p in self.params:
                        operator_pool.append(p * cudaq.SpinOperator(term))
                if only_use_first_pauli:
                    break
        return operator_pool
    
    def get_gate_count(self, seq: list[int]) -> Counter:
        counts = Counter()
        for i in seq:
            operator = self.pool[i]
            for term in operator:
                pauli = term.get_pauli_word(self.n_qubits)
                count = get_pauli_evolution_gate_count(pauli)
                counts.update(count)
        return counts


class ExcitationPool(UCCSDBasedPool):
    def __init__(self, molecule: PySCFMolecule, params: list[float] | None, threshold: float =1e-8):
        super().__init__(molecule, params)
    
    def get_vocab_size(self):
        return len(self.pool)
    
    def build_operator_pool(self, threshold):
        uccsd_ansatz = self.make_uccsd_ansatz(threshold=threshold)
        operator_pool = [self.get_identity_operator()]
        for g in uccsd_ansatz.gates:
            coeff = g.parameter
            operator = None
            for p in g.generator.paulistrings:
                term = convert_pauli_to_cudaq_spin(p)
                operator = term if operator is None else (operator + term * p._coeff)
            if self.params is None:
                operator_pool.append(coeff * cudaq.SpinOperator(operator))
            else:
                for p in self.params:
                    operator_pool.append(p * cudaq.SpinOperator(operator))
        return operator_pool
    
    def get_gate_count(self, seq: list[int]) -> Counter:
        counts = Counter()
        for i in seq:
            operator = self.pool[i]
            for term in operator:
                count = get_pauli_evolution_gate_count(term.get_pauli_word(self.n_qubits))
                counts.update(count)
        return counts