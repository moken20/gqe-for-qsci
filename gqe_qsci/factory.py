import cudaq
from hydra.utils import instantiate

from gqe_qsci.gqe.loss import GRPOLoss, GSPOLoss
from gqe_qsci.gqe.operator_pool import PauliEvolutionPool, ExcitationPool
from gqe_qsci.gqe.sampler import Sampler
from gqe_qsci.qsci.pipeline import QSCIPipeline
from gqe_qsci.wandb_logger import Logger

class Factory:
    def __init__(self):
        self.molecule = None
        self.estimator = None
        self.operator_pool = None

    def create_model(self, cfg):
        vocab_size = self.create_operator_pool(cfg).get_vocab_size()
        print(f"Vocab size: {vocab_size}")
        cfg.vocab_size = vocab_size
        return instantiate(cfg.model, vocab_size=vocab_size, ngates=cfg.ngates)

    def create_temperature_scheduler(self, cfg):
        return instantiate(cfg.trainer.temperature_scheduler)
    
    def create_molecule(self, cfg):
        if self.molecule is not None:
            return self.molecule
        molecule = instantiate(cfg.molecule)
        self.molecule = molecule
        return molecule
    
    def create_wandb_logger(self, cfg):
        reference_keys = cfg.reference_keys
        molecule = self.create_molecule(cfg)
        reference_energies = {}
        for key in reference_keys:
            if key == "hf_energy":
                reference_energies[key] = molecule.hf.e_tot
            elif key == "R-CASCI":
                reference_energies[key] = molecule.compute_casci()
                print(f"CASCI Energy: {reference_energies[key]}")
            elif key == "R-CCSD":
                reference_energies[key] = molecule.compute_ccsd()
                print(f"CCSD Energy: {reference_energies[key]}")
        return Logger(reference_energies=reference_energies)
        
    def create_loss_fn(self, cfg):
        loss_fn_name = cfg.trainer.loss.type
        match loss_fn_name:
            case "grpo":
                assert cfg.trainer.batch_size == cfg.trainer.num_samples, "batch_size must be equal to num_samples for GRPO training"
                return GRPOLoss(cfg.trainer.loss.clip_grpo_low, cfg.trainer.loss.clip_grpo_high)
            case "gspo":
                assert cfg.trainer.batch_size == cfg.trainer.num_samples, "batch_size must be equal to num_samples for GSPO training"
                return GSPOLoss(cfg.trainer.loss.clip_gspo_low, cfg.trainer.loss.clip_gspo_high)
            case _:
                raise ValueError(f"Unknown loss function name: {loss_fn_name}")
    
    def create_operator_pool(self, cfg):
        if self.operator_pool is not None:
            return self.operator_pool
        molecule = self.create_molecule(cfg)
        match cfg.operator_pool.spec:
            case "pauli_evolution":
                operator_pool = PauliEvolutionPool(
                    molecule,
                    params=cfg.operator_pool.params,
                    threshold=cfg.operator_pool.ccsd_threshold,
                    remove_z_ladder=cfg.operator_pool.remove_z_ladder,
                    only_use_first_pauli=cfg.operator_pool.only_use_first_pauli
                )
            case "excitation":
                operator_pool = ExcitationPool(molecule, params=cfg.operator_pool.params, threshold=cfg.operator_pool.ccsd_threshold)
            case _:
                raise ValueError(f"Unknown operator pool specification: {cfg.operator_pool.spec}")
        self.operator_pool = operator_pool
        return operator_pool
    
    def create_qsci_pipeline(self, cfg):
        cudaqTarget = cudaq.get_target()
        numQPUs = cudaqTarget.num_qpus()
        molecule = self.create_molecule(cfg)
        operator_pool = self.create_operator_pool(cfg)
        sampler = Sampler(operator_pool, mpi=cfg.sampler.mpi, numQPUs=numQPUs, shots_count=cfg.sampler.shots)
        return QSCIPipeline(
            molecule, operator_pool, sampler,
            max_dim=cfg.qsci.max_dim,
            enlarge_method=cfg.qsci.enlarge_method,
            max_cycle=cfg.qsci.max_cycle,
            diagonalize_backend=cfg.qsci.diagonalize_backend,
            eigsh_kwargs=cfg.qsci.eigsh_kwargs,
        )