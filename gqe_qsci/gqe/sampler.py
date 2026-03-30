from mpi4py import MPI
import torch, cudaq

from gqe_qsci.gqe.operator_pool import OperatorPool

class Sampler:
    def __init__(self, operator_pool: OperatorPool, mpi: bool, numQPUs: int, shots_count: int):
        self.pool = operator_pool
        self.shots_count = shots_count
        self.mpi = mpi
        self.numQPUs = numQPUs
                
    @torch.no_grad()
    def run(self, state: dict): 
        """Sample quantum states in computational basis.  

        Args:  
            state: Dictionary containing 'idx' key with operator indices  
            shots_count: Number of shots for sampling (default: 1000)  

        Returns:  
            List of sample_result objects or list of probability dictionaries  
        """  
        res = []  
        idx_output = state["idx"]
        pool = self.pool  

        if cudaq.mpi.is_initialized():  
            rank = cudaq.mpi.rank()  
            numRanks = cudaq.mpi.num_ranks()  
            total_elements = len(idx_output)  
            elements_per_rank = total_elements // numRanks  
            remainder = total_elements % numRanks  
            start = rank * elements_per_rank + min(rank, remainder)  
            end = start + elements_per_rank + (1 if rank < remainder else 0)  

            res = [  
                self.sample_state([pool[j] for j in row],   
                                shots_count=self.shots_count,  
                                qpu_id=i % self.numQPUs)  
                for i, row in enumerate(idx_output[start:end])  
            ]  
        else:  
            res = [  
                self.sample_state([pool[j] for j in row],  
                                shots_count=self.shots_count,  
                                qpu_id=i % self.numQPUs)  
                for i, row in enumerate(idx_output)  
            ]  

        # Handle async results if using MPI  
        if self.mpi and isinstance(res[0], tuple) and len(res[0]) == 2:  
            res = [getResultFunctor(handle) for (handle, getResultFunctor) in res]  

        # Gather results from all MPI ranks  
        if cudaq.mpi.is_initialized():  
            res = MPI.COMM_WORLD.allgather(res)  
            res = [x for xs in res for x in xs]  

        return res
    

    def term_coefficients(self, op: cudaq.SpinOperator) -> list[complex]:
        return [term.evaluate_coefficient() for term in op]


    def term_words(self, op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
        return [cudaq.pauli_word(term.get_pauli_word(self.pool.n_qubits)) for term in op]
    
    def sample_state(self, sampled_ops, shots_count, **kwargs):
        """Sample a single quantum state.

        Args:  
            sampled_ops: List of operators to apply  
            shots_count: Number of shots for sampling  
            **kwargs: Additional arguments including qpu_id  

        Returns:  
            sample_result object or tuple (handle, getter) if using async  
        """  
        full_coeffs = []  
        full_words = []  

        for op in sampled_ops:  
            full_coeffs += [c.real for c in self.term_coefficients(op)]  
            full_words += self.term_words(op)

        if self.mpi:  
            handle = cudaq.sample_async(self.kernel,  
                                       self.pool.n_qubits,  
                                       self.pool.n_electrons,  
                                       full_coeffs,  
                                       full_words,  
                                       shots_count=shots_count,
                                       qpu_id=kwargs['qpu_id'])
            return handle, lambda res: res.get()  
        else:  
            return cudaq.sample(self.kernel,  
                              self.pool.n_qubits,  
                              self.pool.n_electrons,  
                              full_coeffs,  
                              full_words,  
                              shots_count=shots_count)

    
    # Kernel that applies the selected operators
    @cudaq.kernel
    def kernel(n_qubits: int, n_electrons: int, coeffs: list[float], words: list[cudaq.pauli_word]):
        q = cudaq.qvector(n_qubits)

        for i in range(n_electrons):
            x(q[i])

        for i in range(len(coeffs)):
            exp_pauli(coeffs[i], q, words[i])
