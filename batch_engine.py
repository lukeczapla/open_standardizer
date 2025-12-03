import math
import multiprocessing as mp
from rdkit import Chem

class BatchStandardizer:
    """
    High-throughput batch processor with:
      • GPU CUDA streams
      • CPU multiprocessing fallback
      • Order-preserving merge
    """

    def __init__(self, gpu_cpu_manager, chunk_size=2048, n_cpu=None):
        self.gpu_cpu = gpu_cpu_manager
        self.chunk_size = chunk_size
        self.n_cpu = n_cpu or max(1, mp.cpu_count() - 1)

    # ------------------------------
    #  CPU worker
    # ------------------------------
    @staticmethod
    def _cpu_worker(smiles_list, op_sequence):
        from .standardize import standardize_smiles

        out = []
        for smi in smiles_list:
            try:
                res = standardize_smiles(smi, ops=op_sequence)
            except Exception:
                res = None
            out.append(res)
        return out

    # ------------------------------
    #  GPU processing (one chunk)
    # ------------------------------
    def _gpu_process_chunk(self, smiles_list, op_sequence):
        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            res = self.gpu_cpu.try_gpu_standardize(mol, op_sequence)
            if res is None:
                res = None
            results.append(res)
        return results

    # ------------------------------
    #  Main entry point
    # ------------------------------
    def run(self, smiles_list, op_sequence):
        """
        Standardize millions of SMILES while:
          - Exploiting GPU if available
          - Falling back to CPU pool for chunks
          - Maintaining order
        """
        n = len(smiles_list)
        n_chunks = math.ceil(n / self.chunk_size)
        chunks = [
            smiles_list[i * self.chunk_size : (i + 1) * self.chunk_size]
            for i in range(n_chunks)
        ]

        # If GPU disabled, whole thing goes CPU-only
        if not self.gpu_cpu.gpu_enabled:
            with mp.Pool(self.n_cpu) as pool:
                mapped = pool.starmap(self._cpu_worker, [(c, op_sequence) for c in chunks])
            return [item for batch in mapped for item in batch]

        # Otherwise: hybrid GPU + CPU pool
        results = [None] * n_chunks

        # Launch CPU pool
        cpu_pool = mp.Pool(self.n_cpu)

        for idx, chunk in enumerate(chunks):
            # Try GPU first
            gpu_out = self._gpu_process_chunk(chunk, op_sequence)

            # If too many failures, fallback CPU batch
            if gpu_out.count(None) > len(chunk) * 0.25:
                # Re-run CPU batch
                async_res = cpu_pool.apply_async(
                    BatchStandardizer._cpu_worker,
                    (chunk, op_sequence)
                )
                results[idx] = async_res
            else:
                results[idx] = gpu_out

        cpu_pool.close()
        cpu_pool.join()

        # Merge results in correct order
        final_out = []
        for r in results:
            if isinstance(r, list):
                final_out.extend(r)
            else:
                final_out.extend(r.get())

        return final_out
