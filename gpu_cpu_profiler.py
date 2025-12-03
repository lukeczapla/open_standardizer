import time
from rdkit import Chem

from .cpu_ops import cpu_execute


class GPUCPUProfiler:
    """
    Profiles each operation (once) to decide:
        - GPU is faster → use GPU for that op
        - CPU is equal/faster → CPU fallback

    Results cached in `self.best_engine[op_name]`.
    """

    def __init__(self, policy):
        """
        policy: GPUCPUPolicyManager (or compatible)
        """
        self.policy = policy
        self.best_engine: dict[str, str] = {}

        # Bigger, more realistic test molecule than a tiny fragment
        # (still small enough to be cheap)
        self._dummy_smiles = "CC1=CC=CC=C1C(=O)OCCN(CC)CC"

    # ------------------------------------------------------------------ #
    # Timing helpers
    # ------------------------------------------------------------------ #

    def _time_cpu(self, op_name: str) -> float:
        mol = Chem.MolFromSmiles(self._dummy_smiles)
        if mol is None:
            return float("inf")

        t0 = time.perf_counter()
        try:
            _ = cpu_execute(op_name, mol)
        except Exception:
            return float("inf")
        return time.perf_counter() - t0

    def _time_gpu(self, op_name: str) -> float:
        smiles = self._dummy_smiles
        gpu_fn = self.policy.gpu_ops.get(op_name)
        if gpu_fn is None:
            return float("inf")

        t0 = time.perf_counter()
        try:
            # Prefer (smiles, [op_name]) signature
            try:
                out = gpu_fn(smiles, [op_name])
            except TypeError:
                # Fallback to simple (smiles)
                out = gpu_fn(smiles)

            if not isinstance(out, str) or not out:
                return float("inf")
        except Exception:
            return float("inf")

        return time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def profile(self, op_name: str) -> str:
        """
        Determine best backend for this specific op.
        Only runs once per op — cached forever.
        """
        # already profiled?
        if op_name in self.best_engine:
            return self.best_engine[op_name]

        # If GPU is disabled globally, or this is a CPU-only op, or no GPU impl
        if (
            not self.policy.enable_gpu
            or op_name in getattr(self.policy, "cpu_only_ops", set())
            or op_name not in self.policy.gpu_ops
        ):
            self.best_engine[op_name] = "cpu"
            return "cpu"

        cpu_t = self._time_cpu(op_name)
        gpu_t = self._time_gpu(op_name)

        if gpu_t < cpu_t:
            engine = "gpu"
        else:
            engine = "cpu"

        self.best_engine[op_name] = engine
        return engine

    def best_for(self, op_name: str) -> str:
        """
        Return "gpu" or "cpu" for a given operation.
        """
        return self.profile(op_name)
