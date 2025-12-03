import pycuda.driver as cuda
import pycuda.autoinit  # ensures context
from pycuda.compiler import SourceModule
from rdkit import Chem

NUM_STREAMS = 16

class MultiStreamExecutor:
    """
    Launches GPU work across N independent CUDA streams.
    Ideal for thousands of molecules with small kernels.
    """

    def __init__(self, cuda_module):
        self.streams = [cuda.Stream() for _ in range(NUM_STREAMS)]
        self.kernels = {
            "stereo": cuda_module.get_function("kernel_stereo"),
            "aromaticity": cuda_module.get_function("kernel_aromaticity"),
            "charge": cuda_module.get_function("kernel_charge"),
            "bond_order": cuda_module.get_function("kernel_bond_order"),
            "mesomerizer": cuda_module.get_function("kernel_mesomerizer"),
        }

    def launch_op(self, op_name, gpu_data_ptrs):
        kernel = self.kernels.get(op_name)
        if kernel is None:
            return False

        n = len(gpu_data_ptrs)
        if n == 0:
            return True

        # Partition into 16 streams
        chunk = (n + NUM_STREAMS - 1) // NUM_STREAMS

        for i in range(NUM_STREAMS):
            start = i * chunk
            end = min(start + chunk, n)
            if start >= end:
                break

            buf_slice = gpu_data_ptrs[start:end]
            kernel(
                cuda.InOut(buf_slice),
                block=(256, 1, 1),
                grid=((len(buf_slice) + 255) // 256, 1),
                stream=self.streams[i]
            )

        # synchronize ALL streams
        for s in self.streams:
            s.synchronize()

        return True
