class GPUNotAvailable(Exception):
    """Raised when GPU hardware or CUDA runtime is missing/unusable."""
    pass


class GPUStepFailed(Exception):
    """Raised when an individual GPU standardization step fails."""
    pass