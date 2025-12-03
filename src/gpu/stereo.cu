#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <vector>

#include "stereo_kernels.hpp"

// ============================================================
// GPU DATA STRUCTURE
// ============================================================

// Minimal representation of atom for stereochemical ops.
// (Bond directions and CIP are CPU → GPU precomputed flags)
struct GAtom {
    uint8_t atomicNum;
    uint8_t chiralFlag;   // 0=none, 1=R, 2=S, 3=unspecified (to be fixed)
    int8_t  hCount;
};

// Per-bond directional/stereo flags
struct GBond {
    uint8_t order;
    uint8_t stereo;   // 0 none, 1=UP, 2=DOWN, 3=E, 4=Z
    int idxA;
    int idxB;
};

// Output fixed stereo flags
struct StereoFixResult {
    uint8_t atomChiral;
    uint8_t bondStereo;
};

// ============================================================
// DEVICE HELPERS
// ============================================================

// Drop RDKit “wedge/hash” if noisy (ChemAxon-style ClearStereo)
__device__ __forceinline__
uint8_t cleanAtomStereo(uint8_t flag) {
    // 1=R, 2=S, 3=unspecified
    if (flag == 3) return 0;
    return flag;
}

// Bond stereo cleaning: keep E/Z, drop noisy up/down wedges
__device__ __forceinline__
uint8_t cleanBondStereo(uint8_t s) {
    if (s <= 2) return 0;    // Remove UP/DOWN
    return s;               // Keep E/Z
}



// ============================================================
// KERNEL — one thread per atom or bond, warp-uniform
// ============================================================

extern "C" __global__
void stereo_kernel(
    const GAtom *atoms,
    const GBond *bonds,
    StereoFixResult *outAtom,
    StereoFixResult *outBond,
    int numAtoms,
    int numBonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Fix atoms
    if (tid < numAtoms) {
        uint8_t a = atoms[tid].chiralFlag;
        StereoFixResult r;
        r.atomChiral = cleanAtomStereo(a);
        r.bondStereo = 0;
        outAtom[tid] = r;
    }

    // Fix bonds
    if (tid < numBonds) {
        uint8_t b = bonds[tid].stereo;
        StereoFixResult r;
        r.bondStereo = cleanBondStereo(b);
        r.atomChiral = 0;
        outBond[tid] = r;
    }
}
