#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "gpu_structs.hpp"
#include "gpu_kernels.hpp"

// ==========================================================
// GPU STRUCTS — minimal but sufficient for aromaticity
// ==========================================================

struct GAtomArom {
    uint8_t atomicNum;
    uint8_t degree;
    uint8_t isAromatic; // output is written here
    uint8_t sp2Like;    // precomputed on CPU
};

struct GBondArom {
    uint8_t order;
    uint8_t isAromatic; // output
    int idxA;
    int idxB;
};

// Ring descriptors (precomputed on CPU)
struct GRing {
    int start;     // index into ringAtoms array
    int size;      // number of atoms
    int ringId;    // for debugging
};


// ==========================================================
// DEVICE HELPERS
// ==========================================================

// π-electron count contribution based on atomic number & hybridization
__device__ __forceinline__
int pi_electron_contribution(uint8_t atomicNum, uint8_t sp2Like) {
    if (!sp2Like) return 0;

    // Simplified RDKit-like rules:
    // Carbon sp2: 1 electron from p-orbital
    if (atomicNum == 6) return 1;

    // Nitrogen cases
    if (atomicNum == 7) return 2; // mimics pyridine/pyrrole general behavior

    // Oxygen or sulfur in aromatic ring
    if (atomicNum == 8 || atomicNum == 16) return 2;

    return 0;
}

// Hückel rule: aromatic if 4n+2 = piCount
__device__ __forceinline__
bool huckel_aromatic(int pi) {
    if (pi < 2) return false;
    return ((pi - 2) % 4) == 0;
}


// ==========================================================
// CUDA KERNEL — ring evaluation + aromatic flag marking
// ==========================================================

extern "C" __global__
void aromaticity_kernel(
    const GAtomArom *atoms,
    GBondArom *bonds,
    const int *ringAtoms,   // flattened ring atom lists
    const GRing *rings,
    uint8_t *outAtomArom,
    uint8_t *outBondArom,
    int numAtoms,
    int numBonds,
    int numRings
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRings) return;

    GRing ring = rings[tid];

    // Step 1: count π electrons
    int piCount = 0;
    for (int i = 0; i < ring.size; i++) {
        int atomIdx = ringAtoms[ring.start + i];
        const GAtomArom &A = atoms[atomIdx];
        piCount += pi_electron_contribution(A.atomicNum, A.sp2Like);
    }

    // Step 2: Hückel aromaticity
    bool isArom = huckel_aromatic(piCount);

    if (!isArom) return;

    // Step 3: Mark atoms & bonds
    // Mark atoms first
    for (int i = 0; i < ring.size; i++) {
        int atomIdx = ringAtoms[ring.start + i];
        outAtomArom[atomIdx] = 1;
    }

    // Mark bonds participating in ring
    for (int i = 0; i < ring.size; i++) {
        int aIdx = ringAtoms[ring.start + i];
        int bIdx = ringAtoms[ring.start + ((i + 1) % ring.size)];

        // Linear scan (small rings) → acceptable, no divergence
        for (int b = 0; b < numBonds; ++b) {
            const GBondArom &B = bonds[b];
            if ((B.idxA == aIdx && B.idxB == bIdx) ||
                (B.idxA == bIdx && B.idxB == aIdx)) {
                outBondArom[b] = 1;
            }
        }
    }
}
