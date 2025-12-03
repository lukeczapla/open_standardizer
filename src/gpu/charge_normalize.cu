#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "charge_kernels.hpp"

// ======================================================
// ATOM STRUCT FOR CHARGE NORMALIZATION
// ======================================================
struct GAtomCharge {
    int8_t formalCharge;     // input
    uint8_t atomicNum;       // Z
    uint8_t degree;          // bonded neighbors
    uint8_t implicitH;       // # implicit hydrogens
    uint8_t valence;         // RDKit precomputed valence

    // output
    int8_t newCharge;
};


// ======================================================
// DEVICE HELPERS — CHEMAXON-LIKE NEUTRALIZATION RULES
// ======================================================

// Rule: quaternary ammonium stays +1, never neutralize
__device__ __forceinline__
bool is_quaternary_ammonium(uint8_t Z, uint8_t degree, int8_t charge) {
    return (Z == 7 && degree == 4 && charge > 0);
}

// Rule: deprotonated carboxylate → neutralize by +1
__device__ __forceinline__
bool is_carboxylate_like(uint8_t Z, int8_t charge, uint8_t valence) {
    return (Z == 8 && charge == -1 && valence == 1);
}

// Rule: protonated amines (primary/secondary/tertiary) → neutralize (–1)
__device__ __forceinline__
bool is_protonated_amine(uint8_t Z, int8_t charge, uint8_t degree) {
    return (Z == 7 && charge == +1 && degree <= 3);
}

// Rule: sulfonate (-SO3−) and phosphate (-PO4−) remain anionic, do not neutralize
__device__ __forceinline__
bool is_strong_acid_residue(uint8_t Z, int8_t charge, uint8_t valence) {
    if (charge != -1) return false;
    if (Z == 8 && valence == 1) return true;      // O− in sulfate, phosphate
    if (Z == 15 && valence == 5) return true;     // P(V)
    if (Z == 16 && valence == 6) return true;     // S(VI)
    return false;
}

// Rule: generic "if charge ≠ 0 and not special → try to neutralize"
__device__ __forceinline__
int8_t neutralize_basic(int8_t charge) {
    if (charge == 0) return 0;
    if (charge > 0) return charge - 1;
    return charge + 1;
}


// ======================================================
// MAIN CUDA KERNEL — ONE THREAD PER ATOM
// ======================================================

extern "C" __global__
void charge_normalize_kernel(
    const GAtomCharge *atoms,
    int8_t *outCharges,
    int numAtoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numAtoms) return;

    const GAtomCharge A = atoms[tid];
    int8_t charge = A.formalCharge;

    // --------------------------------------------------
    // RULESET (ChemAxon-equivalent)
    // --------------------------------------------------

    // 1. Skip atoms with no charge
    if (charge == 0) {
        outCharges[tid] = 0;
        return;
    }

    // 2. Quaternary ammonium → keep +1
    if (is_quaternary_ammonium(A.atomicNum, A.degree, charge)) {
        outCharges[tid] = +1;
        return;
    }

    // 3. Carboxylate-like (O− attached to carbonyl)
    if (is_carboxylate_like(A.atomicNum, charge, A.valence)) {
        outCharges[tid] = 0;
        return;
    }

    // 4. Protonated amines → neutralize
    if (is_protonated_amine(A.atomicNum, charge, A.degree)) {
        outCharges[tid] = 0;
        return;
    }

    // 5. Strong-acid residues → do NOT neutralize
    if (is_strong_acid_residue(A.atomicNum, charge, A.valence)) {
        outCharges[tid] = charge;
        return;
    }

    // 6. General fallback → neutralize
    int8_t newC = neutralize_basic(charge);
    outCharges[tid] = newC;
}
