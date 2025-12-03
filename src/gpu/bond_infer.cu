#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "bond_structs.hpp"

// ======================================================
// DEVICE-LEVEL HELPERS (ChemAxon-like bond normalization)
// ======================================================

// Aromatic adjacency = both atoms aromatic
__device__ __forceinline__
bool both_aromatic(uint8_t a1_arom, uint8_t a2_arom) {
    return (a1_arom & 1) && (a2_arom & 1);
}

// Amide carbonyl C–N pattern
__device__ __forceinline__
bool is_amide_CN(uint8_t Z1, uint8_t Z2,
                 uint8_t val1, uint8_t val2,
                 uint8_t deg1, uint8_t deg2) {

    // C=O–N pattern (very broad ChemAxon shortcut)
    bool carbonylCarbon =
        (Z1 == 6 && val1 == 3 && deg1 == 3);

    bool amideNitrogen =
        (Z2 == 7 && val2 == 3);

    return carbonylCarbon && amideNitrogen;
}

// Nitro group N(=O)-O− form
__device__ __forceinline__
bool is_nitro(uint8_t Z1, uint8_t Z2,
              uint8_t val1, uint8_t val2,
              int8_t c1, int8_t c2) {

    // Broad ChemAxon heuristic:
    //  - N with valence 4 (formal charge +1)
    //  - O with valence 1 (O−)
    bool n_plus = (Z1 == 7 && val1 == 4 && c1 >= 0);
    bool o_minus = (Z2 == 8 && val2 == 1 && c2 <= 0);
    return n_plus && o_minus;
}


// Carboxylate C=O / C–O− (normalize to ideal)
__device__ __forceinline__
bool is_carboxylate_pair(uint8_t Z1, uint8_t Z2,
                         uint8_t val1, uint8_t val2,
                         int8_t c1, int8_t c2) {

    bool carbonyl =
        (Z1 == 6 && val1 == 3);

    bool oxygen =
        (Z2 == 8 && val2 <= 2);

    return carbonyl && oxygen;
}


// ======================================================
// MAIN CUDA BOND NORMALIZATION KERNEL
// (1 thread per bond)
// ======================================================

extern "C" __global__
void bond_infer_kernel(
    const GBond *bonds,
    const GAtom *atoms,
    uint8_t *outBondOrder,
    uint8_t *outAromatic,
    int numBonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBonds) return;

    const GBond B = bonds[tid];
    const GAtom A1 = atoms[B.a1];
    const GAtom A2 = atoms[B.a2];

    uint8_t Z1 = A1.atomicNum;
    uint8_t Z2 = A2.atomicNum;
    uint8_t val1 = A1.valence;
    uint8_t val2 = A2.valence;
    uint8_t deg1 = A1.degree;
    uint8_t deg2 = A2.degree;

    int8_t c1 = A1.formalCharge;
    int8_t c2 = A2.formalCharge;

    uint8_t order = B.bondOrder;
    uint8_t arom = B.aromaticFlag;

    // =============================================
    // 1. Aromaticity override (ChemAxon-like)
    // =============================================
    if (both_aromatic(A1.aromatic, A2.aromatic)) {
        outBondOrder[tid] = 1;    // aromatic bond encoded as 1
        outAromatic[tid] = 1;
        return;
    }

    // =============================================
    // 2. Amide normalization
    // C=O–N → C=O, N single-bonded
    // =============================================
    if (is_amide_CN(Z1, Z2, val1, val2, deg1, deg2)) {
        outBondOrder[tid] = 1; // C–N single bond (amide)
        outAromatic[tid] = 0;
        return;
    }
    if (is_amide_CN(Z2, Z1, val2, val1, deg2, deg1)) {
        outBondOrder[tid] = 1;
        outAromatic[tid] = 0;
        return;
    }

    // =============================================
    // 3. Nitro normalization
    // N(+)(=O)-O(−) → N–O, N=O canonical
    // =============================================
    if (is_nitro(Z1, Z2, val1, val2, c1, c2)) {
        // convert ambiguous patterns to N–O single
        outBondOrder[tid] = (val2 == 1 ? 1 : 2);  // match RDKit style
        outAromatic[tid] = 0;
        return;
    }
    if (is_nitro(Z2, Z1, val2, val1, c2, c1)) {
        outBondOrder[tid] = (val1 == 1 ? 1 : 2);
        outAromatic[tid] = 0;
        return;
    }

    // =============================================
    // 4. Carboxylate / Carbonyl heuristics
    // C=O–O− → normalize to single/double combination
    // =============================================
    if (is_carboxylate_pair(Z1, Z2, val1, val2, c1, c2)) {
        // carbonyl pair
        if (c2 == -1) {
            outBondOrder[tid] = 1;   // C–O(−) single
        } else {
            outBondOrder[tid] = 2;   // C=O
        }
        outAromatic[tid] = 0;
        return;
    }

    if (is_carboxylate_pair(Z2, Z1, val2, val1, c2, c1)) {
        if (c1 == -1) {
            outBondOrder[tid] = 1;
        } else {
            outBondOrder[tid] = 2;
        }
        outAromatic[tid] = 0;
        return;
    }

    // =============================================
    // 5. Keep existing order (fallback)
    // =============================================
    outBondOrder[tid] = order;
    outAromatic[tid] = arom;
}
