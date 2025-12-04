#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include "gpu_structs.hpp"
#include "gpu_kernels.hpp"

// =====================================================
// DEVICE HELPERS FOR TAUTOMER CANONICALIZATION
// =====================================================

// Basic type predicates
__device__ __forceinline__
bool is_carbon(uint8_t Z) {
    return Z == 6;
}

__device__ __forceinline__
bool is_oxygen(uint8_t Z) {
    return Z == 8;
}

__device__ __forceinline__
bool is_nitrogen(uint8_t Z) {
    return Z == 7;
}

// Simple "is carbonyl C" check: C with valence ~3 and at least
// one double bond to O (we only use local info here, so we rely
// on valence and current bond order).
__device__ __forceinline__
bool is_carbonyl_like(const GAtom &A) {
    return (A.atomicNum == 6 && A.valence >= 3);
}

// We bias strongly towards canonical keto (C=O) over enol
// and canonical nitro over aci-nitro, in line with RDKit's
// scoring preferences. :contentReference[oaicite:2]{index=2}


// =====================================================
// KERNEL
//   - atom-phase: pass formal charges through (or lightly
//                 tweak if you later want to move H+)
//   - bond-phase: adjust bond orders in known tautomeric
//                 motifs in a canonical direction.
// =====================================================

extern "C" __global__
void tautomerize_kernel(
    const GAtom *atoms,
    const GBond *bonds,
    int8_t *outCharges,
    uint8_t *outBondOrder,
    uint8_t *outAromatic,
    int numAtoms,
    int numBonds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // -------------------------------------------------
    // ATOM PHASE
    // -------------------------------------------------
    if (tid < numAtoms) {
        const GAtom A = atoms[tid];

        // For now we are conservative: we do not attempt to
        // explicitly move protons in the GPU kernel. We just
        // carry through the formal charges.
        //
        // If you extend GAtom to carry explicit hydrogen
        // counts and want to do real H-shuffles, this is
        // where you'd do it.
        outCharges[tid] = A.formalCharge;
        return;
    }

    // -------------------------------------------------
    // BOND PHASE
    // -------------------------------------------------
    int bid = tid - numAtoms;
    if (bid >= numBonds) return;

    const GBond B = bonds[bid];
    const GAtom A1 = atoms[B.a1];
    const GAtom A2 = atoms[B.a2];

    uint8_t Z1 = A1.atomicNum;
    uint8_t Z2 = A2.atomicNum;

    int8_t c1 = A1.formalCharge;
    int8_t c2 = A2.formalCharge;

    uint8_t val1 = A1.valence;
    uint8_t val2 = A2.valence;

    uint8_t order = B.bondOrder;
    uint8_t arom  = B.aromaticFlag;

    // Default: keep original
    uint8_t newOrder = order;
    uint8_t newArom  = arom;

    // =================================================
    // 1) Nitro / aci-nitro canonicalization
    //
    //    RDKit's scoring scheme explicitly penalizes
    //    aci-nitro ([C]=N+(O-)[OH]) and favors the
    //    canonical nitro form. :contentReference[oaicite:3]{index=3}
    //
    //    Here, we normalize bonds C–N(+)=O / C–N(=O)–O(-)
    //    towards a stable N=O / N–O(-) configuration.
    // =================================================
    // Obvious nitro-like bonds: N–O or C–N
    if ((is_nitrogen(Z1) && is_oxygen(Z2)) ||
        (is_nitrogen(Z2) && is_oxygen(Z1))) {

        // For simplicity we just force N=O / N–O(-)-style bond
        // orders: one bond will be double, the other single.
        // This kernel only sees one bond at a time, so we
        // normalize the *current* bond towards a reasonable
        // nitro resonance state: single or double but not
        // aromatic.
        if (order == 1 || order == 2) {
            // If the O side is negatively charged, it's more
            // like N–O(-) → single bond; otherwise N=O → double.
            bool o_is_neg = (is_oxygen(Z1) && c1 < 0) ||
                            (is_oxygen(Z2) && c2 < 0);
            newOrder = o_is_neg ? 1 : 2;
            newArom  = 0;
        }
    }

    // =================================================
    // 2) Carboxyl / keto-enol normalization around C=O
    //
    //    RDKit strongly rewards carbonyl C=O vs enol forms. :contentReference[oaicite:4]{index=4}
    //    Here we bias bonds between carbon and oxygen towards
    //    C=O rather than C–O if they look like part of a
    //    conjugated carbonyl system.
    // =================================================
    if (is_carbon(Z1) && is_oxygen(Z2)) {
        // C(=O)X preferred over C(-O)X when valence suggests a
        // carbonyl center.
        if (is_carbonyl_like(A1)) {
            // For carbonyl-ish C–O, prefer order=2
            if (order == 1) {
                newOrder = 2;
                newArom  = 0;
            }
        }
    } else if (is_carbon(Z2) && is_oxygen(Z1)) {
        if (is_carbonyl_like(A2)) {
            if (order == 1) {
                newOrder = 2;
                newArom  = 0;
            }
        }
    }

    // =================================================
    // 3) Amide vs imidic normalization
    //
    //    e.g. O=C–NH ↔ HO–C=N. Canonical form is usually the
    //    amide O=C–NH, which is favored by RDKit's scoring
    //    via C=O priority. :contentReference[oaicite:5]{index=5}
    //
    //    We can't see the whole functional group here, but
    //    we can at least make sure the C–O bond in C/N systems
    //    is driven towards double where appropriate.
    // =================================================
    if ((is_carbon(Z1) && is_nitrogen(Z2)) ||
        (is_carbon(Z2) && is_nitrogen(Z1))) {

        // In conjugated C–N–C=O systems, RDKit favors a
        // "classic" amide with a C=O. We don't reconstruct
        // the whole amide here; we just ensure the C–N bond
        // is not arbitrarily elevated to a second double if
        // we know there's already a C=O on that carbon.
        //
        // If a C–N bond is currently double and the carbon
        // is valence-heavy, push it back to single so the
        // carbonyl can stay double to O.
        if (order == 2) {
            const GAtom &C = is_carbon(Z1) ? A1 : A2;
            if (is_carbonyl_like(C) && C.valence > 3) {
                newOrder = 1;
                newArom  = 0;
            }
        }
    }

    // =================================================
    // 4) Default: keep original if no rule fired
    // =================================================
    outBondOrder[bid] = newOrder;
    outAromatic[bid]  = newArom;
}
