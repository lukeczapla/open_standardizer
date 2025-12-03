#include <cuda_runtime.h>
#include <stdint.h>

// Layout assumptions (simplified):
//   - N molecules in batch
//   - total_atoms = sum_i num_atoms[i]
//   - atom_offset[m] = starting index of mol m in atom arrays
//
// Arrays:
//   atom_isotope[total_atoms]      : int (0 = none)
//   atom_flags[total_atoms]        : bitfield for stereo/H, etc.
//   atom_frag_id[total_atoms]      : connected component id per atom
//   atom_keep_mask[total_atoms]    : 1 = keep, 0 = drop (for removeFragment/removeH)
//
// We'll run one block per molecule, threads over atoms.

/* Bit masks (example only; align with your real encoding) */
constexpr uint32_t FLAG_STEREO_MASK   = 0x0000000F;  // low bits store stereo
constexpr uint32_t FLAG_EXPLICIT_H    = 0x00000010;  // explicit H marker


__global__
void clear_isotopes_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int         num_mols,
    int*        atom_isotope  // in-place
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        atom_isotope[idx] = 0;  // clear isotopic label
    }
}


__global__
void clear_stereo_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int         num_mols,
    uint32_t*   atom_flags  // in-place bitfield
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        uint32_t f = atom_flags[idx];
        // zero stereo bits only
        f &= ~FLAG_STEREO_MASK;
        atom_flags[idx] = f;
    }
}


// Simple per-fragment atom counting. Assumes:
//   frag_id in [0, max_frags) per molecule (pre-normalized on CPU)
__global__
void largest_fragment_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int  const* __restrict__ atom_frag_id,
    int         num_mols,
    int         max_frags_per_mol,
    int*        mol_largest_frag_id    // out: per-mol largest fragment id
) {
    extern __shared__ int s_counts[];  // size: max_frags_per_mol
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int tid = threadIdx.x;

    // init shared counts
    for (int f = tid; f < max_frags_per_mol; f += blockDim.x) {
        s_counts[f] = 0;
    }
    __syncthreads();

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];

    // count atoms per frag
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        int frag = atom_frag_id[idx];
        if (frag >= 0 && frag < max_frags_per_mol) {
            atomicAdd(&s_counts[frag], 1);
        }
    }
    __syncthreads();

    // find largest fragment by count (thread 0)
    int best_frag = 0;
    if (tid == 0) {
        int best_count = -1;
        for (int f = 0; f < max_frags_per_mol; ++f) {
            int c = s_counts[f];
            if (c > best_count) {
                best_count = c;
                best_frag = f;
            }
        }
        mol_largest_frag_id[mol_id] = best_frag;
    }
}


__global__
void apply_largest_fragment_mask_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int  const* __restrict__ atom_frag_id,
    int  const* __restrict__ mol_largest_frag_id,
    int         num_mols,
    uint8_t*    atom_keep_mask  // 1 = keep, 0 = drop
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];
    int largest_frag = mol_largest_frag_id[mol_id];

    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        int frag = atom_frag_id[idx];
        atom_keep_mask[idx] = (frag == largest_frag) ? 1 : 0;
    }
}

__global__
void mark_explicit_h_kernel(
    int  const* __restrict__ mol_atom_offset,
    int  const* __restrict__ mol_num_atoms,
    int         num_mols,
    uint32_t*   atom_flags,
    uint8_t*    atom_keep_mask  // 1 = keep, 0 = drop
) {
    int mol_id = blockIdx.x;
    if (mol_id >= num_mols) return;

    int start = mol_atom_offset[mol_id];
    int n     = mol_num_atoms[mol_id];
    int tid   = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) {
        int idx = start + i;
        uint32_t f = atom_flags[idx];

        // If this bit marks explicit H (simplified example),
        // set keep_mask=0 so CPU removes this atom.
        if (f & FLAG_EXPLICIT_H) {
            atom_keep_mask[idx] = 0;
        }
    }
}
