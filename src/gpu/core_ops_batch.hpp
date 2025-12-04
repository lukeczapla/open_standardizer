// src/gpu/core_ops_batch.hpp
#pragma once

#include <vector>
#include <cstdint>

namespace RDKit {
class ROMol;
}  // namespace RDKit

/**
 * CoreOpsBatch
 *
 * Standardized, flattened view of a batch of molecules for use with
 * core_ops.cu kernels (clear_isotopes_kernel, clear_stereo_kernel,
 * largest_fragment_kernel, apply_largest_fragment_mask_kernel,
 * mark_explicit_h_kernel).
 *
 * Layout:
 *   - num_mols          : number of molecules in this batch
 *   - total_atoms       : sum of all per-molecule atom counts
 *   - mol_atom_offset[m]: starting atom index for molecule m in
 *                         the flat arrays
 *   - mol_num_atoms[m]  : number of atoms in molecule m
 *
 * Per-atom arrays of length total_atoms:
 *   - atom_isotope[idx] : RDKit isotope label (0 = none)
 *   - atom_flags[idx]   : bitfield (stereo, explicit H, etc.)
 *   - atom_frag_id[idx] : fragment id (0..max_frags_per_mol-1)
 *   - atom_keep_mask[idx]: 1 = keep atom, 0 = drop (for salt stripping /
 *                          explicit-H removal, etc.)
 *
 * max_frags_per_mol is the maximum fragment count across all mols; this
 * matches the core_ops.cu kernels' expectations.
 */
struct CoreOpsBatch {
    int num_mols = 0;
    int total_atoms = 0;
    int max_frags_per_mol = 0;

    // molecule-level
    std::vector<int> mol_atom_offset;  // [num_mols]
    std::vector<int> mol_num_atoms;    // [num_mols]

    // atom-level (flattened over all molecules)
    std::vector<int>      atom_isotope;    // [total_atoms]
    std::vector<std::uint32_t> atom_flags;    // [total_atoms]
    std::vector<int>      atom_frag_id;    // [total_atoms]
    std::vector<std::uint8_t>  atom_keep_mask; // [total_atoms]
};


/**
 * Build a CoreOpsBatch from a vector of RDKit molecules.
 *
 * Responsibilities:
 *   - Assign offsets / counts per molecule
 *   - Fill atom_isotope from RDKit's atom->getIsotope()
 *   - Initialize atom_flags to 0 (you can later encode stereo/explicit-H bits)
 *   - Compute fragment IDs per atom (atom_frag_id) using RDKit's
 *     MolOps::getMolFrags mapping overload
 *   - Set max_frags_per_mol accordingly
 *   - Initialize atom_keep_mask to 1 (everything kept by default)
 *
 * NOTE: This is purely CPU-side and does NOT touch CUDA directly.
 */
CoreOpsBatch build_core_ops_batch_from_mols(
    const std::vector<const RDKit::ROMol*>& mols
);
