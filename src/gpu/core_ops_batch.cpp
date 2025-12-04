// src/gpu/core_ops_batch.cpp

#include "core_ops_batch.hpp"
#include "core_ops_flags.hpp"

#include <algorithm>
#include <GraphMol/ROMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/MolOps.h>

using RDKit::ROMol;

CoreOpsBatch build_core_ops_batch_from_mols(
    const std::vector<const ROMol*>& mols
) {
    CoreOpsBatch batch;
    batch.num_mols = static_cast<int>(mols.size());

    batch.mol_atom_offset.resize(batch.num_mols);
    batch.mol_num_atoms.resize(batch.num_mols);

    // -----------------------------
    // 1) Compute per-mol offsets and total_atoms
    // -----------------------------
    int offset = 0;
    for (int m = 0; m < batch.num_mols; ++m) {
        const ROMol* mol = mols[m];
        if (!mol) {
            batch.mol_atom_offset[m] = offset;
            batch.mol_num_atoms[m]   = 0;
            continue;
        }
        int nAtoms = static_cast<int>(mol->getNumAtoms());
        batch.mol_atom_offset[m] = offset;
        batch.mol_num_atoms[m]   = nAtoms;
        offset += nAtoms;
    }
    batch.total_atoms = offset;

    batch.atom_isotope.resize(batch.total_atoms);
    batch.atom_flags.resize(batch.total_atoms);
    batch.atom_frag_id.resize(batch.total_atoms);
    batch.atom_keep_mask.resize(batch.total_atoms, 1u); // default: keep everything

    int global_idx = 0;
    int max_frags = 0;

    // -----------------------------
    // 2) Fill per-atom fields
    // -----------------------------
    for (int m = 0; m < batch.num_mols; ++m) {
        const ROMol* mol = mols[m];
        int nAtoms = batch.mol_num_atoms[m];
        if (!mol || nAtoms == 0) {
            continue;
        }

        // --- isotopes & flags (stereo/H bits TBD) ---
        for (int i = 0; i < nAtoms; ++i) {
            int idx = global_idx + i;
            const auto* atom = mol->getAtomWithIdx(i);
            batch.atom_isotope[idx] = atom->getIsotope();
            batch.atom_flags[idx]   = 0;  // TODO: encode stereo / explicit H if desired
        }

        // --- fragment IDs per atom ---
        // we only need the mapping; let RDKit create it for us
        std::vector<int> fragMapping(nAtoms, 0);
        // This overload of getMolFrags fills fragMapping with fragment index
        // for each atom. We don't need the fragment ROMols themselves.
        RDKit::MolOps::getMolFrags(
            *mol,
            nullptr,        // frags: we don't need the fragment molecules here
            &fragMapping,   // atomToFragmentMapping
            nullptr,        // fName
            true,           // asMols
            nullptr,        // numToUse
            false           // sanitizeFrags
        );

        int local_max_frag = 0;
        for (int i = 0; i < nAtoms; ++i) {
            int idx   = global_idx + i;
            int frag  = fragMapping[i];
            batch.atom_frag_id[idx] = frag;
            if (frag > local_max_frag) {
                local_max_frag = frag;
            }
        }
        // fragment IDs are assumed to be 0..(numFrags-1)
        int numFrags = local_max_frag + 1;
        if (numFrags > max_frags) {
            max_frags = numFrags;
        }

        global_idx += nAtoms;
    }

    batch.max_frags_per_mol = (max_frags > 0 ? max_frags : 1);
    return batch;
}
