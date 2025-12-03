#pragma once

#include <GraphMol/ROMol.h>

RDKit::ROMol gpu_kernel_stereo(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_charge(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_bond_order(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_aromaticity(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_mesomerizer(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_final_clean(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_clear_isotopes(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_clear_stereo(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_keep_largest_fragment(const RDKit::ROMol &mol);
RDKit::ROMol gpu_kernel_remove_explicit_h(const RDKit::ROMol &mol);