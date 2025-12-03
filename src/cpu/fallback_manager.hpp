#pragma once

#include <memory>
#include <GraphMol/ROMol.h>

std::unique_ptr<RDKit::ROMol> cpu_fallback_standardize(const RDKit::ROMol &mol);
