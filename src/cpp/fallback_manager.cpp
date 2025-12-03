#include "fallback_engine.hpp"
#include <GraphMol/ROMol.h>
#include <GraphMol/MolOps.h>

FallbackEngine::FallbackEngine() {}

RDKit::ROMol* FallbackEngine::standardize(const RDKit::ROMol* inMol) {
    // Simple CPU safe fallback
    auto mol = new RDKit::ROMol(*inMol);

    RDKit::MolOps::removeHs(*mol);
    RDKit::MolOps::sanitizeMol(*mol);

    return mol;
}
