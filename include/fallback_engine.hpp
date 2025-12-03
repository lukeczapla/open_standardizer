#pragma once
#include <GraphMol/ROMol.h>

class FallbackEngine {
public:
    FallbackEngine();
    RDKit::ROMol* standardize(const RDKit::ROMol* inMol);
};

