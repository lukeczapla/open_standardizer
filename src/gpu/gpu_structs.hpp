#pragma once

#include <cstdint>

// Minimal atom representation for GPU kernels.
// Extend this if you later need more fields (e.g. implicit H count).
struct GAtom {
    std::uint8_t atomicNum;    // Z
    std::int8_t  formalCharge; // RDKit formal charge
    std::uint8_t valence;      // explicit valence
    std::uint8_t degree;       // number of neighbors
    // You can add flags/bitfields here as needed.
};

// Minimal bond representation for GPU kernels.
struct GBond {
    std::int32_t a1;           // begin atom index
    std::int32_t a2;           // end atom index
    std::uint8_t bondOrder;    // 1 = single, 2 = double, 3 = triple, etc.
    std::uint8_t aromaticFlag; // 1 = aromatic, 0 = non-aromatic
    // Add stereochemistry bits here if you need them later.
};
