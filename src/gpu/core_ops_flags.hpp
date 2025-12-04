// src/gpu/core_ops_flags.hpp
#pragma once

#include <cstdint>

// Bit masks for atom_flags used by core_ops.cu and batch builder.
// Keep these in sync with core_ops.cu.
static constexpr std::uint32_t FLAG_STEREO_MASK = 0x0000000F;  // low bits = stereo
static constexpr std::uint32_t FLAG_EXPLICIT_H  = 0x00000010;  // explicit H marker
