#pragma once

#include <string_view>

namespace guiding::physics {

// Beam axis used for the forward cut, the exit window and phase space (x, y or z).
enum class Longitudinal { X, Y, Z };

[[nodiscard]] Longitudinal parse_longitudinal(std::string_view name);
[[nodiscard]] const char* longitudinal_name(Longitudinal axis) noexcept;

}  // namespace guiding::physics
