#pragma once

#include <optional>
#include <string_view>
#include <utility>

// Port of cap_guiding/case_metadata.py.
namespace guiding::campaign {

inline constexpr double kDefaultRampUpMm = 5.0;

// Plateau length encoded as L<length>mm in a case name or path. Mirrors
// _PLATEAU_TOKEN_RE exactly, including its separator class [\\/\\s_-] with
// IGNORECASE: backslash, slash, 's', 'S', underscore and hyphen (not whitespace).
[[nodiscard]] std::optional<double> infer_plateau_length_mm_from_text(std::string_view text);

// (plateau_start_mm, plateau_end_mm) = (ramp_up, ramp_up + length); throws if length <= 0.
[[nodiscard]] std::pair<double, double> plateau_window_from_length_mm(double plateau_length_mm,
                                                                      double ramp_up_mm = kDefaultRampUpMm);

[[nodiscard]] std::optional<std::pair<double, double>> infer_plateau_window_mm_from_text(
    std::string_view text, double ramp_up_mm = kDefaultRampUpMm);

}  // namespace guiding::campaign
