#pragma once

#include <cstdint>
#include <string>

// Text formatting that matches what the Python reference pipeline writes.
namespace guiding::table {

// Python repr() of a float: shortest round-trip digits, fixed notation when the
// decimal exponent is in (-4, 16], exponent notation otherwise ("1e-05",
// "1e+16"), "nan"/"inf"/"-inf" for non-finite values.
void append_py_float_repr(std::string& out, double value);
[[nodiscard]] std::string py_float_repr(double value);

[[nodiscard]] inline const char* py_bool(bool value) noexcept { return value ? "True" : "False"; }

}  // namespace guiding::table
