#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

// Text formatting that matches what the Python reference pipeline writes.
namespace guiding::table {

// repr() of a Python str and of a list of str ("['a', 'b']").
[[nodiscard]] std::string py_str_repr(std::string_view text);
[[nodiscard]] std::string python_list_repr(std::span<const std::string> items);

// str(pathlib.Path(p)) for POSIX paths: collapses repeated separators and
// drops "." components and a trailing separator.
[[nodiscard]] std::string python_path_string(const std::filesystem::path& path);

// Python repr() of a float: shortest round-trip digits, fixed notation when the
// decimal exponent is in (-4, 16], exponent notation otherwise ("1e-05",
// "1e+16"), "nan"/"inf"/"-inf" for non-finite values.
void append_py_float_repr(std::string& out, double value);
[[nodiscard]] std::string py_float_repr(double value);

[[nodiscard]] inline const char* py_bool(bool value) noexcept { return value ? "True" : "False"; }

}  // namespace guiding::table
