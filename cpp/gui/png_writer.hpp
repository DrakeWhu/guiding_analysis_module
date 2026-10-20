#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace guiding::gui {

// Writes 8-bit RGBA pixels (top row first) as a PNG with stored (uncompressed)
// deflate blocks: no zlib dependency, larger files, exact pixels.
void write_png_rgba(const std::filesystem::path& path, int width, int height, std::span<const std::uint8_t> rgba);

}  // namespace guiding::gui
