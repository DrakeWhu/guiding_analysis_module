#include "png_writer.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

namespace guiding::gui {
namespace {

const std::array<std::uint32_t, 256>& crc_table() {
  static const auto table = [] {
    std::array<std::uint32_t, 256> values{};
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
      }
      values[n] = c;
    }
    return values;
  }();
  return table;
}

std::uint32_t crc32(std::span<const std::uint8_t> bytes) {
  std::uint32_t c = 0xFFFFFFFFU;
  for (std::uint8_t byte : bytes) {
    c = crc_table()[(c ^ byte) & 0xFFU] ^ (c >> 8U);
  }
  return c ^ 0xFFFFFFFFU;
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>(value >> static_cast<unsigned>(shift)));
  }
}

void put_chunk(std::vector<std::uint8_t>& out, const char* type, std::span<const std::uint8_t> data) {
  put_u32(out, static_cast<std::uint32_t>(data.size()));
  const std::size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  put_u32(out, crc32({out.data() + start, out.size() - start}));
}

}  // namespace

void write_png_rgba(const std::filesystem::path& path, int width, int height, std::span<const std::uint8_t> rgba) {
  if (width <= 0 || height <= 0 || rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4) {
    throw std::invalid_argument("write_png_rgba: pixel buffer does not match the image size");
  }
  // Scanlines with filter type 0.
  const std::size_t stride = static_cast<std::size_t>(width) * 4;
  std::vector<std::uint8_t> raw;
  raw.reserve((stride + 1) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    const auto* row = rgba.data() + static_cast<std::size_t>(y) * stride;
    raw.insert(raw.end(), row, row + stride);
  }

  // zlib stream of stored blocks.
  std::vector<std::uint8_t> zlib{0x78, 0x01};
  std::size_t offset = 0;
  do {
    const std::size_t length = std::min<std::size_t>(65535, raw.size() - offset);
    const bool last = offset + length == raw.size();
    zlib.push_back(last ? 1 : 0);
    zlib.push_back(static_cast<std::uint8_t>(length & 0xFFU));
    zlib.push_back(static_cast<std::uint8_t>(length >> 8U));
    zlib.push_back(static_cast<std::uint8_t>(~length & 0xFFU));
    zlib.push_back(static_cast<std::uint8_t>((~length >> 8U) & 0xFFU));
    zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                raw.begin() + static_cast<std::ptrdiff_t>(offset + length));
    offset += length;
  } while (offset < raw.size());
  std::uint32_t a = 1;
  std::uint32_t b = 0;
  for (std::uint8_t byte : raw) {
    a = (a + byte) % 65521U;
    b = (b + a) % 65521U;
  }
  put_u32(zlib, (b << 16U) | a);

  std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
  std::vector<std::uint8_t> header;
  put_u32(header, static_cast<std::uint32_t>(width));
  put_u32(header, static_cast<std::uint32_t>(height));
  header.insert(header.end(), {8, 6, 0, 0, 0});  // 8-bit RGBA, no interlace
  put_chunk(png, "IHDR", header);
  put_chunk(png, "IDAT", zlib);
  put_chunk(png, "IEND", {});

  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(fmt::format("cannot write {}", path.string()));
  }
  stream.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
}

}  // namespace guiding::gui
