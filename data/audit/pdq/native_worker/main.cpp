// Portable, source-only boundary around Meta ThreatExchange PDQ.
// The worker accepts canonical interleaved RGB bytes; image decoding stays in Python.

#include <pdq/cpp/hashing/pdqhashing.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr std::array<std::uint8_t, 8> kRequestMagic = {
    'P', 'D', 'Q', 'R', 'E', 'Q', '0', '2'};
constexpr std::array<std::uint8_t, 8> kResponseMagic = {
    'P', 'D', 'Q', 'R', 'S', 'P', '0', '2'};
constexpr std::uint32_t kProtocolVersion = 2;
constexpr std::uint32_t kMaximumDimension = 16384;
constexpr std::uint64_t kMaximumPixels = 33554432;
constexpr std::uint64_t kChannels = 3;
constexpr std::uint32_t kPdqIoResizeDimension = 512;

void apply_pdqio_oversize_resize(
    std::vector<std::uint8_t>& rgb,
    std::uint32_t& width,
    std::uint32_t& height) {
  if (width <= kPdqIoResizeDimension && height <= kPdqIoResizeDimension) {
    return;
  }
  const std::uint32_t source_width = width;
  const std::uint32_t source_height = height;
  const std::size_t output_bytes =
      static_cast<std::size_t>(kPdqIoResizeDimension) *
      kPdqIoResizeDimension * kChannels;
  std::vector<std::uint8_t> resized(output_bytes);
  for (std::uint32_t output_y = 0; output_y < kPdqIoResizeDimension; ++output_y) {
    const std::uint32_t source_y = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(output_y) * source_height /
        kPdqIoResizeDimension);
    for (std::uint32_t output_x = 0; output_x < kPdqIoResizeDimension; ++output_x) {
      const std::uint32_t source_x = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(output_x) * source_width /
          kPdqIoResizeDimension);
      const std::size_t source_offset =
          (static_cast<std::size_t>(source_y) * source_width + source_x) *
          kChannels;
      const std::size_t output_offset =
          (static_cast<std::size_t>(output_y) * kPdqIoResizeDimension +
           output_x) * kChannels;
      resized[output_offset] = rgb[source_offset];
      resized[output_offset + 1] = rgb[source_offset + 1];
      resized[output_offset + 2] = rgb[source_offset + 2];
    }
  }
  rgb.swap(resized);
  width = kPdqIoResizeDimension;
  height = kPdqIoResizeDimension;
}

bool read_exact_or_clean_eof(void* destination, std::size_t size, bool& clean_eof) {
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t offset = 0;
  clean_eof = false;
  while (offset < size) {
    const std::size_t count = std::fread(output + offset, 1, size - offset, stdin);
    if (count > 0) {
      offset += count;
      continue;
    }
    if (std::feof(stdin)) {
      clean_eof = (offset == 0);
      return false;
    }
    return false;
  }
  return true;
}

bool write_exact(const void* source, std::size_t size) {
  const auto* input = static_cast<const std::uint8_t*>(source);
  std::size_t offset = 0;
  while (offset < size) {
    const std::size_t count = std::fwrite(input + offset, 1, size - offset, stdout);
    if (count == 0) {
      return false;
    }
    offset += count;
  }
  return true;
}

std::uint32_t load_u32_le(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
      (static_cast<std::uint32_t>(bytes[1]) << 8) |
      (static_cast<std::uint32_t>(bytes[2]) << 16) |
      (static_cast<std::uint32_t>(bytes[3]) << 24);
}

void store_u32_le(std::uint8_t* bytes, std::uint32_t value) {
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8);
  bytes[2] = static_cast<std::uint8_t>(value >> 16);
  bytes[3] = static_cast<std::uint8_t>(value >> 24);
}

std::uint64_t load_u64_le(const std::uint8_t* bytes) {
  std::uint64_t value = 0;
  for (int index = 7; index >= 0; --index) {
    value = (value << 8) | bytes[index];
  }
  return value;
}

void store_u64_le(std::uint8_t* bytes, std::uint64_t value) {
  for (int index = 0; index < 8; ++index) {
    bytes[index] = static_cast<std::uint8_t>(value);
    value >>= 8;
  }
}

int fail(const char* message) {
  std::fprintf(stderr, "PDQ worker: %s\n", message);
  return 2;
}

bool emit_response(
    std::uint64_t request_sequence,
    const std::array<std::uint8_t, 32>& request_token,
    int quality,
    const std::array<facebook::pdq::hashing::Hash256, 8>& hashes) {
  std::array<std::uint8_t, 316> response{};
  for (std::size_t index = 0; index < kResponseMagic.size(); ++index) {
    response[index] = kResponseMagic[index];
  }
  store_u32_le(response.data() + 8, kProtocolVersion);
  store_u32_le(response.data() + 12, 0);
  store_u32_le(response.data() + 16, static_cast<std::uint32_t>(quality));
  store_u64_le(response.data() + 20, request_sequence);
  for (std::size_t index = 0; index < request_token.size(); ++index) {
    response[28 + index] = request_token[index];
  }
  std::size_t offset = 60;
  for (const auto& hash : hashes) {
    for (int word_index = 15; word_index >= 0; --word_index) {
      const std::uint16_t word = hash.w[word_index];
      response[offset++] = static_cast<std::uint8_t>(word >> 8);
      response[offset++] = static_cast<std::uint8_t>(word);
    }
  }
  return write_exact(response.data(), response.size()) && std::fflush(stdout) == 0;
}

} // namespace

int main() {
  try {
    for (;;) {
      std::array<std::uint8_t, 64> header{};
      bool clean_eof = false;
      if (!read_exact_or_clean_eof(header.data(), header.size(), clean_eof)) {
        return clean_eof ? 0 : fail("truncated request header");
      }
      for (std::size_t index = 0; index < kRequestMagic.size(); ++index) {
        if (header[index] != kRequestMagic[index]) {
          return fail("request magic differs");
        }
      }
      const std::uint32_t version = load_u32_le(header.data() + 8);
      const std::uint64_t request_sequence = load_u64_le(header.data() + 12);
      std::uint32_t width = load_u32_le(header.data() + 20);
      std::uint32_t height = load_u32_le(header.data() + 24);
      const std::uint32_t payload_bytes = load_u32_le(header.data() + 28);
      std::array<std::uint8_t, 32> request_token{};
      for (std::size_t index = 0; index < request_token.size(); ++index) {
        request_token[index] = header[32 + index];
      }
      if (version != kProtocolVersion) {
        return fail("protocol version differs");
      }
      if (width == 0 || height == 0 || width > kMaximumDimension ||
          height > kMaximumDimension) {
        return fail("image dimensions are outside the fixed bounds");
      }
      const std::uint64_t pixels =
          static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (pixels > kMaximumPixels ||
          pixels > std::numeric_limits<std::size_t>::max() / kChannels) {
        return fail("image pixel count is outside the fixed bounds");
      }
      const std::uint64_t expected_bytes = pixels * kChannels;
      if (expected_bytes != payload_bytes) {
        return fail("RGB payload length differs from image geometry");
      }

      std::vector<std::uint8_t> rgb(static_cast<std::size_t>(payload_bytes));
      if (!read_exact_or_clean_eof(rgb.data(), rgb.size(), clean_eof)) {
        return fail("truncated RGB payload");
      }
      apply_pdqio_oversize_resize(rgb, width, height);
      const std::uint64_t working_pixels =
          static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
      if (working_pixels > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return fail("float buffer size overflows size_t");
      }
      std::vector<float> full_buffer_1(static_cast<std::size_t>(working_pixels));
      facebook::pdq::hashing::fillFloatLumaFromRGB(
          rgb.data(), rgb.data() + 1, rgb.data() + 2,
          static_cast<int>(height), static_cast<int>(width),
          static_cast<int>(width * kChannels), static_cast<int>(kChannels),
          full_buffer_1.data());
      std::vector<std::uint8_t>().swap(rgb);
      std::vector<float> full_buffer_2(static_cast<std::size_t>(working_pixels));

      float buffer64x64[64][64]{};
      float buffer16x64[16][64]{};
      float buffer16x16[16][16]{};
      float buffer16x16_aux[16][16]{};
      std::array<facebook::pdq::hashing::Hash256, 8> hashes{};
      int quality = -1;
      const bool ok = facebook::pdq::hashing::pdqDihedralHash256esFromFloatLuma(
          full_buffer_1.data(), full_buffer_2.data(),
          static_cast<int>(height), static_cast<int>(width),
          buffer64x64, buffer16x64, buffer16x16, buffer16x16_aux,
          &hashes[0], &hashes[1], &hashes[2], &hashes[3],
          &hashes[4], &hashes[5], &hashes[6], &hashes[7], quality);
      if (!ok || quality < 0 || quality > 100) {
        return fail("upstream PDQ returned an invalid result");
      }
      if (!emit_response(request_sequence, request_token, quality, hashes)) {
        return fail("response write failed");
      }
    }
  } catch (const std::bad_alloc&) {
    return fail("memory allocation failed");
  } catch (const std::exception&) {
    return fail("unexpected C++ exception");
  } catch (...) {
    return fail("unexpected non-standard exception");
  }
}
