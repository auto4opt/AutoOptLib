// Exact contiguous equality: no reductions of floating-point values, no FMA,
// and no reads beyond the final complete vector. Mixed dtypes use scalar code.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
namespace exact_simd {
inline bool available() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return true;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
inline bool equal(const char *a, const char *b, size_t n, bool integer) {
  size_t j = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
  for (; j + 2 <= n; j += 2) {
    uint64x2_t same =
        integer ? vceqq_s64(vld1q_s64(reinterpret_cast<const int64_t *>(a) + j),
                            vld1q_s64(reinterpret_cast<const int64_t *>(b) + j))
                : vceqq_f64(vld1q_f64(reinterpret_cast<const double *>(a) + j),
                            vld1q_f64(reinterpret_cast<const double *>(b) + j));
    if (vminvq_u32(vreinterpretq_u32_u64(same)) != UINT32_MAX)
      return false;
  }
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  for (; j + 4 <= n; j += 4) {
    if (integer) {
      auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + j * 8));
      auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + j * 8));
      if (_mm256_movemask_epi8(_mm256_cmpeq_epi64(x, y)) != -1)
        return false;
    } else {
      auto x = _mm256_loadu_pd(reinterpret_cast<const double *>(a) + j);
      auto y = _mm256_loadu_pd(reinterpret_cast<const double *>(b) + j);
      if (_mm256_movemask_pd(_mm256_cmp_pd(x, y, _CMP_EQ_OQ)) != 15)
        return false;
    }
  }
#endif
  for (; j < n; ++j) {
    if (integer) {
      int64_t x, y;
      std::memcpy(&x, a + j * 8, 8);
      std::memcpy(&y, b + j * 8, 8);
      if (x != y)
        return false;
    } else {
      double x, y;
      std::memcpy(&x, a + j * 8, 8);
      std::memcpy(&y, b + j * 8, 8);
      if (x != y)
        return false;
    }
  }
  return true;
}
} // namespace exact_simd
