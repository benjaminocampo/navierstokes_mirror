#pragma once
#include <stdio.h>
#include <x86intrin.h>

static inline void dump_pfloats(__m256 ps) {
  float as[8];
  _mm256_storeu_ps(&as[0], ps);
  printf("[%f, %f, %f, %f, %f, %f, %f, %f]\n", as[0], as[1], as[2], as[3],
         as[4], as[5], as[6], as[7]);
}

static inline void dump_pints(__m256i epi) {
  int as[8];
  _mm256_storeu_si256((__m256i*)&as[0], epi);
  printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n", as[0], as[1], as[2], as[3],
         as[4], as[5], as[6], as[7]);
}

static inline __m256i imul(__m256i a, __m256i b) {
  return _mm256_mullo_epi32(a, b);
}

static inline int fiszero(__m256 ps) {
  float as[8];
  _mm256_storeu_ps(&as[0], ps);
  return as[0] == 0 && as[1] == 0 && as[2] == 0 && as[3] == 0 && as[4] == 0 &&
         as[5] == 0 && as[6] == 0 && as[7] == 0;
}

static inline __m256 fload2x4(float const *base_addr) {
  // equivalent to _mm256_loadu2_m128
  __m256 __v256 = _mm256_castps128_ps256(_mm_loadu_ps(base_addr));
  return _mm256_insertf128_ps(__v256, _mm_loadu_ps(base_addr + 4), 1);
}

static inline __m256 fmul(__m256 a, __m256 b) {
  return _mm256_mul_ps(a, b);
}

static inline __m256 fadd(__m256 a, __m256 b) {
  return _mm256_add_ps(a, b);
}

static inline __m256 ffmadd(__m256 a, __m256 x, __m256 y) {
  return _mm256_fmadd_ps(a, x, y);  // a * x + y
}
