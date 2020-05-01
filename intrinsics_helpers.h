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
