#pragma once
#include <stdio.h>
#include <x86intrin.h>

// Floating point helpers

static inline void dump_pfloats(__m256 ps) {
  float as[8];
  _mm256_storeu_ps(&as[0], ps);
  printf("[%f, %f, %f, %f, %f, %f, %f, %f]\n", as[0], as[1], as[2], as[3],
         as[4], as[5], as[6], as[7]);
}

static inline int fiszero(__m256 ps) {
  float as[8];
  _mm256_storeu_ps(&as[0], ps);
  return as[0] == 0 && as[1] == 0 && as[2] == 0 && as[3] == 0 && as[4] == 0 &&
         as[5] == 0 && as[6] == 0 && as[7] == 0;
}

// Unaligned 256b load
static inline __m256 fload(float const *base_addr) {
  return _mm256_loadu_ps(base_addr);
}

// Unaligned 256b store
static inline void fstore(float *base_addr, __m256 data) {
  _mm256_storeu_ps(base_addr, data);
}

// Aligned 256b load
static inline __m256 faload(float const *base_addr) {
  return _mm256_load_ps(base_addr);
}

// Aligned 256b store
static inline void fastore(float *base_addr, __m256 data) {
  _mm256_store_ps(base_addr, data);
}

// equivalent to _mm256_loadu2_m128
static inline __m256 fload2x4(float const *base_addr) {
  __m256 __v256 = _mm256_castps128_ps256(_mm_loadu_ps(base_addr));
  return _mm256_insertf128_ps(__v256, _mm_loadu_ps(base_addr + 4), 1);
}

static inline __m256 fmul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
static inline __m256 fadd(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
static inline __m256 fsub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
// The p prefix is for conflicts with math.h
static inline __m256 pfmax(__m256 a, __m256 b) { return _mm256_max_ps(a, b); }
// The p prefix is for conflicts with math.h
static inline __m256 pfmin(__m256 a, __m256 b) { return _mm256_min_ps(a, b); }

static inline __m256 fclamp(__m256 t, __m256 a, __m256 b) {
  return pfmin(pfmax(a, t), b);
}

// a * x + y
static inline __m256 ffmadd(__m256 a, __m256 x, __m256 y) {
  return _mm256_fmadd_ps(a, x, y);
}

// -(a * x) + y
static inline __m256 ffnmadd(__m256 a, __m256 x, __m256 y) {
  return _mm256_fnmadd_ps(a, x, y);
}

static inline __m256 fset(float x7, float x6, float x5, float x4, float x3,
                          float x2, float x1, float x0) {
  return _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
}

static inline __m256 fset1(float a) { return _mm256_set1_ps(a); }

// shift right: if a = [ 8 7 6 5 | 4 3 2 1 ] -> returns [ 0 8 7 6 | 5 4 3 2 ]
static inline __m256 fshr(__m256 a) {
  __m256 t0 = _mm256_permute_ps(a, 0b00111001);  // [ 5 8 7 6 | 1 4 3 2 ]
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0b10000001);  // [0 0 0 0 5 8 7 6]
  __m256 sr = _mm256_blend_ps(t0, t1, 0b10001000);  // [ 0 8 7 6 | 5 4 3 2 ]
  return sr;
}

// shift left: if a = [ 8 7 6 5 | 4 3 2 1 ] -> returns [ 7 6 5 4 | 3 2 1 0 ]
static inline __m256 fshl(__m256 a) {
  __m256 t0 = _mm256_permute_ps(a, 0b10010011);  // [ 7 6 5 8 | 3 2 1 4 ]
  __m256 t1 = _mm256_permute2f128_ps(t0, t0, 0b00001000);  // [3 2 1 4 0 0 0 0]
  __m256 sl = _mm256_blend_ps(t0, t1, 0b00010001);  // [ 7 6 5 4 | 3 2 1 0 ]
  return sl;
}

// Integer helpers

static inline void dump_pints(__m256i epi) {
  int as[8];
  _mm256_storeu_si256((__m256i *)&as[0], epi);
  printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n", as[0], as[1], as[2], as[3],
         as[4], as[5], as[6], as[7]);
}

static inline __m256i iadd(__m256i a, __m256i b) {
  return _mm256_add_epi32(a, b);
}

static inline __m256i isub(__m256i a, __m256i b) {
  return _mm256_sub_epi32(a, b);
}

static inline __m256i imul(__m256i a, __m256i b) {
  return _mm256_mullo_epi32(a, b);
}

static inline __m256i ixor(__m256i a, __m256i b) {
  return _mm256_xor_si256(a, b);
}

static inline __m256i iand(__m256i a, __m256i b) {
  return _mm256_and_si256(a, b);
}

static inline __m256i ior(__m256i a, __m256i b) {
  return _mm256_or_si256(a, b);
}

static inline __m256i iset(int x7, int x6, int x5, int x4, int x3, int x2,
                           int x1, int x0) {
  return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}

static inline __m256i iset1(int x) { return _mm256_set1_epi32(x); }

// Miscelaneous

static inline __m256 itof(__m256i x) { return _mm256_cvtepi32_ps(x); }
static inline __m256i ftoi(__m256 x) { return _mm256_cvttps_epi32(x); }
