#pragma once
#include <stdio.h>
#include <x86intrin.h>

// Floating point helpers

static inline void dump_pfloats(__m128 ps) {
  float as[8];
  _mm_storeu_ps(&as[0], ps);
  printf("[%f, %f, %f, %f, %f, %f, %f, %f]\n", as[0], as[1], as[2], as[3],
         as[4], as[5], as[6], as[7]);
}

static inline int fiszero(__m128 ps) {
  float as[8];
  _mm_storeu_ps(&as[0], ps);
  return as[0] == 0 && as[1] == 0 && as[2] == 0 && as[3] == 0 && as[4] == 0 &&
         as[5] == 0 && as[6] == 0 && as[7] == 0;
}

// Unaligned 256b load
static inline __m128 fload(float const *base_addr) {
  return _mm_loadu_ps(base_addr);
}

// Unaligned 256b store
static inline void fstore(float *base_addr, __m128 data) {
  _mm_storeu_ps(base_addr, data);
}

// Aligned 256b load
static inline __m128 faload(float const *base_addr) {
  return _mm_load_ps(base_addr);
}

// Aligned 256b store
static inline void fastore(float *base_addr, __m128 data) {
  _mm_store_ps(base_addr, data);
}

// equivalent to _mm_loadu2_m128
static inline __m128 fload2x4(float const *base_addr) {
  return fload(base_addr);
}

static inline __m128 fmul(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
static inline __m128 fadd(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
static inline __m128 fsub(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
// The p prefix is for conflicts with math.h
static inline __m128 pfmax(__m128 a, __m128 b) { return _mm_max_ps(a, b); }
// The p prefix is for conflicts with math.h
static inline __m128 pfmin(__m128 a, __m128 b) { return _mm_min_ps(a, b); }

static inline __m128 fclamp(__m128 t, __m128 a, __m128 b) {
  return pfmin(pfmax(a, t), b);
}

static inline __m128 fset(float x3, float x2, float x1, float x0) {
  return _mm_set_ps(x3, x2, x1, x0);
}

static inline __m128 fset1(float a) { return _mm_set1_ps(a); }

// a * x + y
static inline __m128 ffmadd(__m128 a, __m128 x, __m128 y) {
  // return _mm_fmadd_ps(a, x, y);
  return fadd(fmul(a, x), y);
}

// -(a * x) + y
static inline __m128 ffnmadd(__m128 a, __m128 x, __m128 y) {
  // return _mm_fnmadd_ps(a, x, y);
  return fadd(fmul(fset1(-1.0f), fmul(a, x)), y);
}

// shift right: if a = [ 8 7 6 5 | 4 3 2 1 ] -> returns [ 0 8 7 6 | 5 4 3 2 ]
static inline __m128 fshr(__m128 a) {
  __m128 t0 = _mm_blend_ps(a, fset1(0), 0b0001);  // [ 4 3 2 0 ]
  __m128 t1 = _mm_shuffle_ps(t0, t0, 0b00111001);     // [ 0 4 3 2 ]
  return t1;
}

// shift left: if a = [ 8 7 6 5 | 4 3 2 1 ] -> returns [ 7 6 5 4 | 3 2 1 0 ]
static inline __m128 fshl(__m128 a) {
  __m128 t0 = _mm_blend_ps(a, fset1(0), 0b1000);  // [ 0 3 2 1 ]
  __m128 t1 = _mm_shuffle_ps(t0, t0, 0b10010011);     // [ 3 2 1 0 ]
  return t1;
}

// Integer helpers

static inline void dump_pints(__m128i epi) {
  int as[8];
  _mm_storeu_si128((__m128i *)&as[0], epi);
  printf("[%d, %d, %d, %d, %d, %d, %d, %d]\n", as[0], as[1], as[2], as[3],
         as[4], as[5], as[6], as[7]);
}

static inline __m128i iadd(__m128i a, __m128i b) { return _mm_add_epi32(a, b); }

static inline __m128i isub(__m128i a, __m128i b) { return _mm_sub_epi32(a, b); }

static inline __m128i imul(__m128i a, __m128i b) {
  return _mm_mullo_epi32(a, b);
}

static inline __m128i ixor(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }

static inline __m128i iand(__m128i a, __m128i b) { return _mm_and_si128(a, b); }

static inline __m128i ior(__m128i a, __m128i b) { return _mm_or_si128(a, b); }

static inline __m128i iset(int x3, int x2, int x1, int x0) {
  return _mm_set_epi32(x3, x2, x1, x0);
}

static inline __m128i iset1(int x) { return _mm_set1_epi32(x); }

// Miscelaneous

static inline __m128 itof(__m128i x) { return _mm_cvtepi32_ps(x); }
static inline __m128i ftoi(__m128 x) { return _mm_cvttps_epi32(x); }

static inline __m128 fgather(const float *base, __m128i indexes, float scale) {
  int a = _mm_extract_epi32(indexes, 0);
  int b = _mm_extract_epi32(indexes, 1);
  int c = _mm_extract_epi32(indexes, 2);
  int d = _mm_extract_epi32(indexes, 3);
  return fset(base[d], base[c], base[b], base[a]);
}

// i := j*32
// m := j*32
// addr := base_addr + SignExtend64(vindex[m+31:m]) * ZeroExtend64(scale) * 8
// dst[i+31:i] := MEM[addr+31:addr]
