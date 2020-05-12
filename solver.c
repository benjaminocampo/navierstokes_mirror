#include "solver.h"

#include <stddef.h>
#include <x86intrin.h>

#include "indices.h"
#include "intrinsics_helpers.h"

#define IX(x, y) (rb_idx((x), (y), (n + 2)))
#define SWAP(x0, x)  \
  {                  \
    float *tmp = x0; \
    x0 = x;          \
    x = tmp;         \
  }

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

static void add_source(unsigned int n, float *x, const float *s, float dt) {
  unsigned int size = (n + 2) * (n + 2);
  const __m256 pdt = fset1(dt);
  unsigned int i;
  for (i = 0; i < size - 8; i += 8) {
    __m256 px = faload(&x[i]);
    __m256 ps = faload(&s[i]);
    __m256 product = _mm256_fmadd_ps(pdt, ps, px);  // x + dt * s[i]
    _mm256_store_ps(&x[i], product);                // x[i] += dt * s[i];
  }
  for (; i < size; i++) x[i] += dt * s[i];
}

static void set_bnd(unsigned int n, boundary b, float *x) {
  for (unsigned int i = 1; i <= n; i++) {
    x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
    x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
  }
  x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
  x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
  x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void lin_solve_rb_step(grid_color color, unsigned int n, float a,
                              float c, const float *restrict same0,
                              const float *restrict neigh,
                              float *restrict same) {
  const float invc = 1 / c;
  unsigned int start = color == RED ? 0 : 1;
  unsigned int width = (n + 2) / 2;
  const __m256 pinvc = fset1(invc);
  const __m256 pa = fset1(a);
  float leftmost;  // Left slack of the left-right neighbours load
  for (unsigned int y = 1; y <= n; ++y, start = 1 - start) {
    leftmost = neigh[idx(0, y, width)];
    for (unsigned int x = start; x < width - (1 - start); x += 8) {
      int index = idx(x, y, width);
      // In haswell it is a tad better to load two 128 vectors when unaligned
      // See 14.6.2 at intel IA-32 Architectures Optimization Reference Manual
      __m256 f = fload2x4(&same0[index]);
      __m256 u = fload2x4(&neigh[index - width]);
      __m256 r = fload2x4(&neigh[index - start + 1]);
      __m256 d = fload2x4(&neigh[index + width]);
      __m256 l = _mm256_blend_ps(fshl(r), fset1(leftmost), 0b00000001);

      // t = (f + a * (u + r + d + l)) / c
      __m256 t = fmul(ffmadd(pa, fadd(u, fadd(r, fadd(d, l))), f), pinvc);
      fstore(&same[index], t);

      // extract the rightmost to be the next leftmost
      _MM_EXTRACT_FLOAT(leftmost, _mm256_extractf128_ps(r, 1), 3);
    }
  }
}

static void lin_solve(unsigned int n, boundary b, float *restrict x,
                      const float *restrict x0, float a, float c) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  const float *red0 = x0;
  const float *blk0 = x0 + color_size;
  float *red = x;
  float *blk = x + color_size;

  for (unsigned int k = 0; k < 20; ++k) {
    lin_solve_rb_step(RED, n, a, c, red0, blk, red);
    lin_solve_rb_step(BLACK, n, a, c, blk0, red, blk);
    set_bnd(n, b, x);
  }
}

static void diffuse(unsigned int n, boundary b, float *x, const float *x0,
                    float diff, float dt) {
  float a = dt * diff * n * n;
  lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect_rb(grid_color color, unsigned int n, float *samed,
                      float *sameu, float *samev, const float *samed0,
                      const float *sameu0, const float *samev0,
                      const float *d0, const float *u0, const float *v0,
                      float dt) {
  int shift = color == RED ? 1 : -1;
  int start = color == RED ? 0 : 1;
  const int width = (n + 2) / 2;
  const float dt0 = dt * n;

  // See answers from Peter Cordes as to why we can't use _mm256_loadu_epi32
  // https://stackoverflow.com/questions/59649287/how-to-emulate-mm256-loadu-epi32-with-gcc-or-clang
  // https://stackoverflow.com/questions/53905757/what-is-the-difference-between-mm512-load-epi32-and-mm512-load-si512
  // TODO: See agner fog optimization manuals for generating constants
  // maybe using that we can free some vector registers
  const __m256i psuccs = iset(7, 6, 5, 4, 3, 2, 1, 0);
  const __m256 pdt0 = fset1(dt0);
  const __m256 plowest = fset1(0.5f);
  const __m256 phighest = fset1(n + 0.5f);
  const __m256i pone = iset1(1);
  const __m256 pfone = fset1(1.0);
  const __m256i pnplus2 = iset1(n + 2);
  const __m256i pwidth = iset1(width);
  const __m256i phalfgrid = imul(pnplus2, pwidth);
  for (int iy = 1; iy <= (int)n; iy++, shift = -shift, start = 1 - start) {
    const __m256i pshift = iset1(shift);
    const __m256i pstart = iset1(start);
    const __m256i pi = iset1(iy);
    for (int ix = start; ix < width - (1 - start); ix += 8) {
      __m256i pj = iadd(iset1(ix), psuccs);  // j = x + 0, ..., x + 7

      int index = idx(ix, iy, width);
      const __m256i pgridi = pi;
      const __m256 pfgridi = itof(pgridi);  // (float)gridi
      const __m256i pgridj =                // 2 * j + shift + start
          iadd(_mm256_slli_epi32(pj, 1), iadd(pshift, pstart));
      const __m256 pfgridj = itof(pgridj);  // (float)gridj
      const __m256 psameu0 = fload(&sameu0[index]);
      const __m256 psamev0 = fload(&samev0[index]);
      __m256 px = ffnmadd(pdt0, psameu0, pfgridj);  // gridj - dt0 * sameu0[i]
      __m256 py = ffnmadd(pdt0, psamev0, pfgridi);  // gridi - dt0 * samev0[i]

      // TODO: Idea, as suggested in
      // https://stackoverflow.com/questions/38006616/how-to-use-if-condition-in-intrinsics
      // Use an if instead of these four instructions
      // Usually none of the branches will be taken
      // The hope is the branch predictor will do its thing
      // To help it we could firstly traverse an internal portion of the
      // grid (with some margin) so it almost always guesses correctly
      px = fclamp(px, plowest, phighest);
      py = fclamp(py, plowest, phighest);

      const __m256i pj0 = ftoi(px);  // j0 = (int)x;
      const __m256i pi0 = ftoi(py);  // i0 = (int)y;

      const __m256 ps1 = _mm256_sub_ps(px, itof(pj0));  // s1 = x - j0;
      const __m256 ps0 = _mm256_sub_ps(pfone, ps1);     // s0 = 1 - s1;
      const __m256 pt1 = _mm256_sub_ps(py, itof(pi0));  // t1 = y - i0;
      const __m256 pt0 = _mm256_sub_ps(pfone, pt1);     // t0 = 1 - t1;

      // Let's build IX(j0, i0):
      const __m256i pisoddrow = iand(pi0, pone);         // i % 2
      const __m256i pisevenrow = ixor(pisoddrow, pone);  // !isoddrow

      // (!= parity) === (j0 % 2) ^ (i0 % 2) === (j0 & 1) ^ (i0 & 1)
      const __m256i pisblack = ixor(iand(pj0, pone), pisoddrow);
      const __m256i pisred = ixor(pisblack, pone);     // !isblack
      const __m256i pfshift = isub(pisred, pisblack);  // isred ? 1 : -1

      // TODO: All the bool operations are using entire int registers,
      // can we improve on that?
      // !((isred && isevenrow) || (isblack && isoddrow)), or equivalently
      // (isblack || isoddrow) && (isred || isevenrow)
      const __m256i p_starts_at_zero =
          iand(ior(pisblack, pisoddrow), ior(pisred, pisevenrow));

      // TODO: Maybe instead of a multiplication with
      // pisblack we could do a conditional move?
      // pbase = isblack ? (n+2) * width : 0
      const __m256i pbase = imul(pisblack, phalfgrid);
      const __m256i poffset = iadd(   // (j0 / 2) + i0 * ((n+2) / 2)
          _mm256_srai_epi32(pj0, 1),  // (j0 / 2)
          imul(pi0, pwidth)           // i0 * ((n+2) / 2)
      );

      // i0j0 = // IX(j0, i0)
      const __m256i pi0j0 = iadd(pbase, poffset);
      // i1j1 = i0j0 + width + (1 - isoffstart);
      const __m256i pi1j1 = iadd(pi0j0, iadd(pwidth, p_starts_at_zero));
      // i0j1 = i0j0 + fshift * width * (n + 2) + (1 - isoffstart);
      const __m256i pi0j1 =
          iadd(pi0j0, iadd(p_starts_at_zero, imul(pfshift, phalfgrid)));
      // i1j0 = i0j0 + fshift * width * (n + 2) + width;
      const __m256i pi1j0 = iadd(pi0j0, iadd(pwidth, imul(pfshift, phalfgrid)));

      // TODO: Gather ps seems to be slower on zx81 but faster on i7 7700hq
      // Read and test with:
      // https://stackoverflow.com/questions/24756534/in-what-situation-would-the-avx2-gather-instructions-be-faster-than-individually
      // So maybe we shouldn't use gather
      const __m256 pd0i0j0 = _mm256_i32gather_ps(d0, pi0j0, 4);
      const __m256 pd0i0j1 = _mm256_i32gather_ps(d0, pi0j1, 4);
      const __m256 pd0i1j0 = _mm256_i32gather_ps(d0, pi1j0, 4);
      const __m256 pd0i1j1 = _mm256_i32gather_ps(d0, pi1j1, 4);
      // s0 * (t0 * d0[i0j0] + t1 * d0[i1j0]) +
      // s1 * (t0 * d0[i0j1] + t1 * d0[i1j1])
      const __m256 psamed =
          fadd(fmul(ps0, fadd(fmul(pt0, pd0i0j0), fmul(pt1, pd0i1j0))),
               fmul(ps1, fadd(fmul(pt0, pd0i0j1), fmul(pt1, pd0i1j1))));

      const __m256 pu0i0j0 = _mm256_i32gather_ps(u0, pi0j0, 4);
      const __m256 pu0i0j1 = _mm256_i32gather_ps(u0, pi0j1, 4);
      const __m256 pu0i1j0 = _mm256_i32gather_ps(u0, pi1j0, 4);
      const __m256 pu0i1j1 = _mm256_i32gather_ps(u0, pi1j1, 4);
      // s0 * (t0 * u0[i0j0] + t1 * u0[i1j0]) +
      // s1 * (t0 * u0[i0j1] + t1 * u0[i1j1])
      const __m256 psameu =
          fadd(fmul(ps0, fadd(fmul(pt0, pu0i0j0), fmul(pt1, pu0i1j0))),
               fmul(ps1, fadd(fmul(pt0, pu0i0j1), fmul(pt1, pu0i1j1))));

      const __m256 pv0i0j0 = _mm256_i32gather_ps(v0, pi0j0, 4);
      const __m256 pv0i0j1 = _mm256_i32gather_ps(v0, pi0j1, 4);
      const __m256 pv0i1j0 = _mm256_i32gather_ps(v0, pi1j0, 4);
      const __m256 pv0i1j1 = _mm256_i32gather_ps(v0, pi1j1, 4);
      // s0 * (t0 * v0[i0j0] + t1 * v0[i1j0])
      // s1 * (t0 * v0[i0j1] + t1 * v0[i1j1])
      const __m256 psamev =
          fadd(fmul(ps0, fadd(fmul(pt0, pv0i0j0), fmul(pt1, pv0i1j0))),
               fmul(ps1, fadd(fmul(pt0, pv0i0j1), fmul(pt1, pv0i1j1))));

      fstore(&samed[index], psamed);
      fstore(&sameu[index], psameu);
      fstore(&samev[index], psamev);
    }
  }
}

static void advect(unsigned int n, float *d, float *u, float *v,
                   const float *d0, const float *u0, const float *v0,
                   float dt) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  float *redd = d;
  float *redu = u;
  float *redv = v;
  float *blkd = d + color_size;
  float *blku = u + color_size;
  float *blkv = v + color_size;
  const float *redd0 = d0;
  const float *redu0 = u0;
  const float *redv0 = v0;
  const float *blkd0 = d0 + color_size;
  const float *blku0 = u0 + color_size;
  const float *blkv0 = v0 + color_size;
  advect_rb(RED, n, redd, redu, redv, redd0, redu0, redv0, d0, u0, v0,
            dt);
  advect_rb(BLACK, n, blkd, blku, blkv, blkd0, blku0, blkv0, d0, u0,
            v0, dt);
  set_bnd(n, VERTICAL, u);
  set_bnd(n, HORIZONTAL, v);
}

static void project_rb_step1(unsigned int n, grid_color color,
                             float *restrict sameu0, float *restrict samev0,
                             float *restrict neighu, float *restrict neighv) {
  unsigned int start = color == RED ? 0 : 1;
  unsigned int width = (n + 2) / 2;

  const __m256 zeros = fset1(0.0f);
  for (unsigned int i = 1; i <= n; ++i, start = 1 - start)
    for (unsigned int j = start; j < width - (1 - start); j += 8)
      fstore(&sameu0[idx(j, i, width)], zeros);

  const __m256 ratio = fset1(-0.5f / n);
  for (unsigned int i = 1; i <= n; ++i, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); j += 8) {
      int index = idx(j, i, width);
      __m256 u = fload2x4(&neighv[index - width]);
      __m256 r = fload2x4(&neighu[index - start + 1]);
      __m256 d = fload2x4(&neighv[index + width]);
      __m256 l = fload2x4(&neighu[index - start]);
      __m256 result = fmul(ratio, fadd(fsub(r, l), fsub(d, u)));
      fstore(&samev0[index], result);
    }
  }
}

static void project_rb_step2(unsigned int n, grid_color color,
                             float *restrict sameu, float *restrict samev,
                             float *restrict neighu0) {
  unsigned int start = color == RED ? 0 : 1;
  unsigned int width = (n + 2) / 2;
  const __m256 ratio = fset1(0.5f * n);
  for (unsigned int i = 1; i <= n; ++i, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); j += 8) {
      int index = idx(j, i, width);
      __m256 oldu = fload2x4(&sameu[index]);
      __m256 oldv = fload2x4(&samev[index]);
      __m256 u = fload2x4(&neighu0[index - width]);
      __m256 r = fload2x4(&neighu0[index - start + 1]);
      __m256 d = fload2x4(&neighu0[index + width]);
      __m256 l = fload2x4(&neighu0[index - start]);
      __m256 newu = ffnmadd(ratio, fsub(r, l), oldu);
      __m256 newv = ffnmadd(ratio, fsub(d, u), oldv);
      fstore(&sameu[index], newu);
      fstore(&samev[index], newv);
    }
  }
}

static void project(unsigned int n, float *u, float *v, float *u0, float *v0) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  float *redu = u;
  float *redv = v;
  float *blku = u + color_size;
  float *blkv = v + color_size;
  float *redu0 = u0;
  float *redv0 = v0;
  float *blku0 = u0 + color_size;
  float *blkv0 = v0 + color_size;
  project_rb_step1(n, RED, redu0, redv0, blku, blkv);
  project_rb_step1(n, BLACK, blku0, blkv0, redu, redv);
  set_bnd(n, NONE, v0);
  set_bnd(n, NONE, u0);
  lin_solve(n, NONE, u0, v0, 1, 4);
  project_rb_step2(n, RED, redu, redv, blku0);
  project_rb_step2(n, BLACK, blku, blkv, redu0);
  set_bnd(n, VERTICAL, u);
  set_bnd(n, HORIZONTAL, v);
}

void step(unsigned int n, float *d, float *u, float *v, float *d0,
          float *u0, float *v0, float diff, float visc, float dt) {
  // Density update
  add_source(n, d, d0, dt);
  SWAP(d0, d);
  diffuse(n, NONE, d, d0, diff, dt);
  SWAP(d0, d);
  // density advection will be done afterwards mixed with the velocity advection

  // Velocity update
  add_source(n, u, u0, dt);
  add_source(n, v, v0, dt);
  SWAP(u0, u);
  diffuse(n, VERTICAL, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(n, HORIZONTAL, v, v0, visc, dt);
  project(n, u, v, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  advect(n, d, u, v, d0, u0, v0, dt);
  project(n, u, v, u0, v0);
}
