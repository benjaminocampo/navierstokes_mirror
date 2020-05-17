#include <x86intrin.h>

#include "solver.h"
#include "indices.h"
#include "intrinsics_helpers.h"

void add_source(unsigned int n, float *x, const float *s, float dt,
                const unsigned int from, const unsigned int to) {
  const __m256 pdt = fset1(dt);
  unsigned int i;
  const unsigned int strip_border = idx(0, to, n + 2);
  for (i = idx(0, from, n + 2); i < strip_border - 8; i += 8) {
    __m256 px = fload(&x[i]);
    __m256 ps = fload(&s[i]);
    __m256 product = ffmadd(pdt, ps, px);  // x + dt * s[i]
    fstore(&x[i], product);                // x[i] += dt * s[i];
  }
  for (; i < strip_border; i++) x[i] += dt * s[i];
}

void lin_solve_rb_step(grid_color color, unsigned int n, float a, float c,
                       const float *restrict same0, const float *restrict neigh,
                       float *restrict same, const unsigned int from,
                       const unsigned int to) {
  const float invc = 1 / c;
  unsigned int start = ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  unsigned int width = (n + 2) / 2;
  const __m256 pinvc = fset1(invc);
  const __m256 pa = fset1(a);
  for (unsigned int y = from; y < to; ++y, start = 1 - start) {
    for (unsigned int x = start; x < width - (1 - start); x += 8) {
      int index = idx(x, y, width);
      __m256 f = fload(&same0[index]);
      __m256 u = fload(&neigh[index - width]);
      __m256 r = fload(&neigh[index - start + 1]);
      __m256 d = fload(&neigh[index + width]);
      __m256 l = fload(&neigh[index - start]);

      // t = (f + a * (u + r + d + l)) / c
      __m256 t = fmul(ffmadd(pa, fadd(u, fadd(r, fadd(d, l))), f), pinvc);
      fstore(&same[index], t);
    }
  }
}

void advect_rb(grid_color color, unsigned int n, float *samed, float *sameu,
               float *samev, const float *samed0, const float *sameu0,
               const float *samev0, const float *d0, const float *u0,
               const float *v0, float dt, const unsigned int from,
               const unsigned int to) {
  int start = ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  int shift = 1 - start * 2;
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
  for (int iy = from; iy < (int)to; iy++, shift = -shift, start = 1 - start) {
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

void project_rb_step1(unsigned int n, grid_color color, float *restrict sameu0,
                      float *restrict samev0, float *restrict neighu,
                      float *restrict neighv, const unsigned int from,
                      const unsigned int to) {
  unsigned int start = ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  unsigned int width = (n + 2) / 2;
  const __m256 zeros = fset1(0.0f);
  const __m256 ratio = fset1(-0.5f / n);
  for (unsigned int i = from; i < to; ++i, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); j += 8) {
      int index = idx(j, i, width);
      __m256 u = fload(&neighv[index - width]);
      __m256 r = fload(&neighu[index - start + 1]);
      __m256 d = fload(&neighv[index + width]);
      __m256 l = fload(&neighu[index - start]);
      __m256 result = fmul(ratio, fadd(fsub(r, l), fsub(d, u)));
      fstore(&samev0[index], result);
      fstore(&sameu0[idx(j, i, width)], zeros);
    }
  }
}

void project_rb_step2(unsigned int n, grid_color color, float *restrict sameu,
                      float *restrict samev, float *restrict neighu0,
                      const unsigned int from, const unsigned int to) {
  unsigned int start = ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  unsigned int width = (n + 2) / 2;
  const __m256 ratio = fset1(0.5f * n);
  for (unsigned int i = from; i < to; ++i, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); j += 8) {
      int index = idx(j, i, width);
      __m256 oldu = fload(&sameu[index]);
      __m256 oldv = fload(&samev[index]);
      __m256 u = fload(&neighu0[index - width]);
      __m256 r = fload(&neighu0[index - start + 1]);
      __m256 d = fload(&neighu0[index + width]);
      __m256 l = fload(&neighu0[index - start]);
      __m256 newu = ffnmadd(ratio, fsub(r, l), oldu);
      __m256 newv = ffnmadd(ratio, fsub(d, u), oldv);
      fstore(&sameu[index], newu);
      fstore(&samev[index], newv);
    }
  }
}
