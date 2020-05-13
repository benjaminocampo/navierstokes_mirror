#include "indices.h"
#include "solver.h"

void add_source(unsigned int n, float *x, const float *s, float dt) {
  unsigned int size = (n + 2) * (n + 2);
  for (unsigned int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

void lin_solve_rb_step(grid_color color, unsigned int n, float a, float c,
                       const float *restrict same0, const float *restrict neigh,
                       float *restrict same) {
  int shift = color == RED ? 1 : -1;
  unsigned int start = color == RED ? 0 : 1;

  unsigned int width = (n + 2) / 2;

  for (unsigned int y = 1; y <= n; ++y, shift = -shift, start = 1 - start) {
    for (unsigned int x = start; x < width - (1 - start); ++x) {
      int index = idx(x, y, width);
      same[index] =
          (same0[index] + a * (neigh[index - width] + neigh[index] +
                               neigh[index + shift] + neigh[index + width])) /
          c;
    }
  }
}

void advect_rb(grid_color color, unsigned int n, float *samed, float *sameu,
               float *samev, const float *samed0, const float *sameu0,
               const float *samev0, const float *d0, const float *u0,
               const float *v0, float dt) {
  int i0, j0;
  float x, y, s0, t0, s1, t1;

  int shift = color == RED ? 1 : -1;
  unsigned int start = color == RED ? 0 : 1;
  unsigned int width = (n + 2) / 2;

  float dt0 = dt * n;
  for (unsigned int i = 1; i <= n; i++, shift = -shift, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); j++) {
      int index = idx(j, i, width);
      unsigned int gridi = i;
      unsigned int gridj = 2 * j + shift + start;
      x = gridj - dt0 * sameu0[index];
      y = gridi - dt0 * samev0[index];
      if (x < 0.5f) {
        x = 0.5f;
      } else if (x > n + 0.5f) {
        x = n + 0.5f;
      }
      if (y < 0.5f) {
        y = 0.5f;
      } else if (y > n + 0.5f) {
        y = n + 0.5f;
      }
      j0 = (int)x;
      i0 = (int)y;
      s1 = x - j0;
      s0 = 1 - s1;
      t1 = y - i0;
      t0 = 1 - t1;

      unsigned int i0j0 = rb_idx(j0, i0, n + 2);
      unsigned int isblack = (j0 % 2) ^ (i0 % 2);
      unsigned int isred = !isblack;
      unsigned int iseven = (i0 % 2 == 0);
      unsigned int isodd = !iseven;
      unsigned int fstart = ((isred && iseven) || (isblack && isodd));
      int fshift = isred ? 1 : -1;
      unsigned int i1j1 = i0j0 + width + (1 - fstart);
      unsigned int i0j1 = i0j0 + fshift * width * (n + 2) + (1 - fstart);
      unsigned int i1j0 = i0j0 + fshift * width * (n + 2) + width;

      samed[index] = s0 * (t0 * d0[i0j0] + t1 * d0[i1j0]) +
                     s1 * (t0 * d0[i0j1] + t1 * d0[i1j1]);
      sameu[index] = s0 * (t0 * u0[i0j0] + t1 * u0[i1j0]) +
                     s1 * (t0 * u0[i0j1] + t1 * u0[i1j1]);
      samev[index] = s0 * (t0 * v0[i0j0] + t1 * v0[i1j0]) +
                     s1 * (t0 * v0[i0j1] + t1 * v0[i1j1]);
    }
  }
}

void project_rb_step1(unsigned int n, grid_color color, float *restrict sameu0,
                      float *restrict samev0, float *restrict neighu,
                      float *restrict neighv) {
  int shift = color == RED ? 1 : -1;
  unsigned int start = color == RED ? 0 : 1;
  unsigned int width = (n + 2) / 2;
  for (unsigned int i = 1; i <= n; ++i, shift = -shift, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); ++j) {
      int index = idx(j, i, width);
      samev0[index] = -0.5f *
                      (neighu[index - start + 1] - neighu[index - start] +
                       neighv[index + width] - neighv[index - width]) /
                      n;
      sameu0[index] = 0;
    }
  }
}

void project_rb_step2(unsigned int n, grid_color color, float *restrict sameu,
                      float *restrict samev, float *restrict neighu0) {
  int shift = color == RED ? 1 : -1;
  unsigned int start = color == RED ? 0 : 1;
  unsigned int width = (n + 2) / 2;

  for (unsigned int i = 1; i <= n; ++i, shift = -shift, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); ++j) {
      int index = idx(j, i, width);
      sameu[index] -=
          0.5f * n * (neighu0[index - start + 1] - neighu0[index - start]);
      samev[index] -=
          0.5f * n * (neighu0[index + width] - neighu0[index - width]);
    }
  }
}
