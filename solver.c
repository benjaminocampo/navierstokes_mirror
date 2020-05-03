#include "solver.h"
#include <stddef.h>
#include "indices.h"

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
  for (unsigned int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
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
                      const float *d0, const float *sameu, const float *samev,
                      float dt) {
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
      x = gridj - dt0 * sameu[index];
      y = gridi - dt0 * samev[index];
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

      unsigned int i0j0 = IX(j0, i0);
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
    }
  }
}

static void advect(unsigned int n, boundary b, float *d, const float *d0,
                   const float *u, const float *v, float dt) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);

  float *redd = d;
  const float *redu = u;
  const float *redv = v;
  float *blkd = d + color_size;
  const float *blku = u + color_size;
  const float *blkv = v + color_size;
  advect_rb(RED, n, redd, d0, redu, redv, dt);
  advect_rb(BLACK, n, blkd, d0, blku, blkv, dt);
  set_bnd(n, b, d);
}

static void vel_advect_rb(grid_color color, unsigned int n,
                          float *restrict sameu, float *restrict samev,
                          const float *sameu0, const float *samev0,
                          const float *u0, const float *v0, float dt) {
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

      unsigned int i0j0 = IX(j0, i0);
      unsigned int isblack = (j0 % 2) ^ (i0 % 2);
      unsigned int isred = !isblack;
      unsigned int iseven = (i0 % 2 == 0);
      unsigned int isodd = !iseven;
      unsigned int fstart = ((isred && iseven) || (isblack && isodd));
      int fshift = isred ? 1 : -1;
      unsigned int i1j1 = i0j0 + width + (1 - fstart);
      unsigned int i0j1 = i0j0 + fshift * width * (n + 2) + (1 - fstart);
      unsigned int i1j0 = i0j0 + fshift * width * (n + 2) + width;

      sameu[index] = s0 * (t0 * u0[i0j0] + t1 * u0[i1j0]) +
                     s1 * (t0 * u0[i0j1] + t1 * u0[i1j1]);
      samev[index] = s0 * (t0 * v0[i0j0] + t1 * v0[i1j0]) +
                     s1 * (t0 * v0[i0j1] + t1 * v0[i1j1]);
    }
  }
}

static void vel_advect(unsigned int n, float *restrict u, float *restrict v,
                       const float *restrict u0, const float *restrict v0,
                       float dt) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  float *redu = u;
  float *redv = v;
  float *blku = u + color_size;
  float *blkv = v + color_size;
  const float *redu0 = u0;
  const float *redv0 = v0;
  const float *blku0 = u0 + color_size;
  const float *blkv0 = v0 + color_size;
  vel_advect_rb(RED, n, redu, redv, redu0, redv0, u0, v0, dt);
  vel_advect_rb(BLACK, n, blku, blkv, blku0, blkv0, u0, v0, dt);
  set_bnd(n, VERTICAL, u);
  set_bnd(n, HORIZONTAL, v);
}

static void project_rb_step1(unsigned int n, grid_color color,
                             float *restrict sameu0, float *restrict samev0,
                             float *restrict neighu, float *restrict neighv) {
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

static void project_rb_step2(unsigned int n, grid_color color,
                             float *restrict sameu, float *restrict samev,
                             float *restrict neighu0) {
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

void dens_step(unsigned int n, float *x, float *x0, float *u, float *v,
               float diff, float dt) {
  add_source(n, x, x0, dt);
  SWAP(x0, x);
  diffuse(n, NONE, x, x0, diff, dt);
  SWAP(x0, x);
  advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0,
              float visc, float dt) {
  add_source(n, u, u0, dt);
  add_source(n, v, v0, dt);
  SWAP(u0, u);
  diffuse(n, VERTICAL, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(n, HORIZONTAL, v, v0, visc, dt);
  project(n, u, v, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  vel_advect(n, u, v, u0, v0, dt);
  project(n, u, v, u0, v0);
}
