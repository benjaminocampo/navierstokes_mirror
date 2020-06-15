#include "indices.h"
#include "solver.h"
#include "helper_cuda.h"

#define IX(x, y) (rb_idx((x), (y), (n + 2)))

__global__
void gpu_add_source(unsigned int n, float *dst, const float *src, float dt) {
  const int grid_width = gridDim.x * blockDim.x;
  const int grid_height = gridDim.y * blockDim.y;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  // TODO: Change the boundary of the loop. It should go up to n + 2
  // Another option: Consider the range [1, n] x [1, n]
  for (int y = gtidy; y < n; y += grid_height) {
    for (int x = gtidx; x < n; x += grid_width) {
      int index = y * n + x;
      dst[index] += dt * src[index];
    }
  }
}

__global__
void gpu_set_bnd(unsigned int n, boundary b, float *x) {
  const int grid_width = gridDim.x * blockDim.x;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = 1 + gtid; i <= n; i += grid_width) {
    x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
    x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
  }
  if(gtid == 0) {
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = -0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
  }
}

// Pre: blockDim.y is even, if not, start needs to shift between row increments.
// TODO: Move preconditions "Pre:" to asserts
__global__
void gpu_lin_solve_rb_step(grid_color color, unsigned int n, float a, float c,
                           const float *__restrict__ same0,
                           const float *__restrict__ neigh,
                           float *__restrict__ same) {
  unsigned int width = (n + 2) / 2;
  unsigned int start = (
    (color == RED && ((threadIdx.y + 1) % 2 == 0)) ||
    (color == BLACK && ((threadIdx.y + 1) % 2 == 1))
  );
  const int grid_width = gridDim.x * blockDim.x;
  const int grid_height = gridDim.y * blockDim.y;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  for (int y = 1 + gtidy; y <= n; y += grid_height) {
    for (int x = start + gtidx; x < width - (1 - start); x += grid_width) {
      int index = y * width + x;
      same[index] = (same0[index] + a * (
          neigh[index - width] +
          neigh[index - start] +
          neigh[index - start + 1] +
          neigh[index + width]
      )) / c;
    }
  }
}

// Pre: blockDim.y must be even, `start` calculation will not be recomputed.
__global__
void gpu_advect_rb(grid_color color, unsigned int n, float dt,
                   float *samed, float *sameu, float *samev,
                   const float *samed0, const float *sameu0, const float *samev0,
                   const float *d0, const float *u0, const float *v0) {
  const int grid_width = gridDim.x * blockDim.x;
  const int grid_height = gridDim.y * blockDim.y;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;

  int i0, j0;
  float xx, yy, s0, t0, s1, t1;

  unsigned int width = (n + 2) / 2;
  unsigned int start = (
    (color == RED && ((threadIdx.y + 1) % 2 == 0)) ||
    (color == BLACK && ((threadIdx.y + 1) % 2 == 1))
  );
  float dt0 = dt * n;

  for (int i = 1 + gtidy; i <= n; i += grid_height) {
    for (int j = start + gtidx; j < width - (1 - start); j += grid_width) {
      int index = idx(j, i, width);
      unsigned int fluidgridi = i;
      unsigned int fluidgridj = 2 * j + 1 - start;
      xx = fluidgridj - dt0 * sameu0[index];
      yy = fluidgridi - dt0 * samev0[index];
      if (xx < 0.5f) {
        xx = 0.5f;
      } else if (xx > n + 0.5f) {
        xx = n + 0.5f;
      }
      if (yy < 0.5f) {
        yy = 0.5f;
      } else if (yy > n + 0.5f) {
        yy = n + 0.5f;
      }
      j0 = (int)xx;
      i0 = (int)yy;
      s1 = xx - j0;
      s0 = 1 - s1;
      t1 = yy - i0;
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

__global__
void gpu_project_rb_step1(unsigned int n, grid_color color,
                          float *__restrict__ sameu0, float *__restrict__ samev0,
                          float *__restrict__ neighu, float *__restrict__ neighv) {
  // TODO: Check __restrict__ on parameters, the nonvect versions have them
  const int grid_width = gridDim.x * blockDim.x;
  const int grid_height = gridDim.y * blockDim.y;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int width = (n + 2) / 2;
  unsigned int start = (
    (color == RED && ((threadIdx.y + 1) % 2 == 0)) ||
    (color == BLACK && ((threadIdx.y + 1) % 2 == 1))
  );

  for (int i = 1 + gtidy; i <= n; i += grid_height) {
    for (int j = start + gtidx; j < width - (1 - start); j += grid_width) {
      int index = idx(j, i, width);
      samev0[index] = -0.5f *
                      (neighu[index - start + 1] - neighu[index - start] +
                       neighv[index + width] - neighv[index - width]) /
                      n;
      sameu0[index] = 0;
    }
  }
}

__global__
void gpu_project_rb_step2(unsigned int n, grid_color color,
                          float *sameu, float *samev, float *neighu0) {
  // TODO: Check __restrict__ on parameters, the nonvect versions have them
  const int grid_width = gridDim.x * blockDim.x;
  const int grid_height = gridDim.y * blockDim.y;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int width = (n + 2) / 2;
  unsigned int start = ((color == RED && ((threadIdx.y + 1) % 2 == 0)) ||
                        (color == BLACK && ((threadIdx.y + 1) % 2 == 1)));

  for (int i = 1 + gtidy; i <= n; i += grid_height) {
    for (int j = start + gtidx; j < width - (1 - start); j += grid_width) {
      int index = idx(j, i, width);
      sameu[index] -=
          0.5f * n * (neighu0[index - start + 1] - neighu0[index - start]);
      samev[index] -=
          0.5f * n * (neighu0[index + width] - neighu0[index - width]);
    }
  }
}
