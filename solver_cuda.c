#include "indices.h"
#include "solver.h"
#include "helper_cuda.h"
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

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

// TODO: Make an add_source version using grid stride loops
// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

void add_source(unsigned int n, float *x, const float *s, float dt,
                const unsigned int from, const unsigned int to) {
  for (unsigned int i = idx(0, from, n + 2); i < idx(0, to, n + 2); i++) {
    x[i] += dt * s[i];
  }
}

__global__
void gpu_lin_solve_rb_step(grid_color color, unsigned int n, float a, float c,
                           const float *__restrict__ same0,
                           const float *__restrict__ neigh,
                           float *__restrict__ same) {
  // unsigned int start = color == RED ? 0 : 1;
  // int shift = color == RED ? 1 : -1;
  // TODO: Clean start/shift code in all its usages
  unsigned int width = (n + 2) / 2;

  unsigned int start = ((color == RED && (threadIdx.y % 2 == 0)) ||
                        (color == BLACK && (threadIdx.y % 2 == 1)));
  int shift = 1 - start * 2;

  const int grid_tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_tidy = blockIdx.y * blockDim.y + threadIdx.y;

  int x = start + grid_tidx;  // [start, width - (1 - start))
  int y = 1 + grid_tidy;      // [1, n]

  if(x < width - (1 - start)){
    int index = y * width + x;
    same[index] =
          (same0[index] + a * (neigh[index - width] + neigh[index] +
                               neigh[index + shift] + neigh[index + width])) /
          c;
  }
}

void lin_solve_rb_step(grid_color color, unsigned int n, float a, float c,
                       const float *__restrict__ same0,
                       const float *__restrict__ neigh,
                       float *__restrict__ same, const unsigned int from,
                       const unsigned int to) {
  // unsigned int start = color == RED ? 0 : 1;
  // int shift = color == RED ? 1 : -1;
  // TODO: Clean start/shift code in all its usages
  unsigned int start = ((color == RED && (from % 2 == 0)) ||
                        (color == BLACK && (from % 2 == 1)));
  int shift = 1 - start * 2;

  unsigned int width = (n + 2) / 2;

  for (unsigned int y = from; y < to; ++y, shift = -shift, start = 1 - start) {
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
               const float *v0, float dt, const unsigned int from,
               const unsigned int to) {
  int i0, j0;
  float x, y, s0, t0, s1, t1;

  // TODO: Remove unused comments
  // int shift = color == RED ? 1 : -1;
  // unsigned int start = color == RED ? 0 : 1;
  unsigned int start =
      ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  int shift = 1 - start * 2;

  unsigned int width = (n + 2) / 2;

  float dt0 = dt * n;
  for (unsigned int i = from; i < to; i++, shift = -shift, start = 1 - start) {
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

void project_rb_step1(unsigned int n, grid_color color,
                      float *__restrict__ sameu0, float *__restrict__ samev0,
                      float *__restrict__ neighu, float *__restrict__ neighv,
                      const unsigned int from, const unsigned int to) {
  // int shift = color == RED ? 1 : -1;
  // unsigned int start = color == RED ? 0 : 1;
  unsigned int start =
      ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  int shift = 1 - start * 2;

  unsigned int width = (n + 2) / 2;
  for (unsigned int i = from; i < to; ++i, shift = -shift, start = 1 - start) {
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

void project_rb_step2(unsigned int n, grid_color color,
                      float *__restrict__ sameu, float *__restrict__ samev,
                      float *__restrict__ neighu0, const unsigned int from,
                      const unsigned int to) {
  // int shift = color == RED ? 1 : -1;
  // unsigned int start = color == RED ? 0 : 1;
  unsigned int start =
      ((color == RED && (from % 2 == 0)) || (color != RED && (from % 2 == 1)));
  int shift = 1 - start * 2;

  unsigned int width = (n + 2) / 2;
  for (unsigned int i = from; i < to; ++i, shift = -shift, start = 1 - start) {
    for (unsigned int j = start; j < width - (1 - start); ++j) {
      int index = idx(j, i, width);
      sameu[index] -=
          0.5f * n * (neighu0[index - start + 1] - neighu0[index - start]);
      samev[index] -=
          0.5f * n * (neighu0[index + width] - neighu0[index - width]);
    }
  }
}
