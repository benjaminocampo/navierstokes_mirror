#include "solver.h"

#include <x86intrin.h>

#include "indices.h"
#include "helper_string.h"
#include "helper_cuda.h"

#include <omp.h>

#if defined INTRINSICS
#include "solver_intrinsics.h"
#elif defined ISPC
#include "solver_ispc.h"
#elif defined CUDA
#include "solver_cuda.h"
#else
#include "solver_nonvect.h"
#endif

#define IX(x, y) (rb_idx((x), (y), (n + 2)))
#define SWAP(x0, x)  \
  {                  \
    float *tmp = x0; \
    x0 = x;          \
    x = tmp;         \
  }

static void set_bnd(unsigned int n, boundary b, float *x,
                    const unsigned int from, unsigned int to) {
  for (unsigned int i = from; i < to; i++) {
    x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
  }

  if (from == 1) {
    for (unsigned int i = 1; i < n + 1; i++)
      x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
  }

  if (to == n + 1) {
    for (unsigned int i = 1; i < n + 1; i++)
      x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
  }
}

static void lin_solve(unsigned int n, boundary b, float *__restrict__ x,
                      const float *__restrict__ x0, const float a, const float c,
                      const unsigned int from, const unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  const float *red0 = x0;
  const float *blk0 = x0 + color_size;
  float *red = x;
  float *blk = x + color_size;

  for (unsigned int k = 0; k < 20; ++k) {
      lin_solve_rb_step(RED, n, a, c, red0, blk, red, from, to);
      lin_solve_rb_step(BLACK, n, a, c, blk0, red, blk, from, to);
      #pragma omp barrier
      set_bnd(n, b, x, from, to);
  }
}

static void diffuse(unsigned int n, boundary b, float *x, const float *x0,
                    float diff, float dt, const unsigned int from,
                    const unsigned int to) {
  float a = dt * diff * n * n;
  lin_solve(n, b, x, x0, a, 1 + 4 * a, from, to);
}

static void advect(unsigned int n, float *d, float *u, float *v,
                   const float *d0, const float *u0, const float *v0, float dt,
                   const unsigned int from, const unsigned int to) {
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
  advect_rb(RED, n, redd, redu, redv, redd0, redu0, redv0, d0, u0, v0, dt, from, to);
  advect_rb(BLACK, n, blkd, blku, blkv, blkd0, blku0, blkv0, d0, u0, v0, dt, from, to);
  #pragma omp barrier
  set_bnd(n, VERTICAL, u, from, to);
  set_bnd(n, HORIZONTAL, v, from, to);
}

static void project(unsigned int n, float *u, float *v, float *u0, float *v0,
                    unsigned int from, unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  float *redu = u;
  float *redv = v;
  float *blku = u + color_size;
  float *blkv = v + color_size;
  float *redu0 = u0;
  float *redv0 = v0;
  float *blku0 = u0 + color_size;
  float *blkv0 = v0 + color_size;

  project_rb_step1(n, RED, redu0, redv0, blku, blkv, from, to);
  project_rb_step1(n, BLACK, blku0, blkv0, redu, redv, from, to);
  #pragma omp barrier

  set_bnd(n, NONE, v0, from, to);
  set_bnd(n, NONE, u0, from, to);
  #pragma omp barrier

  lin_solve(n, NONE, u0, v0, 1, 4, from, to);
  #pragma omp barrier

  project_rb_step2(n, RED, redu, redv, blku0, from, to);
  project_rb_step2(n, BLACK, blku, blkv, redu0, from, to);
  #pragma omp barrier

  set_bnd(n, VERTICAL, u, from, to);
  set_bnd(n, HORIZONTAL, v, from, to);
}

void step(unsigned int n, float diff, float visc, float dt,
          float *hd, float *hu, float *hv, float *hd0, float *hu0, float *hv0,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          const unsigned int from, const unsigned int to) {

  //const dim3 block_dim{16, 16};
  //const dim3 grid_dim{n / block_dim.x, n / block_dim.y};

  // TODO: These launches can be unsynchronized inside the device, specify that
  // gpu_add_source<<<grid_dim, block_dim>>>(n, dd, dd0, dt);
  // gpu_add_source<<<grid_dim, block_dim>>>(n, du, dd0, dt);
  // gpu_add_source<<<grid_dim, block_dim>>>(n, dv, dd0, dt);

  //checkCudaErrors(cudaDeviceSynchronize());


  // Old openmp version
  add_source(n, hd, hd0, dt, from, to);
  add_source(n, hu, hu0, dt, from, to);
  add_source(n, hv, hv0, dt, from, to);
  #pragma omp barrier

  SWAP(hd0, hd);
  SWAP(hu0, hu);
  SWAP(hv0, hv);
  diffuse(n, NONE, hd, hd0, diff, dt, from, to);
  diffuse(n, VERTICAL, hu, hu0, visc, dt, from, to);
  diffuse(n, HORIZONTAL, hv, hv0, visc, dt, from, to);
  #pragma omp barrier

  project(n, hu, hv, hu0, hv0, from, to);
  #pragma omp barrier

  SWAP(hd0, hd);
  SWAP(hu0, hu);
  SWAP(hv0, hv);
  advect(n, hd, hu, hv, hd0, hu0, hv0, dt, from, to);
  #pragma omp barrier

  project(n, hu, hv, hu0, hv0, from, to);
  #pragma omp barrier
}
