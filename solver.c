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

static unsigned int div_round_up(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

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

static void lin_solve(unsigned int n, boundary b, const float a, const float c,
                      float *__restrict__ hx, float *__restrict__ hx0,
                      float *__restrict__ dx, float *__restrict__ dx0,
                      const unsigned int from, const unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  // cudaMemcpy does not allow const pointers in dst.
  float *hred0 = hx0;
  float *hblk0 = hx0 + color_size;
  float *hred = hx;
  float *hblk = hx + color_size;

  float *dred0 = dx0;
  float *dblk0 = dx0 + color_size;
  float *dred = dx;
  float *dblk = dx + color_size;

  unsigned int width = (n + 2) / 2;

  const dim3 block_dim{16, 16};
  // The grid size is according to rb in order to cover the range [1, n]x[0, width)

  const dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};
  size_t size = (n + 2) * width * sizeof(float);

  for (unsigned int k = 0; k < 20; ++k) {
    checkCudaErrors(cudaMemcpy(dred0, hred0, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dblk, hblk, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dred, hred, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dblk0, hblk0, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dred, hred, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dblk, hblk, size, cudaMemcpyHostToDevice));
    gpu_lin_solve_rb_step<<<grid_dim, block_dim>>>(RED, n, a, c, dred0, dblk, dred);
    gpu_lin_solve_rb_step<<<grid_dim, block_dim>>>(BLACK, n, a, c, dblk0, dred, dblk);
    checkCudaErrors(cudaMemcpy(hred0, dred0,  size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hblk, dblk, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hred, dred, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hblk0, dblk0,  size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hred, dred, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hblk, dblk, size, cudaMemcpyDeviceToHost));
    // old openmp version
    // lin_solve_rb_step(RED, n, a, c, hred0, hblk, hred, from, to);
    // lin_solve_rb_step(BLACK, n, a, c, hblk0, hred, hblk, from, to);
    // #pragma omp barrier
    set_bnd(n, b, hx, from, to);
  }
}

static void diffuse(unsigned int n, boundary b, float diff, float dt,
                    float *hx, float *hx0,
                    float *dx, float *dx0,
                    const unsigned int from, const unsigned int to) {
  const float a = dt * diff * n * n;
  lin_solve(n, b, a, 1 + 4 * a, hx, hx0, dx, dx0, from, to);
}

// lin_solve version of project. TODO: Change signature inside of project to use
// the other linsolve.
static void project_lin_solve(unsigned int n, boundary b,
                      float *__restrict__ x, const float *__restrict__ x0,
                      const float a, const float c,
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

  project_lin_solve(n, NONE, u0, v0, 1, 4, from, to);
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

  const dim3 block_dim{1, 1};
  const dim3 grid_dim{1, 1};

  // TODO: These launches can be unsynchronized inside the device, specify that

  size_t size = (n + 2) * (n + 2) * sizeof(float);
  checkCudaErrors(cudaMemcpy(dd, hd, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(du, hu, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv, hv, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dd0, hd0, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(du0, hu0, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv0, hv0, size, cudaMemcpyHostToDevice));
  gpu_add_source<<<grid_dim, block_dim>>>(n, dd, dd0, dt);
  gpu_add_source<<<grid_dim, block_dim>>>(n, du, du0, dt);
  gpu_add_source<<<grid_dim, block_dim>>>(n, dv, dv0, dt);
  checkCudaErrors(cudaMemcpy(hd, dd, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hu, du, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hv, dv, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hd0, dd0, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hu0, du0, size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hv0, dv0, size, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());

  // Old openmp version
  // add_source(n, hd, hd0, dt, from, to);
  // add_source(n, hu, hu0, dt, from, to);
  // add_source(n, hv, hv0, dt, from, to);
  // #pragma omp barrier

  SWAP(hd0, hd);
  SWAP(hu0, hu);
  SWAP(hv0, hv);

  diffuse(n, NONE, diff, dt, hd, hd0, dd, dd0, from, to);
  diffuse(n, VERTICAL, visc, dt, hu, hu0, du, du0, from, to);
  diffuse(n, HORIZONTAL, visc, dt, hv, hv0, dv, dv0, from, to);

  // Old openmp version
  //diffuse(n, NONE, hd, hd0, diff, dt, from, to);
  //diffuse(n, VERTICAL, hu, hu0, visc, dt, from, to);
  //diffuse(n, HORIZONTAL, hv, hv0, visc, dt, from, to);
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
