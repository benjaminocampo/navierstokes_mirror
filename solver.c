#include "solver.h"

#include <x86intrin.h>
#include <assert.h>

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

// gtx 1060 maxq
// static dim3 coop_block{24, 16};
// static dim3 coop_grid{5, 2};

// rtx 2080 ti
static dim3 coop_block{32, 16};
static dim3 coop_grid{17, 4};

// Checks wether the hardcoded dimensions are the best for your particular gpu
void check_coop_dims(void) {
  int best_block_size;
  int best_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&best_grid_size, &best_block_size, gpu_lin_solve, 0, 0);
  int block_size = coop_block.x * coop_block.y * coop_block.z;
  int grid_size = coop_grid.x * coop_grid.y * coop_grid.z;
  if (block_size != best_block_size || grid_size != best_grid_size) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);\
    printf("[Error] Subobtimal or invalid coop grid settings detected.\n");
    printf("GPU Detected: %s\n", deviceProp.name);
    printf("Using block_size=%d and grid_size=%d\n", block_size, grid_size);
    printf("But the optimal configuration for running gpu_lin_solve would be with block_size=%d and grid_size=%d\n", best_block_size, best_grid_size);
    printf("Change coop_block and coop_grid to match those sizes, and make them the most square-ish you can for optimal performance.\n");
    assert(false);
  }
}

static unsigned int div_round_up(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

static void diffuse(unsigned int n, boundary b, float diff, float dt,
                    float *dx, float *dx0,
                    const unsigned int from, const unsigned int to) {
  const float a = dt * diff * n * n;

  // TODO: Move {block, grid}_dims up in the call hierarchy
  size_t width = (n + 2) / 2;
  dim3 block_dim{16, 16};
  dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};
  // gpu_lin_solve<<<grid_dim, block_dim>>>(n, b, a, 1 + 4 * a, dx, dx0);

  float c = 1 + 4 * a;
  void *kernel_args[] = { (void*)&n, (void*)&b, (void*)&a, (void*)&c, (void*)&dx, (void*)&dx0 };
  checkCudaErrors(cudaLaunchCooperativeKernel((void*)gpu_lin_solve, coop_grid, coop_block, kernel_args));
}

static void advect(unsigned int n,
                   float *dd, float *du, float *dv,
                   float *dd0, float *du0, float *dv0,
                   float dt, unsigned int from, unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  float *dredd = dd;
  float *dredu = du;
  float *dredv = dv;
  float *dblkd = dd + color_size;
  float *dblku = du + color_size;
  float *dblkv = dv + color_size;
  float *dredd0 = dd0;
  float *dredu0 = du0;
  float *dredv0 = dv0;
  float *dblkd0 = dd0 + color_size;
  float *dblku0 = du0 + color_size;
  float *dblkv0 = dv0 + color_size;

  unsigned int width = (n + 2) / 2;
  const dim3 block_dim{16, 16};
  const dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};

  gpu_advect_rb<<<grid_dim, block_dim>>>(
    RED, n, dt, dredd, dredu, dredv, dredd0, dredu0, dredv0, dd0, du0, dv0
  );
  gpu_advect_rb<<<grid_dim, block_dim>>>(
    BLACK, n, dt, dblkd, dblku, dblkv, dblkd0, dblku0, dblkv0, dd0, du0, dv0
  );
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, VERTICAL, du);
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, HORIZONTAL, dv);
}

static void project(unsigned int n,
                    float *du, float *dv, float *du0, float *dv0,
                    unsigned int from, unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  float *dredu = du;
  float *dredv = dv;
  float *dblku = du + color_size;
  float *dblkv = dv + color_size;
  float *dredu0 = du0;
  float *dredv0 = dv0;
  float *dblku0 = du0 + color_size;
  float *dblkv0 = dv0 + color_size;

  size_t width = (n + 2) / 2;
  dim3 block_dim{16, 16};
  dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};

  gpu_project_rb_step1<<<grid_dim, block_dim>>>(n, RED, dredu0, dredv0, dblku, dblkv);
  gpu_project_rb_step1<<<grid_dim, block_dim>>>(n, BLACK, dblku0, dblkv0, dredu, dredv);
  // TODO: What to do with all the pragma omp barriers?
  #pragma omp barrier

  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, NONE, dv0);
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, NONE, du0);
  #pragma omp barrier

  // gpu_lin_solve<<<grid_dim, block_dim>>>(n, NONE, 1, 4, du0, dv0);
  float a = 1;
  float c = 4;
  boundary b = NONE;
  void *kernel_args[] = { (void*)&n, (void*)&b, (void*)&a, (void*)&c, (void*)&du0, (void*)&dv0 };
  checkCudaErrors(cudaLaunchCooperativeKernel((void*)gpu_lin_solve, coop_grid, coop_block, kernel_args));

  #pragma omp barrier

  gpu_project_rb_step2<<<grid_dim, block_dim>>>(n, RED, dredu, dredv, dblku0);
  gpu_project_rb_step2<<<grid_dim, block_dim>>>(n, BLACK, dblku, dblkv, dredu0);
  #pragma omp barrier

  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, VERTICAL, du);
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, HORIZONTAL, dv);
}

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          const unsigned int from, const unsigned int to) {

  // TODO: Probably this should be the only place where block and grid_dim are defined
  dim3 block_dim{16, 16};
  dim3 grid_dim{n / block_dim.x, n / block_dim.y};

  // TODO: These launches can be unsynchronized inside the device, specify that
  gpu_add_source<<<grid_dim, block_dim>>>(n, dd, dd0, dt);
  gpu_add_source<<<grid_dim, block_dim>>>(n, du, du0, dt);
  gpu_add_source<<<grid_dim, block_dim>>>(n, dv, dv0, dt);
  // TODO : Here would be a barrier of streams

  SWAP(dd0, dd);
  SWAP(du0, du);
  SWAP(dv0, dv);

  diffuse(n, NONE, diff, dt, dd, dd0, from, to);
  diffuse(n, VERTICAL, visc, dt, du, du0, from, to);
  diffuse(n, HORIZONTAL, visc, dt, dv, dv0, from, to);
  #pragma omp barrier

  project(n, du, dv, du0, dv0, from, to);
  #pragma omp barrier

  SWAP(dd0, dd);
  SWAP(du0, du);
  SWAP(dv0, dv);

  advect(n, dd, du, dv, dd0, du0, dv0, dt, from, to);
  #pragma omp barrier

  project(n, du, dv, du0, dv0, from, to);
  #pragma omp barrier
}
