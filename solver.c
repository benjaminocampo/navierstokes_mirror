#include "solver.h"

#include <x86intrin.h>

#include "indices.h"
#include "helper_string.h"
#include "helper_cuda.h"

#include <omp.h>
#include <assert.h>

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

// GTX 1060 MaxQ
static dim3 gpu_add_source_grid{4, 5};
static dim3 gpu_add_source_block{32, 32};

static dim3 gpu_set_bnd_grid{4, 5};
static dim3 gpu_set_bnd_block{32, 32};

static dim3 gpu_lin_solve_rb_step_grid{4, 5};
static dim3 gpu_lin_solve_rb_step_block{32, 32};

static dim3 gpu_advect_rb_grid{4, 5};
static dim3 gpu_advect_rb_block={32, 24};

static dim3 gpu_project_rb_step1_grid{4, 5};
static dim3 gpu_project_rb_step1_block{32, 32};

static dim3 gpu_project_rb_step2_grid{4, 5};
static dim3 gpu_project_rb_step2_block{32, 32};

// RTX 2080 ti
// TODO

template<class T, typename S>
static bool matches_best_dims(S name, dim3 grid, dim3 block, T kernel) {
  int best_block_size;
  int best_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&best_grid_size, &best_block_size, kernel, 0, 0);
  int block_size = block.x * block.y * block.z;
  int grid_size = grid.x * grid.y * grid.z;
  bool valid_config = block_size == best_block_size && grid_size == best_grid_size;
  if (!valid_config) printf(
    "[%s] Using (grid, block) == (%d, %d). Use (%d, %d) instead\n",
    name, block_size, grid_size, best_block_size, best_grid_size
  );
  return valid_config;
}

// Checks wether the hardcoded dimensions are the best for your particular gpu
void check_dims(void) {
  bool all = true;
  all &= matches_best_dims("gpu_add_source", gpu_add_source_grid, gpu_add_source_block, gpu_add_source);
  all &= matches_best_dims("gpu_set_bnd", gpu_set_bnd_grid, gpu_set_bnd_block, gpu_set_bnd);
  all &= matches_best_dims("gpu_lin_solve_rb_step", gpu_lin_solve_rb_step_grid, gpu_lin_solve_rb_step_block, gpu_lin_solve_rb_step);
  all &= matches_best_dims("gpu_advect_rb", gpu_advect_rb_grid, gpu_advect_rb_block, gpu_advect_rb);
  all &= matches_best_dims("gpu_project_rb_step1", gpu_project_rb_step1_grid, gpu_project_rb_step1_block, gpu_project_rb_step1);
  all &= matches_best_dims("gpu_project_rb_step2", gpu_project_rb_step2_grid, gpu_project_rb_step2_block, gpu_project_rb_step2);
  if (!all) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);\
    printf("[Error] Subobtimal or invalid coop grid settings detected.\n");
    printf("GPU Detected: %s\n", deviceProp.name);
    printf("GPU Compute Capability: %d\n", deviceProp.major);
    printf("Change block and grid for those kernel to match those sizes.\n");
    printf("Make them as square-ish as possible and keep powers of two for widths.\n");
    assert(false);
  }
}

static void lin_solve(unsigned int n, boundary b, const float a, const float c,
                      float *__restrict__ dx, float *__restrict__ dx0,
                      const unsigned int from, const unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  // cudaMemcpy does not allow const pointers in dst.
  float *dred0 = dx0;
  float *dblk0 = dx0 + color_size;
  float *dred = dx;
  float *dblk = dx + color_size;

  // TODO: Move up block_dim and grid_dim
  unsigned int width = (n + 2) / 2;
  const dim3 block_dim{16, 16};
  const dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};
  for (unsigned int k = 0; k < 20; ++k) {
    gpu_lin_solve_rb_step<<<gpu_lin_solve_rb_step_grid, gpu_lin_solve_rb_step_block>>>(RED, n, a, c, dred0, dblk, dred);
    gpu_lin_solve_rb_step<<<gpu_lin_solve_rb_step_grid, gpu_lin_solve_rb_step_block>>>(BLACK, n, a, c, dblk0, dred, dblk);
    gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, b, dx);
  }
}

static void diffuse(unsigned int n, boundary b, float diff, float dt,
                    float *dx, float *dx0,
                    const unsigned int from, const unsigned int to) {
  const float a = dt * diff * n * n;
  lin_solve(n, b, a, 1 + 4 * a, dx, dx0, from, to);
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

  gpu_advect_rb<<<gpu_advect_rb_grid, gpu_advect_rb_block>>>(
    RED, n, dt, dredd, dredu, dredv, dredd0, dredu0, dredv0, dd0, du0, dv0
  );
  gpu_advect_rb<<<gpu_advect_rb_grid, gpu_advect_rb_block>>>(
    BLACK, n, dt, dblkd, dblku, dblkv, dblkd0, dblku0, dblkv0, dd0, du0, dv0
  );
  gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, VERTICAL, du);
  gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, HORIZONTAL, dv);
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

  gpu_project_rb_step1<<<gpu_project_rb_step1_grid, gpu_project_rb_step1_block>>>(n, RED, dredu0, dredv0, dblku, dblkv);
  gpu_project_rb_step1<<<gpu_project_rb_step1_grid, gpu_project_rb_step1_block>>>(n, BLACK, dblku0, dblkv0, dredu, dredv);
  // TODO: What to do with all the pragma omp barriers?
  #pragma omp barrier

  gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, NONE, dv0);
  gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, NONE, du0);
  #pragma omp barrier

  lin_solve(n, NONE, 1, 4, du0, dv0, from, to);
  #pragma omp barrier

  gpu_project_rb_step2<<<gpu_project_rb_step2_grid, gpu_project_rb_step2_block>>>(n, RED, dredu, dredv, dblku0);
  gpu_project_rb_step2<<<gpu_project_rb_step2_grid, gpu_project_rb_step2_block>>>(n, BLACK, dblku, dblkv, dredu0);
  #pragma omp barrier

  gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, VERTICAL, du);
  gpu_set_bnd<<<gpu_set_bnd_grid, gpu_set_bnd_block>>>(n, HORIZONTAL, dv);
}

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          const unsigned int from, const unsigned int to) {

  // TODO: Probably this should be the only place where block and grid_dim are defined
  dim3 block_dim{16, 16};
  dim3 grid_dim{n / block_dim.x, n / block_dim.y};

  // TODO: These launches can be unsynchronized inside the device, specify that
  gpu_add_source<<<gpu_add_source_grid, gpu_add_source_block>>>(n, dd, dd0, dt);
  gpu_add_source<<<gpu_add_source_grid, gpu_add_source_block>>>(n, du, du0, dt);
  gpu_add_source<<<gpu_add_source_grid, gpu_add_source_block>>>(n, dv, dv0, dt);
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
