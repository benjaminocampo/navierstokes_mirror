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


static void lin_solve(unsigned int n, boundary b, float a, float c,
                      float *__restrict__ dx, float *__restrict__ dx0,
                      cudaStream_t stream0, cudaStream_t stream1,
                      cudaEvent_t spread, cudaEvent_t join_stream0,
                      cudaEvent_t join_stream1) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  // cudaMemcpy does not allow const pointers in dst.
  float *dred0 = dx0;
  float *dblk0 = dx0 + color_size;
  float *dred = dx;
  float *dblk = dx + color_size;
  unsigned int width = (n + 2) / 2;
  const dim3 block_dim{16, 16};
  const dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};

  for(unsigned int k = 0; k < 20; ++k){
    gpu_lin_solve_rb_step<<<grid_dim, block_dim, 0, stream0>>>(
      RED, n, a, c, dred0, dblk, dred
    );
    gpu_lin_solve_rb_step<<<grid_dim, block_dim, 0, stream1>>>(
      BLACK, n, a, c, dblk0, dred, dblk
    );
    cudaEventRecord(join_stream1, stream1);
    cudaStreamWaitEvent(stream0, join_stream1, 0);
    gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, stream0>>>(
      n, b, dx
    );
    cudaEventRecord(spread, stream0);
    cudaStreamWaitEvent(stream1, spread, 0);
  }
}

static void diffuse(unsigned int n, boundary b, float diff, float dt,
                    float *dx, float *dx0, cudaStream_t stream, cudaEvent_t spread,
                    cudaEvent_t join_stream0, cudaEvent_t join_stream1) {
  const float a = dt * diff * n * n;
  lin_solve(n, b, a, 1 + 4 * a, dx, dx0, stream, stream, spread, join_stream0, join_stream1);
}



static void advect(unsigned int n,
                   float *dd, float *du, float *dv,
                   float *dd0, float *du0, float *dv0,
                   float dt, cudaStream_t stream0, cudaStream_t stream1,
                   cudaEvent_t spread, cudaEvent_t join_stream0,
                   cudaEvent_t join_stream1) {
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

  cudaEventRecord(spread, stream0);
  cudaStreamWaitEvent(stream1, spread, 0);

  gpu_advect_rb<<<grid_dim, block_dim, 0, stream0>>>(
    RED, n, dt, dredd, dredu, dredv, dredd0, dredu0, dredv0, dd0, du0, dv0
  );
  gpu_advect_rb<<<grid_dim, block_dim, 0, stream1>>>(
    BLACK, n, dt, dblkd, dblku, dblkv, dblkd0, dblku0, dblkv0, dd0, du0, dv0
  );

  cudaEventRecord(join_stream1, stream1);
  cudaStreamWaitEvent(stream0, join_stream1, 0);

  cudaEventRecord(spread, stream0);
  cudaStreamWaitEvent(stream1, spread, 0);

  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, stream0>>>(
    n, VERTICAL, du
  );
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, stream1>>>(
    n, HORIZONTAL, dv
  );

  cudaEventRecord(join_stream1, stream1);
  cudaStreamWaitEvent(stream0, join_stream1, 0);
}

static void project(unsigned int n,
                    float *du, float *dv, float *du0, float *dv0,
                    cudaStream_t stream0, cudaStream_t stream1,
                    cudaEvent_t spread, cudaEvent_t join_stream0,
                    cudaEvent_t join_stream1) {
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
  int size_in_mem = (n + 2) * (n + 2) * sizeof(float);

  gpu_project_rb_step1<<<grid_dim, block_dim, 0, stream0>>>(
    n, RED, dredv0, dblku, dblkv
  );
  gpu_project_rb_step1<<<grid_dim, block_dim, 0, stream0>>>(
    n, BLACK, dblkv0, dredu, dredv
  );
  checkCudaErrors(cudaMemsetAsync(du0, 0, size_in_mem, stream1));

  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, stream0>>>(
    n, NONE, dv0
  );

  cudaEventRecord(join_stream1, stream1);
  cudaStreamWaitEvent(stream0, join_stream1, 0);

  lin_solve(n, NONE, 1, 4, du0, dv0, stream0, stream1, spread, join_stream0, join_stream1);

  cudaEventRecord(spread, stream0);
  cudaStreamWaitEvent(stream1, spread, 0);

  gpu_project_rb_step2<<<grid_dim, block_dim, 0, stream0>>>(
    n, RED, dredu, dredv, dblku0
  );
  gpu_project_rb_step2<<<grid_dim, block_dim, 0, stream1>>>(
    n, BLACK, dblku, dblkv, dredu0
  );

  cudaEventRecord(join_stream1, stream1);
  cudaStreamWaitEvent(stream0, join_stream1, 0);

  cudaEventRecord(spread, stream0);
  cudaStreamWaitEvent(stream1, spread, 0);

  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, stream0>>>(
    n, VERTICAL, du
  );
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, stream1>>>(
    n, HORIZONTAL, dv
  );

  cudaEventRecord(join_stream1, stream1);
  cudaStreamWaitEvent(stream0, join_stream1, 0);
}

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          cudaStream_t stream0, cudaStream_t stream1, cudaStream_t stream2,
          cudaEvent_t spread, cudaEvent_t join_stream0, cudaEvent_t join_stream1,
          cudaEvent_t join_stream2) {

  // TODO: These launches can be unsynchronized inside the device, specify that  

  dim3 block_dim{16, 16};
  dim3 grid_dim{n / block_dim.x, n / block_dim.y};

  cudaEventRecord(spread, stream0);
  cudaStreamWaitEvent(stream1, spread, 0);
  cudaStreamWaitEvent(stream2, spread, 0);

  gpu_add_source<<<grid_dim, block_dim, 0, stream0>>>(n, dv, dv0, dt);
  SWAP(dv0, dv);
  diffuse(n, HORIZONTAL, visc, dt, dv, dv0, stream0, spread, join_stream0, join_stream0);
  
  gpu_add_source<<<grid_dim, block_dim, 0, stream1>>>(n, du, du0, dt);
  SWAP(du0, du);
  diffuse(n, VERTICAL, visc, dt, du, du0, stream1, spread, join_stream0, join_stream1);

  gpu_add_source<<<grid_dim, block_dim, 0, stream2>>>(n, dd, dd0, dt);
  SWAP(dd0, dd);
  diffuse(n, NONE, diff, dt, dd, dd0, stream2, spread, join_stream0, join_stream2);

  project(
    n, du, dv, du0, dv0, stream0, stream1,
    spread, join_stream0, join_stream1
  );

  SWAP(dd0, dd);
  SWAP(du0, du);
  SWAP(dv0, dv);

  cudaEventRecord(join_stream2, stream2);
  cudaStreamWaitEvent(stream0, join_stream2, 0);

  advect(
    n, dd, du, dv, dd0, du0, dv0, dt,
    stream0, stream1, spread, join_stream0, join_stream1
  );
  #pragma omp barrier

  cudaEventRecord(spread, stream0);
  cudaStreamWaitEvent(stream1, spread, 0);

  project(
    n, du, dv, du0, dv0, stream0, stream1,
    spread, join_stream0, join_stream1
  );
  #pragma omp barrier
}
