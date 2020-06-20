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

void create_graph_addsource3(cudaGraphExec_t *graph_exec,
                             unsigned int n, float dt,
                             float *dd, float *dd0,
                             float *du, float *du0,
                             float *dv, float *dv0){
  dim3 block_dim{16, 16};
  dim3 grid_dim{n / block_dim.x, n / block_dim.y};
  cudaGraph_t graph;
  cudaEvent_t spread, join_du, join_dv;
  cudaStream_t stream_dd, stream_du, stream_dv;
  checkCudaErrors(cudaEventCreate(&spread));
  checkCudaErrors(cudaEventCreate(&join_du));
  checkCudaErrors(cudaEventCreate(&join_dv));
  checkCudaErrors(cudaStreamCreate(&stream_dd));
  checkCudaErrors(cudaStreamCreate(&stream_du));
  checkCudaErrors(cudaStreamCreate(&stream_dv));
  checkCudaErrors(cudaStreamBeginCapture(stream_dd, cudaStreamCaptureModeGlobal));    
  
  cudaEventRecord(spread, stream_dd);
  cudaStreamWaitEvent(stream_du, spread, 0);
  cudaStreamWaitEvent(stream_dv, spread, 0);
  gpu_add_source<<<grid_dim, block_dim, 0, stream_dd>>>(n, dd, dd0, dt);
  gpu_add_source<<<grid_dim, block_dim, 0, stream_du>>>(n, du, du0, dt);
  gpu_add_source<<<grid_dim, block_dim, 0, stream_dv>>>(n, dv, dv0, dt);
  cudaEventRecord(join_du, stream_du);
  cudaEventRecord(join_dv, stream_dv);
  cudaStreamWaitEvent(stream_dd, join_du, 0);
  cudaStreamWaitEvent(stream_dd, join_dv, 0);

  checkCudaErrors(cudaStreamEndCapture(stream_dd, &graph));
  checkCudaErrors(cudaGraphInstantiate(graph_exec, graph, NULL, NULL, 0));
}

static void lin_solve(unsigned int n, boundary b, float a, float c,
                      float *__restrict__ dx, float *__restrict__ dx0,
                      cudaStream_t *__restrict__ stream) {
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
    gpu_lin_solve_rb_step<<<grid_dim, block_dim, 0, *stream>>>(
      RED, n, a, c, dred0, dblk, dred
    );
    gpu_lin_solve_rb_step<<<grid_dim, block_dim, 0, *stream>>>(
      BLACK, n, a, c, dblk0, dred, dblk
    );
    gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, *stream>>>(
      n, b, dx
    );
  }
}

static void diffuse(unsigned int n, boundary b, float diff, float dt,
                    float *dx, float *dx0, cudaStream_t *__restrict__ stream) {
  const float a = dt * diff * n * n;
  lin_solve(n, b, a, 1 + 4 * a, dx, dx0, stream);
}



static void advect(unsigned int n,
                   float *dd, float *du, float *dv,
                   float *dd0, float *du0, float *dv0,
                   float dt, cudaStream_t *__restrict__ stream) {
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

  gpu_advect_rb<<<grid_dim, block_dim, 0, *stream>>>(
    RED, n, dt, dredd, dredu, dredv, dredd0, dredu0, dredv0, dd0, du0, dv0
  );
  gpu_advect_rb<<<grid_dim, block_dim, 0, *stream>>>(
    BLACK, n, dt, dblkd, dblku, dblkv, dblkd0, dblku0, dblkv0, dd0, du0, dv0
  );
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, *stream>>>(
    n, VERTICAL, du
  );
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, *stream>>>(
    n, HORIZONTAL, dv
  );
}

static void project(unsigned int n,
                    float *du, float *dv, float *du0, float *dv0,
                    cudaStream_t *__restrict__ stream) {
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

  gpu_project_rb_step1<<<grid_dim, block_dim, 0, *stream>>>(
    n, RED, dredv0, dblku, dblkv
  );
  gpu_project_rb_step1<<<grid_dim, block_dim, 0, *stream>>>(
    n, BLACK, dblkv0, dredu, dredv
  );
  // TODO: What to do with all the pragma omp barriers?
  #pragma omp barrier

  checkCudaErrors(cudaMemsetAsync(du0, 0, size_in_mem, *stream));
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, *stream>>>(
    n, NONE, dv0
  );
  #pragma omp barrier

  lin_solve(n, NONE, 1, 4, du0, dv0, stream);
  #pragma omp barrier

  gpu_project_rb_step2<<<grid_dim, block_dim, 0, *stream>>>(
    n, RED, dredu, dredv, dblku0
  );
  gpu_project_rb_step2<<<grid_dim, block_dim, 0, *stream>>>(
    n, BLACK, dblku, dblkv, dredu0
  );
  #pragma omp barrier

  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, *stream>>>(
    n, VERTICAL, du
  );
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, *stream>>>(
    n, HORIZONTAL, dv
  );
}

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          cudaGraphExec_t *add_source3, cudaStream_t *__restrict__ stream) {

  // TODO: These launches can be unsynchronized inside the device, specify that
  checkCudaErrors(cudaGraphLaunch(*add_source3, *stream));
  // TODO : Here would be a barrier of streams

  SWAP(dd0, dd);
  SWAP(du0, du);
  SWAP(dv0, dv);

  diffuse(n, NONE, diff, dt, dd, dd0, stream);
  diffuse(n, VERTICAL, visc, dt, du, du0, stream);
  diffuse(n, HORIZONTAL, visc, dt, dv, dv0, stream);
  #pragma omp barrier

  project(n, du, dv, du0, dv0, stream);
  #pragma omp barrier

  SWAP(dd0, dd);
  SWAP(du0, du);
  SWAP(dv0, dv);

  advect(n, dd, du, dv, dd0, du0, dv0, dt, stream);
  #pragma omp barrier

  project(n, du, dv, du0, dv0, stream);
  #pragma omp barrier
}
