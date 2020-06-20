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

cudaGraph_t *create_lin_solve_graph(unsigned int n, boundary b, float a, float c,
                      float *__restrict__ dred, float *__restrict__ dred0,
                      float *__restrict__ dblk, float *__restrict__ dblk0,
                      float *__restrict__ dx){
  cudaGraph_t *graph = (cudaGraph_t *)malloc(sizeof(cudaGraph_t));
  cudaStream_t origin_stream, forked_stream;
  cudaEvent_t fork, join;
  checkCudaErrors(cudaStreamCreate(&origin_stream));
  checkCudaErrors(cudaStreamCreate(&forked_stream));
  checkCudaErrors(cudaEventCreate(&fork));
  checkCudaErrors(cudaEventCreate(&join));
  // TODO: width should not be computed here.
  // TODO: Avoid re-definition of block_dim and grid_dim.
  unsigned int width = (n + 2) / 2;
  const dim3 block_dim{16, 16};
  const dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};

  checkCudaErrors(cudaStreamBeginCapture(origin_stream, cudaStreamCaptureModeGlobal));
  cudaEventRecord(fork, origin_stream);
  cudaStreamWaitEvent(forked_stream, fork, 0);
  gpu_lin_solve_rb_step<<<grid_dim, block_dim, 0, origin_stream>>>(RED, n, a, c, dred0, dblk, dred);
  gpu_lin_solve_rb_step<<<grid_dim, block_dim, 0, forked_stream>>>(BLACK, n, a, c, dblk0, dred, dblk);
  checkCudaErrors(cudaEventRecord(join, forked_stream));
  checkCudaErrors(cudaStreamWaitEvent(origin_stream, join, 0));
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, origin_stream>>>(n, b, dx);
  
  cudaStreamEndCapture(origin_stream, graph);
  return graph;
}

static void lin_solve(unsigned int n, boundary b, float a, float c,
                      float *__restrict__ dx, float *__restrict__ dx0,
                      const unsigned int from, const unsigned int to) {
  unsigned int color_size = (n + 2) * ((n + 2) / 2);
  // cudaMemcpy does not allow const pointers in dst.
  float *dred0 = dx0;
  float *dblk0 = dx0 + color_size;
  float *dred = dx;
  float *dblk = dx + color_size;
  cudaStream_t stream_for_graph;
  cudaGraph_t *graph;
  cudaGraphExec_t graph_exec;
  // TODO: A new graph is created for each step. It should be created once.
  // TODO: Create just one graph of lin_solve.
  checkCudaErrors(cudaStreamCreate(&stream_for_graph));
  graph = create_lin_solve_graph(n, b, a, c, dred, dred0, dblk, dblk0, dx);
  checkCudaErrors(cudaGraphInstantiate(&graph_exec, *graph, NULL, NULL, 0));
  for(unsigned int k = 0; k < 20; ++k){
    checkCudaErrors(cudaGraphLaunch(graph_exec, stream_for_graph));
  }
  checkCudaErrors(cudaStreamSynchronize(stream_for_graph));
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

  // TODO: A new graph is created for each step. It should be created once.
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  cudaStream_t origin_stream, forked_stream, stream_for_graph;
  cudaEvent_t fork, join;
  checkCudaErrors(cudaStreamCreate(&stream_for_graph));
  checkCudaErrors(cudaStreamCreate(&origin_stream));
  checkCudaErrors(cudaStreamCreate(&forked_stream));
  checkCudaErrors(cudaEventCreate(&fork));
  checkCudaErrors(cudaEventCreate(&join));

  checkCudaErrors(cudaStreamBeginCapture(origin_stream, cudaStreamCaptureModeGlobal));
  cudaEventRecord(fork, origin_stream);
  cudaStreamWaitEvent(forked_stream, fork, 0);
  gpu_advect_rb<<<grid_dim, block_dim, 0, origin_stream>>>(
    RED, n, dt, dredd, dredu, dredv, dredd0, dredu0, dredv0, dd0, du0, dv0
  );
  gpu_advect_rb<<<grid_dim, block_dim, 0, forked_stream>>>(
    BLACK, n, dt, dblkd, dblku, dblkv, dblkd0, dblku0, dblkv0, dd0, du0, dv0
  );
  checkCudaErrors(cudaEventRecord(join, forked_stream));
  checkCudaErrors(cudaStreamWaitEvent(origin_stream, join, 0));
  cudaEventRecord(fork, origin_stream);
  cudaStreamWaitEvent(forked_stream, fork, 0);
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, origin_stream>>>(n, VERTICAL, du);
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x, 0, forked_stream>>>(n, HORIZONTAL, dv);
  checkCudaErrors(cudaEventRecord(join, forked_stream));
  checkCudaErrors(cudaStreamWaitEvent(origin_stream, join, 0));
  cudaStreamEndCapture(origin_stream, &graph);

  // Graph Launching
  checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  for(unsigned int k = 0; k < 20; ++k){
    checkCudaErrors(cudaGraphLaunch(graph_exec, stream_for_graph));
  }
  checkCudaErrors(cudaStreamSynchronize(stream_for_graph));
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
  int size_in_mem = (n + 2) * (n + 2) * sizeof(float);

  gpu_project_rb_step1<<<grid_dim, block_dim>>>(n, RED, dredv0, dblku, dblkv);
  gpu_project_rb_step1<<<grid_dim, block_dim>>>(n, BLACK, dblkv0, dredu, dredv);
  // TODO: What to do with all the pragma omp barriers?
  #pragma omp barrier

  checkCudaErrors(cudaMemsetAsync(du0, 0, size_in_mem));
  gpu_set_bnd<<<div_round_up(n + 2, block_dim.x), block_dim.x>>>(n, NONE, dv0);
  #pragma omp barrier

  lin_solve(n, NONE, 1, 4, du0, dv0, from, to);
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
