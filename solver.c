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
  unsigned int width = (n + 2) / 2;
  cudaGraph_t *graph = (cudaGraph_t *)malloc(sizeof(cudaGraph_t));
  checkCudaErrors(cudaGraphCreate(graph, 0));
  // TODO: Find general way to define kernel params.
  // TODO: Avoid re-definition of block_dim and grid_dim.
  cudaKernelNodeParams kernelNodeParams = {0};
  const dim3 block_dim{16, 16};
  const dim3 grid_dim{div_round_up(width, block_dim.x), n / block_dim.y};
  kernelNodeParams.gridDim = grid_dim;
  kernelNodeParams.blockDim = block_dim;
  kernelNodeParams.extra = NULL;

  // Kernel: gpu_lin_solve_rb_step
  kernelNodeParams.func = (void *)gpu_lin_solve_rb_step;
  // RED Node
  cudaGraphNode_t lin_solve_rnode;
  grid_color color = RED;
  void *rnode_params[7] = {&color, &n, &a, &c, (void *)&dred0, (void *)&dblk, (void *)&dred};
  kernelNodeParams.kernelParams = (void **)rnode_params;
  checkCudaErrors(
    cudaGraphAddKernelNode(
      &lin_solve_rnode,
      *graph,
      NULL,
      0,
      &kernelNodeParams
    )
  );
  // BLACK Node
  cudaGraphNode_t lin_solve_bnode;
  color = BLACK;
  void *bnode_params[7] = {&color, &n, &a, &c, (void *)&dblk0, (void *)&dred, (void *)&dblk};
  kernelNodeParams.kernelParams = (void **)bnode_params;
  checkCudaErrors(
    cudaGraphAddKernelNode(
      &lin_solve_bnode,
      *graph,
      NULL,
      0,
      &kernelNodeParams
    )
  );
  // Kernel: set_bnd
  cudaGraphNode_t set_bnd_node;
  kernelNodeParams.func = (void *)gpu_set_bnd;
  kernelNodeParams.gridDim = dim3(
    div_round_up(n + 2, block_dim.x), 1, 1
  );
  kernelNodeParams.blockDim = dim3(block_dim.x, 1, 1);
  kernelNodeParams.extra = NULL;
  void *set_bnd_node_params[3] = {&n, &b, (void *)&dx};
  kernelNodeParams.kernelParams = (void **)set_bnd_node_params;
  cudaGraphNode_t dependencies[2] = {lin_solve_bnode, lin_solve_rnode};
  size_t ndependencies = 2;
  checkCudaErrors(
    cudaGraphAddKernelNode(
      &set_bnd_node,
      *graph,
      dependencies,
      ndependencies,
      &kernelNodeParams
    )
  );
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

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  cudaGraph_t *graph = create_lin_solve_graph(n, b, a, c, dred, dred0, dblk, dblk0, dx);
  cudaGraphExec_t graph_exec;
  checkCudaErrors(cudaGraphInstantiate(&graph_exec, *graph, NULL, NULL, 0));
  // TODO: Move up block_dim and grid_dim
  for (unsigned int k = 0; k < 20; ++k) {
    checkCudaErrors(cudaGraphLaunch(graph_exec, stream));
  }
  checkCudaErrors(cudaStreamSynchronize(stream));
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
