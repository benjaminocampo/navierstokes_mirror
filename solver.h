#pragma once

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          cudaStream_t stream_dd, cudaStream_t stream_du, cudaStream_t stream_dv,
          cudaEvent_t spread, cudaEvent_t join_du, cudaEvent_t join_dv);

void create_graph_addsource3(cudaGraphExec_t *graph_exec,
                             cudaEvent_t spread, cudaEvent_t join_du,
                             cudaEvent_t join_dv, cudaStream_t stream_dd,
                             cudaStream_t stream_du, cudaStream_t stream_dv,
                             unsigned int n, float dt,
                             float *dd, float *dd0,
                             float *du, float *du0,
                             float *dv, float *dv0);

void create_graph_diffuse3(cudaGraphExec_t *graph_exec,
                           cudaEvent_t spread, cudaEvent_t join_du,
                           cudaEvent_t join_dv, cudaStream_t stream_dd,
                           cudaStream_t stream_du, cudaStream_t stream_dv,
                           unsigned int n, float diff,
                           float visc, float dt,
                           float *dd, float *dd0,
                           float *du, float *du0,
                           float *dv, float *dv0);
