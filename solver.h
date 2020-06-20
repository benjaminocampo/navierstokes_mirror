#pragma once

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          cudaGraphExec_t *add_source3, cudaStream_t *stream);

void create_graph_addsource3(cudaGraphExec_t *graph_exec,
                             unsigned int n, float dt,
                             float *dd, float *dd0,
                             float *du, float *du0,
                             float *dv, float *dv0);
