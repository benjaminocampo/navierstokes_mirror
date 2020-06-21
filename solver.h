#pragma once

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

void step(unsigned int n, float diff, float visc, float dt,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          cudaStream_t stream0, cudaStream_t stream1, cudaStream_t stream2,
          cudaEvent_t spread, cudaEvent_t join_stream0, cudaEvent_t join_stream1,
          cudaEvent_t join_stream2);
