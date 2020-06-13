#pragma once

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

void step(unsigned int n, float diff, float visc, float dt,
          float *hd, float *hu, float *hv, float *hd0, float *hu0, float *hv0,
          float* dd, float *du, float *dv, float *dd0, float *du0, float *dv0,
          const unsigned int from, const unsigned int to);
