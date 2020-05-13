#pragma once

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED, BLACK } grid_color;

void step(unsigned int n, float *d, float *u, float *v, float *d0, float *u0,
          float *v0, float diff, float visc, float dt);
