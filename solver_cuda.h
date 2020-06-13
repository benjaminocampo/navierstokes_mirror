#pragma once

__global__
void gpu_add_source(unsigned int n, float *x, const float *s, float dt);

void add_source(unsigned int n, float *x, const float *s, float dt,
                const unsigned int from, const unsigned int to);

__global__
void gpu_set_bnd(unsigned int n, boundary b, float *x);

__global__
void gpu_lin_solve_rb_step(grid_color color,unsigned int n, float a, float c,
                           const float * same0, const float * neigh,
                           float * same);

void lin_solve_rb_step(grid_color color, unsigned int n, float a, float c,
                       const float *same0, const float *neigh, float *same,
                       const unsigned int from, const unsigned int to);

void advect_rb(grid_color color, unsigned int n, float *samed, float *sameu,
               float *samev, const float *samed0, const float *sameu0,
               const float *samev0, const float *d0, const float *u0,
               const float *v0, float dt, const unsigned int from,
               const unsigned int to);

void project_rb_step1(unsigned int n, grid_color color, float *sameu0,
                      float *samev0, float *neighu, float *neighv,
                      const unsigned int from, const unsigned int to);

void project_rb_step2(unsigned int n, grid_color color, float *sameu,
                      float *samev, float *neighu0, const unsigned int from,
                      const unsigned int to);
