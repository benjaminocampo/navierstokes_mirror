/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

        This code is a simple prototype that demonstrates how to use the
        code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
        for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#include "indices.h"
#include "solver.h"
#include "timing.h"

/* macros */

#define IX(x, y) (rb_idx((x), (y), (N + 2)))

/* global variables */

static int N, steps;
static float dt, diff, visc;
static float force, source;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

__attribute__((unused)) static void dump_grid(float *grid) {
  for (int i = 0; i < N; i++) {
    printf("%d [", i);
    for (int j = 0; j < N; j++) printf("%f, ", grid[IX(i, j)]);
    printf("], \n");
  }
}

/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/

static void free_data(void) {
  if (u) _mm_free(u);
  if (v) _mm_free(v);
  if (u_prev) _mm_free(u_prev);
  if (v_prev) _mm_free(v_prev);
  if (dens) _mm_free(dens);
  if (dens_prev) _mm_free(dens_prev);
}

static void clear_data(void) {
  int i, size = (N + 2) * (N + 2);

  for (i = 0; i < size; i++) {
    u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
  }
}

static int allocate_data(void) {
  int size = (N + 2) * (N + 2);

  u = (float *)_mm_malloc(size * sizeof(float), 32);
  v = (float *)_mm_malloc(size * sizeof(float), 32);
  u_prev = (float *)_mm_malloc(size * sizeof(float), 32);
  v_prev = (float *)_mm_malloc(size * sizeof(float), 32);
  dens = (float *)_mm_malloc(size * sizeof(float), 32);
  dens_prev = (float *)_mm_malloc(size * sizeof(float), 32);

  if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev) {
    fprintf(stderr, "cannot allocate data\n");
    return (0);
  }

  return (1);
}

static void react(float *d, float *uu, float *vv) {
  int i, size = (N + 2) * (N + 2);
  float max_velocity2 = 0.0f;
  float max_density = 0.0f;

  max_velocity2 = max_density = 0.0f;
  for (i = 0; i < size; i++) {
    if (max_velocity2 < uu[i] * uu[i] + vv[i] * vv[i]) {
      max_velocity2 = uu[i] * uu[i] + vv[i] * vv[i];
    }
    if (max_density < d[i]) {
      max_density = d[i];
    }
  }

  // TODO: Unify these two fors
  for (i = 0; i < size; i++) {
    uu[i] = vv[i] = d[i] = 0.0f;
  }

  if (max_velocity2 < 0.0000005f) {
    uu[IX(N / 2, N / 2)] = force * 10.0f;
    vv[IX(N / 2, N / 2)] = force * 10.0f;
    for (int y = 64; y < N; y += 64)
      for (int x = 64; x < N; x += 64) {
        uu[IX(x, y)] = force * 1000.0f * (N / 2 - y) / (N / 2);
        vv[IX(x, y)] = force * 1000.0f * (N / 2 - x) / (N / 2);
      }
  }
  if (max_density < 1.0f) {
    d[IX(N / 2, N / 2)] = source * 10.0f;
    for (int y = 64; y < N; y += 64)
      for (int x = 64; x < N; x += 64) d[IX(x, y)] = source * 1000.0f;
  }
  return;
}

static void one_step(void) {
  // static int times = 1;
  static double start_t = 0.0;
  // static double one_second = 0.0;
  static double react_ns_p_cell = 0.0;
  static double step_ns_p_cell = 0.0;

  start_t = wtime();
  react(dens_prev, u_prev, v_prev);
  react_ns_p_cell = 1.0e9 * (wtime() - start_t) / (N * N);

  start_t = wtime();
  step(N, dens, u, v, dens_prev, u_prev, v_prev, diff, visc, dt, 1, N + 1);
  step_ns_p_cell = 1.0e9 * (wtime() - start_t) / (N * N);

  printf("%lf, %lf, %lf, %lf\n",
         (react_ns_p_cell + step_ns_p_cell), react_ns_p_cell,
         step_ns_p_cell, step_ns_p_cell);
}

/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char **argv) {
  int i = 0;
  setbuf(stdout, NULL);

  if (argc != 1 && argc != 8) {
    fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t N      : grid resolution\n");
    fprintf(stderr, "\t dt     : time step\n");
    fprintf(stderr, "\t diff   : diffusion rate of the density\n");
    fprintf(stderr, "\t visc   : viscosity of the fluid\n");
    fprintf(stderr,
            "\t force  : scales the mouse movement that generate a force\n");
    fprintf(stderr, "\t source : amount of density that will be deposited\n");
    fprintf(stderr,
            "\t steps : amount of steps the program will be executed\n");
    exit(1);
  }

  if (argc == 1) {
    N = 64;
    dt = 0.1f;
    diff = 0.0001f;
    visc = 0.0001f;
    force = 5.0f;
    source = 100.0f;
    steps = 8;
    fprintf(stderr,
            "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g "
            "steps=%d\n",
            N, dt, diff, visc, force, source, steps);
  } else {
    N = atoi(argv[1]);
    dt = atof(argv[2]);
    diff = atof(argv[3]);
    visc = atof(argv[4]);
    force = atof(argv[5]);
    source = atof(argv[6]);
    steps = atoi(argv[7]);
    assert((N / 2) % 8 == 0 && "N/2 must be divisible by avx vector size of 8");
    fprintf(stderr,
            "Using customs : N=%d dt=%g diff=%g visc=%g force = %g source=%g "
            "steps=%d\n",
            N, dt, diff, visc, force, source, steps);
  }

  if (!allocate_data()) exit(1);
  clear_data();
  printf("total_ns,react,vel_step,dens_step\n");
  for (i = 0; i < steps; i++) one_step();
  free_data();
  exit(0);
}
