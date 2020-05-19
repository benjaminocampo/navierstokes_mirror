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

#include <GL/glut.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <omp.h>
#include <assert.h>

#include "indices.h"
#include "solver.h"
#include "timing.h"

/* macros */

#define IX(x, y) (rb_idx((x), (y), (N + 2)))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;

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

  #pragma omp parallel for
  for (i = 0; i < size; i++) {
    // TODO: assert i in [from, to] range
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

/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display(void) {
  glViewport(0, 0, win_x, win_y);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, 1.0, 0.0, 1.0);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
}

static void post_display(void) { glutSwapBuffers(); }

static void draw_velocity(void) {
  int i, j;
  float x, y, h;

  h = 1.0f / N;

  glColor3f(1.0f, 1.0f, 1.0f);
  glLineWidth(1.0f);

  glBegin(GL_LINES);

  for (i = 1; i <= N; i++) {
    x = (i - 0.5f) * h;
    for (j = 1; j <= N; j++) {
      y = (j - 0.5f) * h;

      glVertex2f(x, y);
      glVertex2f(x + u[IX(i, j)], y + v[IX(i, j)]);
    }
  }

  glEnd();
}

static void draw_density(void) {
  int i, j;
  float x, y, h, d00, d01, d10, d11;

  h = 1.0f / N;

  glBegin(GL_QUADS);

  for (i = 0; i <= N; i++) {
    x = (i - 0.5f) * h;
    for (j = 0; j <= N; j++) {
      y = (j - 0.5f) * h;

      d00 = dens[IX(i, j)];
      d01 = dens[IX(i, j + 1)];
      d10 = dens[IX(i + 1, j)];
      d11 = dens[IX(i + 1, j + 1)];

      glColor3f(d00, d00, d00);
      glVertex2f(x, y);
      glColor3f(d10, d10, d10);
      glVertex2f(x + h, y);
      glColor3f(d11, d11, d11);
      glVertex2f(x + h, y + h);
      glColor3f(d01, d01, d01);
      glVertex2f(x, y + h);
    }
  }

  glEnd();
}

/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

static void react(float *d, float *uu, float *vv) {
  int i, j, size = (N + 2) * (N + 2);

  float max_velocity2 = 0.0f; // TODO: Remove initialization because it already starts in -infinity
  float max_density = 0.0f; // TODO: Remove initialization because it already starts in -infinity
  // TODO: Try using default(firstprivate)
  // TODO: Are this parallel fors matching our strip distribution
  #pragma omp parallel for default(none) private(i) firstprivate(size, uu, vv, d) reduction(max: max_velocity2, max_density)
  for (i = 0; i < size; i++) {
    if (max_velocity2 < uu[i] * uu[i] + vv[i] * vv[i]) {
      max_velocity2 = uu[i] * uu[i] + vv[i] * vv[i];
    }
    if (max_density < d[i]) {
      max_density = d[i];
    }
  }

  #pragma omp parallel for
  for (i = 0; i < size; i++) {
    uu[i] = vv[i] = d[i] = 0.0f;
  }

  if (max_velocity2 < 0.0000005f) {
    // TODO: This should be touched by the middle strip thread
    uu[IX(N / 2, N / 2)] = force * 10.0f;
    vv[IX(N / 2, N / 2)] = force * 10.0f;
    #pragma omp parallel for collapse(2)
    for (int y = 64; y < N; y += 64)
      for (int x = 64; x < N; x += 64) {
        uu[IX(x, y)] = force * 1000.0f * (N / 2 - y) / (N / 2);
        vv[IX(x, y)] = force * 1000.0f * (N / 2 - x) / (N / 2);
      }
  }
  if (max_density < 1.0f) {
    // TODO: This should be touched by the middle strip thread
    d[IX(N / 2, N / 2)] = source * 10.0f;
    #pragma omp parallel for collapse(2)
    for (int y = 64; y < N; y += 64)
      for (int x = 64; x < N; x += 64) d[IX(x, y)] = source * 1000.0f;
  }

  if (!mouse_down[0] && !mouse_down[2]) return;

  i = (int)((mx / (float)win_x) * N + 1);
  j = (int)(((win_y - my) / (float)win_y) * N + 1);

  if (i < 1 || i > N || j < 1 || j > N) return;

  if (mouse_down[0]) {
    uu[IX(i, j)] = force * (mx - omx);
    vv[IX(i, j)] = force * (omy - my);
  }

  if (mouse_down[2]) {
    d[IX(i, j)] = source;
  }

  omx = mx;
  omy = my;

  return;
}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func(unsigned char key, int x, int y) {
  switch (key) {
    case 'c':
    case 'C':
      clear_data();
      break;

    case 'q':
    case 'Q':
      free_data();
      exit(0);
      break;

    case 'v':
    case 'V':
      dvel = !dvel;
      break;
  }
}

static void mouse_func(int button, int state, int x, int y) {
  omx = mx = x;
  omx = my = y;

  mouse_down[button] = state == GLUT_DOWN;
}

static void motion_func(int x, int y) {
  mx = x;
  my = y;
}

static void reshape_func(int width, int height) {
  glutSetWindow(win_id);
  glutReshapeWindow(width, height);

  win_x = width;
  win_y = height;
}

static void idle_func(void) {
  static int times = 1;
  static double start_t = 0.0;
  static double one_second = 0.0;
  static double react_ns_p_cell = 0.0;
  static double step_ns_p_cell = 0.0;

  start_t = wtime();
  react(dens_prev, u_prev, v_prev);

  react_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

  start_t = wtime();
  #pragma omp parallel firstprivate(dens, u, v, dens_prev, u_prev, v_prev, diff, visc, dt)
  {
    int threads = omp_get_num_threads();
    int strip_size = (N + threads - 1) / threads;
    #pragma omp for
    for(int tid = 0; tid < threads; tid++){
      int from = tid * strip_size + 1;
      int to = MIN((tid + 1) * strip_size + 1, N + 1);
      step(N, dens, u, v, dens_prev, u_prev, v_prev, diff, visc, dt, from, to);
    }
  }
  step_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

  if (1.0 < wtime() - one_second) { /* at least 1s between stats */
    printf(
        "%lf, %lf, %lf, %lf: ns per cell total, react, vel_step, dens_step\n",
        (react_ns_p_cell + step_ns_p_cell) / times,
        react_ns_p_cell / times, step_ns_p_cell / times, step_ns_p_cell / times);
    one_second = wtime();
    react_ns_p_cell = 0.0;
    step_ns_p_cell = 0.0;
    times = 1;
  } else {
    times++;
  }

  glutSetWindow(win_id);
  glutPostRedisplay();
}

static void display_func(void) {
  pre_display();

  if (dvel)
    draw_velocity();
  else
    draw_density();

  post_display();
}

/*
  ----------------------------------------------------------------------
   open_glut_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glut_window(void) {
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  glutInitWindowPosition(0, 0);
  glutInitWindowSize(win_x, win_y);
  win_id = glutCreateWindow("Alias | wavefront");

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glutSwapBuffers();
  glClear(GL_COLOR_BUFFER_BIT);
  glutSwapBuffers();

  pre_display();

  glutKeyboardFunc(key_func);
  glutMouseFunc(mouse_func);
  glutMotionFunc(motion_func);
  glutReshapeFunc(reshape_func);
  glutIdleFunc(idle_func);
  glutDisplayFunc(display_func);
}

/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char **argv) {
  glutInit(&argc, argv);

  if (argc != 1 && argc != 7) {
    fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t N      : grid resolution\n");
    fprintf(stderr, "\t dt     : time step\n");
    fprintf(stderr, "\t diff   : diffusion rate of the density\n");
    fprintf(stderr, "\t visc   : viscosity of the fluid\n");
    fprintf(stderr,
            "\t force  : scales the mouse movement that generate a force\n");
    fprintf(stderr, "\t source : amount of density that will be deposited\n");
    exit(1);
  }

  if (argc == 1) {
    N = 64;
    dt = 0.1f;
    diff = 0.0001f;
    visc = 0.0001f;
    force = 5.0f;
    source = 100.0f;
    assert((N / 2) % 8 == 0 && "N/2 must be divisible by avx vector size of 8");
    fprintf(
        stderr,
        "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n", N,
        dt, diff, visc, force, source);
  } else {
    N = atoi(argv[1]);
    dt = atof(argv[2]);
    diff = atof(argv[3]);
    visc = atof(argv[4]);
    force = atof(argv[5]);
    source = atof(argv[6]);
  }

  printf("\n\nHow to use this demo:\n\n");
  printf("\t Add densities with the right mouse button\n");
  printf(
      "\t Add velocities with the left mouse button and dragging the mouse\n");
  printf("\t Toggle density/velocity display with the 'v' key\n");
  printf("\t Clear the simulation by pressing the 'c' key\n");
  printf("\t Quit by pressing the 'q' key\n");

  dvel = 0;

  if (!allocate_data()) exit(1);
  clear_data();

  win_x = 512;
  win_y = 512;
  open_glut_window();

  glutMainLoop();

  exit(0);
}
