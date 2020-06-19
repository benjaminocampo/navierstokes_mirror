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
#include <thrust/extrema.h>

/* macros */

#define IX(x, y) (rb_idx((x), (y), (N + 2)))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

#include "indices.h"
#include "solver.h"
#include "timing.h"
#include "helper_cuda.h"
#include "helper_string.h"

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float *hd, *hu, *hv;
static float *hd_prev, *hu_prev, *hv_prev;
static float *dd, *du, *dv;
static float *dd_prev, *du_prev, *dv_prev;

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
  if (hu) _mm_free(hu);
  if (hv) _mm_free(hv);
  if (hu_prev) _mm_free(hu_prev);
  if (hv_prev) _mm_free(hv_prev);
  if (hd) _mm_free(hd);
  if (hd_prev) _mm_free(hd_prev);
  if (du) cudaFree(hu);
  if (dv) cudaFree(hv);
  if (du_prev) cudaFree(hu_prev);
  if (dv_prev) cudaFree(hv_prev);
  if (dd) cudaFree(hd);
  if (dd_prev) cudaFree(hd_prev);
}

static void clear_data(void) {
  int i, size = (N + 2) * (N + 2);

  size_t size_in_mem = size * sizeof(float);
  checkCudaErrors(cudaMemset(du, 0, size_in_mem));
  checkCudaErrors(cudaMemset(dv, 0, size_in_mem));
  checkCudaErrors(cudaMemset(du_prev, 0, size_in_mem));
  checkCudaErrors(cudaMemset(dv_prev, 0, size_in_mem));
  checkCudaErrors(cudaMemset(dd, 0, size_in_mem));
  checkCudaErrors(cudaMemset(dd_prev, 0, size_in_mem));

  #pragma omp parallel for
  for (i = 0; i < size; i++) {
    hu[i] = hv[i] = hu_prev[i] = hv_prev[i] = hd[i] = hd_prev[i] = 0.0f;
  }
}

static int allocate_data(void) {
  int size = (N + 2) * (N + 2) * sizeof(float);
  checkCudaErrors(cudaMalloc(&du, size));
  checkCudaErrors(cudaMalloc(&du_prev, size));
  checkCudaErrors(cudaMalloc(&dv, size));
  checkCudaErrors(cudaMalloc(&dv_prev, size));
  checkCudaErrors(cudaMalloc(&dd, size));
  checkCudaErrors(cudaMalloc(&dd_prev, size));
  hu = (float *)_mm_malloc(size, 32);
  hv = (float *)_mm_malloc(size, 32);
  hu_prev = (float *)_mm_malloc(size, 32);
  hv_prev = (float *)_mm_malloc(size, 32);
  hd = (float *)_mm_malloc(size, 32);
  hd_prev = (float *)_mm_malloc(size, 32);

  if (!hu || !hv || !hu_prev || !hv_prev || !hd || !hd_prev) {
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
      glVertex2f(x + hu[IX(i, j)], y + hv[IX(i, j)]);
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

      d00 = hd[IX(i, j)];
      d01 = hd[IX(i, j + 1)];
      d10 = hd[IX(i + 1, j)];
      d11 = hd[IX(i + 1, j + 1)];

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

using thrust::device_ptr;
using thrust::tuple;
using thrust::zip_iterator;
using thrust::zip_iterator;
using thrust::make_zip_iterator;
using thrust::make_tuple;
using thrust::max_element;

typedef device_ptr<float> dfloatp;
typedef tuple<dfloatp, dfloatp> dfloatp2;
typedef tuple<float, float> tfloat2;

struct compare_dfloatp2 {
  __device__
  bool operator()(tfloat2 lhs, tfloat2 rhs) {
    float lu = lhs.get<0>();
    float lv = lhs.get<1>();
    float ru = rhs.get<0>();
    float rv = rhs.get<1>();
    return lu * lu + lv * lv < ru * ru + rv * rv;
  }
};

static unsigned int div_round_up(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__global__
void gpu_react_velocity(float* u, float* v, float force, int n) {
  int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  int x = (gtidx + 1) * 64;
  int y = (gtidy + 1) * 64;
  if (x < n && y < n) {
    int index = rb_idx(x, y, n + 2);
    u[index] = force * 1000.0f * (n / 2 - y) / (n / 2);
    v[index] = force * 1000.0f * (n / 2 - x) / (n / 2);
  }
  if (gtidx == 0 && gtidy == 0) {
    int mid_index = rb_idx(n / 2, n / 2, n + 2);
    u[mid_index] = force * 10.0f;
    v[mid_index] = force * 10.0f;
  }
}

__global__
void gpu_react_density(float* d, float source, int n) {
  int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  int x = (gtidx + 1) * 64;
  int y = (gtidy + 1) * 64;

  if (x < n && y < n) {
    int index = rb_idx(x, y, n + 2);
    d[index] = source * 1000.0f;
  }
  if (gtidx == 0 && gtidy == 0) {
    int mid_index = rb_idx(n / 2, n / 2, n + 2);
    d[mid_index] = source * 10.0f;
  }
}

static void react(void) {
  int i, j, size = (N + 2) * (N + 2);

  zip_iterator<dfloatp2> uvs_begin = make_zip_iterator(make_tuple(du_prev, dv_prev));
  zip_iterator<dfloatp2> uvs_end = make_zip_iterator(make_tuple(du_prev + size, dv_prev + size));
  // TODO: max_element has an implicit cudaDeviceSynchronize that we should get rid off.
  zip_iterator<dfloatp2> zmaxvel2 = max_element(uvs_begin, uvs_end, compare_dfloatp2());
  dfloatp2 mv2 = zmaxvel2.get_iterator_tuple();
  float mvu = *mv2.get<0>();
  float mvv = *mv2.get<1>();
  float max_velocity2 = mvu * mvu + mvv * mvv;

  dfloatp tdd_prev(dd_prev);
  // TODO: Same as above.
  float max_density = *thrust::max_element(tdd_prev, tdd_prev + size);

  size_t size_in_mem = size * sizeof(float);
  checkCudaErrors(cudaMemsetAsync(du_prev, 0, size_in_mem));
  checkCudaErrors(cudaMemsetAsync(dv_prev, 0, size_in_mem));
  checkCudaErrors(cudaMemsetAsync(dd_prev, 0, size_in_mem));

  dim3 block_dim{16, 16};
  dim3 grid_dim{ // The gridblock mapping is one thread per reactionary point
    div_round_up(div_round_up(N, 64), block_dim.x),
    div_round_up(div_round_up(N, 64), block_dim.y)
  };

  if (max_velocity2 < 0.0000005f)
    gpu_react_velocity<<<grid_dim, block_dim>>>(du_prev, dv_prev, force, N);

  if (max_density < 1.0f)
    gpu_react_density<<<grid_dim, block_dim>>>(dd_prev, source, N);

  if (!mouse_down[0] && !mouse_down[2]) return;

  i = (int)((mx / (float)win_x) * N + 1);
  j = (int)(((win_y - my) / (float)win_y) * N + 1);

  if (i < 1 || i > N || j < 1 || j > N) return;

  // Bring grids from gpu and fill mouse info with cpu
  checkCudaErrors(cudaMemcpy(hd_prev, dd_prev, size_in_mem, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hu_prev, du_prev, size_in_mem, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hv_prev, dv_prev, size_in_mem, cudaMemcpyDeviceToHost));

  if (mouse_down[0]) {
    hu_prev[IX(i, j)] = force * (mx - omx);
    hv_prev[IX(i, j)] = force * (omy - my);
  }

  if (mouse_down[2]) {
    hd_prev[IX(i, j)] = source;
  }

  omx = mx;
  omy = my;

  // Go back to device with mouse input filled
  checkCudaErrors(cudaMemcpy(dd_prev, hd_prev, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(du_prev, hu_prev, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_prev, hv_prev, size_in_mem, cudaMemcpyHostToDevice));

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
  size_t size_in_mem = (N + 2) * (N + 2) * sizeof(float);

  start_t = wtime();
  checkCudaErrors(cudaMemcpy(dd, hd, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(du, hu, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv, hv, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dd_prev, hd_prev, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(du_prev, hu_prev, size_in_mem, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dv_prev, hv_prev, size_in_mem, cudaMemcpyHostToDevice));
  react();

  react_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

  start_t = wtime();
  #pragma omp parallel firstprivate(hd, hu, hv, hd_prev, hu_prev, hv_prev, diff, visc, dt)
  {
    int threads = omp_get_num_threads();
    int strip_size = (N + threads - 1) / threads;
    #pragma omp for
    for(int tid = 0; tid < threads; tid++){
      int from = tid * strip_size + 1;
      int to = MIN((tid + 1) * strip_size + 1, N + 1);
      step(N, diff, visc, dt,
           dd, du, dv, dd_prev, du_prev, dv_prev,
           from, to);
      checkCudaErrors(cudaDeviceSynchronize());
      checkCudaErrors(cudaMemcpy(hd, dd, size_in_mem, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hu, du, size_in_mem, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hv, dv, size_in_mem, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hd_prev, dd_prev, size_in_mem, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hu_prev, du_prev, size_in_mem, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(hv_prev, dv_prev, size_in_mem, cudaMemcpyDeviceToHost));
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
