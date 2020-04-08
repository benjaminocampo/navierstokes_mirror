#include <stddef.h>
#include "solver.h"

#define IX(i,j) ((j)+(n+2)*(i))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;

static void add_source(unsigned int n, float * x, const float * s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)]     = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)]     = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

/* Old lin_solve algorithm
static void lin_solve(unsigned int n, boundary b, float * x, const float * x0, float a, float c)
{
    const float invc = 1 / c;
    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int i = 1; i <= n; i++) {
            for (unsigned int j = 1; j <= n; j++) {
                x[IX(i, j)] = (
                    x0[IX(i, j)] +
                    a * (
                        x[IX(i - 1, j)] +
                        x[IX(i + 1, j)] +
                        x[IX(i, j - 1)] +
                        x[IX(i, j + 1)]
                    )
                ) * invc;
            }
        }
        set_bnd(n, b, x);
    }
}
*/

// basic manual cache blocking
static void lin_solve(unsigned int n, boundary b, float * x, const float * x0, float a, float c)
{
    // TODO: It does not work for n != 2**k, will not traverse all cells
    const float invc = 1 / c;
    const int tile_width = 4; // 8 for i7 7700hq, 4 for e5 2560v3
    const int tile_height = 4; // 4 in both
    const int N = (int) n;
    for (unsigned int k = 0; k < 20; k++) {
        for (int ti = 0; ti < N - 2; ti += tile_width) {
            for (int tj = 0; tj < N - 2; tj += tile_height) {
                for (int ii = 0; ii < tile_width; ii++) {
                    for (int jj = 0; jj < tile_height; jj++) {
                        const int i = 1 + ti + ii;
                        const int j = 1 + tj + jj;
                        x[IX(i, j)] = (
                            x0[IX(i, j)] +
                            a * (
                                x[IX(i - 1, j)] +
                                x[IX(i + 1, j)] +
                                x[IX(i, j - 1)] +
                                x[IX(i, j + 1)]
                            )
                        ) * invc;
                    }
                }
            }
        }
        set_bnd(n, b, x);
    }
}

// Smart lin_solve try
/*
static void lin_solve(unsigned int n, boundary b, float * x, const float * x0, float a, float c) {
    int N = (int) n;
    const int tile_size = 16;
    for (int ti = 1; ti < (N + 2) / tile_size - 1; ti++) {
        for (int tj = 1; tj < (N + 2) / tile_size - 1; tj++) {
            int tile_i = ti * tile_size;
            int tile_j = tj * tile_size;
            int offset = 1;
            for (int counter = 0; counter < 3; counter++) { // do it three times so there are about 7 * 3 = 21 iterations
                while (offset < tile_size / 2) {
                    for (int i = tile_i + offset; i < tile_i + tile_size - offset; i++) {
                        for (int j = tile_j + offset; j < tile_j + tile_size - offset; j++) {
                            x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
                        }
                    }
                    offset++;
                } // Pos: offset = 8
                for (int r = 0; r <= tile_size / 2 - 2; r++) { // perimeters remaining
                    for (int p = tile_size / 2 - 2 - r; p >= 0; p--) { // fill perimeter
                        for (int i = 1; i < tile_size - p * 2; i++) {
                            // TODO: there is a p - 1 index here, that will crash in boundaries
                            x[IX(tile_i + p, p + i + tile_j)] = (x0[IX(tile_i + p, p + i + tile_j)] + a * (x[IX(tile_i + -1 + p, p + i + tile_j)] + x[IX(tile_i + 1 + p, p + i + tile_j)] + x[IX(tile_i + p, p + i - 1 + tile_j)] + x[IX(tile_i + p, p + i + 1 + tile_j)])) / c;
                            x[IX(tile_i + p + i, tile_size - p - 1 + tile_j)] = (x0[IX(tile_i + p + i, tile_size - p - 1 + tile_j)] + a * (x[IX(tile_i + -1 + p + i, tile_size - p - 1 + tile_j)] + x[IX(tile_i + 1 + p + i, tile_size - p - 1 + tile_j)] + x[IX(tile_i + p + i, tile_size - p - 1 - 1 + tile_j)] + x[IX(tile_i + p + i, tile_size - p - 1 + 1 + tile_j)])) / c;
                            x[IX(tile_i + tile_size - p - 1 - i, p + tile_j)] = (x0[IX(tile_i + tile_size - p - 1 - i, p + tile_j)] + a * (x[IX(tile_i + -1 + tile_size - p - 1 - i, p + tile_j)] + x[IX(tile_i + 1 + tile_size - p - 1 - i, p + tile_j)] + x[IX(tile_i + tile_size - p - 1 - i, p - 1 + tile_j)] + x[IX(tile_i + tile_size - p - 1 - i, p + 1 + tile_j)])) / c;
                            x[IX(tile_i + tile_size - p - 1, tile_size - p - 1 - i + tile_j)] = (x0[IX(tile_i + tile_size - p - 1, tile_size - p - 1 - i + tile_j)] + a * (x[IX(tile_i + -1 + tile_size - p - 1, tile_size - p - 1 - i + tile_j)] + x[IX(tile_i + 1 + tile_size - p - 1, tile_size - p - 1 - i + tile_j)] + x[IX(tile_i + tile_size - p - 1, tile_size - p - 1 - i - 1 + tile_j)] + x[IX(tile_i + tile_size - p - 1, tile_size - p - 1 - i + 1 + tile_j)])) / c;
                        }
                    }
                }
            }
        }
    }
}
*/

static void diffuse(unsigned int n, boundary b, float * x, const float * x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect(unsigned int n, boundary b, float * d, const float * d0, const float * u, const float * v, float dt)
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            // TODO: Maybe we have some numerical tricks available here?
            x = i - dt0 * u[IX(i, j)];
            y = j - dt0 * v[IX(i, j)];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(n, b, d);
}

static void project(unsigned int n, float *u, float *v, float *p, float *div)
{
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                     v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
}

void dens_step(unsigned int n, float *x, float *x0, float *u, float *v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
    SWAP(u0, u);
    diffuse(n, VERTICAL, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, HORIZONTAL, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, VERTICAL, u, u0, u0, v0, dt);
    advect(n, HORIZONTAL, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
}
