#ifndef SOLVER_H
#define SOLVER_H

void step(unsigned int n, float *d, float *u, float *v, float *d0,
          float *u0, float *v0, float diff, float visc, float dt);

#endif /* SOLVER_H */
