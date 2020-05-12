#ifndef SOLVER_H
#define SOLVER_H

void step(unsigned int n, float *dens, float *u, float *v, float *dens0,
          float *u0, float *v0, float diff, float visc, float dt);

#endif /* SOLVER_H */
