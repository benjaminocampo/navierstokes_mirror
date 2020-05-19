# Changes from E5-2620v3 to E5-2680v4
TODO: Remove? Included below
From haswell to broadwell:

- Intel Optimization Manual (2.4.7) and Agner Fog Optimization Manual 3 (10.14):
  - Gathers have been improved (./vectortest now tells gathers are better than single loads) (14.16.4)
  - `fpmul` from 5 to 3 cycles
  - `PCLMULQDQ` is one cycle
- Also
  - `fload2x4` is notably worse (~10ns) than `fload8` as expected
  - `shload` is now not worth it, it is better just directly read from memory
  - `stream` is now slower. (Note that reading and writing the same memory
    location is now **really** bad, and in haswell it seemed to be *"free"*)

# Parallel Computing - Lab 3: OpenMP

- Benjamín Ocampo: nicolasbenjaminocampo@gmail.com
- Mateo de Mayo: mateodemayo@gmail.com

# Changes from E5-2620v3 to E5-2680v4

Before adding a line of code we needed to re-measure our previous results,
since they will not be comparable with future versions of the project due 
to the change of architecture from Haswell to Broadwell in zx81 and jupiterace.
We also had to corroborate that our best approaches were still obtaining
similar outcomes. It was also important to decide if our best
approach, which was called *Stream*, would be our *baseline* during this lab.
Therefore, it was compared along with *Shload* and our basic vectorization
in linsolve. Remember that the three of them contains all the improvements
in *advect* and *project* pointed out in the previous laboratory.

We were taken by surprise when stream was worse for the smallest cases
(N = 128, 512, and 2048) but considerably better for the largest ones
(N = 4096 and 8192). Note that reading and writing the same memory
location is now **really** bad, and in haswell it seemed to be *"free"*.
*Shload* was also worse than without it. It is better just read directly
from memory. And the code without these approaches was the best
one for the smaller cases but one of the worst for the largest ones.
That was totally confusing. Which of them was the fastest one?
Which could be used during the lab 3?

We decided to remove our tricks and aces up to the sleeve, i.e, working
without *Shload* and *Stream* but organizing the code in such a way that
works with them in order to choose the fastest one at the end of the
laboratory.

Another changes:

- Intel Optimization Manual (2.4.7) and Agner Fog Optimization Manual 3 (10.14):
  - Gathers have been improved (./vectortest now tells gathers are better than
    single loads) (14.16.4)
  - `fpmul` from 5 to 3 cycles
  - `PCLMULQDQ` is one cycle
- Also
  - `fload2x4` is notably worse (~10ns) than `fload8` as expected

# Tidying up

In order to provide a paralelization which works for the three approaches
mentioned above (implemented in *intrinsics* and *ispc* with the exception
of *Stream* which has not been done in *ispc* yet), a code migration was
performed. The file *solver.c* was changed in such a way that *advect*,
*project* and *linsolve* were implemented by means of functions that share
the same interfaces in intrinsics and ispc. Therefore, it is possible to
compile either intrinsics or ispc code by means of Makefile rules. A
non-vectorized version was also included as an option if its use is required.

Another thing that changed were the functions *dens_step* and *vel_step*.
Remember that another version of *advect* (called *vel_advect*) was implemented
in the previous lab. Even though this approach increases the performance
updating both **u** and **v** instead of doing it separately. It decreases
the reusability and mantainability of code. In order to keep just one
function that performs the advection, *dens_step* and *vel_step* were merged
in just one function called *step*. But now, *advect* will not only update
**u** and **v**, but also **d** which stores the density values.
We will discover later that these changes will be useful for minimizing the
number of omp barriers.

# Thinking in strips

In order to divide equally the work of updating the grid (the space where
the algorithm is running) among a number of threads, we needed to find the
best possible share-out. We found out that an strip-divition, i.e, a set
of rows of the same length for each thread it is the one that feets better.
So, each thread receives a strip of size ceil(N/threads). The last thread
will receive the remaining rows if the number of threads do not divide N.

# From-To divition

Since we wanted a general implementation which works with intrinsics and ispc
(That does not allow the use of omp pragmas), two new parameters were included
in the function *step*. The first argument is *from*, that indicates
from which row a thread will start updating the grid, and the second
one is *to*, that says up to which row will do it. After that, all the
functions that are called in the definition of step were changed to the
from-to format, that includes *add_source*, *advect*, *project*, *linsolve*
(involving the functions implemented in intrinsics and ispc), and *set_bnd*
(which will have an important role in the correctness of the program).

Another possibility of work distribution was an square-divition of the grid.
But in that case we needed to find w and h (the width and heigth of
the square respectively) in such a way that each thread receives one of square.
A simple test in linsolve selecting the "best" w and h was performed but
unfortunately it gives not as good results as strip-divition. Obtaining these
two values could also be messy and not as straightforward as the case in the
strip-divition. That is why it was not implemented in this way.

# OpenMP

Remember that in zx81 and jupiterace its topology consist of two NUMA nodes,
where each node consists of 14 cores. All of them sharing a L3 cache of 35MB.
Finally, each core consists of a L2, L1d and L1i of 256KB, 32KB and 32KB
respectively.

Given a certain number of threads, we needed to think about a placement which
maximize the *affinity* of both nodes. I.e, placing in such a way that the
latency for a memory accesses of a certain core is minimum.

It was not so hard to do so, just indicating omp directives before running
that threads had to be allocated in such a way that they belonged to the same
NUMA node, and they were close together.

Memory allocation was performed in parallel sections in order to reduce
the necessity of accessing memory allocated in another NUMA node (just in
the case of border cases).
Since malloc allocates memory just when it is written and the function
*clear_data* is the first one that initialize arrays *u*, *v*, and *d* we
parallelized it.

```c
static void clear_data(void) {
  int i, size = (N + 2) * (N + 2);

  #pragma omp parallel for
  for (i = 0; i < size; i++) {
    u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
  }
}
```

# React

One main parallel section was in react. Since we needed to compute two
maximum values, a reduction over a parallel for was used.

```c
  float max_velocity2 = 0.0f;
  float max_density = 0.0f;
  
  #pragma omp parallel for default(none) private(i) firstprivate(size, uu, vv, d) reduction(max: max_velocity2, max_density)
  for (i = 0; i < size; i++) {
    if (max_velocity2 < uu[i] * uu[i] + vv[i] * vv[i]) {
      max_velocity2 = uu[i] * uu[i] + vv[i] * vv[i];
    }
    if (max_density < d[i]) {
      max_density = d[i];
    }
  }
```

Finally, since these values are used to update velocity and density,
a parallel for collapse directive was applied in these cases.

```c
  if (max_velocity2 < 0.0000005f) {
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
    d[IX(N / 2, N / 2)] = source * 10.0f;
    #pragma omp parallel for collapse(2)
    for (int y = 64; y < N; y += 64)
      for (int x = 64; x < N; x += 64) d[IX(x, y)] = source * 1000.0f;
  }
```

# Step

In order to stay in parallel sections most of the time, the other main region
is placed in *step* as we said before, so if we have m threads, they will
be distributed along the entire grid, computing the function step from a certain
region. Here is the code of the work distribution performed at calling the
function step.

```c
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
```

TODO's

[] Dependency graph in step.
[] Dependency in advect, linsolve, and project.
[] Code of step, advect, linsolve and project.
