# Navier Stokes Realtime Fluid Solver Optimizations

## Presentations and Reports

For each lab, its corresponding report `docs.pdf` and presentation `slides.pdf` can be found in `docs/labN`.

## Running

- `demo` executable runs a visual and interactive simulation
- `headless` runs the simulation without visuals and is the one referenced in
  the presentations

After compiling, both of these can be run with:

```sh
./demo 64 0.1 0.0001 0.0001 5.0 100.0
./headless 2048 0.1 0.0001 0.0001 5.0 100.0 16
```

Where the parameters are (in order):

- `N`: grid resolution
- `dt`: time step
- `diff`: diffusion rate of the density
- `visc`: viscosity of the fluid
- `force`: scales the mouse movement that generate a force
- `source`: amount of density that will be deposited
- `steps`: (headless only) amount of steps the program will run for

## Labs

There are different tags in the repo which are referenced in the presentations
and reports, you can compile the best ones from each lab as follows: *(remember to `make clean`)*

### Lab 1 - Compiler Optimizations

```sh
# lab1
git checkout lab1
make CFLAGS="-Ofast -march=native -floop-nest-optimize -funroll-loops -flto -g"
```

### Lab 2 - Vectorization

```sh
# lab2/intrinsics: fastest lab2 using stream writes, requires N = 16 * k - 2
git checkout lab2/intrinsics
# Compile with icc
make all
# Or compile with gcc
make all BUILD=common CC=gcc CFLAGS="-Ofast -march=native -floop-nest-optimize -funroll-loops -flto -g"
# Run with N = 16 * k - 2 for proper memory alignment
./headless 2046 0.1 0.0001 0.0001 5.0 100.0 16

# lab2/ispc: best ispc version, requires ispc
git checkout lab2/ispc
make all ISPCFLAGS=--target=avx2-i32x8 # if you have avx2
make all ISPCFLAGS=--target=sse2-i32x4 # if you only have sse
# l2/intrinsics/shload: intrinsics version that is comparable to lab2/ispc
git checkout l2/intrinsics/shload
make all
```

### Lab 3 - OpenMP

```sh
# lab3
git checkout lab3
# Parallel and vectorized with intrinsics, requires avx2
make all CFLAGS=-fopenmp BUILD=intrinsics
# Parallel and vectoried with ispc, requires ispc
# see ISPCFLAGS used in lab2 for setting sse or avx2
make all CFLAGS=-fopenmp BUILD=ispc
# Parallel and non-vectorized, slow and portable
make all CFLAGS=-fopenmp BUILD=nonvect
```

### Lab 4 - CUDA

```sh
# lab4
git checkout lab4
make cudaheadless # generate headless
make cudademo # generate demo
```
