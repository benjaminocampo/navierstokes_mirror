# Navier Stokes Realtime Fluid Solver Optimizations

## Different Builds

```bash
# Non vectorized: slow and portable
make clean && make all BUILD=nonvect
# Manually vectorized with intrinsics, requires avx2
make clean && make all BUILD=intrinsics
# Implicitly vectorized with ispc, requires ispc
make clean && make all BUILD=ispc

# demo is the visual version, with headless you can compare the speed with a
# bigger domain without visual feedback
./demo 64 0.1 0.0001 0.0001 5.0 100.0
./demo 512 0.1 0.0001 0.0001 5.0 100.0
./headless 2048 0.1 0.0001 0.0001 5.0 100.0 16
```
