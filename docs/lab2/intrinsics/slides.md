# Intrinsics Slide Remarks

## linsolve

- `fload2x4` intel optimization manual (14.6.2) before skylake
- Delayed iterations: how imprecise can we go?
- `float16`
- Aligned memory:

## addsource

Nothing to say

## advect

- Debugging edge cases is hard, debugging vectorized edge cases is harder.
- `_mm256_mul_ps` multiplies 8 floats. What `_mm256_mul_epi32` does?
- `_mm256_cvtps_epi32` != C cast

## project

Nothing to say

## blocks

Nothing to say (maybe that we don't understand cache)

## shload

What we did for share loading was a little different for ispc and intrinsics
versions.

## icc

Nothing to say (maybe that we didn't lose time with this one and it worked
better than other "optmemory-imizations")

## stream

- writes fills cache
- non temporals `movnt`, `stream`
- memory alignment with short time
- results. again. underwhelming.
- don't base your hopes on isolated experiments.

## Resources

- [Intel optimization manual](https://software.intel.com/sites/default/files/managed/9e/bc/64-ia-32-architectures-optimization-manual.pd)
- [Agner fog optimization manuals](https://www.agner.org/optimize/)
- [Peter Cordes](https://stackoverflow.com/users/224132/peter-cordes) on stack overflow.
- [Nvidia gpu gems](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)
