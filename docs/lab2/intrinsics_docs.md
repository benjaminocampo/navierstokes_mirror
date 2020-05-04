# Intrinsics Optimizations

In each of the following titles we briefly describe how the modification was
done along some implementations details that may be relevant. The four next
items talk about the main loops found in navierstokes.

## linsolve

This is a linear equation solver and it is the main loop of the program, so it
was our first target to vectorize as well. We applied a simple vectorization
scheme in which 8 elements are computed in the same loop only one would have
been done before.

Some details of the implementation are as follow:

- `fload2x4` is an alias for the intrinsic that loads two unaligned 128b vectors
  to a 256 avx register, we used this for loading instead of the regular
  `_m256_loadu_ps` as recommended in the intel optimization manual (14.6.2) for
  unaligned memory accesses before skylake. It improves the performance a bit.
- Shared load: we will let this one for the `shload` section.

During the development of the project, some ideas came up as to how to improve
the iterative method, for the record we will list some of those:

- As the new redblack algorithm is a little less precise, because in each
  iteration, all the neighbours of the red cells are an iteration behind,
  compared to the previous method in which only half neighbours were behind.
  What would happen if you have neighbors that are 2, 3 or more iterations
  behind? This idea does in fact improve spacial and temporal locality, and thus
  the performance, but we ended up not using it as it needed more iterations of
  grace than what we thought was acceptable for the visual quality of the
  simulation.
- The idea of using float16 to compute over 16 elements instead of 8 was
  tempting, however we discovered that intel does not provide more than two
  operations on float16: encoding and decoding, you can't do arithmetic or
  anything on 16 bit floats. And so the idea fall right there before knowing if
  their precision was enough.
- We did a test branch in which we aligned the memory to 32bytes but it had many
  drawbacks, however we bring this idea back again in the `stream` section.

## addsource

There is not much to say here, a plain vectorization with loop peeling to sum
over all the grid elements.

## advect

After measuring with `perf`, `coz-profiler` and `toplev`, both `advect` and
`project` appeared as the next bottlenecks in the program after improving
`lin_solve`.

The vectorization of advect followed directly the previous improvements made in
the non vectorized version. It was a rather hard to debug vectorization as many
states were in place, about 50 `__m256` variables were needed.

Two insights that came from the tortuous debuggin were:

- `_mm256_mul_epi32` does not multiply 8 elements as one would be tempted to
  believe after using `_mm256_mul_ps`
- The default conversion intrinsic `_mm256_cvtps_epi32` does not behave like a c
  cast, it rounds.

## project

Project vectorization was rather straightforward, and not much to say here more
than we thought it was appropiate to split the loop in two as so the two
accesses didn't thrash the cache.

## blocks

Against all our hopes, loop blocking on `lin_solve` degrade by much the
performance, and it was our main idea for gaining more performance as it have
had a deep impact on the previous project. Our main suspicions is cache
thrashing happening on `lin_solve`. More on this on the `stream` section.

## shload

In `lin_solve`, when reading left and right neighbours vectors, seven elements
overlap, the idea with a shared load is simple: just use one load, and then
extract its last element for reuse within the next cell computation.

The gains again, unfortunately, were small as the load that was being saved was
probably in cache already.

## icc

This was just a quick test on icc with flags `-O3 -xHost -fp-model fast=2
-no-prec-div` and indeed it returned some gains.

## stream

`toplev` informed that indeed, our application is mainly memory bound (~50%),
and after tinkering a little with reads and writes that could be causing
problems we discovered that many write only addresses were being brought to
cache lines.

There is two main ways of making addresses "non temporal" for the intel
microarchitecture, one is setting the memory region as non temporal, which would
be a bad idea as we later in the program need to read from that memory. The
second way is with `movnt` instructions or the `stream` intrinsics which perform
writes directly to `DRAM` without passing through cache. Unfortunately for us
this `stream` instructions require the writes to be aligned, and a great portion
of our code relies on unaligned writes.

As we had not much time for doing a full rewrite of the program we came up with
an idea in which we merge the right and left borders of the grid so as to make
all non-boundary cells be aligned. In this way we were able to make a lot of
write only loops be streamed directly into main memory freeing the cache for
other addresses. Unfortunately this seemed to be not enough to prevent the
thrashing as the gains were again small.

## Resources

- Intel optimization manual:
  https://software.intel.com/sites/default/files/managed/9e/bc/64-ia-32-architectures-optimization-manual.pd
- Agner fog optimization manuals: https://www.agner.org/optimize/
- Peter Cordes on stack overflow, he is in charge of the assembly tag and he's
  been doing an amazing job desmitifying it, each of its answers has a lot of
  useful information.
- Nvidia gpu gems:
  https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids
