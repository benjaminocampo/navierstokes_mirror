# `IX` Optimization

<!-- TODO: Talk about the (i, j) swap:
- N=128 does not affect performance but N=512, 2048 does
- The test program we did
- How we missinterpreted perf output at first
-->

# `lin_solve` Optimization

After trying out the
[`coz-profiler`](https://www.youtube.com/watch?v=r-TLSBdHe1A) with
`navierstokes`, the main line in `lin_solve` stuck out as the one that would
bring us the most *"bang for our buck"* so we focused on it.

<!-- TODO: I'm not sure if that 90% really refers to execution time. We should understand better what perf is measuring there. -->

So here we used `perf record` for the first time on the program, with this we
noticed that indeed *(compiled with `-O1`)*, that line was the one where the
program spent about 90% of its running time. In particular 30% of it was spent
on a `divss` instruction that came from the division by `c` in our program.

After thinking a bit about some algebraic manipulation we could do to remove the
division we figured this transformation would be enough.

```c
// Old version using divss instruction
for i, j, k:
    x[i, j, k] = big_summation_and_multiplication / c

// New one using only mulss (about two times faster than divss)
const float invc = 1 / c;
for i, j, k
    x[i, j, k] = big_summation_and_multiplication * inv;
```

<!-- TODO: Put real percentage, mean and std deviation values -->
We were surprised to see that it did improve the performance about 20-30%. The
mean went from ~900 to ~700 with a standard deviation of `—` to one of `—`.

This made us think that maybe upon the possible lost of precision that
multiplying would represent, `-O3` abstained from implementing this
optimization. And indeed, when using `-Ofast` which includes `-ffast-math` the
compiler implemented the same optimization (`-freciprocal-math`). [A related
stackoverflow answer](https://stackoverflow.com/a/45899202/3358251)
