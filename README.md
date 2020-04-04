# const N optimization
<!-- TODO
float square(short n, float* x) {
    // n = 2147483647;
    //n = 21474836;
    //n = 32767;
    float sum = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            sum += x[i + j*n];
    return sum;
}
-->

# `IX` Optimization

<!-- TODO: Talk about the (i, j) swap:
- N=128 does not affect performance but N=512, 2048 does
- The test program we did
- How we missinterpreted perf output at first
-->

<!-- TODO: Talk about this test
These two compilations of this function produce the exact same output

```bash
gcc-9.3 -Ofast -floop-interchange -floop-interchange -floop-strip-mine -floop-block -fgraphite-identity -floop-nest-optimize -ftree-loop-distribution
gcc-9.3 -O3
```

So gcc did not clever enough to notice
```
float square(int n, float* sum) {
    float* x = malloc(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            *sum += x[i + j * n];
    return *sum;
}
```

But when n is a known constant with -Ofast it did it (but not with O3)
```
float square(float* sum) {
    const n = 2048;
    float* x = malloc(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            *sum += x[i + j * n];
    return *sum;
}

```
Also in this case it did it
```
static float square(int n, float* restrict sum) {
    float* x = malloc(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            *sum += x[i + j * n];
    return *sum;
}

int main(void) {
    float a;
    square(2048, &a);
    return a;
}
```

So we will do it by hand as we can't exactly know n at compile time
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
