# Measurements
Our wish is not only to speed up the program but also to uncover the fastest implementation. 
Nevertheless, it might be reckless or impossible to reach that goal without a proper measure
of what performance is. The measure must also be comparable in two different programs with 
different sizes. So, we look for the main, computer-intensive, and repeatable action
that the program was doing. They were three functions called:

* *react*: Adds density and velocity to a particular fluid in the space (The NxN grid) when they decrease to a particular threshold. It might be called the *source force*.
* *vel_step*: Updates the velocity of all the cells in the space.
* *dens_step*: Updates de density of all the cells in the space.

Since the three of these functions updates squillions of cells every step in the execution of *headless*,
One option might be the time needed to perform each of these functions. Nevertheless, that is a size-dependent 
measure. So decided to use the time needed to update each cell of the grid, which is calculated as

$$time\_p\_cell=function\_time/grid\_size$$

# Test of Hypothesis Based on Samples
In order to increase the performance of our program, a careful analysis must be needed.
We cannot only change lines of our code, see that our measures change a little bit, and conclude
that our program runs faster. It is not a game of increasing or decreasing a number.

So, how can we claim that an improvement has strong evidence to be acknowledged?
In our case, there are two contradictory hypotheses under consideration, a **current program**,
and a **potential program**, where the last one assures an improvement on the previous one.
It is initially assumed that the current program has an average performance. This is the
"prior belief" claim. The assertion that the program will increase in speed by means of our approach
is a contradiction to the prior claim. Nevertheless, an assertion of that contradiction is not wanted
unless and until data can provide strong support for it. Hypothesis, claims, contradictions, strong
supports. It sounds like a **hypothesis-testing problem!**

A test of hypotheses is a method for using sample data to decide whether a **null hypothesis** 
should be rejected in terms of an **alternative hypothesis**. On one hand, the prior belief that
the current program has an average performance $$\mu_0 = \mu_0$$, where $$mu_0$$ is the average ns needed to update a cell, is the null hypothesis denoted by $$H_0$$. On the other hand, the alternative hypothesis, denoted by $$H_a$$ is the claim $$\mu = \mu_0$$, i.e, the average performance of the program decreases with the proposed approach.

Since the true average performance of the population $$\mu$$ is unknown, a sample mean is used by means of
a set of observations in order to make an approximation.
These observations are going to be the number of ns per cell needed to perform the functions mentioned above.

Each of one will lead to sets of observations 
$${x_i: i=1, ..., n}$$ 
$${y_i: 1, ... , n}$$ 
$${w_i: i=1, ..., n}$$ respectively.
Finally we are going to observe the set 
$${z_i: i=1 ,...,n}$$. Where $${z_i = x_i + y_i + z_i}$$
Since these sets of samples might differ at least a bit every execution, there is uncertainty about the value of each sample. Because of this uncertainty, before the data becomes available we view each observation as a random variable.

* $${X_1, X_2, . . . , X_n}$$: Where $$X_i$$ is the number of ns per cell needed to perform react in step i.
* $${Y_1, Y_2, . . . , Y_n}$$: Where $$Y_i$$ is the number of ns per cell needed to perform vel_step in step i.
* $${W_1, W_2, . . . , W_n}$$: Where $$W_i$$ is the number of ns per cell needed to perform dens_step in step i.
* $${Z_1, Z_2, . . . , Z_n}$$: Where $$Z_i$$ is the number of ns per cell needed to perform the three functions in step i.

The objective is to decide, based on sample information, which of the two hypotheses is correct.
Therefore, we need a function of the sample data on which the decision (reject $$H_0$$ or do not reject $$H_0$$)
is to be based. This will be our *test statistic*, and a *rejection region*, which is the set of all
test statistic values for which $$H_0$$ will be rejected. In our case, *p-value* is used instead of the *rejection region*,
but both are totally valid.

Errors in concluding are taken into account. On one hand, we might take our improvement 
leads to a faster solution when that is not true. On the other hand, we might also affirm that the improvement 
is not a better solution when it actually is. These are **type 1 ($$\alpha$$)** 
and **type 2 error ($$\beta$$)** respectively.

Since a type 1 error is worst, we specified the largest value of $$\alpha$$ that can be tolerated and find
a rejection region having that value of $$\alpha$$ rather than anything smaller. In our case, the level of
significance is $$\alpha = 0.01$$

Since the amount of samples that can be produced is large, each of them represented by a random variable equally distributed in every step, a ztest hypothesis process about a population mean can be used. 
Finally, these leads to a test statistic denoted by: 
$$Z = ( \overline X - \mu_0)/(s/ \sqrt(n))$$

Let us put all the things together. This is the process that is used in our project.

1. A heuristic is proposed to make the code faster.
 
2. Two versions of our program (the **current** and the **potential** one) are obtained.
    
3. The functions *react*, *vel_step*, and *dens_step* will be executed during **N** steps. 
       Where **N** is sufficiently large.
    
4. Samples $$x_i's, y_i's, w_i's,  z_i's$$ are computed to obtain the average number of ns needed to perform all the functions, that is $$\overline z$$.
    
5. The observed value is normalized and used to calculate the *p-value*.

6. If the *p-value* is lower than our level of significance $$\alpha$$, $$H_0$$ is rejected. Otherwise, $$H_0$$ is not rejected.     

7. An output is produced according to the decision given by 6.

8. Go to 1 and run the process again until the decision is convincing.

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
