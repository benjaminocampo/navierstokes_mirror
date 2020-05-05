# Ispc optimizations

## linsolve

Details of the ispc implementation are as follow:

- Loop iterators are uniform values. If this is not the case, ispc uses *vmaskmovps*
  in order to evaluate the loop condition (Which is usually true every iteration).

- How can we avoid the use of masks? Multiples ideas came into our minds. 
  One possibility would be the use of cfor, which tells the compiler that the loop
  condition would be usually true.
  Another possibility is stopping at the position *width - programCount + start* and use
  an if clause to check boundaries (Which will be compiled as just one *vmaskmovps* at
  the end of updating a row of the grid).

- Anyway, these *vmaskmovps* were not so problematic. Just a few nanoseconds were
  gained at the expense of readability.

## addsource

There is no so much to talk about this. We just used a foreach clause over all the grid elements.

# advect

The ispc version of advect was straightforward, there was no necessary of a thorougly 
debugging as in the case of intrinsics.

Unfortunately there is an issue in the vectorization of both versions. It is the use of
gathers because of indexes that were calculated by means of velocity values. Since these
accesses would be unpredictable we did not found a way to vectorize advect without this drawback.

## project

The same as intrinsics

## blocks 

The same as intrinsics

## shload

In the case of ISPC, it was slightly different than intrinsics. At the
beggining of each row four loads were performed (the four neighbours of
a cell). Our goal was to compute four vectors:

- Up: [index - width, index - width + programCount)
- Down: [index + width, index + width + programCount)
- Left: [index - start, index - start + programCount)
- Right: [index - start + 1, index - start + 1 + programCount)

The upstears and downstears neighbours are read without change.
Then, in order to compute *Left* we read the interval:
 
- [j * programCount + y * width, 
- j * programCount + y * width + programCount): where j = 0, 1, ... 

Then we read the *Next Left*
 [(j + 1) * programCount + y * width, 
  (j + 1) * programCount + y * width + programCount)

Finally, using *Left* and *Next Left* we can build *Right* by means of ispc operators.

Since *Next Left* was computed, we do not need to re-read it again, so it is re-used
in the next iteration performing 3 reads instead of 4.

An easy examples of this can be found in the slides.

## icc

This was just a quick test on icc with flags `-O3 -xHost -fp-model fast=2
-no-prec-div` and indeed it returned some gains.