#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

static inline size_t rb_idx(size_t x, size_t y, size_t dim) {
  // Precondition: x != 0 && y != 0, else it can access undefined memory
  int isblack = (x % 2) ^ (y % 2); // XXX &1 en vez de %2
  int isred = !isblack;
  int isodd = y % 2;
  int iseven = !isodd;
  int start = (isred && iseven) || (isblack && isodd);
  size_t base = isblack * dim * (dim / 2);
  size_t offset = (x / 2) - start + y * (dim / 2);
  return base + offset;
}

static inline size_t idx(size_t x, size_t y, size_t stride) {
  return x + y * stride;
}

#pragma GCC diagnostic pop
