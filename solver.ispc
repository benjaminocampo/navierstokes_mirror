int idx(int x, int y, int dim)
{
  int base = ((x & 1) ^ (y & 1)) * dim * (dim / 2);
  int offset = (x / 2) + y * (dim / 2);
  return base + offset;
}

export void add_source(uniform unsigned int n,
                       uniform float *uniform x,
                       const uniform float *uniform s,
                       uniform float dt)
{
  uniform int32 size = (n + 2) * (n + 2);
  foreach (i = 0 ... size)
  {
    x[i] += dt * s[i];
  }
}

export void lin_solve_rb_step(uniform int color,
                              uniform int n,
                              uniform float a,
                              uniform float c,
                              const uniform float *uniform same0,
                              const uniform float *uniform neigh,
                              uniform float *uniform same) {
  uniform int start = color == 0 ? 0 : 1;
  uniform int width = (n + 2) / 2;
  uniform float invc = 1 / c;
  for (uniform int y = 1; y <= n; y++, start = 1 - start) {
    varying int index_left = programIndex + y * width;
    varying float left = neigh[index_left];
    uniform int x;
    for (x = start; x < width - programCount; x += programCount){
      varying int index_next_left = programCount + index_left;
      varying float next_left = neigh[index_next_left];
      varying float right = insert(shift(left, 1), programCount - 1, extract(next_left, 0));

      varying int index = (x + programIndex) + y * width;
      same[index] =
          (same0[index] + a * (neigh[index - width] + left +
                               right + neigh[index + width])) *
          invc;
      left = next_left;
      index_left = index_next_left;
    }
    varying int xx = programIndex + x;
    if (xx < width - (1 - start)){
      varying int index = xx + y * width;
      same[index] =
          (same0[index] + a * (neigh[index - width] + neigh[index - start] +
                               neigh[index - start + 1] + neigh[index + width])) *
          invc;
    }
  }
}

export void advect_rb(uniform int color,
                          uniform unsigned int n,
                          uniform float * uniform samed,
                          uniform float * uniform sameu,
                          uniform float * uniform samev,
                          const uniform float * uniform samed0,
                          const uniform float * uniform sameu0,
                          const uniform float * uniform samev0,
                          const uniform float * uniform d0,
                          const uniform float * uniform u0,
                          const uniform float * uniform v0,
                          uniform float dt) {

  uniform int shift = color == 0 ? 1 : -1;
  uniform unsigned int start = color == 0 ? 0 : 1;
  uniform unsigned int width = (n + 2) / 2;

  uniform float dt0 = dt * n;
  for (uniform int i = 1; i <= n; i++, shift = -shift, start = 1 - start){
    for (varying int j = programIndex + start;
         j < width - (1 - start);
         j += programCount){
      varying int i0, j0, i1, j1;
      varying float x, y, s0, t0, s1, t1;

      varying int index = j + i * width;
      varying int gridi = i;
      varying int gridj = 2 * j + shift + start;
      x = gridj - dt0 * sameu0[index];
      y = gridi - dt0 * samev0[index];

      x = clamp(x, 0.5f, n + 0.5f);
      y = clamp(y, 0.5f, n + 0.5f);

      j0 = (int)x;
      i0 = (int)y;
      i1 = i0 + 1;
      j1 = j0 + 1;
      s1 = x - j0;
      s0 = 1 - s1;
      t1 = y - i0;
      t0 = 1 - t1;

      samed[index] = s0 * (t0 * d0[idx(j0, i0, n + 2)] + t1 * d0[idx(j0, i1, n + 2)]) +
                     s1 * (t0 * d0[idx(j1, i0, n + 2)] + t1 * d0[idx(j1, i1, n + 2)]);
      sameu[index] = s0 * (t0 * u0[idx(j0, i0, n + 2)] + t1 * u0[idx(j0, i1, n + 2)]) +
                     s1 * (t0 * u0[idx(j1, i0, n + 2)] + t1 * u0[idx(j1, i1, n + 2)]);
      samev[index] = s0 * (t0 * v0[idx(j0, i0, n + 2)] + t1 * v0[idx(j0, i1, n + 2)]) +
                     s1 * (t0 * v0[idx(j1, i0, n + 2)] + t1 * v0[idx(j1, i1, n + 2)]);
    }
  }
}

export void project_rb_step1(uniform unsigned int n,
                             uniform int color,
                             uniform float * uniform sameu0,
                             uniform float * uniform samev0,
                             uniform float * uniform neighu,
                             uniform float * uniform neighv) {
  uniform unsigned int start = color == 0 ? 0 : 1;
  uniform unsigned int width = (n + 2) / 2;
  for (uniform unsigned int i = 1; i <= n; ++i, start = 1 - start) {
    uniform unsigned int j = start;
    for (j = start; j < width - programCount; j+=programCount) {
      varying int index = (j + programIndex) + i * width;
      samev0[index] = -0.5f *
                      (neighu[index - start + 1] - neighu[index - start] +
                       neighv[index + width] - neighv[index - width]) /
                      n;
      sameu0[index] = 0;
    }
    varying unsigned int jj = programIndex + j;
    if(jj < width - (1 - start)){
      varying int index = jj + i * width;
      samev0[index] = -0.5f *
                      (neighu[index - start + 1] - neighu[index - start] +
                       neighv[index + width] - neighv[index - width]) /
                      n;
      sameu0[index] = 0;
    }
  }
}

export void project_rb_step2(uniform unsigned int n,
                             uniform int color,
                             uniform float * uniform sameu,
                             uniform float * uniform samev,
                             uniform float * uniform neighu0) {
  uniform unsigned int start = color == 0 ? 0 : 1;
  uniform unsigned int width = (n + 2) / 2;

  for (uniform unsigned int i = 1; i <= n; ++i, start = 1 - start) {
    uniform unsigned int j = start;
    for (j = start; j < width - programCount; j+=programCount) {
      varying int index = (j + programIndex) + i * width;
      sameu[index] -=
          0.5f * n * (neighu0[index - start + 1] - neighu0[index - start]);
      samev[index] -=
          0.5f * n * (neighu0[index + width] - neighu0[index - width]);
    }
    varying unsigned int jj = programIndex + j;
    if(jj < width - (1 - start)){
      varying int index = jj + i * width;
      sameu[index] -=
          0.5f * n * (neighu0[index - start + 1] - neighu0[index - start]);
      samev[index] -=
          0.5f * n * (neighu0[index + width] - neighu0[index - width]);
    }
  }
}
