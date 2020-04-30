export void add_source(uniform unsigned int n,
                       uniform float * uniform x,
                       const uniform float * uniform s,
                       uniform float dt) {
  uniform int32 size = (n + 2) * (n + 2);
  foreach(i = 0 ... size){
    x[i] += dt * s[i];
  }
} 

export void lin_solve_rb_step(uniform int color,
                              uniform int n,
                              uniform float a,
                              uniform float c,
                              const uniform float * uniform same0,
                              const uniform float * uniform neigh,
                              uniform float * uniform same) {
  uniform int shift = color == 0 ? 1 : -1;
  uniform int start = color == 0 ? 0 : 1;

  uniform int width = (n + 2) / 2;
  
  for (uniform int y = 1; y <= n; y++, shift = -shift, start = 1 - start) {
    for (varying int x = programIndex + start;
         x < width - (1 - start);
         x+= programCount) {
      
      varying int index = x + y * width;
      same[index] =
          (same0[index] + a * (neigh[index - width] + neigh[index] +
                                neigh[index + shift] + neigh[index + width])) /
          c;
    }
  }
}

export void vel_advect_rb(uniform int color,
                          uniform unsigned int n,
                          uniform float * uniform sameu,
                          uniform float * uniform samev,
                          const uniform float * uniform sameu0,
                          const float uniform * uniform samev0,
                          const float uniform * uniform u0,
                          const float uniform * uniform v0,
                          uniform float dt) {

  uniform int shift = color == 0 ? 1 : -1;
  uniform unsigned int start = color == 0 ? 0 : 1;
  uniform unsigned int width = (n + 2) / 2;

  uniform float dt0 = dt * n;
  for (uniform int i = 1; i <= n; i++, shift = -shift, start = 1 - start) {
    for (varying int j = programIndex + start; 
         j < width - (1 - start);
         j+= programCount) {
      varying int i0, j0;
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
      s1 = x - j0;
      s0 = 1 - s1;
      t1 = y - i0;
      t0 = 1 - t1;

      varying int isblack = (j0 % 2) ^ (i0 % 2);
      varying int i0j0 = (isblack * (n + 2) * width) + (j0/2 + i0 * width);
      varying int isred = !isblack;
      varying int iseven = (i0 % 2 == 0);
      varying int isodd = !iseven;
      varying int fstart = ((isred && iseven) && (isblack && isodd));
      varying int fshift = isred ? 1 : -1;
      varying int i1j1 = i0j0 + width + (1 - fstart);
      varying int i0j1 = i0j0 + fshift * width * (n + 2) + (1 - fstart);
      varying int i1j0 = i0j0 + fshift * width * (n + 2) + width;

      sameu[index] = s0 * (t0 * u0[i0j0] + t1 * u0[i1j0]) +
                     s1 * (t0 * u0[i0j1] + t1 * u0[i1j1]);
      samev[index] = s0 * (t0 * v0[i0j0] + t1 * v0[i1j0]) +
                     s1 * (t0 * v0[i0j1] + t1 * v0[i1j1]);
    }
  }
}