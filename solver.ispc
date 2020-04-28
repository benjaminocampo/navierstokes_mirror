export void add_source(uniform uint32 n,
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
  uniform int tile_width = programCount; // Multiple of programCount
  uniform int tile_height = 4;
  uniform int N = (int)n;

  for(uniform int ty = 1; ty <= N; ty += tile_height) { 
    for(uniform int tx = 0; tx < width - tile_width; tx += tile_width) {
      for(uniform int yy = 0; yy < tile_height; ++yy, 
                                        shift = -shift,
                                        start = 1 - start){
        for(uniform int xx = start; xx < tile_width + start; xx+=programCount){
            varying int index = (xx + tx + programIndex) + (ty + yy) * width;
            same[index] =
                (same0[index] +
                  a * (neigh[index - width] + neigh[index] +
                      neigh[index + shift] + neigh[index + width])) /
                c;
        }
      }
    }
  }
  // The last tile must be updated separately
  // An odd amount of cells to update (3 in the case of N = 2**k)
  for (uniform int y = 1; y <= N; y++, shift = -shift, start = 1 - start) {
    for(varying int x = width - tile_width + programIndex;
        x < width - (1 - start); x++){
      varying int index = x + y * width;
      same[index] =
          (same0[index] + a * (neigh[index - width] + neigh[index] +
                               neigh[index + shift] + neigh[index + width])) /
          c;
    }
  }
}