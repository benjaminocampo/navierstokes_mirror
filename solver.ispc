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
    
    for (uniform int y = 1; y <= n; y++, shift = -shift, start = 1 - start) {
        uniform int x;
        for (x = start;
             x < width - (1 - start) - (width - (1 - start)) % programCount;
             x+= programCount) {
            
            varying int index = (x + programIndex) + y * width;
            same[index] =
                (same0[index] + a * (neigh[index - width] + neigh[index] +
                                     neigh[index + shift] + neigh[index + width])) /
                c;
        }

        for(; x < width - (1 - start); ++x){
            uniform int index = x + y * width;
            same[index] =
                (same0[index] + a * (neigh[index - width] + neigh[index] +
                                    neigh[index + shift] + neigh[index + width])) /
                c;
        }
    }
}