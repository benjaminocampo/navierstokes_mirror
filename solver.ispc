export void add_source(uniform uint32 n,
                       uniform float * uniform x,
                       const uniform float * uniform s,
                       uniform float dt) {
  uniform int32 size = (n + 2) * (n + 2);
  foreach(i = 0 ... size){
    x[i] += dt * s[i];
  }
} 
