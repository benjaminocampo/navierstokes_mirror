<!-- TODO: Hablar de (i, j) swap:
- N=128 no afecta, 512, 2048 sí
- Programita de prueba que hicimos,
- Como interpretamos incorrectamente la salida de perf
-->

# Optimización a `lin_solve`

Luego de probar [`coz-profiler`](https://www.youtube.com/watch?v=r-TLSBdHe1A)
con `navierstokes`, la línea principal de `lin_solve` figuraba como la que nos
daría el mayor *"bang for our buck"* y es por esto que decidimos enfocarnos en
ella.

<!-- TODO: Ese 90% no sé realmente si es cómputo, habría que entender mejor
que mide perf ahí-->

Aquí fue la primera vez que utilizamos `perf record` sobre el programa, y con
esto notamos que efectivamente *(compilado con `-O1`)* el 90% del cómputo
sucedía sobre `lin_solve` y en particular, de ese 90%, el 30% lo usaba una
instrucción `divss` que correspondía a la división por `c` en nuestro programa.

Luego de pensar un rato si había alguna forma algebraica de quitar la división
nos dimos cuenta que con esta transformación bastaba:

```c
// Old version using divss instruction
for i, j, k:
    x[i, j, k] = big_summation_and_multiplication / c

// New one using only mulss (about two times faster than divss)
const float invc = 1 / c;
for i, j, k
    x[i, j, k] = big_summation_and_multiplication * inv;
```

<!-- TODO: Poner valores reales de porcentaje, media y sigma -->
Nos sorprendimos al ver que efectivamente, esto mejoró un 20-30% la performance
la media bajo de ~900 a ~700 con un sigma de `—` a uno de `—`.

Esto nos hizo pensar que quizás ante la posibilidad de pérdida de precisión que
la multiplicación representa `-O3` se abstuvo de implementar esa optimización. Y
efectivamente al usar `-Ofast` que incluye `-ffast-math` el compilador
implementó la misma optimización (`-freciprocal-math`). [Respuesta interesante
al respecto](https://stackoverflow.com/a/45899202/3358251)
