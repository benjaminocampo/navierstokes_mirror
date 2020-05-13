# Changes from E5-2620v3 to E5-2680v4

From haswell to broadwell:

- Intel Optimization Manual (2.4.7) and Agner Fog Optimization Manual 3 (10.14):
  - Gathers have been improved (./vectortest now tells gathers are better than single loads) (14.16.4)
  - `fpmul` from 5 to 3 cycles
  - `PCLMULQDQ` is one cycle
- Also
  - `fload2x4` is notably worse (~10ns) than `fload8` as expected
  - `shload` is now not worth it, it is better just directly read from memory
  - `stream` is now slower. (Note that reading and writing the same memory
    location is now **really** bad, and in haswell it seemed to be *"free"*)
