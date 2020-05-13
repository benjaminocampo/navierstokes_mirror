//
// solver_ispc.h
// (Header automatically generated by the ispc compiler.)
// DO NOT EDIT THIS FILE.
//

#pragma once
#include <stdint.h>



#ifdef __cplusplus
namespace ispc { /* namespace */
#endif // __cplusplus

#ifndef __ISPC_ALIGN__
#if defined(__clang__) || !defined(_MSC_VER)
// Clang, GCC, ICC
#define __ISPC_ALIGN__(s) __attribute__((aligned(s)))
#define __ISPC_ALIGNED_STRUCT__(s) struct __ISPC_ALIGN__(s)
#else
// Visual Studio
#define __ISPC_ALIGN__(s) __declspec(align(s))
#define __ISPC_ALIGNED_STRUCT__(s) __ISPC_ALIGN__(s) struct
#endif
#endif


///////////////////////////////////////////////////////////////////////////
// Functions exported from ispc code
///////////////////////////////////////////////////////////////////////////
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
extern "C" {
#endif // __cplusplus
    extern void add_source(uint32_t n, float * x, const float * s, float dt);
    extern void advect_rb(int32_t color, uint32_t n, float * samed, float * sameu, float * samev, const float * samed0, const float * sameu0, const float * samev0, const float * d0, const float * u0, const float * v0, float dt);
    extern void lin_solve_rb_step(int32_t color, int32_t n, float a, float c, const float * same0, const float * neigh, float * same);
    extern void project_rb_step1(uint32_t n, int32_t color, float * sameu0, float * samev0, float * neighu, float * neighv);
    extern void project_rb_step2(uint32_t n, int32_t color, float * sameu, float * samev, float * neighu0);
#if defined(__cplusplus) && (! defined(__ISPC_NO_EXTERN_C) || !__ISPC_NO_EXTERN_C )
} /* end extern C */
#endif // __cplusplus


#ifdef __cplusplus
} /* namespace */
#endif // __cplusplus