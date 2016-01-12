
#ifndef INCLUDED_UTIL_OPENCL
#define INCLUDED_UTIL_OPENCL

#include <GEGLwrapper.h>

#ifdef __cplusplus
extern "C" {
#endif

void RunInvertGamma(GEGLclass*, float*, float*);
void RunSepia(GEGLclass*, float*, float*, float);

#ifdef __cplusplus
}
#endif

#endif