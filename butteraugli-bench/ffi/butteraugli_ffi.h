// Thin C wrapper around libjxl's butteraugli for benchmarking.
// Takes pre-linearized planar float data, returns the butteraugli distance.

#ifndef BUTTERAUGLI_FFI_H_
#define BUTTERAUGLI_FFI_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute butteraugli distance from two linear-light planar float images.
// Each plane is width*height floats, row-major. Caller does sRGB->linear.
// Returns the distance on success, or -999.0 on error.
double butteraugli_from_linear_planes(const float* src0, const float* src1,
                                      const float* src2, const float* dst0,
                                      const float* dst1, const float* dst2,
                                      size_t width, size_t height);

#ifdef __cplusplus
}
#endif

#endif  // BUTTERAUGLI_FFI_H_
