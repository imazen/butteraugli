// Thin C wrapper around libjxl's butteraugli for benchmarking.
// BSD-2-Clause licensed (matching libjxl).

#include "butteraugli_ffi.h"

#include <cstddef>
#include <cmath>

#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/image.h"
#include "tools/no_memory_manager.h"

static double ButteraugliFromLinearPlanes(
    const float* src0, const float* src1, const float* src2,
    const float* dst0, const float* dst1, const float* dst2, size_t width,
    size_t height, float intensity_target, bool pnorm3) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();

  auto make_image = [&](const float* p0, const float* p1,
                        const float* p2) -> jxl::StatusOr<jxl::Image3F> {
    JXL_ASSIGN_OR_RETURN(jxl::Image3F image,
                         jxl::Image3F::Create(memory_manager, width, height));

    for (size_t y = 0; y < height; ++y) {
      float* JXL_RESTRICT row0 = image.PlaneRow(0, y);
      float* JXL_RESTRICT row1 = image.PlaneRow(1, y);
      float* JXL_RESTRICT row2 = image.PlaneRow(2, y);
      const size_t off = y * width;
      for (size_t x = 0; x < width; ++x) {
        row0[x] = p0[off + x];
        row1[x] = p1[off + x];
        row2[x] = p2[off + x];
      }
    }

    return image;
  };

  auto src_result = make_image(src0, src1, src2);
  if (!src_result.ok()) return -999.0;
  auto dst_result = make_image(dst0, dst1, dst2);
  if (!dst_result.ok()) return -999.0;

  jxl::Image3F src_img = std::move(src_result).value_();
  jxl::Image3F dst_img = std::move(dst_result).value_();

  JXL_ASSIGN_OR_RETURN(jxl::ImageF diffmap,
                       jxl::ImageF::Create(memory_manager, width, height));

  jxl::ButteraugliParams params;
  params.intensity_target = intensity_target;
  if (!jxl::ButteraugliDiffmap(src_img, dst_img, params, diffmap)) {
    return -999.0;
  }

  if (pnorm3) {
    double sums[3] = {0.0, 0.0, 0.0};
    for (size_t y = 0; y < height; ++y) {
      const float* row = diffmap.ConstRow(y);
      for (size_t x = 0; x < width; ++x) {
        const double d = row[x];
        const double d3 = d * d * d;
        const double d6 = d3 * d3;
        sums[0] += d3;
        sums[1] += d6;
        sums[2] += d6 * d6;
      }
    }
    const double pixels = static_cast<double>(width) * height;
    return (std::cbrt(sums[0] / pixels) +
            std::pow(sums[1] / pixels, 1.0 / 6.0) +
            std::pow(sums[2] / pixels, 1.0 / 12.0)) /
           3.0;
  }
  return jxl::ButteraugliScoreFromDiffmap(diffmap, &params);
}

extern "C" double butteraugli_from_linear_planes(
    const float* src0, const float* src1, const float* src2,
    const float* dst0, const float* dst1, const float* dst2, size_t width,
    size_t height) {
  return ButteraugliFromLinearPlanes(src0, src1, src2, dst0, dst1, dst2,
                                    width, height, 80.0f, false);
}

extern "C" double butteraugli_pnorm3_from_linear_planes(
    const float* src0, const float* src1, const float* src2,
    const float* dst0, const float* dst1, const float* dst2, size_t width,
    size_t height, float intensity_target) {
  return ButteraugliFromLinearPlanes(src0, src1, src2, dst0, dst1, dst2,
                                    width, height, intensity_target, true);
}
