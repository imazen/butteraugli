// Thin C wrapper around libjxl's butteraugli for benchmarking.
// BSD-2-Clause licensed (matching libjxl).

#include "butteraugli_ffi.h"

#include <cstddef>

#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/image.h"
#include "tools/no_memory_manager.h"

extern "C" double butteraugli_from_linear_planes(
    const float* src0, const float* src1, const float* src2,
    const float* dst0, const float* dst1, const float* dst2, size_t width,
    size_t height) {
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
  if (!jxl::ButteraugliDiffmap(src_img, dst_img, params, diffmap)) {
    return -999.0;
  }

  return jxl::ButteraugliScoreFromDiffmap(diffmap, &params);
}
