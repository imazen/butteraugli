// Shared metric kernels for the evaluation driver and Margarine command.
extern crate image as image_io;

#[path = "box_blur.rs"]
#[cfg(not(feature = "multirate"))]
mod blur;
#[path = "multirate_blur.rs"]
#[cfg(feature = "multirate")]
mod blur;

const CANDIDATE: &str = if cfg!(feature = "native-uhf") {
    "native-uhf-row-malta"
} else if cfg!(all(feature = "simd-malta", feature = "coarse-gaussian")) {
    if cfg!(feature = "row-malta") {
        "simd-coarse-row-malta"
    } else {
        "simd-coarse-full-malta"
    }
} else if cfg!(feature = "simd-opsin") {
    if cfg!(feature = "wide-malta") {
        if cfg!(feature = "row-malta") {
            "opsin-wide-row-malta"
        } else {
            "opsin-wide-full-malta"
        }
    } else if cfg!(feature = "row-malta") {
        "opsin-row-malta"
    } else {
        "opsin-full-malta"
    }
} else if cfg!(feature = "simd-malta") {
    if cfg!(feature = "wide-malta") {
        if cfg!(feature = "row-malta") {
            "simd-wide-row-malta"
        } else {
            "simd-wide-full-malta"
        }
    } else if cfg!(feature = "row-malta") {
        "simd-row-malta"
    } else {
        "simd-full-malta"
    }
} else if cfg!(feature = "wide-malta") {
    if cfg!(feature = "row-malta") {
        "wide-row-malta"
    } else {
        "wide-full-malta"
    }
} else if cfg!(feature = "row-malta") {
    "row-malta"
} else if cfg!(feature = "full-malta") {
    if cfg!(feature = "coarse-gaussian") {
        "coarse-full-malta"
    } else {
        "full-malta"
    }
} else if cfg!(feature = "native-gaussian") {
    "native-gaussian"
} else if cfg!(feature = "native-mask") {
    "native-mask"
} else if cfg!(feature = "tiles") {
    if cfg!(feature = "planar") {
        "planar-tiles"
    } else {
        "tiles"
    }
} else if cfg!(feature = "lattice") {
    if cfg!(feature = "phase-rows") && cfg!(feature = "row-tiles") {
        "phase-tiles"
    } else if cfg!(feature = "phase-rows") {
        "phase-rows"
    } else if cfg!(feature = "row-tiles") {
        "row-tiles"
    } else if cfg!(feature = "row-psycho") {
        "row-psycho"
    } else if cfg!(feature = "coarse-gaussian") {
        "coarse-gaussian"
    } else if cfg!(feature = "stream-blur") {
        "stream-blur"
    } else if cfg!(feature = "planar") {
        "planar"
    } else {
        "lattice"
    }
} else if cfg!(feature = "bounded") {
    "bounded"
} else if cfg!(feature = "stable-peak") {
    "stable-peak"
} else if cfg!(feature = "reference-regions") {
    "reference-regions"
} else if cfg!(feature = "stratified") {
    if cfg!(feature = "peak-stratified") {
        "peak-stratified"
    } else if cfg!(feature = "anchored-pool") {
        "anchored-pool"
    } else {
        "stratified"
    }
} else if cfg!(feature = "refined2") {
    "refined2"
} else if cfg!(feature = "refined1") {
    "refined1"
} else if cfg!(feature = "refined") {
    "refined"
} else if cfg!(feature = "physical") {
    "physical"
} else if cfg!(feature = "perceptual") {
    "perceptual"
} else if cfg!(feature = "pooled") {
    "pooled"
} else if cfg!(feature = "sparse") {
    "sparse"
} else if cfg!(feature = "compact4") {
    "compact4"
} else if cfg!(feature = "compact") {
    "compact"
} else if cfg!(feature = "multirate") {
    "multirate"
} else {
    "box3"
};
#[path = "../../butteraugli/src/consts.rs"]
#[allow(clippy::inconsistent_digit_grouping, clippy::excessive_precision)]
mod consts;
#[path = "../../butteraugli/src/diff.rs"]
mod diff;
#[path = "../../butteraugli/src/blur.rs"]
#[allow(clippy::implicit_saturating_sub, clippy::needless_range_loop)]
mod exact_blur;
#[path = "../../butteraugli/src/image.rs"]
mod image;
#[path = "../../butteraugli/src/malta.rs"]
#[allow(
    clippy::implicit_saturating_sub,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]
mod shared_malta;
#[cfg(not(any(
    feature = "compact4",
    feature = "sparse",
    feature = "lattice",
    feature = "physical"
)))]
use shared_malta as malta;
#[cfg(feature = "full-malta")]
mod full_malta;
#[cfg(feature = "physical")]
mod half_malta_bank;
#[cfg(all(feature = "full-malta", not(feature = "row-malta")))]
use full_malta as malta;
#[cfg(feature = "row-malta")]
mod malta {
    #[cfg(feature = "native-uhf")]
    pub(crate) use crate::full_malta::native_uhf_diff_map as malta_diff_map;
    #[cfg(not(feature = "native-uhf"))]
    pub(crate) use crate::full_malta::sampled_rows_diff_map as malta_diff_map;
}
#[cfg(all(
    feature = "compact4",
    not(any(feature = "sparse", feature = "lattice", feature = "physical"))
))]
#[path = "directional_malta.rs"]
mod malta;
#[cfg(all(
    any(feature = "sparse", feature = "lattice"),
    not(any(feature = "physical", feature = "full-malta"))
))]
#[path = "sparse_malta.rs"]
mod malta;
#[cfg(all(feature = "physical", not(feature = "full-malta")))]
#[path = "physical_malta.rs"]
mod malta;
#[cfg(any(feature = "sparse", feature = "lattice", feature = "physical"))]
mod malta_bank;
#[path = "../../butteraugli/src/mask.rs"]
mod mask;
#[path = "../../butteraugli/src/opsin.rs"]
#[allow(clippy::excessive_precision, clippy::needless_range_loop)]
mod opsin;
#[path = "../../butteraugli/src/psycho.rs"]
#[allow(clippy::excessive_precision, clippy::needless_range_loop)]
mod shared_psycho;
#[cfg(any(not(feature = "compact"), feature = "physical"))]
use shared_psycho as psycho;
#[cfg(all(feature = "compact", not(feature = "physical")))]
#[path = "compact_psycho.rs"]
mod psycho;

use butteraugli::{ButteraugliError, ButteraugliParams};
use std::error::Error;

mod ingress;

// Same p/2p/4p aggregation as butteraugli/src/lib.rs::pnorm_slice. Iterate
// logical rows because this experiment's ImageF can contain padded storage.
fn pnorm(map: &image::ImageF, p: f64) -> f64 {
    let mut sums = [0.0f64; 3];
    for y in 0..map.height() {
        for &value in map.row(y) {
            let mut acc = f64::from(value).powf(p);
            sums[0] += acc;
            acc *= acc;
            sums[1] += acc;
            acc *= acc;
            sums[2] += acc;
        }
    }
    let inv = 1.0 / (map.width() * map.height()) as f64;
    sums.iter()
        .enumerate()
        .map(|(i, &s)| (inv * s).powf(1.0 / (p * f64::from(1u32 << i))))
        .sum::<f64>()
        / 3.0
}

#[cfg(feature = "bounded")]
mod bounded_diff;
#[cfg(feature = "simd-opsin")]
mod opsin_rows;
#[cfg(any(feature = "pooled", feature = "perceptual"))]
mod paired_pool;
#[cfg(feature = "perceptual")]
mod perceptual_pool;
#[cfg(feature = "refined")]
mod refined;
#[cfg(feature = "stream-blur")]
mod stream_blur;
mod strips;
#[cfg(feature = "tiles")]
mod tiles;

fn candidate_encoded(
    a: &ingress::EncodedRows<'_>,
    b: &ingress::EncodedRows<'_>,
    rows: usize,
    params: &ButteraugliParams,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    #[cfg(feature = "refined")]
    {
        refined::compute(a, b, rows, params)
    }
    #[cfg(all(feature = "perceptual", not(feature = "refined")))]
    {
        perceptual_pool::compute(a, b, rows, params)
    }
    #[cfg(all(
        feature = "pooled",
        not(any(feature = "perceptual", feature = "refined"))
    ))]
    {
        paired_pool::compute(a, b, rows, params)
    }
    #[cfg(not(any(feature = "pooled", feature = "perceptual")))]
    {
        strips::compute_encoded(a, b, rows, params)
    }
}
#[cfg(feature = "row-psycho")]
mod row_psycho;
