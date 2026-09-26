//! Stable instruction-count benchmark for Malta's allocation and filter path.
//!
//! Run: `cargo bench -p butteraugli-bench --bench malta_callgrind`

use butteraugli::image::{BufferPool, ImageF};
use butteraugli::malta::malta_diff_map;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

fn make_pair(width: usize, height: usize) -> (ImageF, ImageF) {
    let mut a = ImageF::new(width, height);
    let mut b = ImageF::new(width, height);
    let mut state = 7_u32;
    for y in 0..height {
        for x in 0..width {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            a.set(x, y, (state >> 8) as f32 / 16_777_216.0);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            b.set(x, y, (state >> 8) as f32 / 16_777_216.0);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::small(args = (128, 128), setup = make_pair)]
#[bench::medium(args = (512, 512), setup = make_pair)]
fn malta((a, b): (ImageF, ImageF)) {
    let pool = BufferPool::new();
    black_box(malta_diff_map(
        black_box(&a),
        black_box(&b),
        1.0,
        1.0,
        2.0,
        false,
        &pool,
    ));
}

library_benchmark_group!(name = malta_group; benchmarks = malta);
main!(library_benchmark_groups = malta_group);
