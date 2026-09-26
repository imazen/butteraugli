//! Original sixteen Malta lines evaluated at half-grid coordinates.
//! Coefficients combine bilinear tap weights; no fitted parameters.
use crate::malta_bank::{V, Window};

#[inline(always)]
pub(crate) fn hf_bank<W: Window<Vector = V>>(window: &W) -> V {
    let mut result = V::splat(0.0);
    // Original pattern 1: 9 taps.
    let sum = window.load(-2, 0) * V::splat(1.5)
        + window.load(-1, 0) * V::splat(2.0)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(1, 0) * V::splat(2.0)
        + window.load(2, 0) * V::splat(1.5);
    result += sum * sum;
    // Original pattern 2: 9 taps.
    let sum = window.load(0, -2) * V::splat(1.5)
        + window.load(0, -1) * V::splat(2.0)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(0, 1) * V::splat(2.0)
        + window.load(0, 2) * V::splat(1.5);
    result += sum * sum;
    // Original pattern 3: 7 taps.
    let sum = window.load(-2, -2) * V::splat(0.25)
        + window.load(-2, -1) * V::splat(0.25)
        + window.load(-1, -2) * V::splat(0.25)
        + window.load(-1, -1) * V::splat(1.5)
        + window.load(-1, 0) * V::splat(0.25)
        + window.load(0, -1) * V::splat(0.25)
        + window.load(0, 0) * V::splat(1.5)
        + window.load(0, 1) * V::splat(0.25)
        + window.load(1, 0) * V::splat(0.25)
        + window.load(1, 1) * V::splat(1.5)
        + window.load(1, 2) * V::splat(0.25)
        + window.load(2, 1) * V::splat(0.25)
        + window.load(2, 2) * V::splat(0.25);
    result += sum * sum;
    // Original pattern 4: 7 taps.
    let sum = window.load(-2, 1) * V::splat(0.25)
        + window.load(-2, 2) * V::splat(0.25)
        + window.load(-1, 0) * V::splat(0.25)
        + window.load(-1, 1) * V::splat(1.5)
        + window.load(-1, 2) * V::splat(0.25)
        + window.load(0, -1) * V::splat(0.25)
        + window.load(0, 0) * V::splat(1.5)
        + window.load(0, 1) * V::splat(0.25)
        + window.load(1, -2) * V::splat(0.25)
        + window.load(1, -1) * V::splat(1.5)
        + window.load(1, 0) * V::splat(0.25)
        + window.load(2, -2) * V::splat(0.25)
        + window.load(2, -1) * V::splat(0.25);
    result += sum * sum;
    // Original pattern 5: 9 taps.
    let sum = window.load(-1, 1) * V::splat(0.75)
        + window.load(-1, 2) * V::splat(0.75)
        + window.load(0, -2) * V::splat(0.75)
        + window.load(0, -1) * V::splat(1.25)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(0, 1) * V::splat(1.25)
        + window.load(0, 2) * V::splat(0.75)
        + window.load(1, -2) * V::splat(0.75)
        + window.load(1, -1) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 6: 9 taps.
    let sum = window.load(-1, -2) * V::splat(0.75)
        + window.load(-1, -1) * V::splat(0.75)
        + window.load(0, -2) * V::splat(0.75)
        + window.load(0, -1) * V::splat(1.25)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(0, 1) * V::splat(1.25)
        + window.load(0, 2) * V::splat(0.75)
        + window.load(1, 1) * V::splat(0.75)
        + window.load(1, 2) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 7: 9 taps.
    let sum = window.load(-2, -1) * V::splat(0.75)
        + window.load(-2, 0) * V::splat(0.75)
        + window.load(-1, -1) * V::splat(0.75)
        + window.load(-1, 0) * V::splat(1.25)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(1, 0) * V::splat(1.25)
        + window.load(1, 1) * V::splat(0.75)
        + window.load(2, 0) * V::splat(0.75)
        + window.load(2, 1) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 8: 9 taps.
    let sum = window.load(-2, 0) * V::splat(0.75)
        + window.load(-2, 1) * V::splat(0.75)
        + window.load(-1, 0) * V::splat(1.25)
        + window.load(-1, 1) * V::splat(0.75)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(1, -1) * V::splat(0.75)
        + window.load(1, 0) * V::splat(1.25)
        + window.load(2, -1) * V::splat(0.75)
        + window.load(2, 0) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 9: 7 taps.
    let sum = window.load(-1, -2) * V::splat(0.5)
        + window.load(-1, -1) * V::splat(1.25)
        + window.load(-1, 0) * V::splat(0.25)
        + window.load(0, -1) * V::splat(0.75)
        + window.load(0, 0) * V::splat(1.5)
        + window.load(0, 1) * V::splat(0.75)
        + window.load(1, 0) * V::splat(0.25)
        + window.load(1, 1) * V::splat(1.25)
        + window.load(1, 2) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 10: 7 taps.
    let sum = window.load(-1, 0) * V::splat(0.25)
        + window.load(-1, 1) * V::splat(1.25)
        + window.load(-1, 2) * V::splat(0.5)
        + window.load(0, -1) * V::splat(0.75)
        + window.load(0, 0) * V::splat(1.5)
        + window.load(0, 1) * V::splat(0.75)
        + window.load(1, -2) * V::splat(0.5)
        + window.load(1, -1) * V::splat(1.25)
        + window.load(1, 0) * V::splat(0.25);
    result += sum * sum;
    // Original pattern 11: 7 taps.
    let sum = window.load(-2, -1) * V::splat(0.5)
        + window.load(-1, -1) * V::splat(1.25)
        + window.load(-1, 0) * V::splat(0.75)
        + window.load(0, -1) * V::splat(0.25)
        + window.load(0, 0) * V::splat(1.5)
        + window.load(0, 1) * V::splat(0.25)
        + window.load(1, 0) * V::splat(0.75)
        + window.load(1, 1) * V::splat(1.25)
        + window.load(2, 1) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 12: 7 taps.
    let sum = window.load(-2, 1) * V::splat(0.5)
        + window.load(-1, 0) * V::splat(0.75)
        + window.load(-1, 1) * V::splat(1.25)
        + window.load(0, -1) * V::splat(0.25)
        + window.load(0, 0) * V::splat(1.5)
        + window.load(0, 1) * V::splat(0.25)
        + window.load(1, -1) * V::splat(1.25)
        + window.load(1, 0) * V::splat(0.75)
        + window.load(2, -1) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 13: 9 taps.
    let sum = window.load(-2, 0) * V::splat(0.75)
        + window.load(-2, 1) * V::splat(0.75)
        + window.load(-1, 0) * V::splat(1.25)
        + window.load(-1, 1) * V::splat(0.75)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(1, -1) * V::splat(0.75)
        + window.load(1, 0) * V::splat(1.25)
        + window.load(2, -1) * V::splat(0.75)
        + window.load(2, 0) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 14: 9 taps.
    let sum = window.load(-2, -1) * V::splat(0.75)
        + window.load(-2, 0) * V::splat(0.75)
        + window.load(-1, -1) * V::splat(0.75)
        + window.load(-1, 0) * V::splat(1.25)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(1, 0) * V::splat(1.25)
        + window.load(1, 1) * V::splat(0.75)
        + window.load(2, 0) * V::splat(0.75)
        + window.load(2, 1) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 15: 9 taps.
    let sum = window.load(-1, -2) * V::splat(0.75)
        + window.load(-1, -1) * V::splat(0.75)
        + window.load(0, -2) * V::splat(0.75)
        + window.load(0, -1) * V::splat(1.25)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(0, 1) * V::splat(1.25)
        + window.load(0, 2) * V::splat(0.75)
        + window.load(1, 1) * V::splat(0.75)
        + window.load(1, 2) * V::splat(0.75);
    result += sum * sum;
    // Original pattern 16: 9 taps.
    let sum = window.load(-1, 1) * V::splat(0.75)
        + window.load(-1, 2) * V::splat(0.75)
        + window.load(0, -2) * V::splat(0.75)
        + window.load(0, -1) * V::splat(1.25)
        + window.load(0, 0) * V::splat(2.0)
        + window.load(0, 1) * V::splat(1.25)
        + window.load(0, 2) * V::splat(0.75)
        + window.load(1, -2) * V::splat(0.75)
        + window.load(1, -1) * V::splat(0.75);
    result += sum * sum;
    result
}

#[inline(always)]
pub(crate) fn lf_bank<W: Window<Vector = V>>(window: &W) -> V {
    let mut result = V::splat(0.0);
    // Original pattern 1: 5 taps.
    let sum = window.load(-2, 0)
        + window.load(-1, 0)
        + window.load(0, 0)
        + window.load(1, 0)
        + window.load(2, 0);
    result += sum * sum;
    // Original pattern 2: 5 taps.
    let sum = window.load(0, -2)
        + window.load(0, -1)
        + window.load(0, 0)
        + window.load(0, 1)
        + window.load(0, 2);
    result += sum * sum;
    // Original pattern 3: 5 taps.
    let sum = window.load(-2, -2) * V::splat(0.25)
        + window.load(-2, -1) * V::splat(0.25)
        + window.load(-1, -2) * V::splat(0.25)
        + window.load(-1, -1) * V::splat(1.25)
        + window.load(0, 0)
        + window.load(1, 1) * V::splat(1.25)
        + window.load(1, 2) * V::splat(0.25)
        + window.load(2, 1) * V::splat(0.25)
        + window.load(2, 2) * V::splat(0.25);
    result += sum * sum;
    // Original pattern 4: 5 taps.
    let sum = window.load(-2, 1) * V::splat(0.25)
        + window.load(-2, 2) * V::splat(0.25)
        + window.load(-1, 1) * V::splat(1.25)
        + window.load(-1, 2) * V::splat(0.25)
        + window.load(0, 0)
        + window.load(1, -2) * V::splat(0.25)
        + window.load(1, -1) * V::splat(1.25)
        + window.load(2, -2) * V::splat(0.25)
        + window.load(2, -1) * V::splat(0.25);
    result += sum * sum;
    // Original pattern 5: 5 taps.
    let sum = window.load(-1, 1) * V::splat(0.5)
        + window.load(-1, 2) * V::splat(0.5)
        + window.load(0, -2) * V::splat(0.5)
        + window.load(0, -1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(0, 1) * V::splat(0.5)
        + window.load(0, 2) * V::splat(0.5)
        + window.load(1, -2) * V::splat(0.5)
        + window.load(1, -1) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 6: 5 taps.
    let sum = window.load(-1, -2) * V::splat(0.5)
        + window.load(-1, -1) * V::splat(0.5)
        + window.load(0, -2) * V::splat(0.5)
        + window.load(0, -1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(0, 1) * V::splat(0.5)
        + window.load(0, 2) * V::splat(0.5)
        + window.load(1, 1) * V::splat(0.5)
        + window.load(1, 2) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 7: 5 taps.
    let sum = window.load(-2, -1) * V::splat(0.5)
        + window.load(-2, 0) * V::splat(0.5)
        + window.load(-1, -1) * V::splat(0.5)
        + window.load(-1, 0) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(1, 0) * V::splat(0.5)
        + window.load(1, 1) * V::splat(0.5)
        + window.load(2, 0) * V::splat(0.5)
        + window.load(2, 1) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 8: 5 taps.
    let sum = window.load(-2, 0) * V::splat(0.5)
        + window.load(-2, 1) * V::splat(0.5)
        + window.load(-1, 0) * V::splat(0.5)
        + window.load(-1, 1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(1, -1) * V::splat(0.5)
        + window.load(1, 0) * V::splat(0.5)
        + window.load(2, -1) * V::splat(0.5)
        + window.load(2, 0) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 9: 5 taps.
    let sum = window.load(-1, -2) * V::splat(0.5)
        + window.load(-1, -1)
        + window.load(0, -1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(0, 1) * V::splat(0.5)
        + window.load(1, 1)
        + window.load(1, 2) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 10: 5 taps.
    let sum = window.load(-1, 1)
        + window.load(-1, 2) * V::splat(0.5)
        + window.load(0, -1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(0, 1) * V::splat(0.5)
        + window.load(1, -2) * V::splat(0.5)
        + window.load(1, -1);
    result += sum * sum;
    // Original pattern 11: 5 taps.
    let sum = window.load(-2, -1) * V::splat(0.5)
        + window.load(-1, -1)
        + window.load(-1, 0) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(1, 0) * V::splat(0.5)
        + window.load(1, 1)
        + window.load(2, 1) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 12: 5 taps.
    let sum = window.load(-2, 1) * V::splat(0.5)
        + window.load(-1, 0) * V::splat(0.5)
        + window.load(-1, 1)
        + window.load(0, 0)
        + window.load(1, -1)
        + window.load(1, 0) * V::splat(0.5)
        + window.load(2, -1) * V::splat(0.5);
    result += sum * sum;
    // Original pattern 13: 5 taps.
    let sum = window.load(-2, 1)
        + window.load(-1, 0) * V::splat(0.5)
        + window.load(-1, 1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(1, -1) * V::splat(0.5)
        + window.load(1, 0) * V::splat(0.5)
        + window.load(2, -1);
    result += sum * sum;
    // Original pattern 14: 5 taps.
    let sum = window.load(-2, -1)
        + window.load(-1, -1) * V::splat(0.5)
        + window.load(-1, 0) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(1, 0) * V::splat(0.5)
        + window.load(1, 1) * V::splat(0.5)
        + window.load(2, 1);
    result += sum * sum;
    // Original pattern 15: 5 taps.
    let sum = window.load(-1, -2)
        + window.load(-1, -1) * V::splat(0.5)
        + window.load(0, -1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(0, 1) * V::splat(0.5)
        + window.load(1, 1) * V::splat(0.5)
        + window.load(1, 2);
    result += sum * sum;
    // Original pattern 16: 5 taps.
    let sum = window.load(-1, 1) * V::splat(0.5)
        + window.load(-1, 2)
        + window.load(0, -1) * V::splat(0.5)
        + window.load(0, 0)
        + window.load(0, 1) * V::splat(0.5)
        + window.load(1, -2)
        + window.load(1, -1) * V::splat(0.5);
    result += sum * sum;
    result
}
