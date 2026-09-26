//! The complete Butteraugli Malta stencils, shared by experimental layouts.
#[derive(Clone, Copy)]
pub(crate) struct V<const N: usize = 8>(pub(crate) [f32; N]);
impl<const N: usize> std::ops::Add for V<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }
}
impl<const N: usize> std::ops::Mul for V<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] * other.0[i]))
    }
}
impl<const N: usize> std::ops::AddAssign for V<N> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const N: usize> V<N> {
    #[inline(always)]
    pub(crate) fn splat(value: f32) -> Self {
        Self([value; N])
    }
}

pub(crate) trait Window<const N: usize = 8> {
    fn load(&self, dx: isize, dy: isize) -> V<N>;
}
macro_rules! w {
    ($window:expr,$dx:expr,$dy:expr) => {
        $window.load($dx, $dy)
    };
}

#[inline(always)]
pub(crate) fn hf_bank<const N: usize, W: Window<N>>(window: &W) -> V<N> {
    let mut retval = V([0.0; N]);

    // Pattern 1: x grows, y constant (horizontal line)
    {
        let sum = w!(window, -4, 0)
            + w!(window, -3, 0)
            + w!(window, -2, 0)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, 0)
            + w!(window, 3, 0)
            + w!(window, 4, 0);
        retval += sum * sum;
    }

    // Pattern 2: y grows, x constant (vertical line)
    {
        let sum = w!(window, 0, -4)
            + w!(window, 0, -3)
            + w!(window, 0, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, 0, 2)
            + w!(window, 0, 3)
            + w!(window, 0, 4);
        retval += sum * sum;
    }

    // Pattern 3: both grow (diagonal \)
    {
        let sum = w!(window, -3, -3)
            + w!(window, -2, -2)
            + w!(window, -1, -1)
            + w!(window, 0, 0)
            + w!(window, 1, 1)
            + w!(window, 2, 2)
            + w!(window, 3, 3);
        retval += sum * sum;
    }

    // Pattern 4: y grows, x shrinks (diagonal /)
    {
        let sum = w!(window, 3, -3)
            + w!(window, 2, -2)
            + w!(window, 1, -1)
            + w!(window, 0, 0)
            + w!(window, -1, 1)
            + w!(window, -2, 2)
            + w!(window, -3, 3);
        retval += sum * sum;
    }

    // Pattern 5: y grows -4 to 4, x shrinks 1 -> -1
    {
        let sum = w!(window, 1, -4)
            + w!(window, 1, -3)
            + w!(window, 1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, -1, 2)
            + w!(window, -1, 3)
            + w!(window, -1, 4);
        retval += sum * sum;
    }

    // Pattern 6: y grows -4 to 4, x grows -1 -> 1
    {
        let sum = w!(window, -1, -4)
            + w!(window, -1, -3)
            + w!(window, -1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, 1, 2)
            + w!(window, 1, 3)
            + w!(window, 1, 4);
        retval += sum * sum;
    }

    // Pattern 7: x grows -4 to 4, y grows -1 to 1
    {
        let sum = w!(window, -4, -1)
            + w!(window, -3, -1)
            + w!(window, -2, -1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, 1)
            + w!(window, 3, 1)
            + w!(window, 4, 1);
        retval += sum * sum;
    }

    // Pattern 8: x grows -4 to 4, y shrinks 1 to -1
    {
        let sum = w!(window, -4, 1)
            + w!(window, -3, 1)
            + w!(window, -2, 1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, -1)
            + w!(window, 3, -1)
            + w!(window, 4, -1);
        retval += sum * sum;
    }

    // Pattern 9: steep diagonal (2:1 slope)
    {
        let sum = w!(window, -2, -3)
            + w!(window, -1, -2)
            + w!(window, -1, -1)
            + w!(window, 0, 0)
            + w!(window, 1, 1)
            + w!(window, 1, 2)
            + w!(window, 2, 3);
        retval += sum * sum;
    }

    // Pattern 10: steep diagonal other way
    {
        let sum = w!(window, 2, -3)
            + w!(window, 1, -2)
            + w!(window, 1, -1)
            + w!(window, 0, 0)
            + w!(window, -1, 1)
            + w!(window, -1, 2)
            + w!(window, -2, 3);
        retval += sum * sum;
    }

    // Pattern 11: shallow diagonal (1:2 slope)
    {
        let sum = w!(window, -3, -2)
            + w!(window, -2, -1)
            + w!(window, -1, -1)
            + w!(window, 0, 0)
            + w!(window, 1, 1)
            + w!(window, 2, 1)
            + w!(window, 3, 2);
        retval += sum * sum;
    }

    // Pattern 12: shallow diagonal other way
    {
        let sum = w!(window, 3, -2)
            + w!(window, 2, -1)
            + w!(window, 1, -1)
            + w!(window, 0, 0)
            + w!(window, -1, 1)
            + w!(window, -2, 1)
            + w!(window, -3, 2);
        retval += sum * sum;
    }

    // Patterns 13-16: duplicates of 8,7,6,5 (9 samples each)

    // Pattern 13: curved line pattern (same as 8)
    {
        let sum = w!(window, -4, 1)
            + w!(window, -3, 1)
            + w!(window, -2, 1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, -1)
            + w!(window, 3, -1)
            + w!(window, 4, -1);
        retval += sum * sum;
    }

    // Pattern 14: curved line other direction (same as 7)
    {
        let sum = w!(window, -4, -1)
            + w!(window, -3, -1)
            + w!(window, -2, -1)
            + w!(window, -1, 0)
            + w!(window, 0, 0)
            + w!(window, 1, 0)
            + w!(window, 2, 1)
            + w!(window, 3, 1)
            + w!(window, 4, 1);
        retval += sum * sum;
    }

    // Pattern 15: very shallow curve (same as 6)
    {
        let sum = w!(window, -1, -4)
            + w!(window, -1, -3)
            + w!(window, -1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, 1, 2)
            + w!(window, 1, 3)
            + w!(window, 1, 4);
        retval += sum * sum;
    }

    // Pattern 16: very shallow curve other direction (same as 5)
    {
        let sum = w!(window, 1, -4)
            + w!(window, 1, -3)
            + w!(window, 1, -2)
            + w!(window, 0, -1)
            + w!(window, 0, 0)
            + w!(window, 0, 1)
            + w!(window, -1, 2)
            + w!(window, -1, 3)
            + w!(window, -1, 4);
        retval += sum * sum;
    }

    retval
}

#[inline(always)]
pub(crate) fn lf_bank<const N: usize, W: Window<N>>(window: &W) -> V<N> {
    let mut retval = V([0.0; N]);

    // Pattern 1: x grows, y constant (sparse horizontal)
    {
        let sum = w!(window, -4, 0)
            + w!(window, -2, 0)
            + w!(window, 0, 0)
            + w!(window, 2, 0)
            + w!(window, 4, 0);
        retval += sum * sum;
    }

    // Pattern 2: y grows, x constant (sparse vertical)
    {
        let sum = w!(window, 0, -4)
            + w!(window, 0, -2)
            + w!(window, 0, 0)
            + w!(window, 0, 2)
            + w!(window, 0, 4);
        retval += sum * sum;
    }

    // Pattern 3: both grow (diagonal)
    {
        let sum = w!(window, -3, -3)
            + w!(window, -2, -2)
            + w!(window, 0, 0)
            + w!(window, 2, 2)
            + w!(window, 3, 3);
        retval += sum * sum;
    }

    // Pattern 4: y grows, x shrinks
    {
        let sum = w!(window, 3, -3)
            + w!(window, 2, -2)
            + w!(window, 0, 0)
            + w!(window, -2, 2)
            + w!(window, -3, 3);
        retval += sum * sum;
    }

    // Pattern 5: y grows, x shifts 1 to -1
    {
        let sum = w!(window, 1, -4)
            + w!(window, 1, -2)
            + w!(window, 0, 0)
            + w!(window, -1, 2)
            + w!(window, -1, 4);
        retval += sum * sum;
    }

    // Pattern 6: y grows, x shifts -1 to 1
    {
        let sum = w!(window, -1, -4)
            + w!(window, -1, -2)
            + w!(window, 0, 0)
            + w!(window, 1, 2)
            + w!(window, 1, 4);
        retval += sum * sum;
    }

    // Pattern 7: x grows, y shifts -1 to 1
    {
        let sum = w!(window, -4, -1)
            + w!(window, -2, -1)
            + w!(window, 0, 0)
            + w!(window, 2, 1)
            + w!(window, 4, 1);
        retval += sum * sum;
    }

    // Pattern 8: x grows, y shifts 1 to -1
    {
        let sum = w!(window, -4, 1)
            + w!(window, -2, 1)
            + w!(window, 0, 0)
            + w!(window, 2, -1)
            + w!(window, 4, -1);
        retval += sum * sum;
    }

    // Pattern 9: steep slope
    {
        let sum = w!(window, -2, -3)
            + w!(window, -1, -2)
            + w!(window, 0, 0)
            + w!(window, 1, 2)
            + w!(window, 2, 3);
        retval += sum * sum;
    }

    // Pattern 10: steep slope other way
    {
        let sum = w!(window, 2, -3)
            + w!(window, 1, -2)
            + w!(window, 0, 0)
            + w!(window, -1, 2)
            + w!(window, -2, 3);
        retval += sum * sum;
    }

    // Pattern 11: shallow slope
    {
        let sum = w!(window, -3, -2)
            + w!(window, -2, -1)
            + w!(window, 0, 0)
            + w!(window, 2, 1)
            + w!(window, 3, 2);
        retval += sum * sum;
    }

    // Pattern 12: shallow slope other way
    {
        let sum = w!(window, 3, -2)
            + w!(window, 2, -1)
            + w!(window, 0, 0)
            + w!(window, -2, 1)
            + w!(window, -3, 2);
        retval += sum * sum;
    }

    // Pattern 13: curved path
    {
        let sum = w!(window, -4, 2)
            + w!(window, -2, 1)
            + w!(window, 0, 0)
            + w!(window, 2, -1)
            + w!(window, 4, -2);
        retval += sum * sum;
    }

    // Pattern 14: curved other direction
    {
        let sum = w!(window, -4, -2)
            + w!(window, -2, -1)
            + w!(window, 0, 0)
            + w!(window, 2, 1)
            + w!(window, 4, 2);
        retval += sum * sum;
    }

    // Pattern 15: vertical with shift
    {
        let sum = w!(window, -2, -4)
            + w!(window, -1, -2)
            + w!(window, 0, 0)
            + w!(window, 1, 2)
            + w!(window, 2, 4);
        retval += sum * sum;
    }

    // Pattern 16: vertical other shift
    {
        let sum = w!(window, 2, -4)
            + w!(window, 1, -2)
            + w!(window, 0, 0)
            + w!(window, -1, 2)
            + w!(window, -2, 4);
        retval += sum * sum;
    }

    retval
}
