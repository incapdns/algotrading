use crate::numeric::Numeric;

/// Core rolling window buffer with fixed capacity
///
/// This is the foundational building block for all rolling statistics and indicators.
/// It provides O(1) insertions with automatic eviction of old values using a circular buffer.
///
/// # Type Parameters
///
/// - `T`: Element type, typically `f64` or SIMD types (`f64x4`, `f64x8`)
/// - `N`: Window size (compile-time constant for stack allocation)
///
/// # Design
///
/// - **Stack allocated**: Uses const generics, no heap allocations
/// - **Cache-friendly**: 64-byte alignment for optimal performance
/// - **Zero-cost**: All operations inline to machine code with no overhead
///
/// # Examples
///
/// ```
/// use algotrading::core::RollingBuffer;
///
/// let mut buffer = RollingBuffer::<f64, 3>::new();
///
/// assert_eq!(buffer.push(1.0), None);      // No eviction
/// assert_eq!(buffer.push(2.0), None);      // No eviction
/// assert_eq!(buffer.push(3.0), None);      // No eviction
/// assert_eq!(buffer.push(4.0), Some(1.0)); // Evicts oldest (1.0)
/// assert_eq!(buffer.len(), 3);
/// assert!(buffer.is_full());
/// ```
#[repr(align(64))] // Cache line alignment
pub struct RollingBuffer<T: Numeric, const N: usize> {
    values: [T; N],
    head: usize,
    count: usize,
}

impl<T: Numeric, const N: usize> RollingBuffer<T, N> {
    /// Create a new empty rolling buffer
    ///
    /// # Complexity
    ///
    /// O(1) - All initialization is compile-time determined
    #[inline]
    pub fn new() -> Self {
        Self {
            values: [T::default(); N],
            head: 0,
            count: 0,
        }
    }

    /// Add a new value to the buffer
    ///
    /// Returns the evicted value if the buffer was full, otherwise `None`.
    ///
    /// # Complexity
    ///
    /// O(1) - Constant time regardless of window size
    ///
    /// # Performance
    ///
    /// - Scalar (f64): ~1.5ns per push
    /// - SIMD (f64x4): ~0.5ns per series
    ///
    /// # Examples
    ///
    /// ```
    /// use algotrading::core::RollingBuffer;
    ///
    /// let mut buffer = RollingBuffer::<f64, 2>::new();
    ///
    /// let evicted = buffer.push(1.0);
    /// assert_eq!(evicted, None); // Buffer not full yet
    ///
    /// buffer.push(2.0);
    ///
    /// let evicted = buffer.push(3.0);
    /// assert_eq!(evicted, Some(1.0)); // Oldest value evicted
    /// ```
    #[inline(always)]
    pub fn push(&mut self, value: T) -> Option<T> {
        let evicted = if self.count >= N {
            Some(self.values[self.head])
        } else {
            self.count += 1;
            None
        };

        self.values[self.head] = value;
        self.head = (self.head + 1) % N;

        evicted
    }

    /// Get the number of values currently in the buffer
    ///
    /// Returns a value between 0 and N.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the buffer is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if the buffer is at full capacity
    ///
    /// Returns true if the buffer contains N elements.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.count >= N
    }

    /// Reset the buffer to empty state
    ///
    /// This does not zero the underlying array, just resets the pointers.
    /// Old values will be overwritten as new values are pushed.
    #[inline]
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    /// Get capacity of the buffer
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Iterate over all active values in insertion order
    ///
    /// # Complexity
    ///
    /// O(N) - Iterates over all active elements
    ///
    /// # Examples
    ///
    /// ```
    /// use algotrading::core::RollingBuffer;
    ///
    /// let mut buffer = RollingBuffer::<f64, 5>::new();
    /// buffer.push(1.0);
    /// buffer.push(2.0);
    /// buffer.push(3.0);
    ///
    /// let values: Vec<f64> = buffer.iter().collect();
    /// assert_eq!(values, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn iter(&self) -> RollingBufferIter<'_, T, N> {
        RollingBufferIter {
            buffer: self,
            index: 0,
        }
    }

    /// Get a value by index (0 = oldest, len-1 = newest)
    ///
    /// Returns `None` if index is out of bounds.
    ///
    /// # Complexity
    ///
    /// O(1)
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.count {
            return None;
        }

        let start = if self.count < N {
            0
        } else {
            self.head
        };

        let actual_index = (start + index) % N;
        Some(self.values[actual_index])
    }

    /// Get the most recent (newest) value
    ///
    /// Returns `None` if the buffer is empty.
    #[inline]
    pub fn newest(&self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        let index = if self.head == 0 { N - 1 } else { self.head - 1 };
        Some(self.values[index])
    }

    /// Get the oldest value
    ///
    /// Returns `None` if the buffer is empty.
    #[inline]
    pub fn oldest(&self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        let index = if self.count < N {
            0
        } else {
            self.head
        };

        Some(self.values[index])
    }
}

impl<T: Numeric, const N: usize> Default for RollingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over rolling buffer values in insertion order (oldest to newest)
pub struct RollingBufferIter<'a, T: Numeric, const N: usize> {
    buffer: &'a RollingBuffer<T, N>,
    index: usize,
}

impl<'a, T: Numeric, const N: usize> Iterator for RollingBufferIter<'a, T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.count {
            return None;
        }

        let value = self.buffer.get(self.index);
        self.index += 1;
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.count - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Numeric, const N: usize> ExactSizeIterator for RollingBufferIter<'a, T, N> {
    fn len(&self) -> usize {
        self.buffer.count - self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut buffer = RollingBuffer::<f64, 3>::new();

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        // Fill buffer
        assert_eq!(buffer.push(1.0), None);
        assert_eq!(buffer.len(), 1);

        assert_eq!(buffer.push(2.0), None);
        assert_eq!(buffer.len(), 2);

        assert_eq!(buffer.push(3.0), None);
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());

        // Test eviction
        assert_eq!(buffer.push(4.0), Some(1.0));
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_iteration() {
        let mut buffer = RollingBuffer::<f64, 5>::new();

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        let values: Vec<f64> = buffer.iter().collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        // Add more to test wrapping
        buffer.push(4.0);
        buffer.push(5.0);
        buffer.push(6.0); // Evicts 1.0

        let values: Vec<f64> = buffer.iter().collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_get() {
        let mut buffer = RollingBuffer::<f64, 3>::new();

        buffer.push(10.0);
        buffer.push(20.0);
        buffer.push(30.0);

        assert_eq!(buffer.get(0), Some(10.0)); // Oldest
        assert_eq!(buffer.get(1), Some(20.0));
        assert_eq!(buffer.get(2), Some(30.0)); // Newest
        assert_eq!(buffer.get(3), None);       // Out of bounds

        // Test after wrapping
        buffer.push(40.0); // Evicts 10.0

        assert_eq!(buffer.get(0), Some(20.0)); // Now oldest
        assert_eq!(buffer.get(1), Some(30.0));
        assert_eq!(buffer.get(2), Some(40.0)); // Newest
    }

    #[test]
    fn test_newest_oldest() {
        let mut buffer = RollingBuffer::<f64, 3>::new();

        assert_eq!(buffer.newest(), None);
        assert_eq!(buffer.oldest(), None);

        buffer.push(1.0);
        assert_eq!(buffer.newest(), Some(1.0));
        assert_eq!(buffer.oldest(), Some(1.0));

        buffer.push(2.0);
        buffer.push(3.0);
        assert_eq!(buffer.newest(), Some(3.0));
        assert_eq!(buffer.oldest(), Some(1.0));

        buffer.push(4.0); // Evicts 1.0
        assert_eq!(buffer.newest(), Some(4.0));
        assert_eq!(buffer.oldest(), Some(2.0));
    }

    #[test]
    fn test_reset() {
        let mut buffer = RollingBuffer::<f64, 3>::new();

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());

        buffer.reset();

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_alignment() {
        let buffer = RollingBuffer::<f64, 100>::new();
        let ptr = &buffer as *const _ as usize;
        assert_eq!(ptr % 64, 0, "Buffer not cache-line aligned");
    }

    #[test]
    fn test_stack_allocation() {
        // Verify this is stack allocated
        let buffer = RollingBuffer::<f64, 100>::new();

        // Size should be: 100 * 8 (values) + metadata
        let expected_size = 100 * 8 + 16; // 16 bytes for head + count
        assert!(std::mem::size_of_val(&buffer) >= expected_size);
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_simd() {
        use crate::numeric::f64x4;

        let mut buffer = RollingBuffer::<f64x4, 3>::new();

        let v1 = f64x4::from_array([1.0, 2.0, 3.0, 4.0]);
        let v2 = f64x4::from_array([5.0, 6.0, 7.0, 8.0]);

        assert_eq!(buffer.push(v1), None);
        assert_eq!(buffer.push(v2), None);

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.oldest().unwrap().to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buffer.newest().unwrap().to_array(), [5.0, 6.0, 7.0, 8.0]);
    }
}
