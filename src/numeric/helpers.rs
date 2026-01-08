//! Helper macros and utilities for creating generic SIMD-compatible types
//!
//! This module provides macros to reduce boilerplate when converting
//! existing scalar types to generic SIMD-compatible types.

/// Macro to create a generic version of an existing struct with numeric fields
///
/// # Example
///
/// ```ignore
/// use algotrading::numeric_struct;
///
/// // Before (scalar only):
/// pub struct EWMA {
///     value: f64,
///     alpha: f64,
/// }
///
/// // After (generic with SIMD support):
/// numeric_struct! {
///     /// Exponentially Weighted Moving Average
///     pub struct EWMAGeneric<T: Numeric> {
///         value: T,
///         alpha: f64,  // Scalar parameters stay f64
///         initialized: bool,
///     }
/// }
/// ```
#[macro_export]
macro_rules! numeric_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident<$generic:ident: Numeric $(, $($const:ident: $cty:ty),*)? > {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident: $field_ty:ty
            ),* $(,)?
        }
    ) => {
        #[repr(align(64))]
        $(#[$meta])*
        $vis struct $name<$generic: $crate::numeric::Numeric $(, $(const $const: $cty),*)?> {
            $(
                $(#[$field_meta])*
                $field_vis $field: $field_ty
            ),*
        }
    };
}

/// Macro to implement Default for generic numeric structs
///
/// # Example
///
/// ```ignore
/// impl_default_numeric! {
///     EWMAGeneric<T> {
///         value: T::default(),
///         alpha: 0.1,
///         initialized: false,
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_default_numeric {
    (
        $name:ident<$generic:ident $(, $($const:ident),*)?> {
            $($field:ident: $value:expr),* $(,)?
        }
    ) => {
        impl<$generic: $crate::numeric::Numeric $(, $(const $const: usize),*)?> Default for $name<$generic $(, $($const),*)?> {
            #[inline]
            fn default() -> Self {
                Self {
                    $($field: $value),*
                }
            }
        }
    };
}

/// Macro to create type aliases for scalar and SIMD versions
///
/// # Example
///
/// ```ignore
/// numeric_aliases! {
///     EWMAGeneric -> EWMA,
///     RollingStatsGeneric<N> -> RollingStats<N>
/// }
/// ```
#[macro_export]
macro_rules! numeric_aliases {
    (
        $($generic_name:ident$(<$($const:ident),+>)? -> $scalar_name:ident$(<$($const2:ident),+>)?),* $(,)?
    ) => {
        $(
            /// Scalar (f64) version
            pub type $scalar_name$(<$(const $const2: usize),+>)? = $generic_name<f64 $(, $($const),+)?>;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            paste::paste! {
                /// SIMD f64x4 version (4 parallel series)
                pub type [<$scalar_name Simd4>]$(<$(const $const2: usize),+>)? = $generic_name<$crate::numeric::f64x4 $(, $($const),+)?>;

                /// SIMD f64x8 version (8 parallel series)
                pub type [<$scalar_name Simd8>]$(<$(const $const2: usize),+>)? = $generic_name<$crate::numeric::f64x8 $(, $($const),+)?>;
            }
        )*
    };
}

/// Create a new generic instance with proper initialization
///
/// Handles the case where arrays need to be initialized with Default
#[macro_export]
macro_rules! init_array {
    ($ty:ty, $size:expr) => {
        [<$ty>::default(); $size]
    };
}

#[cfg(test)]
mod tests {
    use crate::numeric::Numeric;

    numeric_struct! {
        /// Test struct
        pub struct TestStruct<T: Numeric> {
            value: T,
            count: usize,
        }
    }

    impl<T: Numeric> TestStruct<T> {
        pub fn new() -> Self {
            Self {
                value: T::default(),
                count: 0,
            }
        }
    }

    #[test]
    fn test_numeric_struct_creation() {
        let _scalar = TestStruct::<f64>::new();

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            use crate::numeric::f64x4;
            let _simd = TestStruct::<f64x4>::new();
        }
    }
}
