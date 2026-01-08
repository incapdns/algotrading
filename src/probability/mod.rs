mod pdf;
mod bayes;
mod mahalanobis;
mod distributions;
pub mod utils;

pub use pdf::{KernelDensity, Histogram, EmpiricalCDF};
pub use mahalanobis::Mahalanobis;
pub use bayes::BayesianFilter;
pub use distributions::{Distribution, Normal, StudentT, LogNormal, erf};

use crate::numeric::Numeric;

pub trait Process<T: Numeric> {
    /// Evolves one step forward given the current state and random sample.
    fn step(&self, state: T, z: T) -> T;
}

pub trait MonteCarlo<T: Numeric> {
    /// Simulates N paths of length `steps`, given a process.
    fn simulate<P: Process<T>>(
        &self,
        process: &P,
        start: T,
        steps: usize,
        paths: usize,
    ) -> Vec<Vec<T>>;
}