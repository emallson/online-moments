#[cfg(test)]
#[macro_use]
extern crate quickcheck_macros;
use num::integer::binomial;
use num::{Float, FromPrimitive};

/// Estimates the first $k$ moments of a distribution in an online fashion (either each datum being
/// incrementally added, or sets of data being added in batches). There is no need to use this
/// method if you are working with complete data. Convenience accessors for several commonly-used
/// moments are provided.
///
/// The most commonly used moment of a distribution is the *variance*, which is the second
/// standardized moment and provided by the `variance()` method.
///
/// The update formulas from [Pébay 2008] are used. If this is used in academic work, please cite
/// this library and [Pébay 2008].
///
/// # Example
///
/// If you're only interested in the mean, you only need to calculate the first moment.
/// ```
/// use online_moments::MomentEstimator;
/// let mut est = MomentEstimator::new(1);
/// est.update(0.0);
/// est.update(1.0);
/// est.update(-1.0);
/// assert_eq!(est.mean(), 0.0);
/// assert_eq!(est.variance(), None);
/// assert_eq!(est.skewness(), None);
/// ```
///
/// Otherwise, you can supply higher moments to estimate (for example) variance and skewness. Note
/// that higher moments require more samples to estimate accurately.
/// ```
/// # use online_moments::MomentEstimator;
/// let mut est = MomentEstimator::<f64>::new(3);
/// for _ in 0..1_000 {
///   est.update(0.0);
///   est.update(1.0);
///   est.update(-1.0);
/// }
/// assert!(est.mean().abs() < 0.005);
/// assert!((2.0 / 3.0 - est.variance().unwrap()).abs() < 0.005);
/// assert!(est.skewness().unwrap().abs() < 0.005);
/// ```
#[derive(Debug)]
pub struct MomentEstimator<T: FromPrimitive + Copy + Float = f64> {
    mean: T,
    // contains moments 2..p
    moments: Vec<T>,
    samples: usize,
}

impl<T: FromPrimitive + Float + Copy + std::iter::Sum> MomentEstimator<T> {
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "Must calculate at least the first moment (mean).");

        MomentEstimator {
            mean: T::zero(),
            moments: vec![T::zero(); p - 1],
            samples: 0,
        }
    }

    pub fn update(&mut self, sample: T) {
        self.samples = self.samples + 1;
        let delta = sample - self.mean;
        let n = T::from_usize(self.samples).unwrap();
        self.mean = self.mean + delta / n;

        if self.samples < 2 {
            return;
        }

        for p_ix in (0..self.moments.len()).rev() {
            let moment = self.moments[p_ix];
            let p = p_ix + 2;
            let summation = (1..=p - 2)
                .map(|k| {
                    // TODO: this binomial call might be able to be replaced by an iterbinomial if
                    // we go column-wise rather than row-wise
                    self.moments[p_ix - k]
                        * T::from_usize(binomial(k, p)).unwrap()
                        * (-delta / n).powi(k as i32)
                })
                .sum();

            let next_moment = moment
                + summation
                + (delta * (n - T::one()) / n).powi(p as i32)
                    * (T::one() - (-T::one() / (n - T::one())).powi((p - 1) as i32));

            self.moments[p_ix] = next_moment;
        }
    }

    pub fn mean(&self) -> T {
        self.mean
    }

    fn n(&self) -> T {
        T::from_usize(self.samples).unwrap()
    }

    pub fn variance(&self) -> Option<T> {
        self.moment(2)
    }

    pub fn skewness(&self) -> Option<T> {
        self.moment(3).map(|m3| {
            let factor =
                self.n().powi(2) / ((self.n() - T::one()) * (self.n() - T::from_f64(2.0).unwrap()));
            factor * m3 / self.variance().unwrap().sqrt().powi(3)
        })
    }

    pub fn moment(&self, ix: usize) -> Option<T> {
        self.moments.get(ix - 2).map(|m| *m / (self.n() - T::one()))
    }
}

#[cfg(test)]
mod test {
    use super::MomentEstimator;
    use quickcheck::TestResult;

    const SAMPLES: usize = 50_000;
    // at most 5% error. this strikes a balance between tightness and false errors due to
    // randomness. in the future, I should replace this with failure rate logic
    const THRESHOLD: f64 = 0.5;

    #[inline]
    fn pct_err(approx: f64, exact: f64) -> f64 {
        (approx - exact).abs() / exact
    }

    #[quickcheck]
    fn normal(mean: f64, variance: f64) -> TestResult {
        if variance <= 0.0 {
            return TestResult::discard();
        }
        use rand::{distributions::Distribution, thread_rng};
        use statrs::distribution::Normal;
        let dist = Normal::new(mean, variance.sqrt()).unwrap();

        let mut est = MomentEstimator::new(3);

        for sample in dist.sample_iter(thread_rng()).take(SAMPLES) {
            est.update(sample);
        }

        // when the mean is approx. 0, we use additive error instead
        let mean_err = if mean.abs() < THRESHOLD {
            est.mean()
        } else {
            pct_err(est.mean(), mean)
        };
        let var_err = pct_err(est.variance().unwrap(), variance);
        // skewness of the normal distribution is 0 so we can't calculate percentage error.
        // instead, we're going to check that the absolute error is less than THRESHOLD - 1.0
        let skew_err = est.skewness().unwrap();

        // println!(
        // "{} {} {:?} {} {} {}",
        // mean, variance, est, mean_err, var_err, skew_err
        // );

        TestResult::from_bool(
            [mean_err, var_err, skew_err]
                .iter()
                .all(|&e| e <= THRESHOLD),
        )
    }

    #[quickcheck]
    fn exponential(lambda: f64) -> TestResult {
        if lambda <= 0.0 {
            return TestResult::discard();
        }

        use rand::{distributions::Distribution, thread_rng};
        use statrs::distribution::Exponential;
        let dist = Exponential::new(lambda).unwrap();

        let mut est = MomentEstimator::new(3);

        for sample in dist.sample_iter(thread_rng()).take(SAMPLES) {
            est.update(sample);
        }

        let mean_err = pct_err(est.mean(), 1.0 / lambda);
        let var_err = pct_err(est.variance().unwrap(), 1.0 / lambda.powi(2));
        let skew_err = pct_err(est.skewness().unwrap(), 2.0);

        // println!("{} {:?} {} {} {}", lambda, est, mean_err, var_err, skew_err);

        TestResult::from_bool(
            [mean_err, var_err, skew_err]
                .iter()
                .all(|&e| e <= THRESHOLD),
        )
    }
}
