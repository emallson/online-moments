use num::integer::binomial;
use num::{Float, FromPrimitive};

// Estimates the first $k$ moments of a distribution in an online fashion (either each datum being
// incrementally added, or sets of data being added in batches). There is no need to use this
// method if you are working with complete data. Convenience accessors for several commonly-used
// moments are provided.
//
// The most commonly used moment of a distribution is the *variance*, which is the second
// standardized moment and provided by the `variance()` method.
//
// The update formulas from [Pébay 2008] are used. If this is used in academic work, please cite
// this library and [Pébay 2008].
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
        self.moment(2).map(|m| m / (self.n() - T::one()))
    }

    pub fn skewness(&self) -> Option<T> {
        self.moment(3).map(|m3| {
            let factor =
                self.n().powi(2) / ((self.n() - T::one()) * (self.n() - T::from_f64(2.0).unwrap()));
            factor * (m3 / (self.n() - T::one())) / self.variance().unwrap().sqrt().powi(3)
        })
    }

    pub fn moment(&self, ix: usize) -> Option<T> {
        self.moments.get(ix - 2).map(|m| *m)
    }
}

#[cfg(test)]
mod test {
    use super::MomentEstimator;

    #[test]
    fn standard_normal() {
        use rand::{distributions::Distribution, thread_rng};
        use statrs::distribution::Normal;
        let dist = Normal::new(0.0, 1.0).unwrap();

        let mut est = MomentEstimator::new(3);

        println!("{:?}", est);

        for sample in dist.sample_iter(thread_rng()).take(1_000_000) {
            est.update(sample);
        }

        println!("{:?}", est);

        println!(
            "{} {} {}",
            est.mean(),
            est.variance().unwrap(),
            est.skewness().unwrap()
        );
        assert!(est.mean().abs() < 1e-2);
        assert!((est.variance().unwrap().sqrt() - 1.0).abs() < 1e-2);
        assert!(est.skewness().unwrap().abs() < 1e-2);
    }

    #[test]
    fn exponential() {
        use rand::{distributions::Distribution, thread_rng};
        use statrs::distribution::Exponential;
        let dist = Exponential::new(1.0).unwrap();

        let mut est = MomentEstimator::new(3);

        println!("{:?}", est);

        for sample in dist.sample_iter(thread_rng()).take(1_000_000) {
            est.update(sample);
        }

        println!("{:?}", est);
        println!("{:?} {:?} {:?}", est.mean(), est.variance(), est.skewness());

        assert!((est.mean() - 1.0).abs() < 1e-2);
        assert!((est.variance().unwrap() - 1.0).abs() < 1e-2);
        assert!((est.skewness().unwrap() - 2.0).abs() < 1e-2);
    }
}
