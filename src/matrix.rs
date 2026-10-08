use std::fmt;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub n: usize,
    pub m: usize,
    pub mat: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            n,
            m,
            mat: vec![vec![0.0; m]; n],
        }
    }

    pub fn edit(&mut self, n: usize, m: usize) {
        self.n = n;
        self.m = m;
        self.mat = vec![vec![0.0; m]; n];
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.n {
            write!(f, "[ ")?;

            for j in 0..self.m {
                write!(f, "{:8.2}", self.mat[i][j])?;

                if j + 1 < self.m {
                    write!(f, " ")?;
                }
            }

            writeln!(f, " ]")?;
        }

        Ok(())
    }
}

pub struct MatrixOperations;

impl MatrixOperations {
    pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
        assert!(a.n == b.n && a.m == b.m);

        let mut result = Matrix::new(a.n, a.m);

        for i in 0..a.n {
            for j in 0..a.m {
                result.mat[i][j] = a.mat[i][j] + b.mat[i][j];
            }
        }

        result
    }

    pub fn subtract(a: &Matrix, b: &Matrix) -> Matrix {
        assert!(a.n == b.n && a.m == b.m);

        let mut result = Matrix::new(a.n, a.m);

        for i in 0..a.n {
            for j in 0..a.m {
                result.mat[i][j] = a.mat[i][j] - b.mat[i][j];
            }
        }

        result
    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
        assert!(a.m == b.n);

        let mut result = Matrix::new(a.n, b.m);

        for i in 0..a.n {
            for j in 0..b.m {
                for k in 0..a.m {
                    result.mat[i][j] += a.mat[i][k] * b.mat[k][j];
                }
            }
        }

        result
    }

    pub fn divide(a: &Matrix, b: &Matrix) -> Matrix {
        // Element-wise division
        assert!(a.n == b.n && a.m == b.m);

        let mut result = Matrix::new(a.n, a.m);

        for i in 0..a.n {
            for j in 0..a.m {
                result.mat[i][j] = a.mat[i][j] / b.mat[i][j];
            }
        }

        result
    }

    pub fn multiply_with_k(a: &Matrix, k: f64) -> Matrix {
        let mut result = Matrix::new(a.n, a.m);

        for i in 0..a.n {
            for j in 0..a.m {
                result.mat[i][j] = a.mat[i][j] * k;
            }
        }

        result
    }

    pub fn determinant(a: &Matrix) -> f64 {
        assert!(a.n == a.m);

        let n = a.n;

        if n == 1 {
            return a.mat[0][0];
        }

        if n == 2 {
            return a.mat[0][0] * a.mat[1][1] - a.mat[0][1] * a.mat[1][0];
        }

        let mut det = 0.0;

        for j in 0..n {
            let mut minor = Matrix::new(n - 1, n - 1);

            for i in 1..n {
                let mut col = 0;

                for k in 0..n {
                    if k == j {
                        continue;
                    }

                    minor.mat[i - 1][col] = a.mat[i][k];
                    col += 1;
                }
            }

            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };

            det += sign * a.mat[0][j] * Self::determinant(&minor);
        }

        det
    }

    pub fn inverse(a: &Matrix) -> Matrix {
        assert!(a.n == a.m, "Matrix must be square");

        let n = a.n;

        // [ A | I ]
        let mut aug = vec![vec![0.0; 2 * n]; n];

        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a.mat[i][j];
            }

            aug[i][n + i] = 1.0;
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find the row with the largest pivot.
            let mut pivot = i;

            for r in (i + 1)..n {
                if aug[r][i].abs() > aug[pivot][i].abs() {
                    pivot = r;
                }
            }

            // If the best pivot is effectively zero,
            // the matrix is singular.
            assert!(aug[pivot][i].abs() > 1e-12, "Matrix is not invertible");

            // Put the pivot row at position i.
            aug.swap(i, pivot);

            // Normalize pivot row so pivot becomes 1.
            let pivot_value = aug[i][i];

            for j in 0..(2 * n) {
                aug[i][j] /= pivot_value;
            }

            // Eliminate this column from every other row.
            for r in 0..n {
                if r == i {
                    continue;
                }

                let factor = aug[r][i];

                for j in 0..(2 * n) {
                    aug[r][j] -= factor * aug[i][j];
                }
            }
        }

        // Extract the right half: A⁻¹
        let mut result = Matrix::new(n, n);

        for i in 0..n {
            for j in 0..n {
                result.mat[i][j] = aug[i][n + j];
            }
        }

        result
    }

    pub fn transpose(a: &Matrix) -> Matrix {
        let mut result = Matrix::new(a.m, a.n);

        for i in 0..a.n {
            for j in 0..a.m {
                result.mat[j][i] = a.mat[i][j];
            }
        }

        result
    }
    pub fn standardize(x: &mut Matrix, means: &[f64], stds: &[f64]) {
        for i in 0..x.n {
            // start at 1 because column 0 is intercept
            for j in 1..x.m {
                x.mat[i][j] = (x.mat[i][j] - means[j - 1]) / stds[j - 1];
            }
        }
    }
    pub fn calculate_stats(x: &Matrix) -> (Vec<f64>, Vec<f64>) {
        let mut means = vec![0.0; x.m - 1];
        let mut stds = vec![0.0; x.m - 1];

        for j in 1..x.m {
            let mut sum = 0.0;

            for i in 0..x.n {
                sum += x.mat[i][j];
            }

            means[j - 1] = sum / x.n as f64;
        }

        for j in 1..x.m {
            let mut sum = 0.0;

            for i in 0..x.n {
                let diff = x.mat[i][j] - means[j - 1];
                sum += diff * diff;
            }

            stds[j - 1] = (sum / x.n as f64).sqrt();
        }

        (means, stds)
    }
}
