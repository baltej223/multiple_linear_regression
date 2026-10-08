use matrix::Matrix;
mod matrix;

use serde::Deserialize;

use crate::matrix::MatrixOperations;

#[derive(Debug, Deserialize)]
struct HousingRow {
    longitude: f64,
    latitude: f64,
    housing_median_age: f64,
    total_rooms: f64,
    total_bedrooms: f64,
    population: f64,
    households: f64,
    median_income: f64,
    median_house_value: f64,
    is_near_bay: f64,
    is_1hour_ocean: f64,
    is_inland: f64,
    // list_near_ocean: f64, // cause of dummy encoding.
}

impl HousingRow {
    pub fn features(&self) -> Vec<f64> {
        vec![
            self.longitude,
            self.latitude,
            self.housing_median_age,
            self.total_rooms,
            self.total_bedrooms,
            self.population,
            self.households,
            self.median_income,
            self.is_near_bay,
            self.is_1hour_ocean,
            self.is_inland,
            // self.list_near_ocean,
        ]
    }

    pub fn target(&self) -> f64 {
        self.median_house_value
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path("processed.csv")?;
    let mut data: Vec<HousingRow> = Vec::new();

    for result in reader.deserialize() {
        let row: HousingRow = result?;
        data.push(row);
    }

    let split = (data.len() as f64 * 0.8) as usize;

    let train_data = &data[..split];
    let test_data = &data[split..];

    println!("Total: {}", data.len());
    println!("Training: {}", train_data.len());
    println!("Testing: {}", test_data.len());

    let mut x_matrix = Matrix::new(train_data.len(), 12);
    for (i, e) in train_data.iter().enumerate() {
        let mut row = vec![1.0];
        row.extend(e.features());
        x_matrix.mat[i] = row;
    }
    let mut y_matrix = Matrix::new(train_data.len(), 1);

    for (i, e) in train_data.iter().enumerate() {
        y_matrix.mat[i][0] = e.target();
    }
    //
    // let (means, stds) = MatrixOperations::calculate_stats(&x_matrix);
    //
    // MatrixOperations::standardize(&mut x_matrix, &means, &stds);
    // MatrixOperations::standardize(&mut y_matrix, &means, &stds);
    //
    let xt = MatrixOperations::transpose(&x_matrix);
    println!("Some done!:1");
    println!("XT: {} × {}", xt.n, xt.m);
    let mut a = MatrixOperations::multiply(&xt, &x_matrix);
    println!("XTX: {} × {}", a.n, a.m);
    println!("{}", a);
    a = MatrixOperations::inverse(&a);
    println!("Some done!:3");
    let b = MatrixOperations::multiply(&xt, &y_matrix);
    println!("Some done!:4");
    let coeff_matrix = MatrixOperations::multiply(&a, &b);
    println!("Doneee!:5");
    print!("{:?}", coeff_matrix.mat);

    // Now test loop.
    let mut y_actual = Matrix::new(test_data.len(), 1);
    let mut y_predicted = Matrix::new(test_data.len(), 1);

    for (i, element) in test_data.iter().enumerate() {
        let mut row = vec![1.0];
        row.extend(element.features());

        let mut x = Matrix::new(1, 12);
        x.mat[0] = row;

        let predicted = predict(&coeff_matrix, &x).mat[0][0];
        let actual = element.target();

        y_actual.mat[i][0] = actual;
        y_predicted.mat[i][0] = predicted;

        let error = actual - predicted;

        println!(
            "actual: {:.2}, predicted: {:.2}, error: {:.2}",
            actual, predicted, error
        );
    }

    let rmse_score = rmse(&y_actual, &y_predicted);
    let mae_score = mae(&y_actual, &y_predicted);
    let r2_score = r2(&y_actual, &y_predicted);

    println!("RMSE: {:.2}", rmse_score);
    println!("MAE:  {:.2}", mae_score);
    println!("R²:   {:.4}", r2_score);

    Ok(())
}

fn predict(coeff_m: &Matrix, new_matrix: &Matrix) -> Matrix {
    MatrixOperations::multiply(new_matrix, coeff_m)
}

// vibecoded:
fn rmse(actual: &Matrix, predicted: &Matrix) -> f64 {
    assert_eq!(actual.n, predicted.n);
    assert_eq!(actual.m, 1);
    assert_eq!(predicted.m, 1);

    let mut sum_squared_error = 0.0;

    for i in 0..actual.n {
        let error = actual.mat[i][0] - predicted.mat[i][0];

        sum_squared_error += error * error;
    }

    (sum_squared_error / actual.n as f64).sqrt()
}

fn mae(actual: &Matrix, predicted: &Matrix) -> f64 {
    assert_eq!(actual.n, predicted.n);
    assert_eq!(actual.m, 1);
    assert_eq!(predicted.m, 1);

    let mut total_error = 0.0;

    for i in 0..actual.n {
        let error = actual.mat[i][0] - predicted.mat[i][0];

        total_error += error.abs();
    }

    total_error / actual.n as f64
}
fn r2(actual: &Matrix, predicted: &Matrix) -> f64 {
    assert_eq!(actual.n, predicted.n);
    assert_eq!(actual.m, 1);
    assert_eq!(predicted.m, 1);

    // Calculate mean of actual values
    let mut mean = 0.0;

    for i in 0..actual.n {
        mean += actual.mat[i][0];
    }

    mean /= actual.n as f64;

    // Calculate:
    //
    // SS_res = Σ(y - ŷ)²
    //
    // SS_tot = Σ(y - mean(y))²

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for i in 0..actual.n {
        let y = actual.mat[i][0];
        let y_hat = predicted.mat[i][0];

        ss_res += (y - y_hat).powi(2);
        ss_tot += (y - mean).powi(2);
    }

    1.0 - (ss_res / ss_tot)
}
