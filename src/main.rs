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
    // list_near_ocean: f64,
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
    println!("Loaded {} rows", data.len());
    println!("{:?}", data[0]);

    // now constructing the in memory matrix

    let mut x_matrix = Matrix::new(data.len(), 12);

    for (i, e) in data.iter().enumerate() {
        let mut row = vec![1.0];
        row.extend(e.features());

        x_matrix.mat[i] = row;
    }
    let mut y_matrix = Matrix::new(data.len(), 1);

    for (i, e) in data.iter().enumerate() {
        y_matrix.mat[i][0] = e.target();
    }

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
    Ok(())
}
