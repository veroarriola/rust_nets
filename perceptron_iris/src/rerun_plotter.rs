use rerun::{Color, LineStrips2D, Points2D, RecordingStream};
use polars::prelude::*;
use crate::iris_dataset::{IrisClass, IrisDataset};

use std::error::Error;

/// Convierte una lista de coordenadas (x, y) en las líneas que forman un "Tache" (X)
fn generar_taches(puntos: &[(f32, f32)], tamano: f32) -> Vec<Vec<(f32, f32)>> {
    let mut strips = Vec::new();
    let d = tamano / 2.0;
    
    for &(x, y) in puntos {
        // Cada tache necesita dos líneas (diagonales cruzadas)
        strips.push(vec![(x - d, y - d), (x + d, y + d)]);
        strips.push(vec![(x - d, y + d), (x + d, y - d)]);
    }
    strips
}

/// Convierte una lista de coordenadas (x, y) en líneas cerradas que forman un Triángulo
fn generar_triangulos(puntos: &[(f32, f32)], tamano: f32) -> Vec<Vec<(f32, f32)>> {
    let mut strips = Vec::new();
    let d = tamano / 2.0;
    
    for &(x, y) in puntos {
        // Una sola línea continua que vuelve al punto de inicio
        strips.push(vec![
            (x, y + d),         // Punta superior
            (x - d, y - d),     // Esquina inferior izquierda
            (x + d, y - d),     // Esquina inferior derecha
            (x, y + d),         // Cierra arriba
        ]);
    }
    strips
}

/*
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Inicializar el motor de Rerun
    let rec = RecordingStreamBuilder::new("grafica_iris")
        .spawn()?;

    // Tus datos de ejemplo (podrían venir de tu DataFrame de Polars)
    let puntos_clase_1 = vec![(1.0, 1.0), (2.0, 2.0), (1.5, 1.2)];
    let puntos_clase_2 = vec![(4.0, 3.0), (3.5, 4.0), (4.2, 3.8)];

    let tamano_marcador = 0.3;

    // 2. Generar la geometría
    let taches = generar_taches(&puntos_clase_1, tamano_marcador);
    let triangulos = generar_triangulos(&puntos_clase_2, tamano_marcador);

    // 3. Mandar a graficar a Rerun
    rec.log(
        "dataset/setosa",
        &LineStrips2D::new(taches)
            .with_colors([Color::from_rgb(255, 50, 50)]), // Rojo
    )?;

    rec.log(
        "dataset/versicolor",
        &LineStrips2D::new(triangulos)
            .with_colors([Color::from_rgb(50, 255, 50)]), // Verde
    )?;

    Ok(())
}
*/

fn plot_2_characteristics(
    rec: &RecordingStream,
    df: &DataFrame,
    species_colors: &Vec<(&str, rerun::Color)>,
    feature_1_name: &str,
    feature_2_name: &str,
)  -> Result<(), Box<dyn Error>> {
    for (species_name, color) in species_colors {
        let mask = df.column("species")?.equal(*species_name)?;
        let filtered_df = df.filter(&mask)?;

        // --- Características ---
        let feature_1 = filtered_df.column(feature_1_name)?.f64()?;
        let feature_2 = filtered_df.column(feature_2_name)?.f64()?;
        
        let features_points: Vec<[f32; 2]> = feature_1
            .into_iter()
            .zip(feature_2.into_iter())
            .filter_map(|(l, w)| Some([l? as f32, w? as f32]))
            .collect();

        // En Rerun 0.35, el registro de arquetipos como Points2D funciona así de limpio
        rec.log(
            format!("dataset/{} (cm) vs {} (cm)/{}", feature_1_name, feature_2_name, species_name),
            &Points2D::new(features_points)
                .with_colors([*color])
                .with_radii([0.05]),
        )?;
    }

    Ok(())

}


pub fn plot_dataset(rec: &RecordingStream, ds: &IrisDataset, rerun_time: i64) -> Result<(), Box<dyn Error>> {

    let df = &ds.original_df;
    println!("AFTER: {df}");

    // Colores para diferenciar las especies
    let species_colors = vec![
        ("Iris-setosa", Color::from_rgb(255, 50, 50)),
        ("Iris-versicolor", Color::from_rgb(50, 255, 50)),
        ("Iris-virginica", Color::from_rgb(50, 100, 255)),
    ];

    // 4. Procesar y graficar
    //rec.set_duration_secs("stable_time", rerun_time);
    rec.set_time_sequence("stable_time", rerun_time);
    plot_2_characteristics(&rec, &df, &species_colors, "petal_length", "petal_width").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_length", "sepal_length").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_length", "sepal_width").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_width", "sepal_length").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_width", "sepal_width").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "sepal_length", "sepal_width").expect("Problem while plotting feature pairs to ReRun");

    Ok(())

}

pub fn plot_dataset_with_target(rec: &RecordingStream, ds: &IrisDataset, target_class: IrisClass, rerun_time: i64) -> Result<(), Box<dyn Error>> {

    let df = &ds.original_df;
    println!("AFTER: {df}");

    // Colores para diferenciar las especies
    let species_colors = vec![
        ("Iris-setosa", if target_class == IrisClass::Setosa { Color::from_rgb(255, 50, 50) } else { Color::from_rgb(100, 100, 100) }),
        ("Iris-versicolor", if target_class == IrisClass::Versicolour { Color::from_rgb(50, 255, 50) } else { Color::from_rgb(100, 100, 100) }),
        ("Iris-virginica", if target_class == IrisClass::Virginica { Color::from_rgb(50, 100, 255) } else { Color::from_rgb(100, 100, 100) }),
    ];

    // 4. Procesar y graficar
    //rec.set_duration_secs("stable_time", rerun_time);
    rec.set_time_sequence("stable_time", rerun_time);
    plot_2_characteristics(&rec, &df, &species_colors, "petal_length", "petal_width").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_length", "sepal_length").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_length", "sepal_width").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_width", "sepal_length").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "petal_width", "sepal_width").expect("Problem while plotting feature pairs to ReRun");
    plot_2_characteristics(&rec, &df, &species_colors, "sepal_length", "sepal_width").expect("Problem while plotting feature pairs to ReRun");

    Ok(())

}