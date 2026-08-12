use rerun::{Clear, Color, LineStrips2D, Points2D, RecordingStream};
use polars::prelude::*;
use crate::iris_dataset::{IrisClass, IrisDataset, Feature, FEATURE_LABELS};

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::train::ClassificationOutput;
use crate::training_worker::MyBackend;

use std::error::Error;

const POINT_RADIUS: f32 = 0.03;
const MARKER_SIZE: f32 = 0.1;

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
                .with_radii([POINT_RADIUS]),
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
    let num_features = FEATURE_LABELS.len();
    for i in 0..num_features {
        for j in (i + 1)..num_features {
            let label_x = FEATURE_LABELS[i];
            let label_y = FEATURE_LABELS[j];
            plot_2_characteristics(&rec, &df, &species_colors, label_x, label_y).expect("Problem while plotting feature pairs to ReRun");
        }
    }

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
    let num_features = FEATURE_LABELS.len();
    for i in 0..num_features {
        for j in (i + 1)..num_features {
            let label_x = FEATURE_LABELS[i];
            let label_y = FEATURE_LABELS[j];
            plot_2_characteristics(&rec, &df, &species_colors, label_x, label_y).expect("Problem while plotting feature pairs to ReRun");
        }
    }

    Ok(())
}

pub fn pairwise_plot_classification_batch(
    rec: &rerun::RecordingStream, 
    output: &ClassificationOutput<MyBackend>,
    inputs: &Tensor<MyBackend, 2>,
    target_name: &str,
    feature_1_name: &str,
    feature_2_name: &str,
) {
    let idx_x = Feature::from_str(feature_1_name).index();  // <--- Índice para el eje X
    let idx_y = Feature::from_str(feature_2_name).index();  // <--- Índice para el eje Y

    // 1. Extraer los datos
    let preds_data = output.output.clone().into_data();
    let preds: &[f32] = preds_data.as_slice::<f32>().unwrap();

    let targets_data = output.targets.clone().into_data();
    let targets: &[i32] = targets_data.as_slice::<i32>().unwrap();

    let inputs_data = inputs.clone().into_data();
    let features: &[f32] = inputs_data.as_slice::<f32>().unwrap();

    let mut correct_points = Vec::new();
    let mut incorrect_points = Vec::new();

    let batch_size = preds.len();

    // 2. Evaluar cada elemento usando los índices seleccionados
    for i in 0..batch_size {
        let pred_label = if preds[i] >= 0.5 { 1 } else { 0 };
        let true_label = targets[i];
        let is_correct = pred_label == true_label;

        // Seleccionamos las características dinámicamente
        let x = features[i * 4 + idx_x]; 
        let y = features[i * 4 + idx_y]; 

        if is_correct {
            correct_points.push((x, y));
        } else {
            incorrect_points.push((x, y));
        }
    }

    // 3. Generar las figuras
    let size = 0.1;
    let triangulos = generar_triangulos(&correct_points, size);
    let taches = generar_taches(&incorrect_points, size);

    // 4. Crear rutas dinámicas para Rerun
    // Ejemplo: "dataset/0_vs_1/correcta"
    let target_path = format!("dataset/{} (cm) vs {} (cm)/{}", feature_1_name, feature_2_name, target_name);

    // 5. Graficar
    if !triangulos.is_empty() {
        rec.log(
            target_path.clone(),
            &LineStrips2D::new(triangulos)
                .with_colors([Color::from_rgb(0, 255, 0)]),
        ).unwrap();
    }

    if !taches.is_empty() {
        rec.log(
            target_path,
            &LineStrips2D::new(taches)
                .with_colors([Color::from_rgb(255, 0, 0)]),
        ).unwrap();
    }
}

pub fn plot_classification_batch(rec: &RecordingStream, output: &ClassificationOutput<MyBackend>, inputs: &Tensor<MyBackend, 2>, target_class: IrisClass) {
    let binding = target_class.target_name();
    let target_name = binding.as_str();
    let num_features = FEATURE_LABELS.len();
    for i in 0..num_features {
        for j in (i + 1)..num_features {
            let label_x = FEATURE_LABELS[i];
            let label_y = FEATURE_LABELS[j];
            pairwise_plot_classification_batch(rec, &output, &inputs, &target_name, label_x, label_y);
        }
    }
}



pub struct ClassificationPlotter {
    epoch_features: Vec<f32>, // Guardará TODAS las dimensiones aplanadas
    epoch_is_correct: Vec<bool>, // Guardará si el modelo atinó o no
}

impl ClassificationPlotter {
    pub fn new() -> Self {
        let epoch_features = Vec::new();
        let epoch_is_correct = Vec::new();
        Self {
            epoch_features,
            epoch_is_correct,
        }
    }

    pub fn accumulate_batch<B: Backend>(
        &mut self,
        output: &ClassificationOutput<B>,
        inputs: &Tensor<B, 2>,
    ) {
        let preds_data = output.output.clone().into_data();
        let preds: &[f32] = preds_data.as_slice::<f32>().unwrap();

        let targets_data = output.targets.clone().into_data();
        let targets: &[i32] = targets_data.as_slice::<i32>().unwrap(); 

        let inputs_data = inputs.clone().into_data();
        let features_slice: &[f32] = inputs_data.as_slice::<f32>().unwrap();

        let batch_size = preds.len();

        // 1. Guardamos todos los features del lote tal cual (copia rápida en RAM)
        self.epoch_features.extend_from_slice(features_slice);

        // 2. Evaluamos la corrección de cada elemento y lo guardamos
        for i in 0..batch_size {
            let pred_label = if preds[i] >= 0.5 { 1 } else { 0 };
            let is_correct = pred_label == targets[i];
            self.epoch_is_correct.push(is_correct);
        }
    }

    pub fn plot_combinations(
        &mut self,
        rec: &rerun::RecordingStream,
        labels: &[&str; 4], // Tus 4 etiquetas (ej: ["sepal_length", ...])
        target_name: &str,
    ) {
        let num_features = labels.len();
        let size = MARKER_SIZE;

        // Iteramos para generar los 6 pares únicos: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        for i in 0..num_features {
            for j in (i + 1)..num_features {
                let mut correct_points = Vec::new();
                let mut incorrect_points = Vec::new();

                // Clasificamos los puntos para este par específico (i vs j)
                for (idx, &correct) in self.epoch_is_correct.iter().enumerate() {
                    let x = self.epoch_features[idx * num_features + i];
                    let y = self.epoch_features[idx * num_features + j];

                    if correct {
                        correct_points.push((x, y));
                    } else {
                        incorrect_points.push((x, y));
                    }
                }

                // Construimos las rutas de Rerun usando tus etiquetas de texto
                let label_x = labels[i];
                let label_y = labels[j];
                let path_correcta = format!("dataset/{} (cm) vs {} (cm)/correct/{}", label_x, label_y, target_name);
                let path_incorrecta = format!("dataset/{} (cm) vs {} (cm)/incorrect/{}", label_x, label_y, target_name);

                // Aciertos
                let triangles = generar_triangulos(&correct_points, size);
                if triangles.is_empty() {
                    rec.log(path_correcta.clone(), &Clear::flat()).unwrap();
                } else {
                    rec.log(path_correcta.clone(), &LineStrips2D::new(triangles).with_colors([Color::from_rgb(0, 255, 0)])).unwrap();
                }

                // Errores
                let crosses = generar_taches(&incorrect_points, size);
                if crosses.is_empty() {
                    rec.log(path_incorrecta.clone(), &Clear::flat()).unwrap();
                } else {
                    rec.log(path_incorrecta.clone(), &LineStrips2D::new(crosses).with_colors([Color::from_rgb(255, 0, 0)])).unwrap();
                }
            }
        }
    }
}