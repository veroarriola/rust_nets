use polars::prelude::*;
use std::error::Error;

use strum_macros::EnumIter;
use serde::{Deserialize, Serialize};

use burn::tensor::{backend::Backend, Tensor, TensorData};


pub const DATASET_SOURCE_FILE: &str = "data/iris.data";


#[derive(Debug, Copy, Clone, PartialEq, Eq, EnumIter, Deserialize, Serialize)]
pub enum IrisClass {
    Setosa,
    Versicolour,
    Virginica,
}

impl std::fmt::Display for IrisClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::Setosa => "Setosa",
            Self::Versicolour => "Versicolour",
            Self::Virginica => "Virginica",
        };
        write!(f, "{}", text)
    }
}

impl IrisClass {
    pub fn target_name(&self) -> String {
        match self {
            Self::Setosa => String::from("Iris-setosa"),
            Self::Versicolour => String::from("Iris-versicolour"),
            Self::Virginica => String::from("Iris-virginica"),
        }
    }
}


#[derive(Clone, Debug)]
pub struct IrisBatch<B: Backend> {
    // Matriz de [tamaño_lote, 4] con las medidas de la flor
    pub inputs: Tensor<B, 2>,
    
    // Matriz de [tamaño_lote, 1] con valores 1.0 (flor elegida) o 0.0 (las otras dos)
    pub targets: Tensor<B, 2>, 
}

#[derive(Debug, Clone)]
pub struct IrisDataset {
    pub original_df : DataFrame,
}

impl IrisDataset {

    pub fn new(csv_path: &str) -> Result<Self, PolarsError> {
        let mut df = CsvReader::from_path(csv_path)?
            .has_header(false)
            .finish()?;
        df.set_column_names(&[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ])?;

        // 1. Quitar todos los renglones que contengan valores nulos
        // `None` significa que checará todas las columnas
        let df_clean = df.drop_nulls::<String>(None)?;
            
        Ok(Self { 
            original_df: df_clean,
        })
    }
    
}


// ¿Ya no?

pub fn load_dataset_for_burn<B: Backend>(
    csv_path: &str, 
    target_flower: &str, 
    device: &B::Device
) -> Result<IrisBatch<B>, Box<dyn Error>> {
    
    // 1. LEER Y TRANSFORMAR CON POLARS
    // Usamos LazyFrame para procesar los datos eficientemente
    let df = LazyCsvReader::new(csv_path)
        .has_header(false)
        .finish()?
        // 2. Renombramos las columnas genéricas por las correctas
        .rename(
            ["column_1", "column_2", "column_3", "column_4", "column_5"],
            ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        )
        // Lógica Uno-contra-el-Resto: Si es la flor elegida = 1.0, si no = 0.0
        .with_column(
            when(col("species").eq(lit(target_flower)))
            .then(lit(1.0f32))
            .otherwise(lit(0.0f32))
            .alias("label")
        )
        .collect()?; // Ejecutamos la transformación

    let num_rows = df.height();

    // 2. EXTRAER LOS DATOS HACIA VECTORES DE RUST
    // Extraemos las 4 columnas de características (features)
    let features_df = df.select(["sepal_length", "sepal_width", "petal_length", "petal_width"])?;
    
    // Convertimos el DataFrame a un vector plano (flatten) de f32
    // Polars guarda los datos en columnas, así que iteramos fila por fila
    let mut inputs_vec = Vec::with_capacity(num_rows * 4);
    for i in 0..num_rows {
        // En un proyecto real, se extrae mediante ndarray o iteradores por columna para mayor velocidad,
        // pero esto es muy ilustrativo para una demo.
        let sl: f32 = features_df.column("sepal_length")?.f32()?.get(i).unwrap_or(0.0);
        let sw: f32 = features_df.column("sepal_width")?.f32()?.get(i).unwrap_or(0.0);
        let pl: f32 = features_df.column("petal_length")?.f32()?.get(i).unwrap_or(0.0);
        let pw: f32 = features_df.column("petal_width")?.f32()?.get(i).unwrap_or(0.0);
        inputs_vec.extend_from_slice(&[sl, sw, pl, pw]);
    }

    // Extraemos los objetivos (targets: 1.0 o 0.0)
    let targets_series = df.column("label")?.f32()?;
    let targets_vec: Vec<f32> = targets_series.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    // 3. CONVERTIR A TENSORES DE BURN
    // TensorData necesita los datos planos y la forma (shape) de la matriz
    let input_data = TensorData::new(inputs_vec, [num_rows, 4]);
    let target_data = TensorData::new(targets_vec, [num_rows, 1]);

    // Creamos los tensores enviándolos a la CPU o GPU (device)
    let inputs = Tensor::<B, 2>::from_data(input_data, device);
    let targets = Tensor::<B, 2>::from_data(target_data, device);

    // Devolvemos el lote empaquetado
    Ok(IrisBatch { inputs, targets })
}