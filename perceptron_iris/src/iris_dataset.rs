use polars::prelude::*;
//use std::error::Error;

use strum::EnumCount;
use strum_macros::EnumCount as EnumCountMacro;  // Se renombra para evitar conflictos de nombres
use strum_macros::EnumIter;
use serde::{Deserialize, Serialize};

use burn::tensor::{backend::Backend, Tensor, TensorData, Int};
use burn::data::{dataloader::batcher::Batcher, dataset::InMemDataset};

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use burn::data::dataloader::DataLoaderBuilder;
use std::sync::Arc;

use burn::data::dataloader::DataLoader;

pub const DATASET_SOURCE_FILE: &str = "data/iris.data";
pub const BATCH_SIZE: usize = 16;
pub const VALIDATION_INTERVAL: usize = 1;
pub const CHECKPOINT_INTERVAL: usize = 5;


#[derive(Debug, Copy, Clone, PartialEq, Eq, EnumIter, Deserialize, Serialize)]
pub enum IrisClass {
    Setosa = 0,
    Versicolour = 1,
    Virginica = 2,
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

/* 
 * Caracterísiticas
 */
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCountMacro)]
pub enum Feature {
    SepalLength = 0,
    SepalWidth = 1,
    PetalLength = 2,
    PetalWidth = 3,
}

impl Feature {
    pub fn from_str(name: &str) -> Self {
        match name {
            "sepal_length" => Self::SepalLength,
            "sepal_width"  => Self::SepalWidth,
            "petal_length" => Self::PetalLength,
            "petal_width"  => Self::PetalWidth,
            _ => panic!("Característica desconocida: '{name}'"),
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

pub const FEATURE_LABELS: [&str; Feature::COUNT] = [
    "sepal_length", 
    "sepal_width", 
    "petal_length", 
    "petal_width"
];


/*
 * Conjunto de datos para Polars
 */

#[derive(Debug, Clone)]
pub struct IrisDataset {
    pub original_df : DataFrame,
    pub original_vec : Vec<IrisItem>,
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
        let original_vec = Self::df_to_vec(&df_clean)?;
            
        Ok(Self { 
            original_df: df_clean,
            original_vec,
        })
    }

    fn df_to_vec(df_clean: &DataFrame) -> Result<Vec<IrisItem>, PolarsError> {
        //let df_clean = &self.original_df;
        let num_rows = df_clean.height();

        // 1. Extraer columnas (Cast a f32 explícito recomendado para evitar errores)
        let sl = df_clean.column("sepal_length")?.cast(&DataType::Float32)?.f32()?.clone();
        let sw = df_clean.column("sepal_width")?.cast(&DataType::Float32)?.f32()?.clone();
        let pl = df_clean.column("petal_length")?.cast(&DataType::Float32)?.f32()?.clone();
        let pw = df_clean.column("petal_width")?.cast(&DataType::Float32)?.f32()?.clone();
        
        // Nota: Dependiendo de la versión de Polars, puede ser .str() o .utf8()
        let species = df_clean.column("species")?.str()?.clone();

        // 2. Comprimir en un Vec<IrisItem>
        let mut items = Vec::with_capacity(num_rows);
        
        let it_sl = sl.into_no_null_iter();
        let it_sw = sw.into_no_null_iter();
        let it_pl = pl.into_no_null_iter();
        let it_pw = pw.into_no_null_iter();
        let it_sp = species.into_no_null_iter();

        for ((((sl_v, sw_v), pl_v), pw_v), sp_v) in it_sl.zip(it_sw).zip(it_pl).zip(it_pw).zip(it_sp) {
            // Mapeo directo de String a entero para CrossEntropyLoss
            let label = match sp_v {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                _ => 2, // Iris-virginica
            };

            items.push(IrisItem {
                features: [sl_v, sw_v, pl_v, pw_v],
                label,
            });
        }

        Ok(items)
    }
    
}


/*
 * Conjuntos de datos para Burn
 */

#[derive(Clone, Debug)]
pub struct IrisItem {
    pub features: [f32; 4],
    pub label: i32, // Burn usa enteros de 64 o 32 bits para clases
}

#[derive(Clone, Debug)]
pub struct IrisBatch<B: Backend> {
    // Matriz de [tamaño_lote, 4] con las medidas de la flor
    pub inputs: Tensor<B, 2>,
    
    // Matriz de [tamaño_lote, 1] con valores 1 (flor elegida) o 0 (las otras dos)
    pub targets: Tensor<B, 1, Int>, 
}

#[derive(Clone, Default)]
pub struct IrisBatcher {}

// El trait Batcher define cómo convertir un Vec<IrisItem> en un IrisBatch}
impl<B: Backend> Batcher<B, IrisItem, IrisBatch<B>> for IrisBatcher {
    fn batch(&self, items: Vec<IrisItem>, device: &B::Device) -> IrisBatch<B> {
        let batch_size = items.len();
        
        let mut features_vec = Vec::with_capacity(batch_size * Feature::COUNT);
        let mut targets_vec = Vec::with_capacity(batch_size);

        for item in items {
            features_vec.extend_from_slice(&item.features);
            targets_vec.push(item.label);
        }

        let features = Tensor::from_data(TensorData::new(features_vec, [batch_size, Feature::COUNT]), device);
        let targets = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(targets_vec, [batch_size]),
            device,
        );

        IrisBatch { inputs: features, targets }
    }
}

pub fn build_dataloaders<B: Backend>(
    mut items: Vec<IrisItem>,
    seed: u64
) -> PolarsResult<(
    Arc<dyn DataLoader<B, IrisBatch<B>>>, 
    Arc<dyn DataLoader<B, IrisBatch<B>>>
)> {
    // Barajar los datos usando la semilla proporcionada para reproducibilidad
    let mut rng = StdRng::seed_from_u64(seed);
    items.shuffle(&mut rng);

    // Dividir en Entrenamiento (80%) y Validación (20%)
    let split_idx = (items.len() as f32 * 0.8) as usize;
    let (train_items, val_items) = items.split_at(split_idx);

    // Convertir a Dataset de Burn
    let train_dataset = InMemDataset::new(train_items.to_vec());
    let val_dataset = InMemDataset::new(val_items.to_vec());

    // Construir los DataLoaders
    let batcher_train = IrisBatcher::default();
    let batcher_val = IrisBatcher::default();

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(BATCH_SIZE)
        .shuffle(seed) // Mezclado interno por epoch en el dataloader
        .build(train_dataset);

    let val_loader = DataLoaderBuilder::new(batcher_val)
        .batch_size(BATCH_SIZE)
        .build(val_dataset); // Validación usualmente no se baraja

    Ok((train_loader, val_loader))
}
