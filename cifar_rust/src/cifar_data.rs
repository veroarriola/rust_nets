use burn::tensor::{Tensor, TensorData, Int, backend::Backend};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::InMemDataset;
use std::fs;

// -.- Datos

// --- CONSTANTES DE ENTRENAMIENTO ---
pub const BATCH_SIZE: usize = 64;
pub const VALIDATION_INTERVAL: usize = 2; // Extraer métricas de Rerun cada 2 épocas
pub const CHECKPOINT_INTERVAL: usize = 5; // Guardar a disco cada 5 épocas
pub const NUM_CLASSES: usize = 10;        // Para la matriz de confusión
pub const TRAIN_DATASET_PATH: &str = "cifar10_images/train";
pub const TEST_DATASET_PATH: &str = "cifar10_images/test";

// 1. Definimos cómo se ve un elemento de nuestro dataset
#[derive(Clone, Debug)]
pub struct CifarItem {
    pub pixels: Vec<f32>, // Guardaremos la imagen aplanada
    pub label: i64,        // El índice de la clase (0 a 9)
}

// 2. Creamos nuestro clon de ImageFolder
pub fn load_cifar_folder(ruta_base: &str) -> InMemDataset<CifarItem> {
    let mut items = Vec::new();

    // CIFAR-10 carpetas
    let clases = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];
    //let class_names = ["avión", "auto", "ave", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"];

    println!("Cargando imágenes desde {}...", ruta_base);

    for (label_idx, clase) in clases.iter().enumerate() {
        let path_clase = format!("{}/{}", ruta_base, clase);

        // Leemos el directorio de cada clase
        match fs::read_dir(&path_clase) {
            Ok(entradas) => {
                for entrada in entradas.flatten() {
                    let path_archivo = entrada.path();

                    // Solo intentamos procesar si es un archivo (ignoramos subcarpetas ocultas)
                    if path_archivo.is_file() {
                        match image::open(&path_archivo) {
                            // Si es una imagen válida, la abrimos
                            Ok(img) => {
                                let rgb = img.to_rgb8();
                                let mut pixels_f32 = Vec::with_capacity(3 * 32 * 32);
                                
                                // Convertimos los píxeles (0-255) a floats normalizados (0.0 - 1.0)
                                for pixel in rgb.pixels() {
                                    pixels_f32.push(pixel[0] as f32 / 255.0); // R
                                    pixels_f32.push(pixel[1] as f32 / 255.0); // G
                                    pixels_f32.push(pixel[2] as f32 / 255.0); // B
                                }
                                
                                items.push(CifarItem {
                                    pixels: pixels_f32,
                                    label: label_idx as i64,
                                });
                            },
                            Err(e) => println!("Falló al leer la imagen {}: {:?}", path_archivo.display(), e),
                        }
                    }
                }
            },
            Err(e) => {
                eprintln!("Error al leer el directorio {}: {}", path_clase, e);
            }
        }
    }

    println!("¡Se cargaron {} imágenes!", items.len());
    InMemDataset::new(items) // Retornamos el dataset de Burn listo para usarse
}


// -.- Manejo de datos por lotes .-.

#[derive(Clone)]
pub struct CifarBatcher {}

#[derive(Clone, Debug)]
pub struct CifarBatch<B: Backend> {
    pub images: Tensor<B, 2>,         // [batch_size, 3072]
    pub targets: Tensor<B, 1, Int>,   // [batch_size]
}

impl<B: Backend> Batcher<B, CifarItem, CifarBatch<B>> for CifarBatcher {
    fn batch(&self, items: Vec<CifarItem>, device: &B::Device) -> CifarBatch<B> {
        let _batch_size = items.len();

        // Convertir imágenes a tensores, aplanar [3, 32, 32] -> [3072], y normalizar 0-1
        let images = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_data(TensorData::from(item.pixels.as_slice()), device))
            .map(|tensor| tensor.reshape([1, 3072])) // Aplanar
            //.map(|tensor| tensor / 255.0) // Normalizar f32
            .collect();

        let images = Tensor::cat(images, 0);

        // Convertir objetivos (targets)
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(TensorData::from([item.label as i32]), device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        CifarBatch { images, targets }
    }
}