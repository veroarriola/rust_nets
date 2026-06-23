use burn::backend::{Wgpu, wgpu::WgpuDevice, Autodiff};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
//use burn::data::dataset::vision::{Cifar10Dataset, Cifar10Item};
//use burn::data::dataset::source::huggingface::HuggingfaceDataset;


use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

//use ndarray::Array2;
use indicatif::ProgressBar;


// -.- Datos

use burn::data::dataset::InMemDataset;
use std::fs;

// 1. Definimos cómo se ve un elemento de nuestro dataset
#[derive(Clone, Debug)]
pub struct CifarItem {
    pub pixels: Vec<f32>, // Guardaremos la imagen aplanada
    pub label: i64,        // El índice de la clase (0 a 9)
}

// 2. Creamos nuestro clon de ImageFolder
pub fn cargar_image_folder(ruta_base: &str) -> InMemDataset<CifarItem> {
    let mut items = Vec::new();

    // CIFAR-10 carpetas
    let clases = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];

    println!("Cargando imágenes desde {}...", ruta_base);

    for (label_idx, clase) in clases.iter().enumerate() {
        let path_clase = format!("{}/{}", ruta_base, clase);

        // Leemos el directorio de cada clase
        if let Ok(entradas) = fs::read_dir(&path_clase) {
            for entrada in entradas.flatten() {
                let path_archivo = entrada.path();

                // Si es una imagen válida, la abrimos
                if let Ok(img) = image::open(&path_archivo) {
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
                }
            }
        }
    }

    println!("¡Se cargaron {} imágenes!", items.len());
    InMemDataset::new(items) // Retornamos el dataset de Burn listo para usarse
}


// -.- Red densa .-.

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    linear_3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        // CIFAR-10: 32x32x3 = 3072 entradas. 10 clases de salida.
        let linear_1 = LinearConfig::new(3072, 1024).init(device);
        // Oculta: 1024 -> 512 (clases de CIFAR)
        let linear_2 = LinearConfig::new(1024, 512).init(device);
        // Oculta: 512 -> Salida: 10 (clases de CIFAR)
        let linear_3 = LinearConfig::new(512, 10).init(device);

        Self {
            linear_1,
            linear_2,
            linear_3,
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Capa 1 -> Activación Relu
        let x = self.linear_1.forward(input);
        let x = self.relu.forward(x);
        let x = self.linear_2.forward(x);
        let x = self.relu.forward(x);

        // Salida (logits)
        self.linear_3.forward(x)
    }
}


// -.- Manejo de datos por lotes .-.

#[derive(Clone)]
struct CifarBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
struct CifarBatch<B: Backend> {
    images: Tensor<B, 2>,         // [batch_size, 3072]
    targets: Tensor<B, 1, Int>,   // [batch_size]
}

impl<B: Backend> Batcher<B, CifarItem, CifarBatch<B>> for CifarBatcher<B> {
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


// -.- Entrenamiento .-.

// Definimos el Backend con Autodiff para entrenamiento en GPU
type MyBackend = Autodiff<Wgpu>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // a. Inicializar Rerun
    let rec = rerun::RecordingStreamBuilder::new("cifar10_mlp_manual").spawn()?;

    // Nombres de las clases de CIFAR-10 para la matriz de confusión
    //let class_names = ["avion", "auto", "ave", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camion"];

    // b. Configurar Dispositivo (GPU por defecto en Wgpu)
    let device = WgpuDevice::default();
    println!("Entrenando en: {:?}", device);

    // Cargas el dataset de entrenamiento y test
    let dataset_train = cargar_image_folder("cifar10_imagenes/train");
    let dataset_test = cargar_image_folder("cifar10_imagenes/test");

    // c. Preparar Datos

    let batch_size = 64;

    let dataloader_train = DataLoaderBuilder::new(CifarBatcher::<MyBackend> { device: device.clone() })
        .batch_size(batch_size)
        .shuffle(42)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(CifarBatcher::<MyBackend> { device: device.clone() })
        .batch_size(batch_size)
        .build(dataset_test);



    // d. Inicializar Modelo y Optimizador
    let mut model = Model::<MyBackend>::new(&device);
    let mut optimizador = AdamConfig::new().init();
    let criterion = CrossEntropyLossConfig::new().init(&device);

    // e. Bucle de Entrenamiento
    let num_epochs = 20;
    let visualizar_cada_n_epochs = 4; // Frecuencia para pesos y matriz de confusión

    println!("Iniciando entrenamiento manual...");

    for epoch in 1..=num_epochs {
        rec.set_time_sequence("epoca", epoch as i64);

        let mut loss_total = 0.0;
        let mut n_batches = 0;

        let pb = ProgressBar::new(dataloader_train.num_items() as u64 / batch_size as u64);
        pb.set_message(format!("Época {}", epoch));

        for batch in dataloader_train.iter() {
            // --- PASO DE ENTRENAMIENTO MANUAL ---

            // 1. Forward pass
            let logits = model.forward(batch.images);

            // 2. Calcular Pérdida (Cross Entropy)
            // burn::tensor::activation::softmax_cross_entropy_with_logits requiere targets como índices
            let loss = criterion.forward(logits, batch.targets);

            loss_total += loss.clone().into_data().to_vec::<f32>().unwrap()[0]; // Extraer escalar para estadística
            n_batches += 1;

            // 3. Backward pass (Calcular Gradientes)
            let grads = loss.backward();

            // Mapear los gradientes al modelo
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);

            // 4. Actualizar parámetros del optimizador
            model = optimizador.step(1e-3, model, grads);

            pb.inc(1);
        }
        pb.finish_with_message(format!("Época {} completada", epoch));

        // --- VISUALIZACIÓN 1: Pérdida media de la época ---
        let loss_media = loss_total / n_batches as f32;
        rec.log("metricas/loss", &rerun::Scalars::new([loss_media as f64]))?;
        println!("Época {}: Loss Media = {:.4}", epoch, loss_media);


        // --- Visualizaciones pesadas (cada N épocas) ---
        if epoch % visualizar_cada_n_epochs == 0 || epoch == 1 {
            println!("==> Extrayendo datos para Rerun...");

            // --- VISUALIZACIÓN 2: Pesos como Imágenes ---
            // Extraemos los pesos de la capa 1 (shape: [1024, 3072])
            // Usamos inner() para obtener el backend Wgpu puro sin autodiff
            let pesos_tensor = model.linear_1.weight.val().inner();
            let data_pesos = pesos_tensor.into_data().to_vec::<f32>().unwrap();

            // Interpretamos la matriz de [1024, 3072] como una imagen.
            // Rerun es inteligente: si le das [H, W] f32, muestra un mapa de calor.
            rec.log(
                "visualizacion/pesos_capa1_heatmap",
                &rerun::Tensor::new(rerun::TensorData::new(
                    vec![1024_u64, 3072_u64],
                    rerun::TensorBuffer::F32(data_pesos.into())
                ))
            )?;

            // --- VISUALIZACIÓN 3: Matriz de Confusión (Requiere validación) ---

            // Usamos un simple Vec plano de 100 elementos (10x10) inicializado en 0
            let mut confusion_matrix = vec![0.0_f32; 100];

            // Modo evaluación (importante si usaras Dropout o BatchNorm, aunque aquí no hay)
            // En Burn manual, simplemente no calculamos gradientes.

            for batch in dataloader_test.iter() {
                let logits = model.forward(batch.images);

                // Obtener predicciones (índice con valor máximo)
                let predictions = logits.argmax(1).squeeze::<1>(); // [batch_size]

                // Mover predicciones y targets a CPU para contar
                let preds_cpu = predictions.into_data().to_vec::<i32>().unwrap();
                let targets_cpu = batch.targets.into_data().to_vec::<i32>().unwrap();

                // Llenar matriz
                for i in 0..preds_cpu.len() {
                    let pred = preds_cpu[i] as usize;
                    let target = targets_cpu[i] as usize;

                    if pred < 10 && target < 10 {
                        // Mapeamos las coordenadas 2D (target, pred) al índice 1D
                        let index = target * 10 + pred;
                        confusion_matrix[index] += 1.0;
                    }
                }
            }

            // Registramos la matriz pasándole la forma [10, 10]
            rec.log(
                "metricas/matriz_confusion",
                &rerun::Tensor::new(rerun::TensorData::new(
                    vec![10_u64, 10_u64],
                    rerun::TensorBuffer::F32(confusion_matrix.into())
                ))
            )?;
        }
    }

    println!("Entrenamiento finalizado.");
    Ok(())
}
