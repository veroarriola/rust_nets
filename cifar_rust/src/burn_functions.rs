use burn::tensor::backend::{Backend}; // Para poder usar B::seed()
use burn::record::{CompactRecorder, Recorder};
use burn::backend::{Wgpu, wgpu::WgpuDevice, Autodiff};
use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::{CrossEntropyLossConfig, CrossEntropyLoss};
use burn::optim::{AdamConfig, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::Module;

use std::fs;
use indicatif::ProgressBar;

use std::sync::Arc;
use burn::data::dataloader::DataLoader;

use crate::training_state::{ToWorker, TrainingMeta, FromWorker, WorkerConfig};
use crate::cifar_data::{CifarBatcher, load_cifar_folder, CifarBatch};
use crate::cifar_net::{Model};

use crate::cifar_data::{
    BATCH_SIZE,
    VALIDATION_INTERVAL,
    CHECKPOINT_INTERVAL,
    NUM_CLASSES,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
};

// Definimos el Backend con Autodiff para entrenamiento en GPU
type MyBackend = Autodiff<Wgpu>;
pub type MyOptimizer = OptimizerAdaptor<
    burn::optim::Adam,
    Model<MyBackend>,   // trait: burn::module::AutodiffModule<AutodiffBackend>
    MyBackend,          // trait: AutodiffBackend
>;

struct TrainingState {
    device: WgpuDevice,
    is_training: bool,
    current_epoch: usize,
    target_epochs: usize,
    current_lr: f32,
    current_seed: u64,      // Necesitamos mantener la semilla en memoria
    reset_dataloader: bool, // Para usar la semilla
    model: Option<Model<MyBackend>>,
    optimizador: Option<MyOptimizer>,
    criterion: CrossEntropyLoss<MyBackend>,
}

impl TrainingState {
    fn start(&mut self, config: WorkerConfig) {
        self.is_training = true;
        self.target_epochs = config.target_epochs;
        self.current_lr = config.lr;
        
        // Si la UI manda una semilla distinta (o es la primera vez), forzamos recarga
        if self.current_seed != config.seed {
            self.current_seed = config.seed;
            self.reset_dataloader = true;
        }
    }

    fn init_model(&mut self) {
        MyBackend::seed(&self.device, self.current_seed);
        self.model = Some(Model::<MyBackend>::new(&self.device));
        self.optimizador = Some(AdamConfig::new().init());
        self.current_epoch = 0;
    }

    fn pause(&mut self) {
        self.is_training = false;
    }

    fn stop(&mut self) {
        self.is_training = false;
        self.model = None; 
        self.optimizador = None;
        self.current_epoch = 0;
        self.reset_dataloader = true; // Si damos Stop, el próximo Start debe ser limpio
    }

    fn load_checkpoint(&mut self, path: String) -> TrainingMeta {
        println!("Cargando checkpoint determinista desde: {}", path);
        
        let meta_str = fs::read_to_string(format!("{}/meta.json", path))
            .expect("No se encontró meta.json en el checkpoint");
        let meta: TrainingMeta = serde_json::from_str(&meta_str).unwrap();

        self.current_epoch = meta.epoch;
        self.current_seed = meta.seed;
        self.current_lr = meta.lr;
        self.is_training = false;
        self.reset_dataloader = true; // Forzamos usar la semilla del checkpoint

        // Cargar Modelo
        let recorder = CompactRecorder::new();
        let model_record = recorder
            .load(format!("{}/model", path).into(), &self.device)
            .expect("No se pudieron cargar los pesos");

        self.model = Some(Model::<MyBackend>::new(&self.device).load_record(model_record));

        // Cargar Optimizador (Restauramos momentum m, varianza v y contador t)
        let optim_record = recorder
            .load(format!("{}/optim", path).into(), &self.device)
            .expect("No se pudo cargar el estado del optimizador");
        self.optimizador = Some(AdamConfig::new().init().load_record(optim_record));

        return meta;
    }

    fn save_checkpoint(&mut self) -> String {
        // Incluimos semilla y learning rate en el nombre de la carpeta
        let dir_path = format!(
            "checkpoints/seed_{}_lr_{}_epoch_{}",
            self.current_seed,
            self.current_lr,
            self.current_epoch);
        fs::create_dir_all(&dir_path).expect("Fallo al crear directorio de checkpoint");

        let recorder = CompactRecorder::new();

        // 1. Guardamos el Modelo
        recorder
            .record(
                self.model.as_ref().unwrap().clone().into_record(),
                format!("{}/model", dir_path).into(),
            )
            .expect("Fallo al guardar los pesos del modelo");

        // 2. Guardamos el Optimizador (Adam: momentum m, varianza v y paso t)
        recorder
            .record(
                self.optimizador.as_ref().unwrap().to_record(),
                format!("{}/optim", dir_path).into(),
            )
            .expect("Fallo al guardar el estado del optimizador");

        let meta = TrainingMeta {
            epoch: self.current_epoch,
            seed: self.current_seed,
            lr: self.current_lr,
        };
        
        let meta_file = std::fs::File::create(format!("{}/meta.json", dir_path)).unwrap();
        serde_json::to_writer_pretty(meta_file, &meta).expect("Fallo al escribir meta.json");

        return dir_path
    }
}

fn rec_confusion_matrix(
    label: &str,
    m: &Model<MyBackend>,
    dataloader: &Arc<dyn DataLoader<MyBackend, CifarBatch<MyBackend>>>,
    training_state: &TrainingState,
    rec: &rerun::RecordingStream,
) {
    let mut loss_total = 0.0;
    let mut batches = 0;
    let mut confusion_matrix = vec![0u32; NUM_CLASSES * NUM_CLASSES];

    for batch in dataloader.iter() {
        let logits = m.forward(batch.images);
        let loss = training_state.criterion.forward(logits.clone(), batch.targets.clone());
        
        loss_total += loss.into_data().to_vec::<f32>().unwrap()[0];
        batches += 1;

        let predictions = logits.argmax(1).into_data().to_vec::<i32>().unwrap();
        let targets = batch.targets.into_data().to_vec::<i32>().unwrap();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let t = *target as usize;
            let p = *pred as usize;
            confusion_matrix[t * NUM_CLASSES + p] += 1;
        }
    }

    let loss_path = format!("metricas/seed_{}/loss_{label}", training_state.current_seed);
    let val_loss_media = loss_total / batches as f32;
    let _ = rec.log(loss_path, &rerun::Scalars::new([val_loss_media as f64]));

    let matrix_path = format!("evaluacion/seed_{}/matriz_confusion_{label}", training_state.current_seed);
    let shape = vec![NUM_CLASSES as u64, NUM_CLASSES as u64];
    let _ = rec.log(
        matrix_path,
        &rerun::Tensor::new(rerun::TensorData::new(
            shape,
            rerun::TensorBuffer::U32(confusion_matrix.into())
        ))
    );
}

pub fn worker_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<ToWorker>,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
) {
    println!("Iniciando Worker de Entrenamiento...");
    
    // Inicializar Rerun
    let rec = rerun::RecordingStreamBuilder::new("cifar10_mlp_manual")
        .spawn()
        .expect("Fallo al iniciar Rerun");

    // Carga del dataset de entrenamiento y prueba
    let dataset_test = load_cifar_folder(TEST_DATASET_PATH);

    // El conjunto de entrenamiento inicia vacío para asignar después la semilla
    let mut dataloader_train = None;

    let dataloader_test = DataLoaderBuilder::new(CifarBatcher {})
        .batch_size(BATCH_SIZE)
        .build(dataset_test);

    let device = WgpuDevice::default();    // Configurar Dispositivo (GPU por defecto en Wgpu)
    let mut training_state = TrainingState {
        device: device.clone(),
        is_training: false,
        current_epoch: 0,
        target_epochs: 0,
        current_lr: 0.001_f32,
        current_seed: 0,           // <-- No importa, se sobrescribirá
        reset_dataloader: false,   // <-- Inicializamos la bandera
        model: None,
        optimizador: None,
        criterion: CrossEntropyLossConfig::new().init(&device),
    };
    println!("Entrenando en: {:?}", training_state.device);

    loop {
        // 1. Destrucción segura fuera del iterador
        if training_state.reset_dataloader {
            dataloader_train = None;
            training_state.reset_dataloader = false;
        }

        let msg = if training_state.is_training {
            rx.try_recv().ok() 
        } else {
            rx.blocking_recv() 
        };

        if let Some(cmd) = msg {
            match cmd {
                // Extraemos los valores del struct config
                ToWorker::Start(config) => {
                    training_state.start(config);

                    if training_state.model.is_none() {
                        training_state.init_model();
                    }

                    println!("Iniciando entrenamiento...");
                }
                ToWorker::Pause => training_state.pause(),
                ToWorker::Stop => {
                    training_state.stop();

                    let _ = tx.send(FromWorker::Finished);
                    continue;
                }
                ToWorker::LoadCheckpoint(path) => {
                    let meta = training_state.load_checkpoint(path);

                    let _ = tx.send(FromWorker::CheckpointLoaded(meta));
                }
                ToWorker::Exit => {
                    println!("Vaciando buffers de Rerun...");
                    // Esto fuerza a Rerun a enviar cualquier tensor pendiente por la red
                    // antes de destruir la conexión.
                    let _ = rec.flush_blocking(); 

                    // Le avisamos a la UI que ya terminamos de limpiar
                    let _ = tx.send(FromWorker::WorkerExited);
                    break;
                }
            }
        }

        if training_state.is_training {
            if training_state.model.is_some() && training_state.optimizador.is_some() {
                // 2. Construcción Diferida con la semilla correcta
                if dataloader_train.is_none() {
                    // Lo cargamos fresco cada vez que necesitemos reiniciar el iterador
                    let dataset_train_fresco = load_cifar_folder(TRAIN_DATASET_PATH);
                    
                    dataloader_train = Some(
                        DataLoaderBuilder::new(CifarBatcher {})
                            .batch_size(BATCH_SIZE)
                            .shuffle(training_state.current_seed)
                            .build(dataset_train_fresco) 
                    );
                }

                training_state.current_epoch += 1;

                let mut loss_total = 0.0;
                let mut n_batches = 0;

                rec.set_time_sequence("epoca", training_state.current_epoch as i64);

                let pb = ProgressBar::new(dataloader_train.as_ref().unwrap().num_items() as u64 / BATCH_SIZE as u64);
                pb.set_message(format!("Época {}", training_state.current_epoch));

                // Justo antes del for, calculas el total de batches (si no lo tenías ya)
                let total_batches = dataloader_train.as_ref().unwrap().num_items() / BATCH_SIZE;

                // Le ponemos etiqueta al ciclo para poder matarlo desde adentro
                'epoch_loop: for batch in dataloader_train.as_ref().unwrap().iter() {
                    n_batches += 1;

                    // BUCLE 1: Mensajes sobre la marcha
                    while let Ok(cmd) = rx.try_recv() {
                        match cmd {
                            ToWorker::Pause => training_state.pause(),
                            ToWorker::Stop => {
                                training_state.stop();
                                let _ = tx.send(FromWorker::Finished);
                                break 'epoch_loop;
                            }
                            ToWorker::Start(config) => {
                                training_state.start(config);

                                // Si cambiaron la semilla mientras entrenaba, abortamos 
                                // la época para reiniciar el iterador de datos.
                                if training_state.reset_dataloader {
                                    break 'epoch_loop;
                                }
                            }
                            ToWorker::LoadCheckpoint(path) => {
                                let meta = training_state.load_checkpoint(path);
                                let _ = tx.send(FromWorker::CheckpointLoaded(meta));
                                
                                break 'epoch_loop; // Abortamos el lote actual para iniciar limpios
                            }
                            ToWorker::Exit => {
                                let _ = rec.flush_blocking();
                                let _ = tx.send(FromWorker::WorkerExited);
                                return;
                            }
                        }
                    }

                    // BUCLE 2: Espera bloqueante cuando está pausado
                    while !training_state.is_training && training_state.model.is_some() {
                        if let Some(cmd) = rx.blocking_recv() {
                            match cmd {
                                ToWorker::Start(config) => {
                                    training_state.start(config);

                                    // Si cambiaron la semilla mientras entrenaba, abortamos 
                                    // la época para reiniciar el iterador de datos.
                                    if training_state.reset_dataloader {
                                        break 'epoch_loop;
                                    }
                                }
                                ToWorker::Stop => {
                                    training_state.stop();
                                    
                                    let _ = tx.send(FromWorker::Finished);
                                    break 'epoch_loop;
                                }
                                ToWorker::LoadCheckpoint(path) => {
                                    let meta = training_state.load_checkpoint(path);

                                    let _ = tx.send(FromWorker::CheckpointLoaded(meta));
                                    break 'epoch_loop;
                                }
                                ToWorker::Exit => {
                                    let _ = rec.flush_blocking();
                                    let _ = tx.send(FromWorker::WorkerExited);
                                    return;
                                }
                                _ => {}
                            }
                        }
                    }

                    if training_state.model.is_none() { break; }

                    // 2. Extraemos las referencias JUSTO antes de usarlas
                    let m = training_state.model.as_mut().unwrap();
                    let opt = training_state.optimizador.as_mut().unwrap();

                    // 1. Forward pass
                    let logits = m.forward(batch.images);

                    // 2. Calcular Pérdida (Cross Entropy)
                    let loss = training_state.criterion.forward(logits, batch.targets);

                    // Extraer escalar para estadística
                    loss_total += loss.clone().into_data().to_vec::<f32>().unwrap()[0];

                    // 3. Backward pass (Calcular Gradientes)
                    let grads = loss.backward();

                    // Mapear los gradientes al modelo
                    let grads_params = burn::optim::GradientsParams::from_grads(grads, m);
                    
                    // 4. Actualizar parámetros del optimizador
                    *m = opt.step(training_state.current_lr as f64, m.clone(), grads_params);

                    let _ = tx.send(FromWorker::BatchProgress {
                        epoch: training_state.current_epoch,
                        current_batch: n_batches,
                        total_batches,
                    });
                    pb.inc(1);
                } // ¡Aquí termina el alcance de 'm' y 'opt', devolviendo el control!
                pb.finish_with_message(format!("Época {} completada", training_state.current_epoch));

                if training_state.model.is_some() && n_batches > 0 {
                    // --- VISUALIZACIÓN 1: Pérdida media de la época ---
                    let train_loss_media = loss_total / n_batches as f32;
                    let loss_path = format!("metricas/seed_{}/loss_train", training_state.current_seed);
                    let _ = rec.log(loss_path, &rerun::Scalars::new([train_loss_media as f64]));
                    println!("Época {}: Loss Media = {:.4}", training_state.current_epoch, train_loss_media);

                    let _ = tx.send(FromWorker::EpochDone {
                        epoch: training_state.current_epoch,
                        loss: train_loss_media,
                    });

                    // --- FASE DE VALIDACIÓN CONDICIONADA ---
                    if training_state.current_epoch % VALIDATION_INTERVAL == 0 {
                        rec_confusion_matrix(
                            "train",
                            &training_state.model.as_ref().unwrap(),
                            &dataloader_train.as_ref().unwrap(),
                            &training_state,
                            &rec,
                        );
                        rec_confusion_matrix(
                            "val",
                            &training_state.model.as_ref().unwrap(),
                            &dataloader_test,
                            &training_state,
                            &rec,
                        );
                    }

                    // --- GUARDADO DE CHECKPOINT CONDICIONADO ---
                    if training_state.current_epoch % CHECKPOINT_INTERVAL == 0 || training_state.current_epoch == 1 {
                        let dir_path = training_state.save_checkpoint();

                        let _ = tx.send(FromWorker::CheckpointSaved {
                            path: dir_path,
                            epoch: training_state.current_epoch,
                        });

                        // --- Visualizaciones pesadas (cada N épocas) ---
                        println!("==> Extrayendo datos para Rerun...");

                        // --- VISUALIZACIÓN: Pesos como Imágenes ---
                        // Tomamos la referencia del modelo
                        let m = training_state.model.as_ref().unwrap();

                        for (nombre_capa, tensor_pesos) in m.obtener_pesos() {
                            // Extraemos los pesos de la capa. Ej: capa 1 (shape: [1024, 3072])
                            // Usamos inner() para obtener el backend Wgpu puro sin capa Autodiff
                            let data_pesos = tensor_pesos.clone().inner().into_data().to_vec::<f32>().unwrap();
                            
                            // Obtener las dimensiones reales de la matriz
                            let shape: Vec<u64> = tensor_pesos.shape().dims::<2>()
                                .into_iter()
                                .map(|d| d as u64)
                                .collect();
                            
                            // Aquí mandas "datos" a Rerun construyendo la ruta de forma dinámica:
                            let ruta_rerun = format!("visualizacion/seed_{}/pesos/pesos_{}_heatmap", training_state.current_seed, nombre_capa);

                            // Interpretamos la matriz de [1024, 3072] como una imagen.
                            // Rerun es inteligente: si le das [H, W] f32, muestra un mapa de calor.
                            let _ = rec.log(
                                ruta_rerun.as_str(),
                                &rerun::Tensor::new(rerun::TensorData::new(
                                    shape, //vec![1024_u64, 3072_u64],
                                    rerun::TensorBuffer::F32(data_pesos.into())
                                ))
                            );
                        }
                    }

                    if training_state.current_epoch >= training_state.target_epochs {
                        training_state.is_training = false; 
                        let _ = tx.send(FromWorker::Finished);

                        println!("Entrenamiento finalizado.");
                    }
                }
            }
        }
    }
}
