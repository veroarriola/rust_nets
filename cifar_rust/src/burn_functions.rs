use burn::tensor::backend::Backend; // Para poder usar B::seed()
use burn::record::CompactRecorder;
use burn::backend::{Wgpu, wgpu::WgpuDevice, Autodiff};
use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::record::Recorder;
use burn::prelude::Module;

use std::fs;
//use crate::training_state::{ToWorker, TrainingStatus, TrainingMeta, FromWorker};
use crate::training_state::{ToWorker, TrainingMeta, FromWorker};
use crate::cifar_data::{CifarBatcher, load_cifar_folder};
use crate::cifar_net::{Model};

// --- CONSTANTES DE ENTRENAMIENTO ---
const BATCH_SIZE: usize = 64;
const VALIDATION_INTERVAL: usize = 2; // Extraer métricas de Rerun cada 2 épocas
const CHECKPOINT_INTERVAL: usize = 5; // Guardar a disco cada 5 épocas
const NUM_CLASSES: usize = 10;        // Para la matriz de confusión

// Definimos el Backend con Autodiff para entrenamiento en GPU
type MyBackend = Autodiff<Wgpu>;

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub seed: u64,
    pub lr: f32,
    pub target_epochs: usize,
    pub validation_interval: usize,
}

pub fn worker_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<ToWorker>,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
) {
    println!("Iniciando Worker de Entrenamiento...");
    
    let rec = rerun::RecordingStreamBuilder::new("cifar10_mlp_manual")
        .spawn()
        .expect("Fallo al iniciar Rerun");

    let device = WgpuDevice::default();
    let dataset_train = load_cifar_folder("cifar10_images/train");
    let dataset_test = load_cifar_folder("cifar10_images/test");

    let dataloader_train = DataLoaderBuilder::new(CifarBatcher {})
        .batch_size(BATCH_SIZE)
        .shuffle(42) 
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(CifarBatcher {})
        .batch_size(BATCH_SIZE)
        .build(dataset_test);

    let mut model: Option<Model<MyBackend>> = None;
    //let mut optimizador: Option<burn::optim::OptimizerAdaptor<burn::optim::Adam, Model<MyBackend>, MyBackend>> = None;
    let mut optimizador: Option<_> = None;
    let criterion = CrossEntropyLossConfig::new().init(&device);

    let mut is_training = false;
    let mut current_epoch = 0;
    let mut target_epochs = 0;
    let mut current_lr = 0.001_f32;
    let mut current_seed = 42_u64; // <-- AÑADIDO: Necesitamos mantener la semilla en memoria

    loop {
        let msg = if is_training {
            rx.try_recv().ok() 
        } else {
            rx.blocking_recv() 
        };

        if let Some(cmd) = msg {
            match cmd {
                // CORREGIDO: Extraemos los valores del struct config
                ToWorker::Start(config) => {
                    is_training = true;
                    target_epochs = config.target_epochs;
                    current_lr = config.lr;
                    current_seed = config.seed;

                    if model.is_none() {
                        MyBackend::seed(&device, current_seed);
                        model = Some(Model::<MyBackend>::new(&device));
                        optimizador = Some(AdamConfig::new().init());
                        current_epoch = 0;
                    }
                }
                ToWorker::Pause => is_training = false,
                ToWorker::Stop => {
                    is_training = false;
                    model = None; 
                    optimizador = None;
                    current_epoch = 0;
                    let _ = tx.send(FromWorker::Finished);
                    continue;
                }
                ToWorker::LoadCheckpoint(path) => {
                    println!("Cargando checkpoint desde: {}", path);
                    
                    let meta_str = fs::read_to_string(format!("{}/meta.json", path))
                        .expect("No se encontró meta.json en el checkpoint");
                    let meta: TrainingMeta = serde_json::from_str(&meta_str).unwrap();

                    current_epoch = meta.epoch;
                    current_seed = meta.seed;
                    current_lr = meta.lr;
                    MyBackend::seed(&device, current_seed);

                    let recorder = CompactRecorder::new();
                    let record = recorder
                        .load(format!("{}/model", path).into(), &device)
                        .expect("No se pudieron cargar los pesos");

                    model = Some(Model::<MyBackend>::new(&device).load_record(record));
                    optimizador = Some(AdamConfig::new().init());

                    let _ = tx.send(FromWorker::CheckpointLoaded(meta));
                }
            }
        }

        if is_training {
            // 1. Cambiamos el if let por is_some()
            if model.is_some() && optimizador.is_some() {
                current_epoch += 1;
                let mut loss_total = 0.0;
                let mut n_batches = 0;

                rec.set_time_sequence("epoca", current_epoch as i64);

                for batch in dataloader_train.iter() {
                    while let Ok(cmd) = rx.try_recv() {
                        match cmd {
                            ToWorker::Pause => is_training = false,
                            ToWorker::Stop => {
                                is_training = false;
                                model = None; // ¡Ahora Rust sí te deja hacer esto!
                                optimizador = None;
                                current_epoch = 0;
                                let _ = tx.send(FromWorker::Finished);
                                break; 
                            }
                            ToWorker::Start(config) => current_lr = config.lr,
                            _ => {}
                        }
                    }

                    while !is_training && model.is_some() {
                        if let Some(cmd) = rx.blocking_recv() {
                            match cmd {
                                ToWorker::Start(config) => {
                                    is_training = true;
                                    current_lr = config.lr;
                                }
                                ToWorker::Stop => {
                                    is_training = false;
                                    model = None;
                                    optimizador = None;
                                    let _ = tx.send(FromWorker::Finished);
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }

                    if model.is_none() { break; }

                    // 2. Extraemos las referencias JUSTO antes de usarlas
                    let m = model.as_mut().unwrap();
                    let opt = optimizador.as_mut().unwrap();

                    let logits = m.forward(batch.images);
                    let loss = criterion.forward(logits, batch.targets);
                    
                    loss_total += loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                    n_batches += 1;

                    let grads = loss.backward();
                    let grads_params = burn::optim::GradientsParams::from_grads(grads, m);
                    
                    *m = opt.step(current_lr as f64, m.clone(), grads_params);
                } // ¡Aquí termina el alcance de 'm' y 'opt', devolviendo el control!

                if model.is_some() {
                    let train_loss_media = loss_total / n_batches as f32;
                    let _ = rec.log("metricas/loss_train", &rerun::Scalars::new([train_loss_media as f64]));

                    let _ = tx.send(FromWorker::EpochDone {
                        epoch: current_epoch,
                        loss: train_loss_media,
                    });

                    // --- FASE DE VALIDACIÓN CONDICIONADA ---
                    if current_epoch % VALIDATION_INTERVAL == 0 {
                        let mut val_loss_total = 0.0;
                        let mut val_batches = 0;
                        let mut confusion_matrix = vec![0u32; NUM_CLASSES * NUM_CLASSES]; 

                        // 3. Tomamos una referencia inmutable solo para evaluar
                        let m = model.as_ref().unwrap();

                        for batch in dataloader_test.iter() {
                            let logits = m.forward(batch.images);
                            let loss = criterion.forward(logits.clone(), batch.targets.clone());
                            
                            val_loss_total += loss.into_data().to_vec::<f32>().unwrap()[0];
                            val_batches += 1;

                            let predictions = logits.argmax(1).into_data().to_vec::<i32>().unwrap();
                            let targets = batch.targets.into_data().to_vec::<i32>().unwrap();

                            for (pred, target) in predictions.iter().zip(targets.iter()) {
                                let t = *target as usize;
                                let p = *pred as usize;
                                confusion_matrix[t * NUM_CLASSES + p] += 1;
                            }
                        }

                        let val_loss_media = val_loss_total / val_batches as f32;
                        let _ = rec.log("metricas/loss_val", &rerun::Scalars::new([val_loss_media as f64]));

                        let shape = vec![NUM_CLASSES as u64, NUM_CLASSES as u64];
                        rec.log(
                            "evaluacion/matriz_confusion",
                            &rerun::Tensor::new(rerun::TensorData::new(
                                shape,
                                rerun::TensorBuffer::U32(confusion_matrix.into())
                            ))
                        );
                    }

                    // --- GUARDADO DE CHECKPOINT CONDICIONADO ---
                    if current_epoch % CHECKPOINT_INTERVAL == 0 {
                        let dir_path = format!("checkpoints/epoch_{}", current_epoch);
                        fs::create_dir_all(&dir_path).expect("Fallo al crear directorio de checkpoint");

                        let recorder = CompactRecorder::new();
                        recorder
                            .record(
                                model.as_ref().unwrap().clone().into_record(),
                                format!("{}/model", dir_path).into(),
                            )
                            .expect("Fallo al guardar los pesos del modelo");

                        let meta = TrainingMeta {
                            epoch: current_epoch,
                            seed: current_seed,
                            lr: current_lr,
                        };
                        
                        let meta_file = std::fs::File::create(format!("{}/meta.json", dir_path)).unwrap();
                        serde_json::to_writer_pretty(meta_file, &meta).expect("Fallo al escribir meta.json");

                        let _ = tx.send(FromWorker::CheckpointSaved {
                            path: dir_path,
                            epoch: current_epoch,
                        });
                    }

                    if current_epoch >= target_epochs {
                        is_training = false; 
                        let _ = tx.send(FromWorker::Finished);
                    }
                }
            }
        }
    }
}
