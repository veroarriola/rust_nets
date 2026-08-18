

use burn::tensor::backend::Backend;
/* Conjuntos de datos */
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use burn::data::dataloader::DataLoader;

use crate::iris_dataset;
use crate::iris_dataset::{IrisBatch, IrisClass, IrisDataset};

/* Visualización */
use rerun::RecordingStream;
use crate::rerun_plotter;
use crate::rerun_plotter::ClassificationPlotter;

/* Modelo */
use crate::burn_perceptron::Perceptron;
use burn::train::TrainStep;
use burn::train::InferenceStep;

/* Entrenamiento */
use burn::backend::{Wgpu, wgpu::WgpuDevice, Autodiff};
use burn::optim::Optimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use crate::iris_dataset::{
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    VALIDATION_INTERVAL,
};

/* Persistencia */
use std::fs;
use burn::prelude::Module;
use burn::record::{CompactRecorder, Recorder};


/*
 * Comunicación entre hilos
 */
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    // Cuando el hilo arranca, nos entrega el "transmisor" para enviarle comandos
    Ready(tokio::sync::mpsc::UnboundedSender<ToWorker>),
    // Actualizaciones de estado desde el worker
    Update(FromWorker),
}

#[derive(Debug, Clone)]
pub enum ToWorker {
    TargetSelected(IrisClass),
    Start(TrainingConfig),
    Pause,
    Stop,
    LoadCheckpoint(String),
    Exit,
}

#[derive(Debug, Clone)]
pub enum FromWorker {
    EpochDone { epoch: usize, loss: f32 },
    CheckpointSaved { path: String, epoch: usize },
    TrainingFinished,
    Error(String),
    CheckpointLoaded(TrainingConfig),
    WorkerExited,
    // Para enviar el progreso de la época actual
    BatchProgress { 
        epoch: usize, 
        current_batch: usize, 
        total_batches: usize 
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingConfig {
    // Clase objetivo
    pub target_class: IrisClass,
    pub seed: u64,
    pub lr: f64,
    pub target_epochs: usize,
    pub current_epoch: usize,
    pub current_batch: usize,
    //pub validation_interval: usize,
}


// Definimos el Backend con Autodiff para entrenamiento en GPU
pub type MyBackend = Autodiff<Wgpu>;

pub type MyOptimizer = OptimizerAdaptor<
    burn::optim::Adam,
    //burn::optim::Sgd<MyBackend>,
    Perceptron<MyBackend>,   // trait: burn::module::AutodiffModule<AutodiffBackend>
    MyBackend,               // trait: AutodiffBackend
>;

struct Trainer {
    // Rerun recording stream
    rec: Option<RecordingStream>,
    rerun_time: i64,
    // Conjunto de datos original (sin filtrar)
    original_dataset: Option<IrisDataset>,
    
    device: WgpuDevice,
    model: Option<Perceptron<MyBackend>>,
    optimizer: MyOptimizer,

    train_data_loader: Option<Arc<dyn DataLoader<MyBackend, IrisBatch<MyBackend>>>>,
    val_data_loader: Option<Arc<dyn DataLoader<MyBackend, IrisBatch<MyBackend>>>>,

    plot_at_start: bool,
}

impl Trainer {
    fn new(rec: Option<RecordingStream>) -> Self {
        // Dentro de una función async (worker_loop)
        let device = WgpuDevice::default();
        Self {
            rec: rec,
            rerun_time: 0,
            original_dataset: None,

            device: device.clone(),
            model: None,
            //optim: burn::optim::SgdConfig::new()
            //    .init::<MyBackend, Perceptron<MyBackend>>(),
            optimizer: burn::optim::AdamConfig::new().init::<MyBackend, Perceptron<MyBackend>>(),
            
            train_data_loader: None,
            val_data_loader: None,

            plot_at_start: true,
        }
    }

    fn start(&mut self, training_config: &TrainingConfig) {
        if self.train_data_loader.is_none() || self.val_data_loader.is_none() || self.model.is_none() {
            MyBackend::seed(&self.device, training_config.seed);

            self.model = Some(Perceptron::new(&self.device));

            let (train_data_loader, val_data_loader) = iris_dataset::build_dataloaders::<MyBackend>(
                self.original_dataset.as_ref().unwrap().original_vec.clone(),
                training_config.seed,
                training_config.target_class,
            ).unwrap(); // Aseguramos que el dataset original esté cargado
            
            self.train_data_loader = Some(train_data_loader);
            self.val_data_loader = Some(val_data_loader);
        }
        
        if self.plot_at_start {
            self.plot_dataset_with_target(training_config.target_class);
        }
        // Aquí podrías reiniciar el optimizador si es necesario
        // self.optim = burn::optim::AdamConfig::new().init::<MyBackend, Perceptron<MyBackend>>();
    }

    fn stop(&mut self) {
        // Parece ser que no hace falta nada por ahora.
        // Quiero conservar el estado del generador aleatorio y los datos, por si se ejecuta otra ronda con los mismos
        self.model = None;
        self.train_data_loader = None;
        self.val_data_loader = None;
    }

    fn validate(&self) -> f32 {
        if let Some(data_loader) = &self.val_data_loader {
            let total_batches = data_loader.num_items();

            let mut total_loss = 0.0;

            for batch in data_loader.iter() {
                let output = InferenceStep::step(self.model.as_ref().expect("Ejecutando validación sin ejecutar Start"), batch);
                total_loss += output.loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            }
            total_loss / total_batches as f32
        } else {
            panic!("Se intentó validar sin haber iniciado entrenamiento");
        }
    }

    /*
     * Carga los datos en memoria, pero no crea conjuntos para entrenamiento.
     */
    fn load_dataset(&mut self) -> Result<(), String> {
        // Cargar conjunto de datos
        match IrisDataset::new(iris_dataset::DATASET_SOURCE_FILE) {
            Ok(dataset) => {
                self.original_dataset = Some(dataset);
                Ok(())
            },
            Err(e) => {
                Err(format!("⚠️ Error al cargar iris.csv: {}", e))
            },
        }
    }

    fn load_checkpoint(&mut self) {
        
    }

    fn save_checkpoint(&mut self, training_config: &TrainingConfig) -> String {
        // Incluimos semilla y learning rate en el nombre de la carpeta
        let dir_path = format!(
            "checkpoints/target_{}/lr_{}/seed_{}/epoch_{}",
            training_config.target_class.target_name(),
            training_config.lr,
            training_config.seed,
            training_config.current_epoch);
        fs::create_dir_all(&dir_path).expect("Fallo al crear directorio de checkpoint");

        let recorder = CompactRecorder::new();

        // 1. Guardamos el Modelo
        recorder
            .record(
                self.model.as_ref().expect("Ejecutando save_checkpoint sin ejecutar Start").clone().into_record(),
                format!("{}/model", dir_path).into(),
            )
            .expect("Fallo al guardar los pesos del modelo");

        // 2. Guardamos el Optimizador (Adam: momentum m, varianza v y paso t)
        recorder
            .record(
                self.optimizer.to_record(),
                format!("{}/optimizer", dir_path).into(),
            )
            .expect("Fallo al guardar el estado del optimizador");
        
        let meta_file = std::fs::File::create(format!("{}/meta.json", dir_path)).unwrap();
        serde_json::to_writer_pretty(meta_file, &training_config).expect("Fallo al escribir meta.json");

        return dir_path
    }

    fn plot_original_dataset(&mut self) {
        if let Some(rec) = &self.rec {
            if let Some(dataset) = &self.original_dataset {
                if let Err(e) = rerun_plotter::plot_dataset(&rec, &dataset, self.rerun_time) {
                    println!("⚠️ Fallo al graficar datos: {}", e);
                }
                self.rerun_time += 1;
            }
        }
    }

    fn plot_dataset_with_target(&mut self, target_class: IrisClass) {
        if let Some(rec) = &self.rec {
            if let Some(dataset) = &self.original_dataset {
                if let Err(e) = rerun_plotter::plot_dataset_with_target(&rec, &dataset, target_class, self.rerun_time) {
                    println!("⚠️ Fallo al graficar datos con clase objetivo: {}", e);
                }
                self.rerun_time += 1;
                self.plot_at_start = false;
            }
        }
    }

    fn plot_current_classification_status(&mut self, training_config: &TrainingConfig) {
        if let Some(rec) = &self.rec {
            let mut plotter = ClassificationPlotter::new(String::from("train"));

            if let Some(train_data_loader) = &self.train_data_loader {
                for batch in train_data_loader.iter() {
                    // Hacemos una copia superficial del handle de inputs (ultrarrápida y sin costo de RAM/GPU)
                    let inputs = batch.inputs.clone();
                    let output = InferenceStep::step(self.model.as_ref().expect("Graficando sin ejecutar Start"), batch);
                    plotter.accumulate_batch(&output, &inputs);
                }
            }
            plotter.plot_accuracy(rec, training_config);
            plotter.plot_combinations(&rec, &iris_dataset::FEATURE_LABELS, &training_config.target_class.target_name());

            plotter = ClassificationPlotter::new(String::from("val"));
            if let Some(val_data_loader) = &self.val_data_loader {
                for batch in val_data_loader.iter() {
                    let inputs = batch.inputs.clone();
                    let output = InferenceStep::step(self.model.as_ref().expect("Graficando sin ejecutar Start"), batch);
                    plotter.accumulate_batch(&output, &inputs);
                }
            }
            plotter.plot_accuracy(rec, training_config);
            plotter.plot_combinations(&rec, &iris_dataset::FEATURE_LABELS, &training_config.target_class.target_name());
            //self.rerun_time += 1;
        }
    }
}



pub async fn worker_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<ToWorker>,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
    rec: Option<RecordingStream>,
) {
    // Se ejecuta cuando main se suscribe al trabajador.
    println!("Trabajador iniciado");

    let mut trainer = Trainer::new(rec);

    // Cargar conjunto de datos
    if let Err(e) = trainer.load_dataset() {
        let _ = tx.send(FromWorker::Error(e));
        let _ = tx.send(FromWorker::WorkerExited);
        return;
    }

    // Graficar conjunto de datos original
    trainer.plot_original_dataset();

    let mut stop_training = false;
    while let Some(msg) = rx.recv().await {
        if let ToWorker::TargetSelected(target_class) = msg {
            println!("Trabajador recibió clase objetivo: {:?}", target_class);
            trainer.stop();
            stop_training = false;
            trainer.plot_dataset_with_target(target_class);
        }
        else if let ToWorker::LoadCheckpoint(_string) = msg {

        }
        // Si rec es None no se graficará el progreso
        else if let ToWorker::Start(mut training_config) = msg {
            println!("Trabajador iniciando entrenamiento...");
            trainer.start(&training_config);  // Inicia el dataloader
            //let mut last_persisted_epoch = training_config.current_epoch;
            
            // Bucle manual de épocas
            'epoch_loop: while training_config.current_epoch < training_config.target_epochs {
                training_config.current_epoch += 1;

                let train_data_loader = trainer.train_data_loader.as_mut().unwrap();
                let total_batches = train_data_loader.num_items() / BATCH_SIZE;

                // Iteramos sobre los lotes (asumiendo que ya se tiene el dataloader)
                let mut total_loss = 0.0;
                let mut n_batches = 0;
                for batch in train_data_loader.iter() {
                    n_batches += 1;

                    // Revisamos si el usuario envió mensajes
                    if let Ok(msg) = rx.try_recv() {
                        if let ToWorker::Pause = msg {
                            // Transmitimos estado actual para ser guardado en la IGU
                            break 'epoch_loop;
                        }
                        else if let ToWorker::Stop = msg {
                            println!("Entrenamiento cancelado por el usuario.");
                            // Transmitimos estado actual para ser guardado en la IGU
                            let _ = tx.send(FromWorker::BatchProgress {
                                //epoch: last_persisted_epoch,
                                epoch: 0,
                                current_batch: 0,
                                total_batches,
                            });
                            stop_training = true;
                            break 'epoch_loop; // Rompes el ciclo de entrenamiento
                        }
                        else if let ToWorker::Exit = msg {
                            println!("Trabajador saliendo por petición del usuario.");
                            let _ = tx.send(FromWorker::WorkerExited);
                            break 'epoch_loop;
                        }
                    }
                    
                    // grads, (output: loss, predictions, batch.targets)
                    let output = TrainStep::step(trainer.model.as_ref().expect("Entrenando sin ejecutar Start"), batch);
                    //print!("Salidas: {}\nObjetivos: {}\n", output.item.output, output.item.targets);
                    total_loss += output.item.loss.clone().into_data().to_vec::<f32>().unwrap()[0];

                    trainer.model = Some(trainer.optimizer.step(training_config.lr, trainer.model.expect("Entrenando sin iniciar."), output.grads));

                    let _ = tx.send(FromWorker::BatchProgress {
                        epoch: training_config.current_epoch,
                        current_batch: n_batches,
                        total_batches,
                    });
                }

                // --- VISUALIZACIÓN 1: Pérdida media de la época ---
                let average_train_loss = total_loss / n_batches as f32;
                

                if let Some(rec) = &trainer.rec {
                    trainer.rerun_time += 1;
                    rec.set_time_sequence("stable_time", trainer.rerun_time);
                    rec.set_time_sequence("epoca", training_config.current_epoch as i64);

                    let loss_path = format!(
                        "metrics/loss_train/target_{}/lr_{}/seed_{}",
                        training_config.target_class.target_name(),
                        training_config.lr,
                        training_config.seed,
                    );
                    let _ = rec.log(loss_path, &rerun::Scalars::new([average_train_loss as f64]));
                    rerun_plotter::graficar_parametros(&rec, trainer.model.as_ref().expect("Graficando durante entrenamiento"), &iris_dataset::FEATURE_LABELS);
                }

                println!("Época {}: Loss Media = {:.4}", training_config.current_epoch, average_train_loss);

                let _ = tx.send(FromWorker::EpochDone {
                    epoch: training_config.current_epoch,
                    loss: average_train_loss,
                });


                // Validación
                if training_config.current_epoch % VALIDATION_INTERVAL == 0 {
                    let val_error = trainer.validate();
                    println!("     Validation error = {:.4}", val_error);

                    if let Some(rec) = &trainer.rec {
                        let loss_path = format!(
                            "metrics/loss_val/target_{}/lr_{}/seed_{}",
                            training_config.target_class.target_name(),
                            training_config.lr,
                            training_config.seed,
                        );
                        let _ = rec.log(loss_path, &rerun::Scalars::new([val_error as f64]));
                    }

                    trainer.plot_current_classification_status(&training_config);
                }

                // --- INTEGRACIÓN CON RERUN E ICED ---
                // Aquí tienes acceso directo al modelo en cada época
                // Puedes extraer los pesos y el sesgo sin pelear con el Learner:
                
                //let pesos = trainer.model.linear.weight.val().into_data().convert::<f32>().value;
                //let sesgo = trainer.model.linear.bias.unwrap().val().into_data().convert::<f32>().value;
                
            }
            if stop_training {
                trainer.stop();
                stop_training = false;
            }
            
            println!("Trabajador terminó entrenamiento.");
            let _ = tx.send(FromWorker::TrainingFinished);
        }
        else if let ToWorker::Exit = msg {
            println!("Trabajador sin trabajo saliendo.");
            let _ = tx.send(FromWorker::WorkerExited);
            break;
        }
    }
    println!("Trabajador terminado.");
    let _ = tx.send(FromWorker::WorkerExited);
}