// Red
use burn::{
    //config,
    module::Module,
    nn::{Linear, LinearConfig}, tensor::{Tensor, activation::sigmoid, backend::Backend}
};

// Entrenamiento
use burn::{
    //nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    nn::loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig},
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, InferenceStep, ClassificationOutput,},
    //optim::Optimizer,
};

use burn::backend::{Wgpu, wgpu::WgpuDevice, Autodiff};
use burn::optim::adaptor::OptimizerAdaptor;

use rerun::RecordingStream;

use serde::{Deserialize, Serialize};

use crate::iris_dataset;
use crate::iris_dataset::{IrisBatch, IrisClass, IrisDataset};
use crate::rerun_plotter;


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
    pub target_class: IrisClass,
    pub seed: u64,
    pub lr: f32,
    pub target_epochs: usize,
    pub validation_interval: usize,
}



/*
 * Perceptrón
 */
#[derive(Module, Debug)]
pub struct Perceptron<B: Backend> {
    // La capa lineal maneja los pesos (W) y el sesgo (b)
    pub linear: Linear<B>,
    pub loss_fn: BinaryCrossEntropyLoss<B>,
}

impl<B: Backend> Perceptron<B> {
    /// Inicializa el perceptrón en el dispositivo especificado (CPU o GPU)
    pub fn new(device: &B::Device) -> Self {
        // 4 entradas (características de Iris) y 1 salida (clasificación binaria)
        let linear = LinearConfig::new(4, 1).init(device);
        let loss_fn = BinaryCrossEntropyLossConfig::new().init(device);
        Self { linear, loss_fn }
    }

    /// Pasada hacia adelante (Forward pass)
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        // Multiplica por los pesos, suma el sesgo y aplica la función sigmoide
        let x = self.linear.forward(inputs);
        sigmoid(x)
    }
}

/*
 * Entrenamiento
 */

// AutodiffBackend es crucial aquí porque necesitamos calcular gradientes
impl<B: AutodiffBackend> TrainStep for Perceptron<B> {
    type Input = IrisBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let predictions = self.forward(batch.inputs);
        let batch_size = batch.targets.dims()[0];

        // Convertimos targets a Float y a 2D: [batch_size, 1] para complacer al Loss
        let targets_2d = batch.targets.clone().reshape([batch_size, 1]);

        let loss = self.loss_fn
            // Clonamos predictions y targets si es necesario para usarlos en ClassificationOutput
            .forward(predictions.clone(), targets_2d);

        let grads = loss.backward();

        // 2. Empaquetamos los datos en ClassificationOutput
        let output = ClassificationOutput::new(loss, predictions, batch.targets);

        // 3. TrainOutput recibe el módulo, los gradientes y nuestro 'output' de métricas
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for Perceptron<B> {
    type Input = IrisBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let predictions = self.forward(batch.inputs);
        let batch_size = batch.targets.dims()[0];

        let targets_2d = batch.targets.clone().reshape([batch_size, 1]);
        let loss = self.loss_fn.forward(predictions.clone(), targets_2d);

        // En inferencia/validación solo devolvemos las métricas (sin gradientes)
        ClassificationOutput::new(loss, predictions, batch.targets)
    }
}


const RERUN_TIME_DELTA: f32 = 0.2;  // segundos
// Definimos el Backend con Autodiff para entrenamiento en GPU
type MyBackend = Autodiff<Wgpu>;

pub type MyOptimizer = OptimizerAdaptor<
    burn::optim::Adam,
    //burn::optim::Sgd<MyBackend>,
    Perceptron<MyBackend>,   // trait: burn::module::AutodiffModule<AutodiffBackend>
    MyBackend,               // trait: AutodiffBackend
>;

struct Trainer {
    // Rerun recording stream
    rec: Option<RecordingStream>,
    rerun_time: f32,
    // Conjunto de datos original (sin filtrar)
    original_dataset: Option<IrisDataset>,
    // Clase objetivo
    target_class: IrisClass,

    model: Perceptron<MyBackend>,
    optim: MyOptimizer,

    current_epoch: usize,
}

impl Trainer {
    fn new(rec: Option<RecordingStream>) -> Self {
        // Dentro de una función async (worker_loop)
        let device = WgpuDevice::default();
        Self {
            rec: rec,
            rerun_time: 0.0,
            original_dataset: None,
            target_class: IrisClass::Setosa, // Valor por defecto igual que el de la IU
            model: Perceptron::new(&device),
            current_epoch: 0,
            //optim: burn::optim::SgdConfig::new()
            //    .init::<MyBackend, Perceptron<MyBackend>>(),
            optim: burn::optim::AdamConfig::new().init::<MyBackend, Perceptron<MyBackend>>(),
        }
    }
    
    fn stop(&mut self) {
        self.current_epoch = 0;
    }

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

    fn plot_original_dataset(&mut self) {
        if let Some(rec) = &self.rec {
            if let Some(dataset) = &self.original_dataset {
                if let Err(e) = rerun_plotter::plot_dataset(&rec, &dataset, self.rerun_time) {
                    println!("⚠️ Fallo al graficar datos: {}", e);
                }
                self.rerun_time += RERUN_TIME_DELTA;
            }
        }
    }

    fn plot_dataset_with_target(&mut self) {
        if let Some(rec) = &self.rec {
            if let Some(dataset) = &self.original_dataset {
                if let Err(e) = rerun_plotter::plot_dataset_with_target(&rec, &dataset, self.target_class, self.rerun_time) {
                    println!("⚠️ Fallo al graficar datos con clase objetivo: {}", e);
                }
                self.rerun_time += RERUN_TIME_DELTA;
            }
        }
    }

    fn set_target_class(&mut self, target_class: IrisClass) {
        self.target_class = target_class;
        self.plot_dataset_with_target();
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

    while let Some(msg) = rx.recv().await {
        if let ToWorker::TargetSelected(target_class) = msg {
            println!("Trabajador recibió clase objetivo: {:?}", target_class);
            trainer.set_target_class(target_class);
        }
        else if let ToWorker::LoadCheckpoint(_string) = msg {

        }
        // Si rec es None no se graficará el progreso
        else if let ToWorker::Start(config) = msg {
            if config.target_class != trainer.target_class {
                trainer.set_target_class(config.target_class);
            }
            println!("Trabajador iniciando entrenamiento...");
            
            // Bucle manual de épocas
            'epoch_loop: while trainer.current_epoch < config.target_epochs {
                // Revisamos si el usuario envió mensajes
                if let Ok(msg) = rx.try_recv() {
                    if let ToWorker::Pause = msg {
                        break 'epoch_loop;
                    }
                    else if let ToWorker::Stop = msg {
                        println!("Entrenamiento cancelado por el usuario.");
                        trainer.stop();
                        break;
                        //break 'epoch_loop'; // Rompes el ciclo de entrenamiento
                    }
                    else if let ToWorker::Exit = msg {
                        println!("Trabajador saliendo por petición del usuario.");
                        let _ = tx.send(FromWorker::WorkerExited);
                        break 'epoch_loop;
                    }
                }
                // Iteramos sobre los lotes (asumiendo que tienes tu dataloader)
                // for batch in train_dataloader.iter() {
                //     let output = model.step(batch);
                //     model = optim.step(learning_rate, model, output.grads);
                // }

                // --- INTEGRACIÓN CON RERUN E ICED ---
                // Aquí tienes acceso directo al modelo en cada época
                // Puedes extraer los pesos y el sesgo sin pelear con el Learner:
                
                // let pesos = model.linear.weight.val().into_data().convert::<f32>().value;
                // let sesgo = model.linear.bias.unwrap().val().into_data().convert::<f32>().value;
                
                // Le envías los datos frescos a tu hilo principal para graficar:
                // let _ = tx.send(FromWorker::EpochUpdate { epoch, pesos, sesgo });
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