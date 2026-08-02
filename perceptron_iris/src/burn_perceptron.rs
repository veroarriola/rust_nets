// Red
use burn::{
    config, module::Module, nn::{Linear, LinearConfig, loss::BinaryCrossEntropyLoss}, tensor::{Tensor, activation::sigmoid, backend::Backend}
};

// Entrenamiento
use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, InferenceStep, RegressionOutput,},
    optim::Optimizer,
};

use burn::backend::{Wgpu, wgpu::WgpuDevice, Autodiff};
use burn::optim::adaptor::OptimizerAdaptor;

use rerun::RecordingStream;

use serde::{Deserialize, Serialize};

use crate::iris_dataset::IrisBatch;

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
    Start(TrainingConfig, Option<RecordingStream>),
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
    pub target_class: String,
    pub seed: u64,
    pub lr: f32,
    pub target_epochs: usize,
    pub validation_interval: usize,
}

// Definimos el Backend con Autodiff para entrenamiento en GPU
type MyBackend = Autodiff<Wgpu>;
pub type MyOptimizer = OptimizerAdaptor<
    burn::optim::Adam,
    Perceptron<MyBackend>,   // trait: burn::module::AutodiffModule<AutodiffBackend>
    MyBackend,               // trait: AutodiffBackend
>;

pub struct TrainingState {
    device: WgpuDevice,
    is_training: bool,
    pub current_epoch: usize,
    reset_dataloader: bool, // Para usar la semilla
    model: Option<Perceptron<MyBackend>>,
    optimizador: Option<MyOptimizer>,
    criterion: BinaryCrossEntropyLoss<MyBackend>,
}



/*
 * Perceptrón
 */
#[derive(Module, Debug)]
pub struct Perceptron<B: Backend> {
    // La capa lineal maneja los pesos (W) y el sesgo (b)
    pub linear: Linear<B>, 
}

impl<B: Backend> Perceptron<B> {
    /// Inicializa el perceptrón en el dispositivo especificado (CPU o GPU)
    pub fn new(device: &B::Device) -> Self {
        // 4 entradas (características de Iris) y 1 salida (clasificación binaria)
        let linear = LinearConfig::new(4, 1).init(device);
        Self { linear }
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
    
    // 1. Ahora el Output es RegressionOutput, el cual sí implementa ItemLazy
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let predictions = self.forward(batch.inputs);

        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&predictions.device())
            // Clonamos predictions y targets si es necesario para usarlos en RegressionOutput
            .forward(predictions.clone(), batch.targets.clone().int());

        let grads = loss.backward();

        // 2. Empaquetamos los datos en RegressionOutput
        let output = RegressionOutput::new(loss, predictions, batch.targets);

        // 3. TrainOutput recibe el módulo, los gradientes y nuestro 'output' de métricas
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep for Perceptron<B> {
    type Input = IrisBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let predictions = self.forward(batch.inputs);
        
        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&predictions.device())
            .forward(predictions.clone(), batch.targets.clone().int());

        // En inferencia/validación solo devolvemos las métricas (sin gradientes)
        RegressionOutput::new(loss, predictions, batch.targets)
    }
}

struct Trainer {

}

impl Trainer {
    fn new() -> Self {
        Self {

        }
    }
}

pub async fn worker_loop(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<ToWorker>,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
) {
    println!("Trabajador iniciado");
    while let Some(msg) = rx.recv().await {
        /*
        match msg {
            ToWorker::Start(config) => {

            }
        }
        */
        //if let Some(cmd) = msg {
            //match cmd {
                //ToWorker::Start(config) => {
                
                // Si rec es None no se graficará el progreso
                if let ToWorker::Start(config, rec) = msg {
                    println!("Trabajador iniciando entrenamiento...");
                    /*
                    let device = WgpuDevice::default();
                    let mut model: Perceptron<MyBackend> = Perceptron::new(&device);
                    let mut optim = burn::optim::SgdConfig::new()
                        .init::<MyBackend, Perceptron<MyBackend>>();
                    
                    // Bucle manual de épocas
                    'epoch_loop: for epoch in 0..config.target_epochs {
                        // Das un "vistazo rápido" para ver si el usuario pidió detener
                        if let Ok(ToWorker::Stop) = rx.try_recv() {
                            println!("Entrenamiento cancelado por el usuario.");
                            break;
                            //break 'epoch_loop'; // Rompes el ciclo de entrenamiento
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
                    */
                    
                    println!("Trabajador terminó entrenamiento.");
                    let _ = tx.send(FromWorker::TrainingFinished);
                }
                else if let ToWorker::Exit = msg {
                    println!("Trabajador sin trabajo saliendo.");
                    let _ = tx.send(FromWorker::WorkerExited);
                    break;
                }
            //}
        //}
    }
    println!("Trabajador terminado.");
    let _ = tx.send(FromWorker::WorkerExited);
}