// Red
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
    tensor::activation::sigmoid,
};

// Entrenamiento
use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, InferenceStep, RegressionOutput},
};

use crate::iris_dataset::IrisBatch;


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