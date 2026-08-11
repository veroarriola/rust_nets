use crate::iris_dataset::IrisBatch;

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



/*
 * Perceptrón
 */
#[derive(Module, Debug)]
pub struct Perceptron<B: Backend> {
    // La capa lineal maneja los pesos (W) y el sesgo (b)
    linear: Linear<B>,
    loss_fn: BinaryCrossEntropyLoss<B>,
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

