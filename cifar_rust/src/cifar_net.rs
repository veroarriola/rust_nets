//use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::{backend::Backend, Tensor};
use burn::config::Config;

// -.- Red densa .-.

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear_layers: Vec<Linear<B>>,
    activation: Relu,
}

impl<B: Backend> Model<B> {

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for layer in &self.linear_layers {
            x = layer.forward(x);
            x = self.activation.forward(x);
        }

        // Salida (logits)
        x
    }

    pub fn read_weights(&self) -> Vec<(String, burn::tensor::Tensor<B, 2>)> {
        let mut weights = Vec::new();
        for (index, layer) in self.linear_layers.iter().enumerate() {
            weights.push((format!("capa_{}", index), layer.weight.val()));
        }

        weights
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig{
    input_size: usize,
    hidden_sizes: Vec<usize>,
    num_classes: usize,
}

impl ModelConfig {
    // Configuración por defecto para demo de perceptrón multicapa con CIFAR-10
    pub fn new_cifar10(hidden_sizes: Vec<usize>) -> Self {
        Self {
            input_size: 3072,               // CIFAR-10: 32x32x3 = 3072 entradas. 10 clases de salida.
            hidden_sizes: hidden_sizes,     // Ej: vec![1024, 512]
            num_classes: 10                 // Salida: 10 (clases de CIFAR)
        }
    }

    // Devuelve el modelo inicializado
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let mut linear_layers: Vec<Linear<B>> = Vec::new();
        if self.hidden_sizes.is_empty() {
            linear_layers.push(LinearConfig::new(self.input_size, self.num_classes).init(device));
        } else {
            linear_layers.push(LinearConfig::new(self.input_size, self.hidden_sizes[0]).init(device));
            // Ocultas
            for i in 0..self.hidden_sizes.len() - 1 {
                linear_layers.push(LinearConfig::new(self.hidden_sizes[i], self.hidden_sizes[i + 1]).init(device));
            }
            linear_layers.push(LinearConfig::new(*self.hidden_sizes.last().expect("Ya chequé que haya algo."), self.num_classes).init(device));
        }

        Model {
            linear_layers,
            activation: Relu::new(),
        }
    }
}