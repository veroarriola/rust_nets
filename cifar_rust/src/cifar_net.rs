//use burn::data::dataloader::DataLoaderBuilder;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::{backend::Backend, Tensor};


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

    pub fn obtener_pesos(&self) -> Vec<(&'static str, burn::tensor::Tensor<B, 2>)> {
        vec![
            ("capa_1", self.linear_1.weight.val()),
            ("capa_2", self.linear_2.weight.val()),
            ("capa_3", self.linear_3.weight.val()), // Añade aquí las capas que tengas
        ]
    }
}

