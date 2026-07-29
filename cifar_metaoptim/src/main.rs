use std::path::Path;
use walkdir::WalkDir;
use image::io::Reader as ImageReader;

use optirs_nas::{
    ArchitectureSpace, SearchSpace, ReinforcementLearningNAS, NASController
};

/// 1. PREPARACIÓN DEL DATASET
/// Escanea tu directorio "cifar10_images" (o mnist) y prepara las rutas o tensores.
fn load_dataset_metadata(dir_path: &str) -> Vec<String> {
    println!("Escaneando imágenes en: {}...", dir_path);
    let mut image_paths = Vec::new();

    for entry in WalkDir::new(dir_path).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            // Aquí puedes añadir filtros por extensión si es necesario
            image_paths.push(path.to_string_lossy().to_string());
        }
    }
    
    println!("✅ Se encontraron {} imágenes listas para la evaluación.", image_paths.len());
    image_paths
}

/// 2. DEFINICIÓN Y EVALUACIÓN DEL MODELO
/// Esta función recibe los parámetros sugeridos por OptiRS-NAS en la iteración actual
fn train_and_evaluate_model(
    _image_paths: &[String],
    layers: Vec<String>,
    channels: usize,
    kernel: usize,
    lr: f64,
    batch_size: usize,
) -> f64 {
    // ----------------------------------------------------------------------
    // AQUÍ VA TU LÓGICA DE DEEP LEARNING (Ej: usando `burn` o `candle-core`)
    // 1. Cargar las imágenes en tensores usando los `_image_paths`
    // 2. Construir la red dinámica basada en el vector `layers`
    // 3. Entrenar por X épocas usando `lr` y `batch_size`
    // 4. Retornar la precisión de validación (validation accuracy)
    // ----------------------------------------------------------------------

    // A modo de ejemplo, simulamos el resultado (accuracy) de la evaluación
    // Penalizamos arquitecturas excesivamente pequeñas o gigantes
    let acc_base = 0.60;
    let depth_bonus = if layers.len() > 2 { 0.25 } else { 0.10 };
    let final_acc = acc_base + depth_bonus - (lr * 0.1); 
    
    println!("  -> [Evaluando Trial] Capas: {:?}, Canales: {}, LR: {:.5}, Acc: {:.2}%", 
             layers, channels, lr, final_acc * 100.0);
             
    final_acc // Retornamos la precisión (métrica a maximizar)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cargamos los metadatos de tus imágenes
    let dataset_dir = "cifar10_images";
    let dataset_paths = load_dataset_metadata(dataset_dir);

    if dataset_paths.is_empty() {
        eprintln!("⚠️ No se encontraron imágenes en {}. Revisa la ruta.", dataset_dir);
        return Ok(());
    }

    // 3. CONFIGURAR EL ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
    let hp_space = SearchSpace::new()
        .add_continuous("learning_rate", 1e-5, 1e-2)
        .add_discrete("batch_size", &[16, 32, 64, 128])
        .add_categorical("optimizer", &["adamw", "sgd"]);

    // 4. CONFIGURAR EL ESPACIO DE BÚSQUEDA DE ARQUITECTURA (NAS)
    // Definimos los bloques de construcción que el algoritmo puede usar
    let arch_space = ArchitectureSpace::new()
        .add_layer_types(&["conv2d", "depthwise_conv2d", "separable_conv2d", "attention"])
        .add_kernel_sizes(&[3, 5, 7])
        .add_channel_sizes(&[32, 64, 128, 256]);

    // 5. INICIALIZAR EL CONTROLADOR NAS BASADO EN APRENDIZAJE POR REFUERZO
    // ReinforcementLearningNAS usa una red controladora interna para "predecir" 
    // la mejor arquitectura a ensamblar
    let mut nas_agent = ReinforcementLearningNAS::new(arch_space, hp_space);
    let max_trials = 50;

    println!("🚀 Iniciando Búsqueda de Arquitectura de Red (NAS) para {} trials...", max_trials);

    // 6. LOOP DE OPTIMIZACIÓN
    for trial_idx in 0..max_trials {
        // Pedir al agente RL que genere una arquitectura a probar
        let mut trial = nas_agent.suggest_trial()?;

        // Extraer los parámetros propuestos para este trial
        let lr = trial.get_continuous("learning_rate").unwrap_or(0.001);
        let batch_size = trial.get_discrete("batch_size").unwrap_or(64) as usize;
        
        // Simular que el trial nos da una estructura de N capas, canales y kernels
        let current_layers = trial.get_layer_sequence().unwrap_or_else(|| vec!["conv2d".to_string()]);
        let current_channels = trial.get_discrete("channel_size").unwrap_or(64) as usize;
        let current_kernel = trial.get_discrete("kernel_size").unwrap_or(3) as usize;

        // Evaluar el modelo (Entrenarlo en tu dataset)
        let accuracy = train_and_evaluate_model(
            &dataset_paths, 
            current_layers, 
            current_channels, 
            current_kernel, 
            lr, 
            batch_size
        );

        // Retroalimentar al Agente NAS con la recompensa (Accuracy)
        // Esto permite que el agente aprenda qué combinaciones de capas funcionan mejor
        trial.report_accuracy(accuracy);
        nas_agent.tell(trial)?;
    }

    // 7. OBTENER LA MEJOR ARQUITECTURA ENCONTRADA
    let best_arch = nas_agent.best_architecture()?;
    println!("🏆 ¡Búsqueda finalizada!");
    println!("Mejor arquitectura encontrada: {:#?}", best_arch);

    Ok(())
}