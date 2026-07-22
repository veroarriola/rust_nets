use serde::{Deserialize, Serialize};
//use std::fs;


// --- MENSAJES DE COMUNICACIÓN ---
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    // Cuando el hilo arranca, nos entrega el "transmisor" para enviarle comandos
    Ready(tokio::sync::mpsc::UnboundedSender<ToWorker>),
    // Actualizaciones de estado desde el worker
    Update(FromWorker),
}

#[derive(Debug, Clone)]
pub enum ToWorker {
    Start(WorkerConfig),
    Pause,
    Stop,
    LoadCheckpoint(String),
    Exit,
}

#[derive(Debug, Clone)]
pub enum FromWorker {
    EpochDone { epoch: usize, loss: f32 },
    CheckpointSaved { path: String, epoch: usize },
    Finished,
    Error(String),
    CheckpointLoaded(TrainingMeta),
    WorkerExited,
}

// El esquema de nuestro meta.json
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingMeta {
    pub epoch: usize,
    pub seed: u64,
    pub lr: f32,
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub seed: u64,
    pub lr: f32,
    pub target_epochs: usize,
    pub validation_interval: usize,
}

#[derive(PartialEq)]
pub enum TrainingStatus {
    Idle,
    Training,
    Paused,
}
