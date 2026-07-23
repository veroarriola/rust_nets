use iced::{Task, Element, Theme, Length};
use iced::widget::{button, column, row, text, text_input, container, Space, progress_bar, scrollable};
use iced::color;
use iced::futures::SinkExt; // Necesario para hacer output.send(...).await
//use tokio::sync::mpsc; // Usaremos canales asíncronos para Iced <-> Worker

mod training_state;
mod cifar_data;
mod cifar_net;
mod burn_functions; 

use training_state::{WorkerEvent, ToWorker, FromWorker, TrainingStatus, WorkerConfig};
use burn_functions::{worker_loop};


// --- EVENTOS DEL SUBSCRIPTOR ---


#[derive(Debug, Clone)]
pub enum UiMessage {
    InputSeedChanged(String),
    InputLrChanged(String),
    InputEpochsChanged(String),
    BtnStartPressed,
    BtnPausePressed,
    BtnStopPressed,
    BtnLoadPressed(String),
    WorkerStatusChanged(WorkerEvent),
    WindowCloseRequested,
    BtnLoadCheckpointPressed,
    CheckpointSelected(Option<String>), // Option porque el usuario puede cancelar la ventana
}


// IU

pub struct CifarExperimenter {
    status: TrainingStatus,
    input_seed: String,
    input_lr: String,
    input_epochs: String,
    current_epoch: usize,
    current_loss: f32,
    current_batch: usize,
    total_batches: usize,
    checkpoints_disponibles: Vec<String>,
    
    // El transmisor para enviarle comandos (Pausa, Iniciar) al hilo de Burn
    worker_tx: Option<tokio::sync::mpsc::UnboundedSender<ToWorker>>,
}

impl CifarExperimenter {
    // Constructor
    pub fn new() -> (Self, Task<UiMessage>) {
        (
            Self {
                status: TrainingStatus::Idle,
                input_seed: "42".to_string(),
                input_lr: "0.001".to_string(),
                input_epochs: "10".to_string(),
                current_epoch: 0,
                current_loss: 0.0,
                current_batch: 0,
                total_batches: 0,
                checkpoints_disponibles: vec![],
                worker_tx: None, // Se conectará al iniciar
            },
            Task::none(),
        )
    }

    pub fn update(&mut self, message: UiMessage) -> Task<UiMessage> {
        match message {
            UiMessage::BtnLoadCheckpointPressed => {
                // Abrimos el explorador de archivos sin bloquear la UI
                return iced::Task::perform(
                    async {
                        let folder = rfd::AsyncFileDialog::new()
                            .set_title("Selecciona la carpeta del checkpoint (ej. epoch_10)")
                            .pick_folder()
                            .await;
                        
                        folder.map(|f| f.path().display().to_string())
                    },
                    UiMessage::CheckpointSelected,
                );
            }

            UiMessage::CheckpointSelected(Some(path)) => {
                if let Some(tx) = &self.worker_tx {
                    println!("Solicitando al Worker cargar: {}", path);
                    let _ = tx.send(ToWorker::LoadCheckpoint(path));
                }
                iced::Task::none()
            }

            UiMessage::CheckpointSelected(None) => {
                // El usuario cerró la ventana sin elegir nada, no hacemos nada.
                iced::Task::none()
            }

            UiMessage::InputSeedChanged(val) => {
                self.input_seed = val;
                Task::none()
            }

            UiMessage::InputLrChanged(val) => {
                self.input_lr = val;
                Task::none()
            }

            UiMessage::InputEpochsChanged(val) => {
                self.input_epochs = val;
                Task::none()
            }
            
            UiMessage::BtnStartPressed => {
                self.status = TrainingStatus::Training;
                
                // Parseamos los parámetros
                let seed = self.input_seed.parse::<u64>().unwrap_or(42);
                let lr = self.input_lr.parse::<f32>().unwrap_or(0.001);
                let epochs = self.input_epochs.parse::<usize>().unwrap_or(10);

                if let Some(tx) = &self.worker_tx {
                    // 1. Instanciamos el struct con los datos recogidos de la UI
                    let config = WorkerConfig {
                        seed,
                        lr,
                        target_epochs: epochs,
                        validation_interval: 2, // Aquí asignas tu intervalo de validación
                    };

                    // 2. Lo pasamos como parámetro a la variante Start
                    let _ = tx.send(ToWorker::Start(config));
                } else {
                    // El worker se inicializa en el 'subscription' de Iced al abrir la app.
                    // Si llegamos aquí, el canal aún no está listo.
                    println!("Advertencia: Se presionó Start pero el Worker no está conectado aún.");
                }
                Task::none()
            }
            
            UiMessage::BtnPausePressed => {
                self.status = TrainingStatus::Paused;
                if let Some(tx) = &self.worker_tx {
                    let _ = tx.send(ToWorker::Pause);
                }
                Task::none()
            }
            
            UiMessage::BtnStopPressed => {
                self.status = TrainingStatus::Idle;
                if let Some(tx) = &self.worker_tx {
                    let _ = tx.send(ToWorker::Stop);
                }
                Task::none()
            }
            
            UiMessage::BtnLoadPressed(path) => {
                // Pasamos la ruta exacta al worker para que cargue los tensores desde el disco
                if let Some(tx) = &self.worker_tx {
                    let _ = tx.send(ToWorker::LoadCheckpoint(path));
                }
                Task::none()
            }

            UiMessage::WorkerStatusChanged(worker_event) => {
                match worker_event {
                    // 1. El worker apenas nació y nos da su canal de comunicación
                    WorkerEvent::Ready(tx) => {
                        self.worker_tx = Some(tx);
                        Task::none()
                    }
                    
                    // 2. El worker nos envía una actualización durante su ciclo de vida
                    WorkerEvent::Update(from_worker_msg) => {
                        match from_worker_msg {
                            FromWorker::BatchProgress { epoch, current_batch, total_batches } => {
                                self.current_epoch = epoch;
                                self.current_batch = current_batch;
                                self.total_batches = total_batches;
                            }
                            FromWorker::EpochDone { epoch, loss } => {
                                self.current_epoch = epoch;
                                self.current_loss = loss;
                                // Opcional: llenar la barra al 100% cuando termine la época
                                self.current_batch = self.total_batches; 
                            }
                            FromWorker::CheckpointSaved { path, .. } => {
                                self.checkpoints_disponibles.push(path);
                            }
                            FromWorker::CheckpointLoaded(meta) => {
                                // Sincronizamos la UI con los datos del JSON
                                self.current_epoch = meta.epoch;
                                self.input_seed = meta.seed.to_string();
                                self.input_lr = meta.lr.to_string();
                                
                                // Lo ponemos en pausa para que el usuario decida cuándo seguir
                                self.status = TrainingStatus::Paused;
                                println!("¡Checkpoint cargado con éxito! Época actual: {}", meta.epoch);
                            }
                            FromWorker::Finished => {
                                self.status = TrainingStatus::Idle;
                            }
                            FromWorker::Error(e) => {
                                println!("Error en worker: {}", e);
                                self.status = TrainingStatus::Idle;
                            }
                            FromWorker::WorkerExited => {
                                println!("Worker terminado de forma segura. Apagando...");
                                std::process::exit(0);
                            }
                        }
                        Task::none()
                    }
                }
            }

            UiMessage::WindowCloseRequested => {
                if let Some(tx) = &self.worker_tx {
                    println!("Pidiendo al Worker que termine...");
                    let _ = tx.send(ToWorker::Exit);
                } else {
                    // Si el Worker nunca se conectó, cerramos de inmediato
                    std::process::exit(0);
                }
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, UiMessage> {
        // --- PANEL IZQUIERDO ---
        let controles = column![
            text("Parámetros de Entrenamiento").size(20),
            text("Semilla (Seed):"),
            text_input("Ej: 42", &self.input_seed).on_input(UiMessage::InputSeedChanged),
            text("Tasa de Aprendizaje (LR):"),
            text_input("Ej: 0.001", &self.input_lr).on_input(UiMessage::InputLrChanged),
            text("Épocas de la serie:"),
            text_input("Ej: 10", &self.input_epochs).on_input(UiMessage::InputEpochsChanged),
        ].spacing(10).padding(20);

        let botones = match self.status {
            TrainingStatus::Idle => {
                let btn_iniciar = button("Iniciar Serie");
                let btn_cargar = button("Cargar Checkpoint");

                // Solo activamos el botón si el canal de comunicación está listo
                let (btn_iniciar, btn_cargar) = if self.worker_tx.is_some() {
                    (
                        btn_iniciar.on_press(UiMessage::BtnStartPressed),
                        btn_cargar.on_press(UiMessage::BtnLoadCheckpointPressed)
                    )
                } else {
                    (btn_iniciar, btn_cargar)
                };
                
                row![btn_iniciar, btn_cargar]
            }
            
            TrainingStatus::Training => row![
                button("Pausar").on_press(UiMessage::BtnPausePressed),
                button("Detener").on_press(UiMessage::BtnStopPressed)
            ],
            
            TrainingStatus::Paused => {
                let btn_reanudar = button("Reanudar");
                let btn_cargar = button("Cargar Otro");

                // También protegemos la reanudación por si el canal se perdiera
                let (btn_reanudar, btn_cargar) = if self.worker_tx.is_some() {
                    (
                        btn_reanudar.on_press(UiMessage::BtnStartPressed),
                        btn_cargar.on_press(UiMessage::BtnLoadCheckpointPressed)
                    )
                } else {
                    (btn_reanudar, btn_cargar)
                };
                
                row![
                    btn_reanudar,
                    btn_cargar,
                    button("Detener").on_press(UiMessage::BtnStopPressed)
                ]
            }
        }
        .spacing(15)
        .padding(20);

        let panel_izquierdo = column![controles, botones].width(Length::Fixed(300.0));

        // --- PANEL PRINCIPAL ---
        // Calculamos el porcentaje del 0.0 al 100.0
        let porcentaje_progreso = if self.total_batches > 0 {
            (self.current_batch as f32 / self.total_batches as f32) * 100.0
        } else {
            0.0
        };

        // Construimos la lista visual de checkpoints
        let mut lista_checkpoints = column![].spacing(8);
        
        if self.checkpoints_disponibles.is_empty() {
            lista_checkpoints = lista_checkpoints.push(text("Ninguno todavía...").size(16));
        } else {
            for path in &self.checkpoints_disponibles {
                // Agregamos cada ruta como un texto a la columna
                lista_checkpoints = lista_checkpoints.push(text(path).size(16));
            }
        }

        // Envolvemos la lista en un área con scroll
        let checkpoints_scroll = scrollable(lista_checkpoints).height(Length::Fill);

        let panel_principal = column![
            text("Estado de la Red").size(24),
            text(format!("Época actual: {}", self.current_epoch)).size(40),
            text(format!("Loss (Pérdida): {:.4}", self.current_loss)).size(40),
            // LA BARRITA DE PROGRESO
            text(format!("Progreso del Lote (Batch): {} / {}", self.current_batch, self.total_batches)),
            progress_bar(0.0..=100.0, porcentaje_progreso),
            
            Space::new().height(Length::Fixed(20.0)), // Separador visual
            text("Checkpoints Guardados:").size(20),
            checkpoints_scroll,
        ].spacing(20).padding(40).width(Length::Fill);

        // --- LAYOUT FINAL CON ESTILO MODERNO (Iced 0.14) ---
        let layout = row![
            panel_izquierdo,
            // Divisor usando el nuevo sistema de closures para estilos
            container(Space::new().width(Length::Fixed(2.0)).height(Length::Fill))
                .style(|_theme: &Theme| {
                    container::Style::default().background(color!(0x333333))
                }),
            panel_principal
        ];

        container(layout)
            .width(Length::Fill)
            .height(Length::Fill)
            // Fondo general tipo Rerun (Oscuro)
            .style(|_theme: &Theme| {
                container::Style::default().background(color!(0x1A1A1A))
            })
            .into()
    }

    // Aquí es donde Iced escucha al Worker permanentemente
    pub fn subscription(&self) -> iced::Subscription<UiMessage> {
        // 1. Escuchamos los eventos nativos de la ventana
        // AÑADIDO: El tercer parámetro `_window_id` que exige Iced 0.14
        let eventos_ventana = iced::event::listen_with(|event, _status, _window_id| {
            // CORREGIDO: Eliminamos el `_,` porque Window ahora solo tiene 1 campo
            if let iced::Event::Window(iced::window::Event::CloseRequested) = event {
                Some(UiMessage::WindowCloseRequested)
            } else {
                None
            }
        });

        // 2. Suscribir al trabajador
        let worker_sub = iced::Subscription::run(
            || iced::stream::channel(
                100, // Buffer de mensajes
                |mut output: iced::futures::channel::mpsc::Sender<WorkerEvent>| async move {
                    // Creamos dos canales
                    let (tx_to_worker, rx_to_worker) = tokio::sync::mpsc::unbounded_channel();
                    let (tx_from_worker, mut rx_from_worker) = tokio::sync::mpsc::unbounded_channel();

                    // Desacoplamos el entrenamiento
                    std::thread::spawn(move || {
                        worker_loop(rx_to_worker, tx_from_worker);
                    });

                    // 1. Enviamos el "control remoto" a la UI
                    let _ = output.send(WorkerEvent::Ready(tx_to_worker)).await;

                    // 2. Bucle infinito escuchando a Burn
                    while let Some(msg) = rx_from_worker.recv().await {
                        let _ = output.send(WorkerEvent::Update(msg)).await;
                    }
                }
            )
        ).map(UiMessage::WorkerStatusChanged); // <-- CORREGIDO: Faltaba este punto y coma

        // 3. Agrupamos ambas suscripciones
        iced::Subscription::batch(vec![eventos_ventana, worker_sub])
    }
    
}

fn main() -> iced::Result {
    iced::application(
        CifarExperimenter::new,
        CifarExperimenter::update,
        CifarExperimenter::view,
    )
    // El título ahora es un closure. Definimos el tipo explícitamente.
    .title(|_state: &CifarExperimenter| {
        String::from("CIFAR-10 Experimenter - Burn & Rerun")
    })
    // Al añadir ": &CifarExperimenter", Rust entiende perfectamente los tiempos de vida
    .theme(|_state: &CifarExperimenter| Theme::Dark)
    .window(iced::window::Settings {
        exit_on_close_request: false, // <-- IMPORTANTE: Desactiva la guillotina automática
        ..Default::default()
    })
    .subscription(CifarExperimenter::subscription)
    .run()
}
