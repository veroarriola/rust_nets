use iced::{Task, Element, Theme, Length, Color};
use iced::widget::{button, combo_box, column, container, progress_bar, row, scrollable, Space, text, text_input};
use iced::color;
use iced::widget::text::Style as TextStyle; // Opcional, para el lambda
use iced::widget::container::Style as ContainerStyle; // Opcional, para el fondo
use iced::{window};
use iced::window::icon;
use iced::futures::SinkExt; // Necesario para hacer output.send(...).await

use strum::IntoEnumIterator; 

use crate::iris_dataset::{IrisClass, IrisDataset};
use crate::training_worker::{FromWorker, ToWorker, TrainingConfig, WorkerEvent, worker_loop};

mod iris_dataset;
mod rerun_plotter;
mod burn_perceptron;
mod training_worker;



fn cargar_icono() -> icon::Icon {
    // 1. Incluimos el archivo en el binario al compilar
    let bytes_imagen = include_bytes!("../data/icono.ico");
    
    // 2. Decodificamos la imagen desde la memoria
    let imagen = image::load_from_memory(bytes_imagen)
        .expect("Error al cargar la imagen del icono")
        .into_rgba8();
        
    let (ancho, alto) = imagen.dimensions();
    let pixeles_rgba = imagen.into_raw();
    
    // 3. Creamos el icono para iced
    icon::from_rgba(pixeles_rgba, ancho, alto)
        .expect("Error al convertir los píxeles al formato del icono")
}


#[derive(Debug, Clone)]
pub enum UiMessage {
    TargetClassSelected(IrisClass),
    InputSeedChanged(String),
    InputLrChanged(String),
    InputEpochsChanged(String),
    BtnStartPressed,
    BtnPausePressed,
    BtnStopPressed,
    BtnLoadPressed(String),
    WindowCloseRequested,
    BtnLoadCheckpointPressed,
    CheckpointSelected(Option<String>), // Option porque el usuario puede cancelar la ventana
    WorkerStatusChanged(WorkerEvent),
}

#[derive(PartialEq)]
pub enum TrainingStatus {
    Idle,
    Training,
    Paused,
}


pub struct PerceptronExperimenter {
    //status: TrainingStatus,
    // Lista de opciones
    target_classes: combo_box::State<IrisClass>,
    // Opción seleccionada actualmente
    target_class: Option<IrisClass>,
    // Original dataset
    original_dataset: Option<IrisDataset>,
    // Mensaje de error en caso de haberlo
    error_message: Option<String>,
    status_bar_message: Option<String>,

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

impl PerceptronExperimenter {
    pub fn new() -> (Self, Task<UiMessage>) {
        let all_target_classes: Vec<IrisClass> = IrisClass::iter().collect();
        (
            Self {
                target_classes: combo_box::State::new(all_target_classes),
                target_class: Some(IrisClass::Setosa),
                original_dataset: None,
                error_message: None,
                status_bar_message: None,

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
            Task::none()
        )
    }

    pub fn update(&mut self, message: UiMessage) -> Task<UiMessage> {
        match message {
            UiMessage::TargetClassSelected(iris_class) => {
                self.target_class = Some(iris_class);
                if let Some(tx) = &self.worker_tx {
                    let _ = tx.send(ToWorker::TargetSelected(iris_class));
                }
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

            

            UiMessage::BtnLoadPressed(path) => {
                // Pasamos la ruta exacta al worker para que cargue los tensores desde el disco
                if let Some(tx) = &self.worker_tx {
                    let _ = tx.send(ToWorker::LoadCheckpoint(path));
                }
                self.status = TrainingStatus::Idle;
                Task::none()
            }

            
            UiMessage::BtnLoadCheckpointPressed => todo!(),
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


            // Mensajes enviados al hilo trabajador
            UiMessage::BtnStartPressed => {
                self.status = TrainingStatus::Training;
                
                // Recuperamos los parámetros
                let target_class = self.target_class.unwrap_or(IrisClass::Setosa);
                let seed = self.input_seed.parse::<u64>().unwrap_or(42);
                let lr = self.input_lr.parse::<f64>().unwrap_or(0.001);
                let epochs = self.input_epochs.parse::<usize>().unwrap_or(10);

                if let Some(tx) = &self.worker_tx {
                    // 1. Instanciamos el struct con los datos recogidos de la UI
                    let config = TrainingConfig {
                        target_class,
                        seed,
                        lr,
                        target_epochs: epochs,
                        //validation_interval: 2, // Aquí asignas tu intervalo de validación
                    };

                    // 2. Lo pasamos como parámetro a la variante Start
                    let _ = tx.send(ToWorker::Start(config));
                } else {
                    // El worker se inicializa en el 'subscription' de Iced al abrir la app.
                    // Si llegamos aquí, el canal aún no está listo.
                    println!("⚠️ Advertencia: Se presionó Start pero el Worker no está conectado aún.");
                    self.status_bar_message = Some(String::from("⚠️ Advertencia: Se presionó Start pero el Worker no está conectado aún."));
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

            // Mensajes recibidos desde el hilo trabajador
            UiMessage::WorkerStatusChanged(worker_event) => {
                match worker_event {
                    // 1. El worker apenas nació y nos da su canal de comunicación
                    WorkerEvent::Ready(tx) => {
                        self.worker_tx = Some(tx);
                        Task::none()
                    }

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
                                self.current_epoch = meta.target_epochs;
                                self.input_seed = meta.seed.to_string();
                                self.input_lr = meta.lr.to_string();
                                
                                // Lo ponemos en pausa para que el usuario decida cuándo seguir
                                self.status = TrainingStatus::Paused;
                                println!("¡Checkpoint cargado con éxito! Época actual: {}", self.current_epoch);
                            }
                            FromWorker::TrainingFinished => {
                                self.status = TrainingStatus::Idle;
                            }
                            FromWorker::Error(e) => {
                                println!("Error en worker: {}", e);
                                self.status = TrainingStatus::Idle;
                            }
                            FromWorker::WorkerExited => {
                                println!("Worker terminado de forma segura. Apagando...");
                                /*
                                if let Some(rec) = self.rec.take() {
                                    let _ = rec.flush_blocking();
                                }
                                    */
                                std::process::exit(0);
                            }
                        }
                        Task::none()
                    }
                }
            }
        }
    }

    pub fn view(&self) -> Element<'_, UiMessage> {
        // 1. Si hay un error, mostramos una pantalla de error
        if let Some(err) = &self.error_message {
            return container(
                text(err)
                    .size(12)
                    .style(|_theme: &Theme| TextStyle {
                            color: Some(Color::from_rgb(0.8, 0.1, 0.1)),
                        })
            )
            .width(Length::Fill)
            .height(Length::Fill)
            //.center_x()
            //.center_y()
            .into();
        }

        // 2. Si no hay error, construimos la UI principal

        /*
         * Controles
         */
        let controles = column![
            combo_box(
                &self.target_classes,
                "Selecciona la clase a identificar",
                self.target_class.as_ref(),
                UiMessage::TargetClassSelected
            ),
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

        // Panel para controles
        let panel_izquierdo = column![
            text("Parámetros de Entrenamiento").size(24),
            controles,
            botones
        ]
        .spacing(10)
        .padding(20)
        .width(Length::Fixed(400.0));

        /*
         * Principal
         */

        // Calculamos el porcentaje de avance en el lote del 0.0 al 100.0
        let porcentaje_progreso = if self.total_batches > 0 {
            (self.current_batch as f32 / self.total_batches as f32) * 100.0
        } else {
            0.0
        };

        // Construimos la lista visual de checkpoints
        let mut checkpoints_list = column![].spacing(8);
        
        if self.checkpoints_disponibles.is_empty() {
            checkpoints_list = checkpoints_list.push(text("Ninguno todavía...").size(16));
        } else {
            for path in &self.checkpoints_disponibles {
                // Agregamos cada ruta como un texto a la columna
                checkpoints_list = checkpoints_list.push(text(path).size(16));
            }
        }

        // Envolvemos la lista en un área con scroll
        let checkpoints_scroll = scrollable(checkpoints_list).height(Length::Fill);


        // Panel principal
        let main_content = column![
            text("Estado de la Red").size(24),

            Space::new().height(Length::Fixed(10.0)), // Separador visual

            text(format!("Época actual: {}", self.current_epoch)),
            text(format!("Pérdida: {:.4}", self.current_loss)),
            // Barrita de progreso
            text(format!("Progreso del lote: {} / {}", self.current_batch, self.total_batches)),
            progress_bar(0.0..=100.0, porcentaje_progreso),
            
            Space::new().height(Length::Fixed(20.0)), // Separador visual

            text("Puntos de control guardados:").size(20),
            checkpoints_scroll,
        ]
        .spacing(10)
        .padding(20)
        .width(Length::Fill)
        .height(Length::Fill);


        // Barra de estado
        let status_text = "";

        let status_bar = container(
            text(status_text)
                .size(14)
                // SOLUCIÓN 1 (El Lambda que pediste):
                .style(|_theme: &Theme| TextStyle {
                    color: Some(Color::from_rgb(0.8, 0.1, 0.1)),
                })
        )
        .width(Length::Fill)
        .padding(5)
        // Opcional: Darle un color de fondo a la barra usando un lambda también
        .style(|_theme: &Theme| ContainerStyle {
            background: Some(iced::Background::Color(Color::from_rgb(0.1, 0.1, 0.1))),
            ..Default::default()
        });

        let row_layout = row![
            panel_izquierdo,
            main_content,
        ];

        let layout = column![
            row_layout,
            status_bar,
        ];

        container(layout)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme: &Theme| {
                container::Style::default().background(color!(0x1A1A1A))
            })
            .into()
    }

    // Aquí es donde Iced escucha al Worker permanentemente
    pub fn subscription(&self) -> iced::Subscription<UiMessage> {

        // 1. Escuchamos los eventos nativos de la ventana
        let window_events = iced::event::listen_with(|event, _status, _window_id| {
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
                    let rec = rerun::RecordingStreamBuilder::new("perceptron_iris")
                        .spawn()
                        .expect("⚠️ Fallo al iniciar ReRun.  Se continuará sin graficar datos.");
                    
                    // Desacoplamos el entrenamiento
                    std::thread::spawn(move || {
                        // 1. Creamos un motor de Tokio dedicado exclusivamente a este hilo
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .expect("Fallo al crear el runtime de Tokio para el worker");

                        // 2. Bloqueamos el hilo usando el motor para ejecutar nuestra función asíncrona
                        rt.block_on(worker_loop(rx_to_worker, tx_from_worker, Some(rec)));
                    });

                    // 1. Enviamos el "control remoto" a la UI
                    let _ = output.send(WorkerEvent::Ready(tx_to_worker)).await;

                    // 2. Bucle infinito escuchando a Burn
                    while let Some(msg) = rx_from_worker.recv().await {
                        let _ = output.send(WorkerEvent::Update(msg)).await;
                    }
                }
            )
        ).map(UiMessage::WorkerStatusChanged);

        // 3. Agrupamos ambas suscripciones
        iced::Subscription::batch(vec![window_events, worker_sub])
    }
}

fn main() -> iced::Result {
    iced::application(
        PerceptronExperimenter::new,
        PerceptronExperimenter::update,
        PerceptronExperimenter::view,
    )
    .title(|_state: &PerceptronExperimenter| {
        String::from("Irist Experimenter - Burn & Rerun")
    })
    .theme(|_state: &PerceptronExperimenter| Theme::Dark)
    .window(window::Settings {
        icon: Some(cargar_icono()),
        exit_on_close_request: false,
        ..window::Settings::default()
    })
    .subscription(PerceptronExperimenter::subscription)
    .run()
}
