use iced::{Task, Element, Theme, Length, Color};
use iced::widget::{button, combo_box, column, row, text, text_input, container, Space, progress_bar, scrollable};
use iced::color;
use iced::widget::text::Style as TextStyle; // Opcional, para el lambda
use iced::widget::container::Style as ContainerStyle; // Opcional, para el fondo

use rerun::RecordingStream;
use strum::IntoEnumIterator; 

use crate::iris_dataset::{IrisClass, IrisDataset};
use crate::burn_perceptron::{FromWorker, Perceptron, ToWorker, TrainingConfig, TrainingStatus, WorkerEvent, worker_loop};

mod iris_dataset;
mod burn_perceptron;
mod rerun_plotter;

use iced::{window};
use iced::window::icon;

use iced::futures::SinkExt; // Necesario para hacer output.send(...).await


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

pub struct PerceptronExperimenter {
    //status: TrainingStatus,
    // Lista de opciones
    target_classes: combo_box::State<IrisClass>,
    // Opción seleccionada actualmente
    target_class: Option<IrisClass>,
    // Original dataset
    original_dataset: Option<IrisDataset>,
    // Rerun recording stream
    rec: Option<RecordingStream>,
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
        let mut obj = Self {
            target_classes: combo_box::State::new(all_target_classes),
            target_class: None,
            original_dataset: None,
            rec: None,
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
        };
        // Cargar conjunto de datos
        match IrisDataset::new(iris_dataset::DATASET_SOURCE_FILE) {
            Ok(dataset) => {
                match rerun::RecordingStreamBuilder::new("perceptron_iris")
                    .spawn() {
                        Ok(rec) => {
                            // Graficar conjunto de datos inicial en rerun
                            match rerun_plotter::plot_dataset(&rec, &dataset) {
                                Ok(_) => {
                                    
                                },
                                Err(e) => {
                                    obj.status_bar_message = Some(format!("Fallo al graficar datos: {}", e));
                                }
                            }
                            obj.original_dataset = Some(dataset);
                            obj.rec = Some(rec);
                            (obj, Task::none())
                        },
                        Err(e) => {
                            obj.error_message = Some(format!("Fallo al iniciar Rerun: {}", e));
                            (obj, Task::none())
                        }
                    }
            },
            Err(e) => {
                obj.error_message = Some(format!("Error al cargar iris.csv: {}", e));
                (obj, Task::none())
            },
        }
        
    }

    pub fn update(&mut self, message: UiMessage) -> Task<UiMessage> {
        match message {
            UiMessage::TargetClassSelected(iris_class) => {
                self.target_class = Some(iris_class);
                iced::Task::none()
            },
            UiMessage::InputSeedChanged(String) => todo!(),
            UiMessage::InputLrChanged(String) => todo!(),
            UiMessage::InputEpochsChanged(String) => todo!(),
            UiMessage::BtnStartPressed => todo!(),
            UiMessage::BtnPausePressed => todo!(),
            UiMessage::BtnStopPressed => todo!(),
            UiMessage::BtnLoadPressed(String) => todo!(),
            UiMessage::WindowCloseRequested => todo!(),
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
                    //.style(|_them:  &Theme| { Color::from_rgb(0.8, 0.1, 0.1) }) // Texto rojo para alertar
            )
            .width(Length::Fill)
            .height(Length::Fill)
            //.center_x()
            //.center_y()
            .into();
        }

        // 2. Si no hay error, construimos la UI principal
        let panel_izquierdo = column![
            text("Parámetros").size(24),
            combo_box(
                &self.target_classes,
                "Selecciona la clase a identificar",
                self.target_class.as_ref(),
                UiMessage::TargetClassSelected
            ),
        ].spacing(20).padding(40).width(Length::Fill);

        let layout = row![
            panel_izquierdo,
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
