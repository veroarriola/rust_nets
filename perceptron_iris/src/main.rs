use iced::{Task, Element, Theme, Length, Color};
use iced::widget::{button, combo_box, column, row, text, text_input, container, Space, progress_bar, scrollable};
use iced::color;
use iced::widget::text::Style as TextStyle; // Opcional, para el lambda
use iced::widget::container::Style as ContainerStyle; // Opcional, para el fondo

use rerun::RecordingStream;
use strum::IntoEnumIterator; 
use crate::iris_dataset::{IrisClass, IrisDataset};

mod iris_dataset;
mod burn_perceptron;
mod rerun_plotter;

use iced::{window};
use iced::window::icon;


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

    pub fn update(&mut self, message: UiMessage) {
        match message {
            UiMessage::TargetClassSelected(iris_class) => {
                self.target_class = Some(iris_class);
            },
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
        ..window::Settings::default()
    })
    .run()
}
