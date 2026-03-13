use iced::keyboard::{self, Key};
use iced::widget::{column, rule};
use iced::{Color, Element, Subscription, Task, Theme, window};
use iced_plot::{LineStyle, MarkerStyle, PlotUiMessage, PlotWidget, Series};
use std::time::Instant;

use crate::components;
use crate::plots;
use crate::store;
use crate::typing::{Profile, Session, random_prompt};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Main,
    Profiles,
}

pub struct App {
    pub mode: Mode,
    pub scale: f32,
    pub is_fullscreen: bool,
    pub name_input: String,
    pub session: Session,
    pub profiles: Vec<Profile>,
    pub current_prompt: &'static str,
    pub live_plot: PlotWidget,
    pub live_plot_has_data: bool,
    pub id_plot: Option<PlotWidget>,
}

impl Default for App {
    fn default() -> Self {
        let mut live_plot = PlotWidget::new();
        live_plot.set_x_axis_label("bigram #");
        live_plot.set_y_axis_label("ms");
        live_plot.autoscale_on_updates(true);

        Self {
            mode: Mode::Main,
            scale: 1.0,
            is_fullscreen: false,
            name_input: String::new(),
            session: Session::default(),
            profiles: store::load(),
            current_prompt: random_prompt(),
            live_plot,
            live_plot_has_data: false,
            id_plot: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    ModeChanged(Mode),
    DeleteProfile(usize),
    ToggleFullscreen,
    NameChanged(String),
    KeyPressed(char, u32, Instant),
    KeyReleased(char, Instant),
    Backspace(Instant),
    BackspaceReleased(Instant),
    Enroll,
    Identify,
    Clear,
    ScaleUp,
    ScaleDown,
    ScaleReset,
    Noop,
    LivePlotMsg(PlotUiMessage),
    IdPlotMsg(PlotUiMessage),
}

impl App {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ModeChanged(mode) => {
                self.mode = mode;
            }
            Message::DeleteProfile(i) => {
                if i < self.profiles.len() {
                    self.profiles.remove(i);
                    store::save(&self.profiles);
                }
            }
            Message::ToggleFullscreen => {
                self.is_fullscreen = !self.is_fullscreen;
                let mode = if self.is_fullscreen {
                    window::Mode::Fullscreen
                } else {
                    window::Mode::Windowed
                };
                return window::oldest().then(move |id| match id {
                    Some(id) => window::set_mode(id, mode),
                    None => Task::none(),
                });
            }
            Message::NameChanged(name) => {
                self.name_input = name;
            }
            Message::KeyPressed(ch, keycode, t) => {
                if self.session.text.len() < self.current_prompt.len() {
                    self.session.push_char(ch, keycode, t);
                    self.session.text.push(ch);
                    self.update_live_plot();
                }
            }
            Message::KeyReleased(ch, t) => {
                self.session.push_release(ch, t);
            }
            Message::Backspace(t) => {
                self.session.push_backspace(t);
                self.update_live_plot();
            }
            Message::BackspaceReleased(t) => {
                self.session.push_backspace_release(t);
            }
            Message::Enroll => {
                if self.name_input.trim().is_empty() || self.session.is_empty() {
                    return Task::none();
                }
                let name = self.name_input.trim().to_string();
                self.profiles
                    .push(Profile::from_session(name, &self.session));
                store::save(&self.profiles);
                self.name_input.clear();
                self.session.clear();
                self.reset_live_plot();
                self.current_prompt = random_prompt();
            }
            Message::Identify => {
                if self.profiles.is_empty() || self.session.is_empty() {
                    return Task::none();
                }
                self.id_plot = plots::build_id_plot(&self.session, &self.profiles);
                self.session.clear();
                self.reset_live_plot();
                self.current_prompt = random_prompt();
            }
            Message::Clear => {
                self.session.clear();
                self.reset_live_plot();
                self.id_plot = None;
                self.current_prompt = random_prompt();
            }
            Message::ScaleUp => self.scale = (self.scale + 0.1).min(3.0),
            Message::ScaleDown => self.scale = (self.scale - 0.1).max(0.5),
            Message::ScaleReset => self.scale = 1.0,
            Message::Noop => {}
            Message::LivePlotMsg(msg) => {
                self.live_plot.update(msg);
            }
            Message::IdPlotMsg(msg) => {
                if let Some(plot) = &mut self.id_plot {
                    plot.update(msg);
                }
            }
        }
        Task::none()
    }

    fn update_live_plot(&mut self) {
        let positions: Vec<[f64; 2]> = self
            .session
            .log
            .iter()
            .enumerate()
            .map(|(i, (_, ms))| [i as f64, *ms])
            .collect();

        if positions.is_empty() {
            if self.live_plot_has_data {
                self.live_plot.remove_series("intervals");
                self.live_plot_has_data = false;
            }
        } else if self.live_plot_has_data {
            self.live_plot.set_series_positions("intervals", &positions);
        } else {
            let _ = self.live_plot.add_series(
                Series::new(positions, MarkerStyle::circle(4.0), LineStyle::Solid)
                    .with_label("intervals")
                    .with_color(Color::from_rgb(0.38, 0.82, 0.48)),
            );
            self.live_plot_has_data = true;
        }
    }

    fn reset_live_plot(&mut self) {
        if self.live_plot_has_data {
            self.live_plot.remove_series("intervals");
            self.live_plot_has_data = false;
        }
    }

    pub fn subscription(&self) -> Subscription<Message> {
        keyboard::listen().map(|event| match event {
            keyboard::Event::KeyPressed {
                key: Key::Character(c),
                ..
            } if c == "=" || c == "+" => Message::ScaleUp,
            keyboard::Event::KeyPressed {
                key: Key::Character(c),
                ..
            } if c == "-" => Message::ScaleDown,
            keyboard::Event::KeyPressed {
                key: Key::Character(c),
                ..
            } if c == "0" => Message::ScaleReset,
            _ => Message::Noop,
        })
    }

    pub fn view(&self) -> Element<'_, Message> {
        if self.mode == Mode::Profiles {
            return column![
                components::top_bar::view(self.mode, self.is_fullscreen),
                rule::horizontal(1),
                components::profiles::view(&self.profiles),
            ]
            .into();
        }

        column![
            components::top_bar::view(self.mode, self.is_fullscreen),
            rule::horizontal(1),
            components::typing_panel::view(
                &self.name_input,
                &self.session,
                self.profiles.len(),
                self.current_prompt,
            ),
            rule::horizontal(1),
            components::info_panel::view(&self.session, &self.live_plot, self.id_plot.as_ref(),),
        ]
        .into()
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}
