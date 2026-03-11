use iced::widget::{column, rule};
use iced::{Element, Task, Theme, window};
use std::time::Instant;

use crate::components;
use crate::store;
use crate::typing::{Profile, Session, identify, random_prompt};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Enroll,
    Identify,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InfoTab {
    #[default]
    Data,
    Profiles,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BigramSort {
    #[default]
    Chronological,
    Alphabetical,
    ByMs,
}

#[derive(Default)]
pub enum Status {
    #[default]
    Idle,
    Enrolled,
    Identified(Vec<(String, f64)>),
    NotEnoughData,
    NoProfiles,
}

pub struct App {
    pub mode: Mode,
    pub info_tab: InfoTab,
    pub bigram_sort: BigramSort,
    pub is_fullscreen: bool,
    pub name_input: String,
    pub session: Session,
    pub profiles: Vec<Profile>,
    pub status: Status,
    pub current_prompt: &'static str,
}

impl Default for App {
    fn default() -> Self {
        Self {
            mode: Mode::Enroll,
            info_tab: InfoTab::default(),
            bigram_sort: BigramSort::default(),
            is_fullscreen: false,
            name_input: String::new(),
            session: Session::default(),
            profiles: store::load(),
            status: Status::Idle,
            current_prompt: random_prompt(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    ModeChanged(Mode),
    InfoTabChanged(InfoTab),
    SortChanged(BigramSort),
    DeleteProfile(usize),
    ToggleFullscreen,
    NameChanged(String),
    KeyPressed(char, u32, Instant),
    KeyReleased(char, Instant),
    Backspace(Instant),
    BackspaceReleased(Instant),
    Submit,
    Clear,
}

impl App {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ModeChanged(mode) => {
                self.mode = mode;
                self.session.clear();
            }
            Message::InfoTabChanged(tab) => {
                self.info_tab = tab;
            }
            Message::SortChanged(sort) => {
                self.bigram_sort = sort;
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
                }
            }
            Message::KeyReleased(ch, t) => {
                self.session.push_release(ch, t);
            }
            Message::Backspace(t) => {
                self.session.push_backspace(t);
            }
            Message::BackspaceReleased(t) => {
                self.session.push_backspace_release(t);
            }
            Message::Submit => match self.mode {
                Mode::Enroll => {
                    if self.name_input.trim().is_empty() || self.session.is_empty() {
                        self.status = Status::NotEnoughData;
                        return Task::none();
                    }
                    let name = self.name_input.trim().to_string();
                    self.profiles
                        .push(Profile::from_session(name, &self.session));
                    store::save(&self.profiles);
                    self.status = Status::Enrolled;
                    self.name_input.clear();
                    self.session.clear();
                    self.current_prompt = random_prompt();
                }
                Mode::Identify => {
                    if self.profiles.is_empty() {
                        self.status = Status::NoProfiles;
                        return Task::none();
                    }
                    if self.session.is_empty() {
                        self.status = Status::NotEnoughData;
                        return Task::none();
                    }
                    let ranked = identify(&self.session, &self.profiles);
                    self.status = Status::Identified(ranked);
                    self.session.clear();
                    self.current_prompt = random_prompt();
                }
            },
            Message::Clear => {
                self.session.clear();
                self.current_prompt = random_prompt();
                self.status = Status::Idle;
            }
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        column![
            components::top_bar::view(self.mode, self.is_fullscreen),
            rule::horizontal(1),
            components::typing_panel::view(
                self.mode,
                &self.name_input,
                &self.session,
                self.current_prompt
            ),
            rule::horizontal(1),
            components::info_panel::view(
                &self.status,
                &self.session,
                &self.profiles,
                self.info_tab,
                self.bigram_sort
            ),
        ]
        .into()
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}
