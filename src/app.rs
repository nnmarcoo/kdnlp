use iced::keyboard::{self, Key};
use iced::widget::{column, rule};
use iced::{Element, Subscription, Task, Theme, window};
use std::time::Instant;

use crate::components;
use crate::plots::{self, IdentificationMethod};
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
    pub id_results: Vec<(String, f64)>,
    pub method: IdentificationMethod,
}

impl Default for App {
    fn default() -> Self {
        Self {
            mode: Mode::Main,
            scale: 1.0,
            is_fullscreen: false,
            name_input: String::new(),
            session: Session::default(),
            profiles: store::load(),
            current_prompt: random_prompt(),
            id_results: Vec::new(),
            method: IdentificationMethod::FlightTime,
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
    MethodChanged(IdentificationMethod),
    Clear,
    ScaleUp,
    ScaleDown,
    ScaleReset,
    Noop,
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
            Message::Enroll => {
                if self.name_input.trim().is_empty() || self.session.is_empty() {
                    return Task::none();
                }
                let name = self.name_input.trim().to_lowercase();
                if let Some(existing) = self.profiles.iter_mut().find(|p| p.name == name) {
                    let new_bigrams = self.session.averaged();
                    for (bigram, new_avg) in &new_bigrams {
                        let new_count = self.session.bigrams[bigram].len();
                        let new_sum = new_avg * new_count as f64;
                        let old_count = existing.bigram_counts.get(bigram).copied().unwrap_or(1);
                        let total_count = old_count + new_count;
                        existing
                            .bigrams
                            .entry(*bigram)
                            .and_modify(|old_avg| {
                                *old_avg =
                                    (*old_avg * old_count as f64 + new_sum) / total_count as f64;
                            })
                            .or_insert(*new_avg);
                        existing
                            .bigram_counts
                            .entry(*bigram)
                            .and_modify(|c| *c += new_count)
                            .or_insert(new_count);
                    }
                    let new_chars = self.session.text.len();
                    let total_chars = existing.char_count + new_chars;
                    if total_chars > 0 {
                        existing.wpm = (existing.wpm * existing.char_count as f64
                            + self.session.wpm() * new_chars as f64)
                            / total_chars as f64;
                    }
                    let new_dwell_count = self
                        .session
                        .events
                        .iter()
                        .filter(|e| e.release_ms.is_some())
                        .count();
                    let total_dwell = existing.dwell_count + new_dwell_count;
                    if total_dwell > 0 {
                        existing.avg_dwell_ms = (existing.avg_dwell_ms
                            * existing.dwell_count as f64
                            + self.session.avg_dwell_ms() * new_dwell_count as f64)
                            / total_dwell as f64;
                    }
                    existing.dwell_count += new_dwell_count;
                    existing.char_count += new_chars;
                    existing.interval_count += self.session.interval_count();
                } else {
                    self.profiles
                        .push(Profile::from_session(name, &self.session));
                }
                store::save(&self.profiles);
                self.name_input.clear();
                self.session.clear();
                self.current_prompt = random_prompt();
            }
            Message::MethodChanged(method) => {
                self.method = method;
            }
            Message::Identify => {
                if self.profiles.is_empty() || self.session.is_empty() {
                    return Task::none();
                }
                self.id_results = plots::rank_profiles(self.method, &self.session, &self.profiles);
            }
            Message::Clear => {
                self.session.clear();
                self.id_results.clear();
                self.current_prompt = random_prompt();
            }
            Message::ScaleUp => self.scale = (self.scale + 0.1).min(3.0),
            Message::ScaleDown => self.scale = (self.scale - 0.1).max(0.5),
            Message::ScaleReset => self.scale = 1.0,
            Message::Noop => {}
        }
        Task::none()
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
                self.method,
            ),
            rule::horizontal(1),
            components::info_panel::view(&self.session, &self.id_results),
        ]
        .into()
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}
