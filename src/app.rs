use iced::keyboard::{self, Key};
use iced::widget::{column, rule};
use iced::{Element, Subscription, Task, Theme, window};
use std::time::Instant;

use crate::components;
use crate::plots;
use crate::store;
use crate::typing::{Profile, Session, next_prompt};

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
    pub profiles: Vec<Profile>,      // user-enrolled, persisted
    pub demo_profiles: Vec<Profile>, // loaded from embedded JSON, not persisted
    pub current_prompt: &'static str,
    pub id_results: Vec<(String, f64)>,
    pub profile_search: String,
    pub scatter_points: Vec<(String, [f32; 2])>,
    pub scatter_session: Option<[f32; 2]>,
    pub scatter_cache: iced::widget::canvas::Cache,
    pub fixed_prompt: bool,
    ranking_in_flight: bool,
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
            demo_profiles: Vec::new(),
            current_prompt: next_prompt(""),
            id_results: Vec::new(),
            profile_search: String::new(),
            scatter_points: Vec::new(),
            scatter_session: None,
            scatter_cache: iced::widget::canvas::Cache::new(),
            fixed_prompt: false,
            ranking_in_flight: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    ModeChanged(Mode),
    DeleteProfile(usize),
    ToggleFullscreen,
    NameChanged(String),
    ProfileSearchChanged(String),
    KeyPressed(char, u32, Instant),
    KeyReleased(char, Instant),
    Backspace(Instant),
    BackspaceReleased(Instant),
    Enroll,
    Identify,
    LoadDemo(usize),
    RankingsDone(
        Vec<(String, f64)>,
        Vec<(String, [f32; 2])>,
        Option<[f32; 2]>,
    ),
    Clear,
    ClearProfiles,
    ToggleFixedPrompt,
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
            Message::ProfileSearchChanged(query) => {
                self.profile_search = query;
            }
            Message::KeyPressed(ch, keycode, t) => {
                if self.session.text.len() < self.current_prompt.len() {
                    self.session.push_char(ch, keycode, t);
                    self.session.text.push(ch);
                }
            }
            Message::KeyReleased(ch, t) => {
                self.session.push_release(ch, t);
                let all: Vec<Profile> = self.all_profiles().cloned().collect();
                if !all.is_empty() && self.session.interval_count() >= 5 && !self.ranking_in_flight
                {
                    self.ranking_in_flight = true;
                    let session = self.session.clone();
                    let profiles = all;
                    return Task::perform(
                        async move {
                            let rankings = plots::rank_profiles(&session, &profiles);
                            let embedded: Vec<(String, &[f32; 128])> = profiles
                                .iter()
                                .filter_map(|p| {
                                    p.embedding.as_ref().map(|e| (p.name.clone(), e.as_ref()))
                                })
                                .collect();
                            let session_emb = crate::embedder::embed(&session);
                            let (mut points, session_pt) = if embedded.len() >= 2 {
                                crate::pca::project(&embedded, session_emb.as_ref())
                            } else {
                                (Vec::new(), None)
                            };
                            // Order scatter points to match ranking order so top 5 get labels
                            let rank_order: std::collections::HashMap<&str, usize> = rankings
                                .iter()
                                .enumerate()
                                .map(|(i, (name, _))| (name.as_str(), i))
                                .collect();
                            points.sort_by_key(|(name, _)| {
                                rank_order.get(name.as_str()).copied().unwrap_or(usize::MAX)
                            });
                            (rankings, points, session_pt)
                        },
                        |(rankings, points, session_pt)| {
                            Message::RankingsDone(rankings, points, session_pt)
                        },
                    );
                }
            }
            Message::RankingsDone(results, points, session_pt) => {
                self.id_results = results;
                self.scatter_points = points;
                self.scatter_session = session_pt;
                self.scatter_cache.clear();
                self.ranking_in_flight = false;
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
                    // Update embedding: average old and new, then re-normalize
                    if let Some(new_emb) = crate::embedder::embed(&self.session) {
                        existing.embedding = Some(Box::new(match &existing.embedding {
                            Some(old_emb) => {
                                let mut avg = [0f32; 128];
                                for i in 0..128 {
                                    avg[i] = (old_emb[i] + new_emb[i]) * 0.5;
                                }
                                let norm: f32 =
                                    avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                                avg.iter_mut().for_each(|x| *x /= norm);
                                avg
                            }
                            None => new_emb,
                        }));
                    }
                } else {
                    self.profiles
                        .push(Profile::from_session(name, &self.session));
                }
                store::save(&self.profiles);
                self.name_input.clear();
                self.session.clear();
                self.current_prompt = self.next_prompt();
            }
            Message::Identify => {
                let all: Vec<Profile> = self.all_profiles().cloned().collect();
                if all.is_empty() || self.session.is_empty() {
                    return Task::none();
                }
                self.id_results = plots::rank_profiles(&self.session, &all);
            }
            Message::LoadDemo(n) => {
                let enrolled_names: std::collections::HashSet<String> =
                    self.profiles.iter().map(|p| p.name.clone()).collect();
                self.demo_profiles = store::load_demo(n)
                    .into_iter()
                    .filter(|p| !enrolled_names.contains(&p.name))
                    .collect();
            }
            Message::Clear => {
                self.session.clear();
                self.id_results.clear();
                self.scatter_points.clear();
                self.scatter_session = None;
                self.scatter_cache.clear();
                self.ranking_in_flight = false;
                self.current_prompt = self.next_prompt();
            }
            Message::ClearProfiles => {
                self.profiles.clear();
                self.demo_profiles.clear();
                store::save(&self.profiles);
                self.id_results.clear();
                self.scatter_points.clear();
                self.scatter_session = None;
                self.scatter_cache.clear();
            }
            Message::ToggleFixedPrompt => {
                self.fixed_prompt = !self.fixed_prompt;
                self.session.clear();
                self.id_results.clear();
                self.current_prompt = self.next_prompt();
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
                components::top_bar::view(self.mode),
                rule::horizontal(1),
                components::profiles::view(
                    &self.profiles,
                    &self.demo_profiles,
                    &self.profile_search
                ),
            ]
            .into();
        }

        column![
            components::top_bar::view(self.mode),
            rule::horizontal(1),
            components::typing_panel::view(&self.name_input, &self.session, self.current_prompt,),
            rule::horizontal(1),
            components::info_panel::view(
                &self.id_results,
                &self.scatter_points,
                self.scatter_session,
                &self.scatter_cache
            ),
        ]
        .into()
    }

    pub fn all_profiles(&self) -> impl Iterator<Item = &Profile> {
        self.profiles.iter().chain(self.demo_profiles.iter())
    }

    fn next_prompt(&self) -> &'static str {
        if self.fixed_prompt {
            crate::typing::PROMPTS[0]
        } else {
            next_prompt(self.current_prompt)
        }
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }
}
