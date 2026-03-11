use iced::alignment::Vertical;
use iced::widget::scrollable::{Direction, Scrollbar};
use iced::widget::{Space, button, column, container, row, rule, scrollable, text};
use iced::{Background, Color, Element, Length};

use crate::app::{BigramSort, InfoTab, Message, Status};
use crate::styles;
use crate::typing::{KeyEvent, Profile, Session, display_char};

const BAR_W: f32 = 150.0;
const MAX_FLIGHT_MS: f64 = 600.0;

pub fn view<'a>(
    status: &'a Status,
    session: &'a Session,
    profiles: &'a [Profile],
    tab: InfoTab,
    sort: BigramSort,
) -> Element<'a, Message> {
    let tab_bar = row![
        if tab == InfoTab::Data {
            button("Data").style(styles::mode_btn_active)
        } else {
            button("Data")
                .style(styles::mode_btn)
                .on_press(Message::InfoTabChanged(InfoTab::Data))
        },
        if tab == InfoTab::Profiles {
            button(text(format!("Profiles ({})", profiles.len()))).style(styles::mode_btn_active)
        } else {
            button(text(format!("Profiles ({})", profiles.len())))
                .style(styles::mode_btn)
                .on_press(Message::InfoTabChanged(InfoTab::Profiles))
        },
    ]
    .spacing(styles::SPACING);

    let body = match tab {
        InfoTab::Data => {
            if !session.is_empty() {
                view_live(session, sort)
            } else {
                match status {
                    Status::Enrolled => profiles
                        .last()
                        .map(view_profile_summary)
                        .unwrap_or_else(view_placeholder),
                    Status::Identified(ranked) => view_identified(ranked),
                    Status::NotEnoughData => {
                        view_message("Not enough data — type more of the passage.")
                    }
                    Status::NoProfiles => {
                        view_message("No profiles enrolled. Switch to Enroll mode first.")
                    }
                    Status::Idle => view_placeholder(),
                }
            }
        }
        InfoTab::Profiles => view_profiles(profiles),
    };

    container(
        column![
            container(tab_bar)
                .width(Length::Fill)
                .padding([6.0, styles::PAD]),
            rule::horizontal(1),
            container(body)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(styles::PAD),
        ]
        .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn view_placeholder<'a>() -> Element<'a, Message> {
    column![
        section_label("Live Session"),
        text("Start typing in the passage above to see your keystroke data.")
            .size(12)
            .color(dim()),
    ]
    .spacing(8)
    .into()
}

fn view_message<'a>(msg: &str) -> Element<'a, Message> {
    text(msg.to_string()).size(12).color(warn()).into()
}

fn view_live<'a>(session: &'a Session, sort: BigramSort) -> Element<'a, Message> {
    let n_backspaces = session.events.iter().filter(|e| e.key == '\x08').count();
    let avg_dwell = avg_dwell_ms(&session.events);
    let wpm = estimate_wpm(&session.events, session.text.len());

    let stats = row![
        chip(format!("{} intervals", session.interval_count())),
        chip(format!("{} corrections", n_backspaces)),
        chip(avg_dwell.map_or("dwell —".into(), |d| format!("dwell {:.0}ms", d))),
        chip(wpm.map_or("— wpm".into(), |w| format!("{:.0} wpm", w))),
    ]
    .spacing(6);

    let items: Vec<((char, char), f64)> = match sort {
        BigramSort::Chronological => session.log.iter().copied().rev().collect(),
        BigramSort::Alphabetical => {
            let mut v: Vec<_> = session.log.iter().copied().collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        }
        BigramSort::ByMs => {
            let mut v: Vec<_> = session.log.iter().copied().collect();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            v
        }
    };

    let max_ms = items
        .iter()
        .map(|&(_, ms)| ms)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        .min(MAX_FLIGHT_MS);

    let bars: Vec<Element<'_, Message>> = items
        .into_iter()
        .map(|((a, b), ms)| {
            flight_bar(
                format!("{}-{}", display_char(a), display_char(b)),
                ms,
                max_ms,
            )
        })
        .collect();

    column![
        section_label("Live Session"),
        stats,
        sort_bar(sort),
        scrollable(column(bars).spacing(4))
            .direction(Direction::Vertical(
                Scrollbar::new().width(4).scroller_width(4)
            ))
            .style(styles::invisible_scroll)
            .width(Length::Shrink)
            .height(Length::Fill),
    ]
    .spacing(8)
    .height(Length::Fill)
    .into()
}

fn view_profile_summary<'a>(profile: &'a Profile) -> Element<'a, Message> {
    let n_keys = profile.events.iter().filter(|e| e.key != '\x08').count();
    let n_backspaces = profile.events.iter().filter(|e| e.key == '\x08').count();
    let avg_dwell = avg_dwell_ms(&profile.events);

    let stats = row![
        chip(format!("{} bigrams", profile.bigrams.len())),
        chip(format!("{} keystrokes", n_keys)),
        chip(format!("{} corrections", n_backspaces)),
        chip(avg_dwell.map_or("dwell —".into(), |d| format!("dwell {:.0}ms", d))),
    ]
    .spacing(6);

    let top = profile.top_bigrams(profile.bigrams.len());
    let max_ms = top
        .iter()
        .map(|(_, ms)| *ms)
        .fold(0.0_f64, f64::max)
        .max(1.0)
        .min(MAX_FLIGHT_MS);

    let bars: Vec<Element<'_, Message>> = top
        .into_iter()
        .map(|((a, b), ms)| {
            flight_bar(
                format!("{}-{}", display_char(a), display_char(b)),
                ms,
                max_ms,
            )
        })
        .collect();

    column![
        section_label(format!("Profile — {}", profile.name)),
        stats,
        column(bars).spacing(4),
    ]
    .spacing(8)
    .into()
}

fn view_identified<'a>(ranked: &'a [(String, f64)]) -> Element<'a, Message> {
    if ranked.is_empty() {
        return view_message("No profiles to compare against. Enroll some profiles first.");
    }

    let (winner, best_dist) = &ranked[0];
    let max_dist = ranked
        .iter()
        .map(|(_, d)| *d)
        .fold(0.0_f64, f64::max)
        .max(1.0);

    let winner_row = row![
        text("Best match: ").size(13).color(dim()),
        text(winner.as_str()).size(13).color(green()),
        Space::new().width(Length::Fill),
        text(format!("{:.1} ms RMS", best_dist))
            .size(11)
            .color(dim()),
    ]
    .align_y(Vertical::Center);

    let rank_rows: Vec<Element<'_, Message>> = ranked
        .iter()
        .enumerate()
        .map(|(i, (name, dist))| rank_row(i + 1, name, *dist, max_dist, i == 0))
        .collect();

    column![
        section_label("Identification Result"),
        winner_row,
        column(rank_rows).spacing(5),
    ]
    .spacing(8)
    .into()
}

fn view_profiles<'a>(profiles: &'a [Profile]) -> Element<'a, Message> {
    if profiles.is_empty() {
        return column![
            section_label("Saved Profiles"),
            text("No profiles enrolled yet. Switch to Enroll mode to add one.")
                .size(12)
                .color(dim()),
        ]
        .spacing(8)
        .into();
    }

    let card_rows: Vec<Element<'_, Message>> = profiles
        .chunks(3)
        .enumerate()
        .map(|(chunk_i, chunk)| {
            let mut items: Vec<Element<'_, Message>> = chunk
                .iter()
                .enumerate()
                .map(|(j, p)| profile_card(chunk_i * 3 + j, p))
                .collect();
            items.push(Space::new().width(Length::Fill).into());
            row(items).spacing(10).into()
        })
        .collect();

    let content = column![
        section_label(format!("{} Saved Profile(s)", profiles.len())),
        column(card_rows).spacing(10),
    ]
    .spacing(8);

    scrollable(content)
        .direction(Direction::Vertical(
            Scrollbar::new().width(4).scroller_width(4),
        ))
        .style(styles::invisible_scroll)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn profile_card<'a>(index: usize, profile: &'a Profile) -> Element<'a, Message> {
    let n_backspaces = profile.events.iter().filter(|e| e.key == '\x08').count();
    let avg_dwell = avg_dwell_ms(&profile.events);

    let del_btn = button(text("Delete").size(11))
        .style(styles::delete_btn)
        .padding([2, 6])
        .on_press(Message::DeleteProfile(index));

    let header = row![
        text(profile.name.as_str())
            .size(13)
            .color(Color::from_rgb(0.88, 0.88, 0.88)),
        Space::new().width(Length::Fill),
        del_btn,
    ]
    .align_y(Vertical::Center);

    let meta = text(format!(
        "{} bigrams  ·  {} corr  ·  {}",
        profile.bigrams.len(),
        n_backspaces,
        avg_dwell.map_or("—".into(), |d| format!("{:.0}ms dwell", d)),
    ))
    .size(11)
    .color(dim());

    let top = profile.top_bigrams(profile.bigrams.len());
    let min_ms = top.first().map(|(_, ms)| *ms).unwrap_or(0.0);
    let max_ms = top.last().map(|(_, ms)| *ms).unwrap_or(1.0);
    let range = (max_ms - min_ms).max(1.0);

    let flight_rows: Vec<Element<'_, Message>> = top
        .into_iter()
        .map(|((a, b), ms)| {
            let ratio = ((ms - min_ms) / range).clamp(0.0, 1.0) as f32;
            let swatch_color = Color::from_rgb(
                0.38 + 0.54 * ratio,
                0.82 - 0.47 * ratio,
                0.48 - 0.13 * ratio,
            );
            row![
                text(format!("{}-{}", display_char(a), display_char(b)))
                    .size(11)
                    .color(Color::from_rgb(0.72, 0.72, 0.72)),
                container(Space::new().height(10.0))
                    .width(Length::Fill)
                    .style(move |_: &iced::Theme| iced::widget::container::Style {
                        background: Some(Background::Color(swatch_color)),
                        border: iced::border::rounded(2.0),
                        ..Default::default()
                    }),
                text(format!("{:.0}ms", ms)).size(11).color(dim()),
            ]
            .align_y(Vertical::Center)
            .spacing(5)
            .into()
        })
        .collect();

    let flight_list = scrollable(column(flight_rows).spacing(3))
        .direction(Direction::Vertical(
            Scrollbar::new().width(3).scroller_width(3),
        ))
        .style(styles::invisible_scroll)
        .height(Length::Fixed(80.0));

    container(
        column![header, meta, flight_list]
            .spacing(8)
            .padding(styles::PAD),
    )
    .style(styles::card_style)
    .width(Length::Fixed(220.0))
    .into()
}

fn section_label<'a>(s: impl ToString) -> Element<'a, Message> {
    text(s.to_string())
        .size(13)
        .color(Color::from_rgb(0.88, 0.88, 0.88))
        .into()
}

fn chip<'a>(s: impl ToString) -> Element<'a, Message> {
    container(
        text(s.to_string())
            .size(11)
            .color(Color::from_rgb(0.75, 0.75, 0.75)),
    )
    .style(styles::chip_style)
    .padding([3, 8])
    .into()
}

fn sort_bar<'a>(current: BigramSort) -> Element<'a, Message> {
    let btn = |label: &'static str, sort: BigramSort| {
        let is_active = current == sort;
        let style: fn(&iced::Theme, iced::widget::button::Status) -> iced::widget::button::Style =
            if is_active {
                styles::sort_btn_active
            } else {
                styles::sort_btn
            };
        let b = button(text(label).size(11)).style(style).padding([3, 8]);
        if is_active {
            b
        } else {
            b.on_press(Message::SortChanged(sort))
        }
    };
    row![
        btn("Recent", BigramSort::Chronological),
        btn("A-Z", BigramSort::Alphabetical),
        btn("Speed", BigramSort::ByMs),
    ]
    .spacing(2)
    .into()
}

fn flight_bar<'a>(label: String, ms: f64, max_ms: f64) -> Element<'a, Message> {
    let fill_w = ((ms / max_ms).min(1.0) as f32 * BAR_W).max(2.0);

    row![
        text(label)
            .size(11)
            .width(Length::Fixed(38.0))
            .color(Color::from_rgb(0.75, 0.75, 0.75)),
        container(Space::new().width(fill_w).height(8.0)).style(styles::bar_fill),
        text(format!("{:.0}ms", ms))
            .size(10)
            .color(Color::from_rgb(0.5, 0.5, 0.5)),
    ]
    .align_y(Vertical::Center)
    .spacing(5)
    .into()
}

fn rank_row<'a>(
    rank: usize,
    name: &'a str,
    dist: f64,
    max_dist: f64,
    is_best: bool,
) -> Element<'a, Message> {
    let sim = 1.0 - (dist / (max_dist + 1e-9));
    let fill_w = (sim as f32 * BAR_W).max(2.0);
    let fill_style: fn(&iced::Theme) -> iced::widget::container::Style = if is_best {
        styles::bar_fill
    } else {
        styles::bar_fill_dim
    };
    let name_color = if is_best {
        green()
    } else {
        Color::from_rgb(0.6, 0.6, 0.6)
    };

    row![
        text(format!("{}.", rank))
            .size(11)
            .width(Length::Fixed(18.0))
            .color(dim()),
        text(name)
            .size(12)
            .width(Length::Fixed(90.0))
            .color(name_color),
        container(Space::new().width(fill_w).height(8.0)).style(fill_style),
        Space::new().width(Length::Fixed(4.0)),
        text(format!("{:.1}ms", dist)).size(10).color(dim()),
    ]
    .align_y(Vertical::Center)
    .spacing(4)
    .into()
}

fn avg_dwell_ms(events: &[KeyEvent]) -> Option<f64> {
    let vals: Vec<f64> = events
        .iter()
        .filter(|e| e.key != '\x08')
        .filter_map(|e| e.dwell_ms())
        .collect();
    (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
}

fn estimate_wpm(events: &[KeyEvent], typed_len: usize) -> Option<f64> {
    let first = events.first()?.press_ms;
    let last = events.last()?.press_ms;
    let elapsed_ms = last.checked_sub(first).filter(|&e| e > 0)? as f64;
    let words = typed_len as f64 / 5.0;
    Some(words / (elapsed_ms / 60_000.0))
}

fn dim() -> Color {
    Color::from_rgb(0.5, 0.5, 0.5)
}

fn warn() -> Color {
    Color::from_rgb(0.8, 0.5, 0.3)
}

fn green() -> Color {
    Color::from_rgb(0.38, 0.82, 0.48)
}
