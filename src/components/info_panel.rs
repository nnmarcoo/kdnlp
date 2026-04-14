use iced::alignment::Vertical;
use iced::widget::{Space, column, container, row, rule, text};
use iced::{Background, Border, Color, Element, Length};

use crate::app::Message;
use crate::pca;
use crate::styles;
use crate::typing::{Profile, Session};
use crate::widgets::bar_chart::Heatmap;
use crate::widgets::scatter::ScatterPlot;

pub fn view<'a>(
    session: &'a Session,
    profiles: &'a [Profile],
    id_results: &'a [(String, f64)],
) -> Element<'a, Message> {
    row![
        dash_card("Flight Times", flight_times_content(session)),
        dash_card("Rankings", rankings_content(id_results)),
        dash_card("Model Output", model_output_content(id_results)),
        dash_card("Fingerprint Space", fingerprint_space_content(session, profiles)),
    ]
    .spacing(styles::PAD)
    .padding(styles::PAD)
    .height(Length::Fill)
    .into()
}

fn dash_card<'a>(title: &'a str, content: Element<'a, Message>) -> Element<'a, Message> {
    container(
        column![
            text(title)
                .size(12)
                .color(Color::from_rgb(0.75, 0.75, 0.75)),
            rule::horizontal(1),
            content,
        ]
        .spacing(8),
    )
    .style(styles::card_style)
    .padding(styles::PAD)
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn flight_times_content<'a>(session: &'a Session) -> Element<'a, Message> {
    if session.is_empty() {
        return placeholder_content("Start typing to see your keystroke intervals.");
    }

    let stats = stats_row(session);

    column![stats, Heatmap::from_vecs(&session.bigrams)]
        .spacing(6)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn stats_row<'a>(session: &'a Session) -> Element<'a, Message> {
    row![
        stat_pill("WPM", &format!("{:.0}", session.wpm())),
        stat_pill("avg", &format!("{:.0}ms", session.avg_interval_ms())),
        stat_pill("dwell", &format!("{:.0}ms", session.avg_dwell_ms())),
        stat_pill("bigrams", &format!("{}", session.unique_bigram_count())),
        stat_pill("total", &format!("{}", session.interval_count())),
        Space::new().width(Length::Fill),
    ]
    .spacing(8)
    .into()
}

fn stat_pill<'a>(label: &'a str, value: &str) -> Element<'a, Message> {
    column![
        text(value.to_string())
            .size(12)
            .color(Color::from_rgb(0.85, 0.85, 0.85)),
        text(label).size(9).color(dim()),
    ]
    .spacing(1)
    .into()
}

fn rankings_content<'a>(id_results: &'a [(String, f64)]) -> Element<'a, Message> {
    if id_results.is_empty() {
        return placeholder_content("Run Identify to see ranked matches.");
    }

    let max_dist = id_results
        .last()
        .map(|r| r.1)
        .unwrap_or(1.0)
        .max(1.0);

    let rows: Vec<Element<'_, Message>> = id_results
        .iter()
        .enumerate()
        .map(|(i, (name, dist))| {
            let t = (dist / max_dist).clamp(0.0, 1.0) as f32;
            let color = Color::from_rgb(0.38 + 0.54 * t, 0.82 - 0.47 * t, 0.48 - 0.13 * t);
            let filled = (((1.0 - t) * 90.0 + 5.0) as u16).max(1);
            let empty = 100u16.saturating_sub(filled);

            let bar = row![
                container(Space::new())
                    .style(move |_: &iced::Theme| container::Style {
                        background: Some(Background::Color(color)),
                        border: Border {
                            radius: 3.0.into(),
                            ..Border::default()
                        },
                        ..container::Style::default()
                    })
                    .width(Length::FillPortion(filled))
                    .height(Length::Fixed(6.0)),
                Space::new().width(Length::FillPortion(empty)),
            ];

            row![
                text(format!("{}.", i + 1))
                    .size(11)
                    .color(dim())
                    .width(Length::Fixed(18.0)),
                text(name.as_str())
                    .size(12)
                    .color(Color::from_rgb(0.85, 0.85, 0.85))
                    .width(Length::Fixed(80.0)),
                bar,
                Space::new().width(Length::Fixed(6.0)),
                text(format!("{:.0}ms", dist)).size(11).color(dim()),
            ]
            .align_y(Vertical::Center)
            .spacing(6)
            .into()
        })
        .collect();

    column(rows)
        .spacing(10)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn model_output_content<'a>(id_results: &'a [(String, f64)]) -> Element<'a, Message> {
    if id_results.is_empty() {
        return placeholder_content("Run Identify to see the best match.");
    }

    let (name, dist) = &id_results[0];

    column![
        text(name.as_str())
            .size(28)
            .color(Color::from_rgb(0.90, 0.90, 0.90)),
        text(format!("{:.0}ms avg distance", dist))
            .size(11)
            .color(dim()),
    ]
    .spacing(4)
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn fingerprint_space_content<'a>(
    session: &'a Session,
    profiles: &'a [Profile],
) -> Element<'a, Message> {
    if profiles.len() < 2 {
        return placeholder_content("Enroll 2+ profiles to see the projection.");
    }
    if session.is_empty() {
        return placeholder_content("Start typing to see where you land.");
    }

    let global_mean = {
        let all: Vec<f64> = profiles
            .iter()
            .flat_map(|p| p.bigrams.values().copied())
            .collect();
        if all.is_empty() {
            200.0
        } else {
            all.iter().sum::<f64>() / all.len() as f64
        }
    };

    let profile_vecs: Vec<(String, std::collections::HashMap<(char, char), f64>)> = profiles
        .iter()
        .map(|p| (p.name.clone(), p.bigrams.clone()))
        .collect();

    let session_avg = session.averaged();
    let session_input = if session.is_empty() {
        None
    } else {
        Some(&session_avg)
    };

    let (projected, session_pt) =
        pca::project_profiles(&profile_vecs, session_input, global_mean);

    ScatterPlot::new(projected, session_pt).into()
}

fn placeholder_content<'a>(msg: &'static str) -> Element<'a, Message> {
    container(text(msg).size(11).color(dim()))
        .width(Length::Fill)
        .height(Length::Fill)
        .center(Length::Fill)
        .into()
}

fn dim() -> Color {
    Color::from_rgb(0.45, 0.45, 0.45)
}
