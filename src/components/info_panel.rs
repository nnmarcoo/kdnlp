use iced::alignment::Vertical;
use iced::widget::scrollable::{Direction, Scrollbar};
use iced::widget::{Space, column, container, row, rule, scrollable, text};
use iced::{Color, Element, Length};

use iced::widget::canvas::Cache;

use crate::app::Message;
use crate::styles;
use crate::widgets::scatter::ScatterPlot;

pub fn view<'a>(
    id_results: &'a [(String, f64)],
    scatter_points: &'a [(String, [f32; 2])],
    scatter_session: Option<[f32; 2]>,
    scatter_cache: &'a Cache,
) -> Element<'a, Message> {
    row![
        dash_card("Rankings", rankings_content(id_results)),
        dash_card(
            "Embedding Space",
            embedding_space_content(scatter_points, scatter_session, scatter_cache)
        ),
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

fn rankings_content<'a>(id_results: &'a [(String, f64)]) -> Element<'a, Message> {
    if id_results.is_empty() {
        return placeholder_content("Start typing to see ranked matches.");
    }

    let rows: Vec<Element<'_, Message>> = id_results
        .iter()
        .enumerate()
        .map(|(i, (name, score))| {
            row![
                text(format!("{}.", i + 1))
                    .size(11)
                    .color(dim())
                    .width(Length::Fixed(22.0)),
                text(name.as_str())
                    .size(12)
                    .color(Color::from_rgb(0.85, 0.85, 0.85)),
                Space::new().width(Length::Fill),
                text(format!("{:.3}", score))
                    .size(11)
                    .color(dim())
                    .width(Length::Fixed(40.0)),
            ]
            .align_y(Vertical::Center)
            .spacing(6)
            .into()
        })
        .collect();

    scrollable(
        column(rows)
            .spacing(10)
            .width(Length::Fill)
            .padding(iced::Padding {
                top: 0.0,
                right: 12.0,
                bottom: 0.0,
                left: 0.0,
            }),
    )
    .direction(Direction::Vertical(
        Scrollbar::new().width(4).scroller_width(4),
    ))
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn embedding_space_content<'a>(
    points: &'a [(String, [f32; 2])],
    session_pt: Option<[f32; 2]>,
    cache: &'a Cache,
) -> Element<'a, Message> {
    if points.len() < 2 {
        return placeholder_content("Enroll at least 2 users to see the embedding space.");
    }
    ScatterPlot::new(points.to_vec(), session_pt, cache).into_element()
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
