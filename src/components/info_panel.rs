use iced::widget::{column, container, row, rule, text};
use iced::{Color, Element, Length};
use iced_plot::PlotWidget;

use crate::app::Message;
use crate::styles;
use crate::typing::Session;

pub fn view<'a>(
    session: &'a Session,
    live_plot: &'a PlotWidget,
    id_plot: Option<&'a PlotWidget>,
) -> Element<'a, Message> {
    row![
        dash_card("Flight Times", flight_times_content(session, live_plot)),
        dash_card("Fingerprint Space", fingerprint_content(id_plot)),
        dash_card("Model Output", placeholder_content("No model loaded yet.")),
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

fn flight_times_content<'a>(
    session: &'a Session,
    live_plot: &'a PlotWidget,
) -> Element<'a, Message> {
    if session.is_empty() {
        return placeholder_content("Start typing to see your keystroke intervals.");
    }
    live_plot.view().map(Message::LivePlotMsg)
}

fn fingerprint_content<'a>(id_plot: Option<&'a PlotWidget>) -> Element<'a, Message> {
    match id_plot {
        Some(plot) => plot.view().map(Message::IdPlotMsg),
        None => placeholder_content(
            "Run Identify to plot your typing fingerprint against enrolled profiles.",
        ),
    }
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
