use iced::widget::{Space, column, container, row, rule, text};
use iced::{Color, Element, Length};
use iced_plot::PlotWidget;

use crate::app::Message;
use crate::styles;
use crate::typing::Session;
use crate::widgets::bar_chart::Heatmap;

pub fn view<'a>(session: &'a Session, id_plot: Option<&'a PlotWidget>) -> Element<'a, Message> {
    row![
        dash_card("Flight Times", flight_times_content(session)),
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

fn flight_times_content<'a>(session: &'a Session) -> Element<'a, Message> {
    if session.is_empty() {
        return placeholder_content("Start typing to see your keystroke intervals.");
    }

    let stats = stats_row(session);

    column![stats, Heatmap::from_vecs(&session.bigrams),]
        .spacing(6)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn stats_row<'a>(session: &'a Session) -> Element<'a, Message> {
    let wpm = session.wpm();
    let avg_iki = session.avg_interval_ms();
    let avg_dwell = session.avg_dwell_ms();
    let unique = session.unique_bigram_count();
    let total = session.interval_count();

    row![
        stat_pill("WPM", &format!("{:.0}", wpm)),
        stat_pill("avg", &format!("{:.0}ms", avg_iki)),
        stat_pill("dwell", &format!("{:.0}ms", avg_dwell)),
        stat_pill("bigrams", &format!("{}", unique)),
        stat_pill("total", &format!("{}", total)),
        Space::new().width(Length::Fill),
    ]
    .spacing(8)
    .into()
}

fn stat_pill<'a>(label: &'a str, value: &str) -> Element<'a, Message> {
    let label_el = text(label).size(9).color(dim());
    let value_el = text(value.to_string())
        .size(12)
        .color(Color::from_rgb(0.85, 0.85, 0.85));
    column![value_el, label_el].spacing(1).into()
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
