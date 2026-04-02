use iced::alignment::Vertical;
use iced::widget::scrollable::{Direction, Scrollbar};
use iced::widget::{Space, button, column, container, row, scrollable, text};
use iced::{Color, Element, Length};

use crate::app::Message;
use crate::styles;
use crate::typing::Profile;
use crate::widgets::bar_chart::Heatmap;

pub fn view<'a>(profiles: &'a [Profile]) -> Element<'a, Message> {
    if profiles.is_empty() {
        return container(
            text("No profiles enrolled yet. Type on the Demo page and press Enroll.")
                .size(12)
                .color(dim()),
        )
        .padding(styles::PAD)
        .width(Length::Fill)
        .height(Length::Fill)
        .into();
    }

    let cards: Vec<Element<'_, Message>> = profiles
        .iter()
        .enumerate()
        .map(|(i, p)| profile_card(i, p))
        .collect();

    let content = row(cards).spacing(12).wrap();

    scrollable(container(content).padding(styles::PAD).width(Length::Fill))
        .direction(Direction::Vertical(
            Scrollbar::new().width(4).scroller_width(4),
        ))
        .style(styles::invisible_scroll)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn profile_card<'a>(index: usize, profile: &'a Profile) -> Element<'a, Message> {
    let del_btn = button(text("Delete").size(11))
        .style(styles::delete_btn)
        .padding([2, 6])
        .on_press(Message::DeleteProfile(index));

    let header = row![
        text(profile.name.as_str())
            .size(14)
            .color(Color::from_rgb(0.90, 0.90, 0.90)),
        Space::new().width(Length::Fill),
        del_btn,
    ]
    .align_y(Vertical::Center);

    let stats = row![
        stat_label(&format!("{:.0}", profile.wpm), "WPM"),
        Space::new().width(8.0),
        stat_label(&format!("{:.0}ms", profile.avg_interval_ms()), "avg"),
        Space::new().width(8.0),
        stat_label(&format!("{:.0}ms", profile.avg_dwell_ms), "dwell"),
        Space::new().width(8.0),
        stat_label(&format!("{}", profile.bigrams.len()), "bigrams"),
        Space::new().width(8.0),
        stat_label(&format!("{}", profile.interval_count), "total"),
        Space::new().width(Length::Fill),
    ]
    .align_y(Vertical::Center);

    let heatmap = Heatmap::new(&profile.bigrams);

    container(
        column![
            header,
            stats,
            container(heatmap).width(Length::Fill).height(Length::Fill),
        ]
        .spacing(8)
        .padding(styles::PAD),
    )
    .style(styles::card_style)
    .width(Length::Fixed(280.0))
    .height(Length::Fixed(300.0))
    .into()
}

fn stat_label<'a>(value: &str, label: &'a str) -> Element<'a, Message> {
    column![
        text(value.to_string())
            .size(12)
            .color(Color::from_rgb(0.80, 0.80, 0.80)),
        text(label).size(9).color(dim()),
    ]
    .spacing(1)
    .into()
}

fn dim() -> Color {
    Color::from_rgb(0.5, 0.5, 0.5)
}
