use iced::alignment::Vertical;
use iced::widget::scrollable::{Direction, Scrollbar};
use iced::widget::tooltip::Position;
use iced::widget::{Space, button, column, container, row, scrollable, text, tooltip};
use iced::{Background, Color, Element, Length};

use crate::app::Message;
use crate::styles;
use crate::typing::{KeyEvent, Profile, display_char};

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

    let content = row(cards).spacing(10).wrap();

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

    let tip = |t: &'static str| {
        container(text(t).size(11))
            .padding([4, 8])
            .style(styles::tooltip_style)
    };
    let meta = row![
        tooltip(
            text(format!("{}", profile.bigrams.len()))
                .size(11)
                .color(dim()),
            tip("bigrams"),
            Position::Bottom,
        ),
        text("  ").size(11),
        tooltip(
            text(format!("{}", n_backspaces)).size(11).color(dim()),
            tip("corrections"),
            Position::Bottom,
        ),
        text("  ").size(11),
        tooltip(
            text(avg_dwell.map_or("—".into(), |d| format!("{:.0}ms", d)))
                .size(11)
                .color(dim()),
            tip("avg dwell time"),
            Position::Bottom,
        ),
    ]
    .align_y(Vertical::Center);

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
                text(format!("{} {}", display_char(a), display_char(b)))
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
    .width(Length::Fixed(200.0))
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

fn dim() -> Color {
    Color::from_rgb(0.5, 0.5, 0.5)
}
