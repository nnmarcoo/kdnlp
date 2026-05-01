use iced::alignment::Vertical;
use iced::widget::{Space, button, column, container, row, scrollable, text, text_input};
use iced::{Color, Element, Length};

use crate::app::Message;
use crate::styles;
use crate::typing::Profile;
use crate::widgets::bar_chart::Heatmap;

pub fn view<'a>(
    profiles: &'a [Profile],
    demo_profiles: &'a [Profile],
    search: &'a str,
) -> Element<'a, Message> {
    let total = profiles.len() + demo_profiles.len();
    if total == 0 {
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

    let search_bar = text_input("Search...", search)
        .on_input(Message::ProfileSearchChanged)
        .style(styles::name_input_style)
        .padding([5, 10])
        .width(Length::Fixed(200.0));

    let query = search.to_lowercase();

    let enrolled_filtered: Vec<(usize, &Profile)> = profiles
        .iter()
        .enumerate()
        .filter(|(_, p)| query.is_empty() || p.name.contains(&query))
        .collect();

    let demo_filtered: Vec<&Profile> = demo_profiles
        .iter()
        .filter(|p| query.is_empty() || p.name.contains(&query))
        .collect();

    let filtered_count = enrolled_filtered.len() + demo_filtered.len();
    let count_label = text(format!("{} / {}", filtered_count, total))
        .size(11)
        .color(dim());

    let clear_btn = button(text("Clear All").size(11))
        .style(styles::delete_btn)
        .padding([4, 10])
        .on_press(Message::ClearProfiles);

    let header = container(
        row![
            search_bar,
            Space::new().width(8.0),
            count_label,
            Space::new().width(Length::Fill),
            clear_btn,
        ]
        .align_y(Vertical::Center),
    )
    .padding(styles::PAD);

    let mut cards: Vec<Element<'_, Message>> = enrolled_filtered
        .into_iter()
        .map(|(i, p)| profile_card(i, p, true))
        .collect();

    for p in demo_filtered {
        cards.push(profile_card(0, p, false));
    }

    let content = row(cards).spacing(12).wrap();

    column![
        header,
        scrollable(container(content).padding(styles::PAD).width(Length::Fill))
            .style(styles::invisible_scroll)
            .width(Length::Fill)
            .height(Length::Fill),
    ]
    .into()
}

fn profile_card<'a>(index: usize, profile: &'a Profile, deletable: bool) -> Element<'a, Message> {
    let name_text = text(profile.name.as_str())
        .size(14)
        .color(Color::from_rgb(0.90, 0.90, 0.90));

    let header = if deletable {
        let del_btn = button(text("Delete").size(11))
            .style(styles::delete_btn)
            .padding([2, 6])
            .on_press(Message::DeleteProfile(index));
        row![name_text, Space::new().width(Length::Fill), del_btn,].align_y(Vertical::Center)
    } else {
        row![name_text, Space::new().width(Length::Fill),].align_y(Vertical::Center)
    };

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
