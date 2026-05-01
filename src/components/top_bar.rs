use iced::alignment::Vertical;
use iced::widget::{button, container, row};
use iced::{Element, Length};

use crate::app::{Message, Mode};
use crate::styles;

pub fn view<'a>(mode: Mode) -> Element<'a, Message> {
    let btn = |label: &'static str, m: Mode| {
        if mode == m {
            button(label).style(styles::mode_btn_active)
        } else {
            button(label)
                .style(styles::mode_btn)
                .on_press(Message::ModeChanged(m))
        }
    };

    container(
        row![btn("Demo", Mode::Main), btn("Profiles", Mode::Profiles),]
            .spacing(styles::SPACING)
            .align_y(Vertical::Center),
    )
    .style(styles::bar_style)
    .width(Length::Fill)
    .padding([8.0, styles::PAD])
    .into()
}
