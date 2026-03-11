use iced::alignment::Vertical;
use iced::widget::{Space, button, container, row};
use iced::{Element, Length};

use crate::app::{Message, Mode};
use crate::styles;

pub fn view<'a>(mode: Mode, is_fullscreen: bool) -> Element<'a, Message> {
    let enroll_btn = if mode == Mode::Enroll {
        button("Enroll").style(styles::mode_btn_active)
    } else {
        button("Enroll")
            .style(styles::mode_btn)
            .on_press(Message::ModeChanged(Mode::Enroll))
    };

    let identify_btn = if mode == Mode::Identify {
        button("Identify").style(styles::mode_btn_active)
    } else {
        button("Identify")
            .style(styles::mode_btn)
            .on_press(Message::ModeChanged(Mode::Identify))
    };

    let fullscreen_btn = button(if is_fullscreen {
        "Restore"
    } else {
        "Fullscreen"
    })
    .style(styles::mode_btn)
    .on_press(Message::ToggleFullscreen);

    container(
        row![
            enroll_btn,
            identify_btn,
            Space::new().width(Length::Fill),
            fullscreen_btn,
        ]
        .spacing(styles::SPACING)
        .align_y(Vertical::Center),
    )
    .style(styles::bar_style)
    .width(Length::Fill)
    .padding([8.0, styles::PAD])
    .into()
}
