use iced::alignment::Vertical;
use iced::widget::{Space, button, column, container, row, text, text_input};
use iced::{Color, Element, Length};

use crate::app::{Message, Mode};
use crate::styles;
use crate::typing::Session;
use crate::widgets::typing_widget::TypingWidget;

pub fn view<'a>(
    mode: Mode,
    name_input: &'a str,
    session: &'a Session,
    prompt: &'a str,
) -> Element<'a, Message> {
    let typing = TypingWidget::new(
        prompt,
        &session.text,
        |ch, keycode, t| Message::KeyPressed(ch, keycode, t),
        |ch, t| Message::KeyReleased(ch, t),
        |t| Message::Backspace(t),
        |t| Message::BackspaceReleased(t),
        Message::Submit,
    );

    let controls = view_controls(mode, name_input, session);

    container(
        column![typing, controls]
            .spacing(styles::SPACING)
            .padding(styles::PAD),
    )
    .width(Length::Fill)
    .into()
}

fn view_controls<'a>(
    mode: Mode,
    name_input: &'a str,
    session: &'a Session,
) -> Element<'a, Message> {
    let can_submit =
        !session.text.is_empty() && (mode == Mode::Identify || !name_input.trim().is_empty());

    let submit_label = match mode {
        Mode::Enroll => "Save Profile",
        Mode::Identify => "Identify",
    };

    let submit_btn = if can_submit {
        button(submit_label)
            .style(styles::mode_btn_active)
            .on_press(Message::Submit)
    } else {
        button(submit_label).style(styles::mode_btn)
    };

    let clear_btn = button("Clear")
        .style(styles::mode_btn)
        .on_press(Message::Clear);

    let intervals = text(format!("{} intervals", session.interval_count()))
        .size(12)
        .color(Color::from_rgb(0.45, 0.45, 0.45));

    match mode {
        Mode::Enroll => {
            let name_field = text_input("Your name", name_input)
                .on_input(Message::NameChanged)
                .on_submit(Message::Submit)
                .padding([8, styles::PAD as u16]);

            row![
                name_field,
                submit_btn,
                clear_btn,
                Space::new().width(Length::Fill),
                intervals,
            ]
            .spacing(styles::SPACING)
            .align_y(Vertical::Center)
            .into()
        }
        Mode::Identify => row![
            submit_btn,
            clear_btn,
            Space::new().width(Length::Fill),
            intervals,
        ]
        .spacing(styles::SPACING)
        .align_y(Vertical::Center)
        .into(),
    }
}
