use iced::alignment::Vertical;
use iced::widget::{Space, button, column, container, pick_list, row, text_input};
use iced::{Element, Length};

use crate::app::Message;
use crate::plots::{IdentificationMethod, METHODS};
use crate::styles;
use crate::typing::Session;
use crate::widgets::typing_widget::TypingWidget;

pub fn view<'a>(
    name_input: &'a str,
    session: &'a Session,
    profiles_count: usize,
    prompt: &'a str,
    method: IdentificationMethod,
    fixed_prompt: bool,
) -> Element<'a, Message> {
    let typing = TypingWidget::new(
        prompt,
        &session.text,
        |ch, keycode, t| Message::KeyPressed(ch, keycode, t),
        |ch, t| Message::KeyReleased(ch, t),
        |t| Message::Backspace(t),
        |t| Message::BackspaceReleased(t),
        Message::Enroll,
    );

    let controls = view_controls(name_input, session, profiles_count, method, fixed_prompt);

    container(
        column![typing, controls]
            .spacing(styles::SPACING)
            .padding(styles::PAD),
    )
    .width(Length::Fill)
    .into()
}

fn view_controls<'a>(
    name_input: &'a str,
    session: &'a Session,
    profiles_count: usize,
    method: IdentificationMethod,
    fixed_prompt: bool,
) -> Element<'a, Message> {
    let has_session = !session.is_empty();
    let has_name = !name_input.trim().is_empty();

    let name_field = text_input("Name", name_input)
        .on_input(Message::NameChanged)
        .on_submit(Message::Enroll)
        .style(styles::name_input_style)
        .padding([5, 10])
        .width(Length::Fixed(140.0));

    let enroll_btn = if has_session && has_name {
        button("Enroll")
            .style(styles::mode_btn_active)
            .on_press(Message::Enroll)
    } else {
        button("Enroll").style(styles::mode_btn)
    };

    let identify_btn = if has_session && profiles_count > 0 {
        button("Identify")
            .style(styles::mode_btn_active)
            .on_press(Message::Identify)
    } else {
        button("Identify").style(styles::mode_btn)
    };

    let clear_btn = button("Clear")
        .style(styles::mode_btn)
        .on_press(Message::Clear);

    let fixed_btn = if fixed_prompt {
        button("fixed prompt").style(styles::mode_btn_active).on_press(Message::ToggleFixedPrompt)
    } else {
        button("fixed prompt").style(styles::mode_btn).on_press(Message::ToggleFixedPrompt)
    };

    let method_picker = pick_list(METHODS, Some(method), Message::MethodChanged);

    row![
        name_field,
        enroll_btn,
        identify_btn,
        clear_btn,
        Space::new().width(Length::Fill),
        fixed_btn,
        method_picker,
    ]
    .spacing(styles::SPACING)
    .align_y(Vertical::Center)
    .into()
}
