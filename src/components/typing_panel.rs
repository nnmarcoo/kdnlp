use iced::alignment::Vertical;
use iced::widget::{Space, button, column, container, pick_list, row, text_input};
use iced::{Element, Length};

use crate::app::Message;
use crate::styles;
use crate::typing::Session;
use crate::widgets::typing_widget::TypingWidget;

const DEMO_OPTIONS: &[&str] = &["5 users", "50 users", "100 users", "200 users", "500 users"];
const DEMO_COUNTS: &[usize] = &[5, 50, 100, 200, 500];

pub fn view<'a>(
    name_input: &'a str,
    session: &'a Session,
    prompt: &'a str,
) -> Element<'a, Message> {
    let typing = TypingWidget::new(
        prompt,
        &session.text,
        Message::KeyPressed,
        Message::KeyReleased,
        Message::Backspace,
        Message::BackspaceReleased,
        Message::Enroll,
    );

    let controls = view_controls(name_input, session);

    container(
        column![typing, controls]
            .spacing(styles::SPACING)
            .padding(styles::PAD),
    )
    .width(Length::Fill)
    .into()
}

fn view_controls<'a>(name_input: &'a str, session: &'a Session) -> Element<'a, Message> {
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

    let clear_btn = button("Clear")
        .style(styles::mode_btn)
        .on_press(Message::Clear);

    let demo_picker = pick_list(DEMO_OPTIONS, None::<&str>, |selected| {
        let n = DEMO_OPTIONS
            .iter()
            .position(|&o| o == selected)
            .map(|i| DEMO_COUNTS[i])
            .unwrap_or(0);
        Message::LoadDemo(n)
    })
    .placeholder("demo users")
    .width(Length::Fixed(150.0));

    row![
        name_field,
        enroll_btn,
        clear_btn,
        Space::new().width(Length::Fill),
        demo_picker,
    ]
    .spacing(styles::SPACING)
    .align_y(Vertical::Center)
    .into()
}
