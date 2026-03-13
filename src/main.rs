mod app;
mod components;
mod plots;
mod store;
mod styles;
mod typing;
mod widgets;

use app::{App, Message};
use iced::{Font, Size, Task, window};

fn main() -> iced::Result {
    iced::application(init, App::update, App::view)
        .title("Keystroke Dynamics")
        .theme(App::theme)
        .scale_factor(|app| app.scale)
        .subscription(App::subscription)
        .default_font(Font::MONOSPACE)
        .window(window::Settings {
            min_size: Some(Size::new(640.0, 420.0)),
            ..Default::default()
        })
        .centered()
        .run()
}

fn init() -> (App, Task<Message>) {
    (App::default(), Task::none())
}
