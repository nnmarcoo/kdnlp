mod app;
mod components;
mod store;
mod styles;
mod typing;
mod widgets;

use app::{App, Message};
use iced::{Size, Task, window};

fn main() -> iced::Result {
    iced::application(init, App::update, App::view)
        .title("Keystroke Dynamics")
        .theme(App::theme)
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
