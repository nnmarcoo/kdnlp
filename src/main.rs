mod app;
mod components;
mod embedder;
mod pca;
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
    // Try loading the LSTM model from the default path next to the executable,
    // or from D:/kdnlp_model as a fallback for development.
    let model_loaded = {
        let exe_model = std::env::current_exe().ok()
            .and_then(|p| p.parent().map(|d| d.join("model")));
        let fallback = std::path::Path::new("D:/kdnlp_model");
        let model_path = exe_model.as_deref()
            .filter(|p| p.join("norm_stats.json").exists())
            .unwrap_or(fallback);
        embedder::load(model_path)
    };
    if model_loaded {
        eprintln!("LSTM model loaded successfully.");
    } else {
        eprintln!("LSTM model not found — Neural Network identification unavailable.");
    }
    (App::default(), Task::none())
}
