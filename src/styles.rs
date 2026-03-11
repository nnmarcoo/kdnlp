use iced::{
    Background, Color, Theme,
    widget::{button, container, scrollable},
};

pub const PAD: f32 = 12.0;
pub const SPACING: f32 = 8.0;

pub fn bar_style(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.strong.color)),
        ..Default::default()
    }
}

pub fn mode_btn(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let background = match status {
        button::Status::Hovered => Some(Background::Color(palette.background.weak.color)),
        button::Status::Pressed => Some(Background::Color(palette.background.base.color)),
        _ => None,
    };
    button::Style {
        background,
        border: iced::border::rounded(6.0),
        text_color: palette.background.base.text,
        ..Default::default()
    }
}

pub fn bar_fill(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.primary.base.color)),
        border: iced::border::rounded(3.0),
        ..Default::default()
    }
}

pub fn bar_fill_dim(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.strong.color)),
        border: iced::border::rounded(3.0),
        ..Default::default()
    }
}

pub fn card_style(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.weak.color)),
        border: iced::Border {
            color: palette.background.strong.color,
            width: 1.0,
            radius: 8.0.into(),
        },
        ..Default::default()
    }
}

pub fn delete_btn(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    button::Style {
        background: match status {
            button::Status::Hovered | button::Status::Pressed => {
                Some(Background::Color(palette.danger.base.color))
            }
            _ => None,
        },
        border: iced::border::rounded(4.0),
        text_color: match status {
            button::Status::Hovered | button::Status::Pressed => palette.danger.base.text,
            _ => Color::from_rgb(0.5, 0.5, 0.5),
        },
        ..Default::default()
    }
}

pub fn chip_style(theme: &Theme) -> container::Style {
    let palette = theme.extended_palette();
    container::Style {
        background: Some(Background::Color(palette.background.strong.color)),
        border: iced::Border {
            color: palette.background.strong.color,
            width: 1.0,
            radius: 4.0.into(),
        },
        ..Default::default()
    }
}

pub fn sort_btn(theme: &Theme, status: button::Status) -> button::Style {
    let bg = match status {
        button::Status::Hovered | button::Status::Pressed => Some(Background::Color(
            theme.extended_palette().background.strong.color,
        )),
        _ => None,
    };
    button::Style {
        background: bg,
        border: iced::border::rounded(4.0),
        text_color: Color::from_rgb(0.45, 0.45, 0.45),
        ..Default::default()
    }
}

pub fn sort_btn_active(theme: &Theme, _status: button::Status) -> button::Style {
    button::Style {
        background: Some(Background::Color(
            theme.extended_palette().background.strong.color,
        )),
        border: iced::border::rounded(4.0),
        text_color: Color::from_rgb(0.88, 0.88, 0.88),
        ..Default::default()
    }
}

pub fn invisible_scroll(theme: &Theme, status: scrollable::Status) -> scrollable::Style {
    let base = scrollable::default(theme, status);
    let transparent_rail = scrollable::Rail {
        background: None,
        border: iced::Border::default(),
        scroller: scrollable::Scroller {
            background: iced::Background::Color(Color::TRANSPARENT),
            border: iced::Border::default(),
        },
    };
    scrollable::Style {
        vertical_rail: transparent_rail,
        horizontal_rail: transparent_rail,
        gap: None,
        ..base
    }
}

pub fn mode_btn_active(theme: &Theme, status: button::Status) -> button::Style {
    let palette = theme.extended_palette();
    let background = match status {
        button::Status::Hovered | button::Status::Pressed => {
            Some(Background::Color(palette.primary.strong.color))
        }
        _ => Some(Background::Color(palette.primary.base.color)),
    };
    button::Style {
        background,
        border: iced::border::rounded(6.0),
        text_color: palette.primary.base.text,
        ..Default::default()
    }
}
