use iced::advanced::layout;
use iced::advanced::renderer::Quad;
use iced::advanced::text::{self, Paragraph, Text};
use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::{self, Clipboard, Layout, Shell, Widget};
use iced::keyboard::{self, Key, key::Named, key::Physical};
use iced::mouse;
use iced::{
    Background, Border, Color, Element, Event, Font, Length, Pixels, Point, Rectangle, Renderer,
    Size,
};
use std::time::Instant;

// Monospace character metrics at FONT_SIZE
const FONT_SIZE: f32 = 16.0;
const LINE_H: f32 = 30.0;
const H_PAD: f32 = 14.0;
const V_PAD: f32 = 12.0;

pub struct TypingWidget<'a, Message> {
    passage: &'a str,
    typed: &'a str,
    on_key: Box<dyn Fn(char, u32, Instant) -> Message + 'a>,
    on_release: Box<dyn Fn(char, Instant) -> Message + 'a>,
    on_backspace: Box<dyn Fn(Instant) -> Message + 'a>,
    on_backspace_release: Box<dyn Fn(Instant) -> Message + 'a>,
    on_submit: Message,
}

impl<'a, Message> TypingWidget<'a, Message> {
    pub fn new(
        passage: &'a str,
        typed: &'a str,
        on_key: impl Fn(char, u32, Instant) -> Message + 'a,
        on_release: impl Fn(char, Instant) -> Message + 'a,
        on_backspace: impl Fn(Instant) -> Message + 'a,
        on_backspace_release: impl Fn(Instant) -> Message + 'a,
        on_submit: Message,
    ) -> Self {
        Self {
            passage,
            typed,
            on_key: Box::new(on_key),
            on_release: Box::new(on_release),
            on_backspace: Box::new(on_backspace),
            on_backspace_release: Box::new(on_backspace_release),
            on_submit,
        }
    }
}

#[derive(Default)]
struct State {
    focused: bool,
}

impl<'a, Message: Clone> Widget<Message, iced::Theme, Renderer> for TypingWidget<'a, Message> {
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn size(&self) -> Size<Length> {
        Size {
            width: Length::Fill,
            height: Length::Shrink,
        }
    }

    fn layout(
        &mut self,
        _tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let max_w = limits.max().width;
        let usable_w = (max_w - 2.0 * H_PAD).max(1.0);
        // Use approximate char width for layout estimation; draw corrects with measured value
        let approx_char_w = 9.6_f32;
        let positions = layout_chars(self.passage, 0.0, 0.0, usable_w, approx_char_w);
        let max_line_y = positions.iter().map(|p| p.y).fold(0.0_f32, f32::max);
        let lines = (max_line_y / LINE_H).ceil() as u32 + 1;
        let h = lines as f32 * LINE_H + 2.0 * V_PAD;
        layout::atomic(limits, max_w, h)
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<State>();
        let bounds = layout.bounds();

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let was = state.focused;
                state.focused = cursor.is_over(bounds);
                if state.focused != was {
                    shell.request_redraw();
                }
                if state.focused {
                    shell.capture_event();
                }
            }
            Event::Keyboard(keyboard::Event::KeyPressed {
                key,
                physical_key,
                text,
                ..
            }) if state.focused => {
                // Capture timestamp as close to the event as possible
                let t = Instant::now();
                match key {
                    Key::Named(Named::Backspace) => {
                        shell.publish((self.on_backspace)(t));
                        shell.capture_event();
                    }
                    Key::Named(Named::Enter) => {
                        shell.publish(self.on_submit.clone());
                        shell.capture_event();
                    }
                    _ => {
                        if let Some(s) = text.as_ref() {
                            if let Some(ch) = s.chars().next() {
                                if !ch.is_control() {
                                    let keycode = physical_keycode(physical_key);
                                    shell.publish((self.on_key)(
                                        ch.to_ascii_lowercase(),
                                        keycode,
                                        t,
                                    ));
                                    shell.capture_event();
                                }
                            }
                        }
                    }
                }
                shell.request_redraw();
            }
            Event::Keyboard(keyboard::Event::KeyReleased { key, .. }) if state.focused => {
                let t = Instant::now();
                match key_to_char(key) {
                    Some('\x08') => {
                        shell.publish((self.on_backspace_release)(t));
                        shell.capture_event();
                    }
                    Some(ch) => {
                        shell.publish((self.on_release)(ch, t));
                        shell.capture_event();
                    }
                    None => {}
                }
            }
            _ => {}
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &iced::Theme,
        _style: &iced::advanced::renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        use advanced::Renderer as _;
        use advanced::text::Renderer as _;

        let state = tree.state.downcast_ref::<State>();
        let bounds = layout.bounds();
        let palette = theme.extended_palette();

        // Background + border
        renderer.fill_quad(
            Quad {
                bounds,
                border: Border {
                    color: if state.focused {
                        palette.primary.base.color
                    } else {
                        palette.background.strong.color
                    },
                    width: 1.0,
                    radius: 6.0.into(),
                },
                ..Quad::default()
            },
            Background::Color(palette.background.weak.color),
        );

        // Measure actual char width once per draw using the renderer
        let char_w = {
            let para = <Renderer as text::Renderer>::Paragraph::with_text(Text {
                content: "m",
                bounds: Size::new(f32::INFINITY, f32::INFINITY),
                size: Pixels(FONT_SIZE),
                line_height: text::LineHeight::default(),
                font: Font::MONOSPACE,
                align_x: iced::alignment::Horizontal::Left.into(),
                align_y: iced::alignment::Vertical::Top.into(),
                shaping: text::Shaping::Basic,
                wrapping: text::Wrapping::None,
            });
            para.min_bounds().width.max(1.0)
        };

        let start_x = bounds.x + H_PAD;
        let start_y = bounds.y + V_PAD;
        let usable_w = bounds.width - 2.0 * H_PAD;

        let positions = layout_chars(self.passage, start_x, start_y, start_x + usable_w, char_w);

        let passage_chars: Vec<char> = self.passage.chars().collect();
        let typed_chars: Vec<char> = self.typed.chars().collect();

        let color_correct = Color::from_rgb(0.38, 0.82, 0.48);
        let color_wrong = Color::from_rgb(0.92, 0.35, 0.35);
        let color_dim = palette.background.base.text.scale_alpha(0.25);
        let color_cursor = palette.background.base.text;

        let draw_cursor = |renderer: &mut Renderer, x: f32, line_y: f32| {
            let cursor_h = FONT_SIZE + 4.0;
            let cursor_y = line_y + (LINE_H - cursor_h) / 2.0;
            renderer.fill_quad(
                Quad {
                    bounds: Rectangle {
                        x: x - 1.0,
                        y: cursor_y,
                        width: 2.0,
                        height: cursor_h,
                    },
                    ..Quad::default()
                },
                Background::Color(color_cursor),
            );
        };

        for (i, (&ch, pos)) in passage_chars.iter().zip(positions.iter()).enumerate() {
            // Draw cursor before the character at typed_len
            if i == typed_chars.len() && state.focused {
                draw_cursor(renderer, pos.x, pos.y);
            }

            let color = if i < typed_chars.len() {
                if typed_chars[i] == ch {
                    color_correct
                } else {
                    color_wrong
                }
            } else {
                color_dim
            };

            // Render spaces only when wrong (as red underscore-like display)
            let render_ch = if ch == ' ' {
                if i < typed_chars.len() && typed_chars[i] != ' ' {
                    '·'
                } else {
                    continue;
                }
            } else {
                ch
            };

            renderer.fill_text(
                Text {
                    content: render_ch.to_string(),
                    bounds: Size::new(char_w + 2.0, LINE_H),
                    size: Pixels(FONT_SIZE),
                    line_height: text::LineHeight::default(),
                    font: Font::MONOSPACE,
                    align_x: iced::alignment::Horizontal::Left.into(),
                    align_y: iced::alignment::Vertical::Center.into(),
                    shaping: text::Shaping::Basic,
                    wrapping: text::Wrapping::None,
                },
                Point::new(pos.x, pos.y + LINE_H / 2.0),
                color,
                *viewport,
            );
        }

        // Cursor after last character (passage complete)
        if typed_chars.len() >= passage_chars.len() {
            if let Some(last) = positions.last() {
                if state.focused {
                    draw_cursor(renderer, last.x + char_w, last.y);
                }
            }
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        if cursor.is_over(layout.bounds()) {
            mouse::Interaction::Text
        } else {
            mouse::Interaction::default()
        }
    }
}

impl<'a, Message: Clone + 'a> From<TypingWidget<'a, Message>>
    for Element<'a, Message, iced::Theme, Renderer>
{
    fn from(w: TypingWidget<'a, Message>) -> Self {
        Self::new(w)
    }
}

/// Extract a numeric keycode from a physical key, matching the Aalto dataset's KEYCODE column.
fn physical_keycode(key: &Physical) -> u32 {
    match key {
        Physical::Code(code) => *code as u32,
        Physical::Unidentified(_) => 0,
    }
}

/// Map an iced Key to the char we track — used for KeyReleased where `text` is often None.
fn key_to_char(key: &Key) -> Option<char> {
    match key {
        Key::Character(s) => s.chars().next().map(|c| c.to_ascii_lowercase()),
        Key::Named(Named::Space) => Some(' '),
        Key::Named(Named::Backspace) => Some('\x08'),
        _ => None,
    }
}

/// Compute the top-left Point of each character in `passage` given wrapping constraints.
/// `start_x/y` is the origin. `max_x` is the right boundary. `char_w` is character advance width.
fn layout_chars(passage: &str, start_x: f32, start_y: f32, max_x: f32, char_w: f32) -> Vec<Point> {
    let mut positions = Vec::with_capacity(passage.len());
    let mut x = start_x;
    let mut y = start_y;
    let chars: Vec<char> = passage.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        if chars[i] == ' ' {
            // Look ahead: how wide is the next word?
            let word_start = i + 1;
            let mut word_end = word_start;
            while word_end < n && chars[word_end] != ' ' {
                word_end += 1;
            }
            let next_w = (word_end - word_start) as f32 * char_w;

            if x + char_w + next_w > max_x && x > start_x {
                // Wrap: place the space off-screen to the right, start next word on new line
                positions.push(Point::new(x, y));
                x = start_x;
                y += LINE_H;
            } else {
                positions.push(Point::new(x, y));
                x += char_w;
            }
        } else {
            positions.push(Point::new(x, y));
            x += char_w;
        }
        i += 1;
    }

    positions
}
