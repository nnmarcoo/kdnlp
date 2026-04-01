use std::collections::{BTreeSet, HashMap};

use iced::advanced::layout;
use iced::advanced::renderer::Quad;
use iced::advanced::text::{self, Text};
use iced::advanced::{self, Layout, Widget};
use iced::mouse;
use iced::{
    Background, Border, Color, Element, Font, Length, Pixels, Point, Rectangle, Renderer, Size,
};

use crate::typing::display_char;

const LABEL_SIZE: f32 = 10.0;
const CELL_PAD: f32 = 1.0;
const HEADER_PAD: f32 = 2.0;

pub struct Heatmap {
    avgs: HashMap<(char, char), f64>,
}

impl Heatmap {
    pub fn new(avgs: &HashMap<(char, char), f64>) -> Self {
        Self { avgs: avgs.clone() }
    }

    pub fn from_vecs(bigrams: &HashMap<(char, char), Vec<f64>>) -> Self {
        let avgs = bigrams
            .iter()
            .map(|(k, v)| (*k, v.iter().sum::<f64>() / v.len() as f64))
            .collect();
        Self { avgs }
    }
}

impl<Message: 'static> Widget<Message, iced::Theme, Renderer> for Heatmap {
    fn size(&self) -> Size<Length> {
        Size {
            width: Length::Fill,
            height: Length::Fill,
        }
    }

    fn layout(
        &mut self,
        _tree: &mut iced::advanced::widget::Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let max = limits.max();
        layout::Node::new(Size::new(max.width, max.height))
    }

    fn draw(
        &self,
        _tree: &iced::advanced::widget::Tree,
        renderer: &mut Renderer,
        theme: &iced::Theme,
        _style: &iced::advanced::renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        use advanced::Renderer as _;
        use advanced::text::Renderer as _;

        let bounds = layout.bounds();
        if self.avgs.is_empty() {
            return;
        }

        // Collect unique chars that appear, sorted
        let mut chars = BTreeSet::new();
        for &(a, b) in self.avgs.keys() {
            chars.insert(a);
            chars.insert(b);
        }
        let chars: Vec<char> = chars.into_iter().collect();
        let n = chars.len();
        if n == 0 {
            return;
        }

        let avgs = &self.avgs;

        // Global min/max for color scale
        let all_vals: Vec<f64> = avgs.values().copied().collect();
        let global_min = all_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let global_max = all_vals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(global_min + 1.0);

        let palette = theme.extended_palette();

        // Measure header width: widest char label
        let header_w = LABEL_SIZE * 0.7 + HEADER_PAD;

        // Cell size: fit grid into available space
        let avail_w = bounds.width - header_w;
        let avail_h = bounds.height - header_w; // top header row takes same space
        let cell = ((avail_w / n as f32).min(avail_h / n as f32))
            .max(2.0)
            .floor();

        let grid_w = n as f32 * cell;
        let grid_h = n as f32 * cell;

        // Center the grid in available space
        let grid_x = bounds.x + header_w + ((avail_w - grid_w).max(0.0)) / 2.0;
        let grid_y = bounds.y + header_w + ((avail_h - grid_h).max(0.0)) / 2.0;

        let dim_color = Color::from_rgb(0.50, 0.50, 0.50);
        let empty_color = palette.background.strong.color;

        // Column headers (top)
        for (ci, &ch) in chars.iter().enumerate() {
            let x = grid_x + ci as f32 * cell;
            let y = grid_y - header_w;
            renderer.fill_text(
                Text {
                    content: display_char(ch).to_string(),
                    bounds: Size::new(cell, header_w),
                    size: Pixels(LABEL_SIZE),
                    line_height: text::LineHeight::default(),
                    font: Font::MONOSPACE,
                    align_x: iced::alignment::Horizontal::Center.into(),
                    align_y: iced::alignment::Vertical::Center.into(),
                    shaping: text::Shaping::Basic,
                    wrapping: text::Wrapping::None,
                },
                Point::new(x + cell / 2.0, y + header_w / 2.0),
                dim_color,
                *viewport,
            );
        }

        // Row headers (left)
        for (ri, &ch) in chars.iter().enumerate() {
            let x = grid_x - header_w;
            let y = grid_y + ri as f32 * cell;
            renderer.fill_text(
                Text {
                    content: display_char(ch).to_string(),
                    bounds: Size::new(header_w, cell),
                    size: Pixels(LABEL_SIZE),
                    line_height: text::LineHeight::default(),
                    font: Font::MONOSPACE,
                    align_x: iced::alignment::Horizontal::Center.into(),
                    align_y: iced::alignment::Vertical::Center.into(),
                    shaping: text::Shaping::Basic,
                    wrapping: text::Wrapping::None,
                },
                Point::new(x + header_w / 2.0, y + cell / 2.0),
                dim_color,
                *viewport,
            );
        }

        // Grid cells
        for (ri, &row_ch) in chars.iter().enumerate() {
            for (ci, &col_ch) in chars.iter().enumerate() {
                let cx = grid_x + ci as f32 * cell + CELL_PAD;
                let cy = grid_y + ri as f32 * cell + CELL_PAD;
                let cw = cell - CELL_PAD * 2.0;

                let color = match avgs.get(&(row_ch, col_ch)) {
                    Some(&ms) => heat_color(ms, global_min, global_max),
                    None => empty_color,
                };

                renderer.fill_quad(
                    Quad {
                        bounds: Rectangle {
                            x: cx,
                            y: cy,
                            width: cw,
                            height: cw,
                        },
                        border: Border {
                            radius: 2.0.into(),
                            ..Border::default()
                        },
                        ..Quad::default()
                    },
                    Background::Color(color),
                );

                // Show ms value in cells if they're big enough
                if cw >= 22.0 {
                    if let Some(&ms) = avgs.get(&(row_ch, col_ch)) {
                        let text_color = if is_dark(color) {
                            Color::from_rgba(1.0, 1.0, 1.0, 0.7)
                        } else {
                            Color::from_rgba(0.0, 0.0, 0.0, 0.7)
                        };
                        renderer.fill_text(
                            Text {
                                content: format!("{:.0}", ms),
                                bounds: Size::new(cw, cw),
                                size: Pixels(8.0),
                                line_height: text::LineHeight::default(),
                                font: Font::MONOSPACE,
                                align_x: iced::alignment::Horizontal::Center.into(),
                                align_y: iced::alignment::Vertical::Center.into(),
                                shaping: text::Shaping::Basic,
                                wrapping: text::Wrapping::None,
                            },
                            Point::new(cx + cw / 2.0, cy + cw / 2.0),
                            text_color,
                            *viewport,
                        );
                    }
                }
            }
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &iced::advanced::widget::Tree,
        _layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        mouse::Interaction::default()
    }
}

impl<'a, Message: 'static> From<Heatmap> for Element<'a, Message, iced::Theme, Renderer> {
    fn from(w: Heatmap) -> Self {
        Self::new(w)
    }
}

/// Maps a value in [min, max] to a green→yellow→red gradient.
fn heat_color(ms: f64, min: f64, max: f64) -> Color {
    let range = max - min;
    let t = if range > 0.0 {
        ((ms - min) / range).clamp(0.0, 1.0) as f32
    } else {
        0.5
    };

    // 0.0 = fastest (green), 0.5 = mid (yellow), 1.0 = slowest (red)
    if t < 0.5 {
        let s = t * 2.0;
        Color::from_rgb(0.20 + 0.75 * s, 0.78 - 0.08 * s, 0.40 - 0.32 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        Color::from_rgb(0.95, 0.70 - 0.40 * s, 0.08 - 0.04 * s)
    }
}

fn is_dark(c: Color) -> bool {
    0.299 * c.r + 0.587 * c.g + 0.114 * c.b < 0.55
}
