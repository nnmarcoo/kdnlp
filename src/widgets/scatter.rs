use iced::mouse;
use iced::widget::canvas::{self, Canvas, Frame, Geometry, Path};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Theme};

use crate::app::Message;

pub struct ScatterPlot {
    profiles: Vec<(String, [f32; 2])>,
    session: Option<[f32; 2]>,
}

impl ScatterPlot {
    pub fn new(profiles: Vec<(String, [f32; 2])>, session: Option<[f32; 2]>) -> Self {
        Self { profiles, session }
    }

    pub fn into_element<'a>(self) -> Element<'a, Message> {
        Canvas::new(self)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

impl canvas::Program<Message> for ScatterPlot {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());

        if self.profiles.is_empty() {
            return vec![frame.into_geometry()];
        }

        let all_x: Vec<f32> = self
            .profiles
            .iter()
            .map(|(_, p)| p[0])
            .chain(self.session.map(|s| s[0]))
            .collect();
        let all_y: Vec<f32> = self
            .profiles
            .iter()
            .map(|(_, p)| p[1])
            .chain(self.session.map(|s| s[1]))
            .collect();

        let min_x = all_x.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_x = all_x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_y = all_y.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_y = all_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let pad = 24.0f32;
        let w = bounds.width - pad * 2.0;
        let h = bounds.height - pad * 2.0;

        let range_x = (max_x - min_x).max(1e-6);
        let range_y = (max_y - min_y).max(1e-6);

        let to_screen = |px: f32, py: f32| -> Point {
            Point {
                x: pad + (px - min_x) / range_x * w,
                y: pad + (1.0 - (py - min_y) / range_y) * h,
            }
        };

        let palette: &[Color] = &[
            Color::from_rgb(0.38, 0.82, 0.48),
            Color::from_rgb(0.38, 0.62, 0.92),
            Color::from_rgb(0.92, 0.72, 0.32),
            Color::from_rgb(0.82, 0.42, 0.82),
            Color::from_rgb(0.92, 0.42, 0.42),
            Color::from_rgb(0.42, 0.82, 0.82),
            Color::from_rgb(0.92, 0.62, 0.42),
            Color::from_rgb(0.72, 0.82, 0.38),
        ];

        for (i, (name, pt)) in self.profiles.iter().enumerate() {
            let color = palette[i % palette.len()];
            let sp = to_screen(pt[0], pt[1]);
            let r = 5.0f32;

            frame.fill(
                &Path::circle(sp, r),
                canvas::Fill {
                    style: canvas::Style::Solid(color),
                    ..canvas::Fill::default()
                },
            );

            frame.fill_text(canvas::Text {
                content: name.clone(),
                position: Point {
                    x: sp.x + r + 3.0,
                    y: sp.y - 6.0,
                },
                color: Color::from_rgb(0.75, 0.75, 0.75),
                size: iced::Pixels(10.0),
                ..canvas::Text::default()
            });
        }

        if let Some(sp_pt) = self.session {
            let sp = to_screen(sp_pt[0], sp_pt[1]);
            let half = 5.0f32;
            let cross = Path::new(|b| {
                b.move_to(Point { x: sp.x - half, y: sp.y - half });
                b.line_to(Point { x: sp.x + half, y: sp.y + half });
                b.move_to(Point { x: sp.x + half, y: sp.y - half });
                b.line_to(Point { x: sp.x - half, y: sp.y + half });
            });
            frame.stroke(
                &cross,
                canvas::Stroke {
                    style: canvas::Style::Solid(Color::WHITE),
                    width: 2.0,
                    ..canvas::Stroke::default()
                },
            );
            frame.fill_text(canvas::Text {
                content: "session".to_string(),
                position: Point {
                    x: sp.x + half + 3.0,
                    y: sp.y - 6.0,
                },
                color: Color::WHITE,
                size: iced::Pixels(10.0),
                ..canvas::Text::default()
            });
        }

        vec![frame.into_geometry()]
    }
}

impl<'a> From<ScatterPlot> for Element<'a, Message> {
    fn from(s: ScatterPlot) -> Self {
        s.into_element()
    }
}
