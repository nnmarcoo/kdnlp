use iced::mouse;
use iced::widget::canvas::{self, Cache, Canvas, Frame, Geometry, Path};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Theme};

use crate::app::Message;

pub struct ScatterPlot<'a> {
    profiles: Vec<(String, [f32; 2])>,
    session: Option<[f32; 2]>,
    cache: &'a Cache,
}

impl<'a> ScatterPlot<'a> {
    pub fn new(
        profiles: Vec<(String, [f32; 2])>,
        session: Option<[f32; 2]>,
        cache: &'a Cache,
    ) -> Self {
        Self {
            profiles,
            session,
            cache,
        }
    }

    pub fn into_element(self) -> Element<'a, Message> {
        Canvas::new(self)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }
}

impl<'a> canvas::Program<Message> for ScatterPlot<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            draw_scatter(frame, &self.profiles, self.session);
        });
        vec![geometry]
    }
}

fn draw_scatter(frame: &mut Frame, profiles: &[(String, [f32; 2])], session: Option<[f32; 2]>) {
    if profiles.is_empty() {
        return;
    }

    let bounds = frame.size();
    let all_x = profiles
        .iter()
        .map(|(_, p)| p[0])
        .chain(session.map(|s| s[0]));
    let all_y = profiles
        .iter()
        .map(|(_, p)| p[1])
        .chain(session.map(|s| s[1]));

    let (min_x, max_x) = all_x.fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
        (mn.min(v), mx.max(v))
    });
    let (min_y, max_y) = all_y.fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
        (mn.min(v), mx.max(v))
    });

    let pad = 24.0f32;
    let w = bounds.width - pad * 2.0;
    let h = bounds.height - pad * 2.0;

    let range = (max_x - min_x).max(max_y - min_y).max(1e-6);
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    let scale = w.min(h) / range;

    let to_screen = |px: f32, py: f32| -> Point {
        Point {
            x: pad + w / 2.0 + (px - cx) * scale,
            y: pad + h / 2.0 - (py - cy) * scale,
        }
    };

    for (i, (name, pt)) in profiles.iter().enumerate() {
        let color = name_color(name);
        let sp = to_screen(pt[0], pt[1]);
        let r = 5.0f32;

        frame.fill(
            &Path::circle(sp, r),
            canvas::Fill {
                style: canvas::Style::Solid(color),
                ..canvas::Fill::default()
            },
        );

        if i < 5 {
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
    }

    if let Some(sp_pt) = session {
        let sp = to_screen(sp_pt[0], sp_pt[1]);
        let half = 5.0f32;
        let cross = Path::new(|b| {
            b.move_to(Point {
                x: sp.x - half,
                y: sp.y - half,
            });
            b.line_to(Point {
                x: sp.x + half,
                y: sp.y + half,
            });
            b.move_to(Point {
                x: sp.x + half,
                y: sp.y - half,
            });
            b.line_to(Point {
                x: sp.x - half,
                y: sp.y + half,
            });
        });
        frame.stroke(
            &cross,
            canvas::Stroke {
                style: canvas::Style::Solid(Color::WHITE),
                width: 2.0,
                ..canvas::Stroke::default()
            },
        );
    }
}

fn name_color(name: &str) -> Color {
    // Hash the name to a stable hue, then pick a saturated color
    let hash = name
        .bytes()
        .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
    let hue = (hash % 360) as f32;
    hsl_to_rgb(hue, 0.65, 0.60)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color::from_rgb(r + m, g + m, b + m)
}

impl<'a> From<ScatterPlot<'a>> for Element<'a, Message> {
    fn from(s: ScatterPlot<'a>) -> Self {
        s.into_element()
    }
}
