use std::collections::HashMap;
use std::time::Instant;

pub const PROMPTS: &[&str] = &[
    "the morning light filtered through the curtains as she made her coffee and carefully read the news before heading out to meet her friends at the park",
    "he opened the old wooden door and stepped into the quiet library where rows of books lined the shelves from floor to ceiling in every direction",
    "the children played in the garden while their parents sat on the porch and watched the clouds drift slowly across the bright blue summer sky",
    "she typed the last few words of her report and saved the file before closing her laptop and walking over to the window to look out at the street",
    "after a long day at work he decided to take a different route home and stopped at the small bakery on the corner to pick up some fresh bread",
    "the train pulled into the station just as the sun began to set and the passengers gathered their bags and prepared to step out onto the platform",
    "they walked along the riverbank collecting stones and talking about their plans for the coming months as the water flowed gently past the old bridge",
    "the coffee shop on the corner was always busy on weekday mornings with people stopping in before work to grab a drink and a quick breakfast",
    "she found the old photograph at the back of the drawer and sat down at the kitchen table to look at it more carefully in the afternoon light",
    "the team worked late into the night to finish the project before the deadline and everyone was relieved when they finally submitted the files",
    "he decided to spend the afternoon reading in the garden and made himself a cup of tea before settling into his favorite chair by the window",
    "she had always wanted to learn how to play the piano and finally signed up for lessons at the music school near her apartment that spring",
    "the restaurant was fully booked for the evening so they decided to cook at home and opened a bottle of wine while preparing dinner together",
    "he checked his phone and saw several missed calls from his sister before realizing he had left it on silent since the early morning meeting",
    "the small bookshop at the end of the street had been there for decades and was known for its carefully chosen selection of second hand novels",
    "she noticed the sky had turned dark while she was working and quickly gathered her things before the rain started to fall on the empty street",
    "they spent the weekend hiking through the forest and came across a narrow path that led down to a quiet lake hidden among the trees below",
    "the students gathered in the hall early to review their notes before the exam started and the room was almost completely silent for once",
    "he spent most of the evening sorting through old boxes in the attic and discovered letters and photographs he had completely forgotten about",
    "the road into town was closed for repairs so they had to take the longer route through the valley which added nearly half an hour to the trip",
    "she remembered leaving her keys on the counter but when she went back to look for them they were nowhere to be found anywhere in the flat",
    "the dog ran across the field and jumped over the fence before disappearing into the tall grass near the edge of the old farmhouse property",
    "he wrote a short letter to his friend explaining why he had missed the meeting and promised to call later that same evening to apologize",
    "the conference was held in a large hotel near the edge of the city and brought together speakers and researchers from universities all around",
    "she packed her bag the night before and set her alarm early so she would have enough time to catch the first train downtown in the morning",
];

pub fn random_prompt() -> &'static str {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as usize;
    PROMPTS[nanos % PROMPTS.len()]
}

#[derive(Clone, Debug)]
pub struct KeyEvent {
    pub key: char,
    pub keycode: u32,
    pub press_ms: u64,
    pub release_ms: Option<u64>,
}

impl KeyEvent {
    pub fn dwell_ms(&self) -> Option<f64> {
        self.release_ms.map(|r| (r - self.press_ms) as f64)
    }
}

pub struct Session {
    pub text: String,
    pub events: Vec<KeyEvent>,
    pub bigrams: HashMap<(char, char), Vec<f64>>,
    pub log: Vec<((char, char), f64)>,
    start_time: Instant,
    last_char: Option<char>,
    last_press: Option<Instant>,
    pending_releases: HashMap<char, Vec<usize>>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            text: String::new(),
            events: Vec::new(),
            bigrams: HashMap::new(),
            log: Vec::new(),
            start_time: Instant::now(),
            last_char: None,
            last_press: None,
            pending_releases: HashMap::new(),
        }
    }
}

impl Session {
    pub fn push_char(&mut self, ch: char, keycode: u32, t: Instant) {
        let press_ms = t.duration_since(self.start_time).as_millis() as u64;

        if let (Some(prev), Some(prev_t)) = (self.last_char, self.last_press) {
            let flight_ms = t.duration_since(prev_t).as_secs_f64() * 1000.0;
            self.bigrams.entry((prev, ch)).or_default().push(flight_ms);
            self.log.push(((prev, ch), flight_ms));
        }

        let idx = self.events.len();
        self.events.push(KeyEvent {
            key: ch,
            keycode,
            press_ms,
            release_ms: None,
        });
        self.pending_releases.entry(ch).or_default().push(idx);

        self.last_char = Some(ch);
        self.last_press = Some(t);
    }

    pub fn push_release(&mut self, ch: char, t: Instant) {
        let release_ms = t.duration_since(self.start_time).as_millis() as u64;
        if let Some(stack) = self.pending_releases.get_mut(&ch) {
            if let Some(idx) = stack.pop() {
                if let Some(ev) = self.events.get_mut(idx) {
                    ev.release_ms = Some(release_ms);
                }
            }
            if stack.is_empty() {
                self.pending_releases.remove(&ch);
            }
        }
    }

    pub fn push_backspace(&mut self, t: Instant) {
        if self.text.is_empty() {
            return;
        }

        let press_ms = t.duration_since(self.start_time).as_millis() as u64;
        let idx = self.events.len();
        self.events.push(KeyEvent {
            key: '\x08',
            keycode: 8,
            press_ms,
            release_ms: None,
        });
        self.pending_releases.entry('\x08').or_default().push(idx);

        self.text.pop();
        if let Some(entry) = self.log.pop() {
            let bigram = entry.0;
            if let Some(v) = self.bigrams.get_mut(&bigram) {
                v.pop();
                if v.is_empty() {
                    self.bigrams.remove(&bigram);
                }
            }
            self.last_char = Some(bigram.0);
        } else {
            self.last_char = None;
        }
        self.last_press = Some(t);
    }

    pub fn push_backspace_release(&mut self, t: Instant) {
        self.push_release('\x08', t);
    }

    pub fn interval_count(&self) -> usize {
        self.bigrams.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.bigrams.is_empty()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn averaged(&self) -> HashMap<(char, char), f64> {
        self.bigrams
            .iter()
            .map(|(k, v)| (*k, v.iter().sum::<f64>() / v.len() as f64))
            .collect()
    }
}

pub struct Profile {
    pub name: String,
    pub events: Vec<KeyEvent>,
    pub bigrams: HashMap<(char, char), f64>,
}

impl Profile {
    pub fn from_session(name: String, session: &Session) -> Self {
        Self {
            name,
            events: session.events.clone(),
            bigrams: session.averaged(),
        }
    }

    pub fn top_bigrams(&self, limit: usize) -> Vec<((char, char), f64)> {
        let mut sorted: Vec<_> = self.bigrams.iter().map(|(k, v)| (*k, *v)).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(limit);
        sorted
    }
}

#[allow(dead_code)]
pub fn identify(_session: &Session, profiles: &[Profile]) -> Vec<(String, f64)> {
    profiles.iter().map(|p| (p.name.clone(), 0.0)).collect()
}

pub fn display_char(c: char) -> char {
    if c == ' ' { '·' } else { c }
}
