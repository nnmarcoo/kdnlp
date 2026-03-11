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

// Raw keystroke event, mirroring the Aalto 136M dataset columns.
// key is '\x08' for backspace, matching BKSP in the dataset.
#[derive(Clone, Debug)]
pub struct KeyEvent {
    pub key: char,
    pub keycode: u32,  // hardware keycode matching the KEYCODE column in Aalto
    pub press_ms: u64, // milliseconds from session start
    pub release_ms: Option<u64>, // milliseconds from session start, None until released
}

impl KeyEvent {
    pub fn dwell_ms(&self) -> Option<f64> {
        self.release_ms.map(|r| (r - self.press_ms) as f64)
    }
}

// Session holds all state for one typing session: raw events, bigram flight times,
// and the display string shown in the typing widget.
pub struct Session {
    pub text: String,

    // Full raw event log including backspaces in chronological press order.
    // This is the sequence the model will consume at inference time.
    pub events: Vec<KeyEvent>,

    // Aggregated bigram flight times (press-to-press interval) per character pair,
    // used for display and the current placeholder identifier.
    pub bigrams: HashMap<(char, char), Vec<f64>>,

    // Chronological log of bigram entries for the live bar chart.
    pub log: Vec<((char, char), f64)>,

    // Anchor timestamp for converting Instant values to session-relative milliseconds.
    start_time: Instant,

    // Tracks the previous character and its press time for bigram computation.
    last_char: Option<char>,
    last_press: Option<Instant>,

    // Maps each character to a stack of event indices waiting for a release.
    // A stack handles the rare case of the same key being pressed again before releasing.
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
    // Record a key press. t is captured at event time in the widget for accuracy.
    pub fn push_char(&mut self, ch: char, keycode: u32, t: Instant) {
        let press_ms = t.duration_since(self.start_time).as_millis() as u64;

        // Compute bigram flight time (press-to-press interval).
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

    // Record a key release. Matches to the most recent outstanding press of this key
    // to fill in dwell time. t is captured at event time in the widget.
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

    // Record a backspace press. The event goes into the raw log (corrections are a
    // biometric signal) but the display text and bigram aggregates are unwound.
    // Timing continuity is preserved through backspace so that backspace-to-next-char
    // flight times are recorded, matching the Aalto dataset format.
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

    // Returns averaged bigram flight times, used for display and the placeholder identifier.
    pub fn averaged(&self) -> HashMap<(char, char), f64> {
        self.bigrams
            .iter()
            .map(|(k, v)| (*k, v.iter().sum::<f64>() / v.len() as f64))
            .collect()
    }
}

// A saved user profile containing the raw event sequence and averaged bigram times.
pub struct Profile {
    pub name: String,
    // Full raw event sequence from enrollment. This is what the model trains and infers from.
    pub events: Vec<KeyEvent>,
    // Averaged bigram flight times used for display in the info panel.
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

    // Returns bigrams sorted fastest to slowest, up to limit entries.
    pub fn top_bigrams(&self, limit: usize) -> Vec<((char, char), f64)> {
        let mut sorted: Vec<_> = self.bigrams.iter().map(|(k, v)| (*k, *v)).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(limit);
        sorted
    }
}

// Identification stub. Will be replaced with a trained encoder model.
//
// Planned pipeline:
//   1. Encode session.events into a fixed-size embedding vector via an LSTM or
//      Transformer trained on the Aalto 136M dataset with triplet loss.
//   2. Compute cosine similarity between the query embedding and each profile embedding.
//   3. Return profiles ranked by similarity score, highest first.
//
// The model is text-independent: it works across different enrollment and identification
// prompts because it learns the user's temporal rhythm, not character-specific bigram times.
pub fn identify(_session: &Session, profiles: &[Profile]) -> Vec<(String, f64)> {
    profiles.iter().map(|p| (p.name.clone(), 0.0)).collect()
}

pub fn display_char(c: char) -> char {
    if c == ' ' { '·' } else { c }
}
