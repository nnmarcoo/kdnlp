use std::collections::HashMap;
use std::time::Instant;

pub const PROMPTS: &[&str] = &[
    "sphinx of black quartz judge my vow and exempt the dwarves from working on fixed lazy summer days objective journalism requires exceptionally bold workers who verify all fuzzy details with skepticism the bulky freezer was jammed awkwardly behind six crates of expensive cognac and dry vermouth bottles back in my quaint garden jaunty zinnias vie with flaxen blooming orchids and plump kumquats on hedges my grandmother fixed the broken jukebox with quiet zeal and played jazzy gospel hymns every weeknight",
    "bravely exploring jungles of zurich and queensland the kayaker found six wedge shaped topaz crystals the job requires extra pluck and zeal from every young wage earner who is ambitious for advancement a wrathful zephyr blew dark clouds above the tranquil bay just mixing up every sailboat in the fjord jackhammers vibrated wildly as the expert dozed off by the frozen lake gazing up quietly at six moons we publicized the quirky festival by fixing a jumbo sign above the roadway behind the old gymnasium",
    "the grumpy wizard makes toxic brew for the evil queen and jovial dwarfs who explore a hidden kingdom a lynx crept above the frozen quagmire deftly jumping over wet shrubs and catching six plump voles the morning light filtered through the curtains as she made her coffee and carefully read the news oxygen is required for combustion but the frozen liquid vaporized abruptly causing extreme hazards the complex fudge recipe was analyzed by experts who rejected adding kumquat flavor to the brown glaze",
    "crazy frederick bought many very exquisite opal jewels and hid them in a quaint shop by the lakefront she explored the fjord at dusk when heavy fog enveloped the rocky coastline and obscured her kayak path pack my box with five dozen liquor jugs and ship them quickly before the amazing sunset fades away the students gathered in the hall early to review their notes before the exam started that afternoon the coffee shop on the corner was always busy on weekday mornings with people stopping in before work",
    "the quick brown fox jumps over a lazy dog while searching for exotic herbs and crazy byzantine jewels few quips galvanized the mock jury pool and the brazen witness left the stand in extreme shock today the anxious boy gazed up at the jovial queen who was adjusting her bronze crown and long velvet cloak the perplexed janitor quietly fixed two broken gavels while the amazed jury observed from the balcony we promptly judged antique ivory buckles for the next prize and the crazy shopkeeper was exuberant",
    "they spent the weekend hiking through the forest and came across a narrow path leading to a quiet lake he checked his phone and saw several missed calls from his sister before realizing it was on silent he opened the old wooden door and stepped into the quiet library where rows of books lined the shelves the train pulled into the station just as the sun began to set and the passengers gathered their bags she typed the last few words of her report and saved the file before closing her laptop for the day",
    "the dog ran across the field and jumped over the fence before disappearing into the tall grass nearby they walked along the riverbank collecting stones and talking about their plans for the coming months she noticed the sky had turned dark while she was working and quickly gathered her things for the walk after a long day at work he decided to take a different route home and stopped at the small bakery the restaurant was fully booked for the evening so they decided to cook at home and opened some wine",
    "she remembered leaving her keys on the counter but when she went back they were nowhere to be found",
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
#[allow(dead_code)]
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

struct PendingFlight {
    bigram: (char, char),
    vec_idx: usize,
    log_idx: usize,
    next_press: Instant,
    prev_char: char,
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
    pending_flight: Option<PendingFlight>,
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
            pending_flight: None,
        }
    }
}

impl Session {
    pub fn push_char(&mut self, ch: char, keycode: u32, t: Instant) {
        let press_ms = t.duration_since(self.start_time).as_millis() as u64;

        if let (Some(prev), Some(prev_t)) = (self.last_char, self.last_press) {
            let bigram = (prev, ch);
            let iki_ms = t.duration_since(prev_t).as_secs_f64() * 1000.0;

            let vec = self.bigrams.entry(bigram).or_default();
            let vec_idx = vec.len();
            vec.push(iki_ms);

            let log_idx = self.log.len();
            self.log.push((bigram, iki_ms));

            self.pending_flight = Some(PendingFlight {
                bigram,
                vec_idx,
                log_idx,
                next_press: t,
                prev_char: prev,
            });
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

        if let Some(pf) = self.pending_flight.take() {
            if ch == pf.prev_char {
                let flight_ms = match pf.next_press.checked_duration_since(t) {
                    Some(dur) => dur.as_secs_f64() * 1000.0,
                    None => -(t.duration_since(pf.next_press).as_secs_f64() * 1000.0),
                };
                if let Some(val) = self
                    .bigrams
                    .get_mut(&pf.bigram)
                    .and_then(|v| v.get_mut(pf.vec_idx))
                {
                    *val = flight_ms;
                }
                if let Some(entry) = self.log.get_mut(pf.log_idx) {
                    entry.1 = flight_ms;
                }
            } else {
                self.pending_flight = Some(pf);
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

        if self
            .pending_flight
            .as_ref()
            .is_some_and(|pf| pf.log_idx + 1 == self.log.len())
        {
            self.pending_flight = None;
        }

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

    pub fn wpm(&self) -> f64 {
        if self.events.len() < 2 {
            return 0.0;
        }
        let last = &self.events[self.events.len() - 1];
        let first = &self.events[0];
        let elapsed_ms = last.press_ms.saturating_sub(first.press_ms) as f64;
        if elapsed_ms < 1.0 {
            return 0.0;
        }
        let words = self.text.len() as f64 / 5.0;
        words / (elapsed_ms / 60_000.0)
    }

    pub fn avg_interval_ms(&self) -> f64 {
        if self.log.is_empty() {
            return 0.0;
        }
        self.log.iter().map(|(_, ms)| ms).sum::<f64>() / self.log.len() as f64
    }

    pub fn avg_dwell_ms(&self) -> f64 {
        let dwells: Vec<f64> = self.events.iter().filter_map(|e| e.dwell_ms()).collect();
        if dwells.is_empty() {
            return 0.0;
        }
        dwells.iter().sum::<f64>() / dwells.len() as f64
    }

    pub fn unique_bigram_count(&self) -> usize {
        self.bigrams.len()
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
    pub bigrams: HashMap<(char, char), f64>,
    pub bigram_counts: HashMap<(char, char), usize>,
    pub char_count: usize,
    pub interval_count: usize,
    pub wpm: f64,
    pub avg_dwell_ms: f64,
    pub dwell_count: usize,
}

impl Profile {
    pub fn from_session(name: String, session: &Session) -> Self {
        let dwell_count = session
            .events
            .iter()
            .filter(|e| e.release_ms.is_some())
            .count();
        Self {
            name,
            bigrams: session.averaged(),
            bigram_counts: session.bigrams.iter().map(|(&k, v)| (k, v.len())).collect(),
            char_count: session.text.len(),
            interval_count: session.interval_count(),
            wpm: session.wpm(),
            avg_dwell_ms: session.avg_dwell_ms(),
            dwell_count,
        }
    }

    pub fn avg_interval_ms(&self) -> f64 {
        if self.bigrams.is_empty() {
            return 0.0;
        }
        self.bigrams.values().sum::<f64>() / self.bigrams.len() as f64
    }
}

#[allow(dead_code)]
pub fn identify(_session: &Session, profiles: &[Profile]) -> Vec<(String, f64)> {
    profiles.iter().map(|p| (p.name.clone(), 0.0)).collect()
}

pub fn display_char(c: char) -> char {
    if c == ' ' { '·' } else { c }
}
