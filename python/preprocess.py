import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

MODIFIERS = frozenset(("SHIFT", "CTRL", "ALT", "CAPSLOCK", "TAB", "BACKSPACE", "BKSP"))

QWERTY_POS = {}
for _r, _row in enumerate(["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]):
    for _c, _ch in enumerate(_row):
        QWERTY_POS[_ch] = (_r, _c + _r * 0.25)
QWERTY_POS[" "] = (3, 4.5)


def key_distance(a, b):
    pa, pb = QWERTY_POS.get(a), QWERTY_POS.get(b)
    if pa is None or pb is None:
        return 0.0
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def read_sessions(filepath):
    sessions = defaultdict(list)
    with open(filepath, errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                press = int(row["PRESS_TIME"])
                release = int(row["RELEASE_TIME"])
            except (ValueError, KeyError, TypeError):
                continue
            sessions[row["TEST_SECTION_ID"]].append((press, release, row["LETTER"]))
    return sessions


def extract_bigrams(keystrokes):
    records = []
    for i in range(len(keystrokes) - 1):
        p1, r1, k1 = keystrokes[i]
        p2, _, k2 = keystrokes[i + 1]

        if k1 in MODIFIERS or k2 in MODIFIERS:
            continue

        c1, c2 = k1.lower(), k2.lower()
        iki = p2 - p1
        dwell = r1 - p1
        flight = p2 - r1

        if iki < 0 or iki > 2000 or dwell < 0 or dwell > 1000:
            continue

        records.append((c1 + c2, iki, dwell, flight, key_distance(c1, c2)))
    return records


def process_all(data_dir, out_dir, min_sessions=3):
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(data_path.glob("*_keystrokes.txt"))
    total = len(files)
    print(f"Found {total} participant files")

    print(f"\n[Pass 1/2] Scanning for participants with >= {min_sessions} sessions...")
    eligible_files = []
    for i, f in enumerate(files):
        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"  {i + 1:>6}/{total}  ({100 * (i + 1) / total:.0f}%)  "
                  f"eligible so far: {len(eligible_files)}")
        sessions = read_sessions(f)
        if sum(1 for v in sessions.values() if len(v) >= 5) >= min_sessions:
            eligible_files.append(f)

    n_eligible = len(eligible_files)
    print(f"  -> {n_eligible} eligible participants\n")

    print("[Pass 2/2] Extracting bigram features...")
    header = ["participant_id", "session_id", "bigram",
              "iki_ms", "dwell_ms", "flight_ms", "key_dist"]
    train_count = 0
    test_count = 0

    with open(out_path / "train.csv", "w", newline="") as tf, \
         open(out_path / "test.csv", "w", newline="") as ef:
        tw = csv.writer(tf)
        ew = csv.writer(ef)
        tw.writerow(header)
        ew.writerow(header)

        for i, f in enumerate(eligible_files):
            if (i + 1) % 2000 == 0 or (i + 1) == n_eligible:
                print(f"  {i + 1:>6}/{n_eligible}  ({100 * (i + 1) / n_eligible:.0f}%)  "
                      f"train: {train_count}  test: {test_count}")

            pid = f.stem.removesuffix("_keystrokes")
            sessions = read_sessions(f)

            valid = []
            for sid, keystrokes in sessions.items():
                if len(keystrokes) >= 5:
                    keystrokes.sort()
                    valid.append((sid, keystrokes))
            valid.sort(key=lambda s: s[1][0][0])

            for sid, keystrokes in valid[:-1]:
                for bg, iki, dwell, flight, dist in extract_bigrams(keystrokes):
                    tw.writerow([pid, sid, bg, iki, dwell, flight, f"{dist:.4f}"])
                    train_count += 1

            for sid, keystrokes in valid[-1:]:
                for bg, iki, dwell, flight, dist in extract_bigrams(keystrokes):
                    ew.writerow([pid, sid, bg, iki, dwell, flight, f"{dist:.4f}"])
                    test_count += 1

    print(f"\nTrain: {train_count} bigram records")
    print(f"Test:  {test_count} bigram records")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../keystrokes/files")
    parser.add_argument("--out_dir", default="./processed")
    parser.add_argument("--min_sessions", type=int, default=3)
    args = parser.parse_args()
    process_all(args.data_dir, args.out_dir, args.min_sessions)
