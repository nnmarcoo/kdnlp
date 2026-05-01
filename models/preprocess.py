import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2147483647))

MODIFIERS = frozenset(("SHIFT", "CTRL", "ALT", "CAPSLOCK", "TAB", "BACKSPACE", "BKSP"))


def read_sessions(filepath):
    sessions = defaultdict(list)
    with open(filepath, errors="ignore") as f:
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

        iki = p2 - p1
        dwell = r1 - p1
        flight = p2 - r1

        if iki < 0 or iki > 2000 or dwell < 0 or dwell > 1000:
            continue

        bigram = k1.lower() + k2.lower()
        records.append((bigram, iki, dwell, flight))
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
    header = ["participant_id", "session_id", "bigram", "iki_ms", "dwell_ms", "flight_ms"]
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
                for bigram, iki, dwell, flight in extract_bigrams(keystrokes):
                    tw.writerow([pid, sid, bigram, iki, dwell, flight])
                    train_count += 1

            for sid, keystrokes in valid[-1:]:
                for bigram, iki, dwell, flight in extract_bigrams(keystrokes):
                    ew.writerow([pid, sid, bigram, iki, dwell, flight])
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
