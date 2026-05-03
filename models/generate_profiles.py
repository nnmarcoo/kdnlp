"""
Generate a profiles.json from Aalto dataset participants using the trained ONNX model.
Usage: python generate_profiles.py --data_dir ../keystrokes/files --model_dir D:/kdnlp_model --n 10
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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
        if len(k1) != 1 or len(k2) != 1:
            continue
        iki = p2 - p1
        dwell = r1 - p1
        flight = p2 - r1
        if iki < 0 or iki > 2000 or dwell < 0 or dwell > 1000:
            continue
        records.append(((k1.lower(), k2.lower()), float(iki), float(dwell), float(flight)))
    return records


def embed(session_bigrams, model, means, stds, max_len=50):
    if not session_bigrams:
        return None
    import torch
    import torch.nn.functional as F
    import torch.nn.utils.rnn as rnn_utils

    # Use same max_len crop as training — randomly sample if longer
    data = np.array([[iki, dwell, flight] for _, iki, dwell, flight in session_bigrams], dtype=np.float32)
    data = (data - means) / stds

    if len(data) > max_len:
        data = data[-max_len:]

    length = len(data)
    x = torch.tensor(data).unsqueeze(0)  # (1, length, 3)
    lengths = torch.tensor([length])

    model.eval()
    with torch.no_grad():
        emb = model(x, lengths)  # uses the real training forward pass

    emb = emb[0].numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    emb = emb / max(norm, 1e-8)
    return emb.tolist()


NAMES = [
    "aaron", "abby", "abel", "ada", "adam", "adela", "adele", "adeline",
    "adrien", "agnes", "aiden", "aileen", "alan", "alana", "alba", "albert",
    "alec", "alexa", "alex", "alfie", "alice", "alicia", "alina", "alison",
    "alma", "amos", "amy", "ana", "anders", "andre", "andrea", "andy",
    "angel", "angie", "anna", "anne", "ansel", "anton", "april", "arlo",
    "arno", "arthur", "asha", "asher", "ashley", "astrid", "atlas", "aubrey",
    "august", "aurora", "austin", "ava", "axel", "ayda", "baron", "beatrix",
    "beck", "bella", "ben", "benedict", "bianca", "blake", "blanche", "bob",
    "brad", "bram", "bree", "brett", "briar", "brody", "brook", "brynn",
    "bruno", "caleb", "calla", "calvin", "camille", "cara", "carl", "carla",
    "carol", "carson", "casey", "cassie", "cedar", "celeste", "celia", "chloe",
    "chris", "cian", "claire", "clara", "clark", "claude", "clement", "cleo",
    "clover", "cody", "cole", "colin", "cora", "corin", "cy", "cyrus",
    "dale", "dana", "daniel", "daphne", "dave", "dean", "delia", "della",
    "diana", "diego", "dion", "dom", "dominic", "donna", "dora", "dove",
    "drew", "dylan", "eden", "edgar", "edith", "elena", "eli", "elias",
    "eliot", "elisa", "ella", "ellen", "elio", "elise", "ember", "emil",
    "emile", "emily", "emma", "eric", "erica", "erin", "ernest", "esme",
    "ethan", "eva", "evan", "eve", "ezra", "faith", "faye", "felix",
    "fern", "finn", "fiona", "flora", "frank", "fran", "freya", "gabe",
    "gabriel", "gem", "george", "gia", "gilbert", "glen", "goldie", "grace",
    "grant", "greta", "grey", "griffin", "hana", "hannah", "harriet", "harry",
    "haven", "hazel", "heath", "hector", "heidi", "henry", "holden", "holly",
    "honor", "hope", "hugo", "hunter", "ida", "imogen", "indie", "ingrid",
    "iona", "iris", "irene", "isaac", "isabel", "isadora", "ivan", "ivy",
    "jack", "jade", "jake", "james", "jamie", "jan", "jasper", "javier",
    "jean", "jed", "jenna", "jesse", "joel", "jonah", "jorge", "josie",
    "julia", "julian", "june", "juno", "karen", "kai", "kate", "kevin",
    "kit", "knox", "kurt", "kyle", "lara", "lars", "laura", "lauren",
    "layla", "lea", "leah", "leila", "leo", "leon", "leona", "lewis",
    "liam", "lila", "lily", "lin", "lina", "lisa", "lola", "lotte",
    "lou", "louis", "luca", "lucia", "lucy", "luke", "luna", "lyra",
    "mabel", "mae", "magnus", "mara", "marc", "marco", "margo", "maria",
    "marie", "marina", "mark", "martin", "max", "maya", "mia", "miles",
    "milo", "mira", "miriam", "mo", "morgan", "nadine", "nadia", "nate",
    "nell", "nico", "nina", "noah", "noel", "nolan", "nora", "norma",
    "olive", "olivia", "orion", "oscar", "otis", "otto", "owen", "paige",
    "paul", "penny", "petra", "phoebe", "pia", "pierce", "pip", "quinn",
    "rafael", "reed", "rex", "rhea", "rio", "rob", "robin", "roman",
    "ron", "rosa", "rose", "rowan", "ruby", "ruth", "ryan", "sable",
    "sage", "sam", "sara", "selene", "seth", "shea", "sid", "sierra",
    "simon", "skye", "sloane", "sofia", "sol", "stella", "sven", "sylvie",
    "tara", "teo", "theo", "thea", "thomas", "tia", "tobias", "tom",
    "tove", "troy", "uma", "una", "vale", "vera", "victor", "violet",
    "vince", "wade", "ward", "wendy", "willa", "wren", "xander", "xena",
    "yara", "york", "yusuf", "yvonne", "zelda", "zoe", "zara", "zeb",
    "abner", "acey", "aden", "adlai", "adriel", "agatha", "ahanu", "aiken",
    "ailis", "ainsley", "aira", "airic", "aislin", "alaire", "alaric", "alban",
    "alberic", "albin", "alcott", "alda", "aldric", "alena", "aleron", "aleta",
    "alev", "algar", "algot", "alida", "alinta", "alira", "alise", "alivia",
    "aliz", "alka", "allard", "alleen", "allene", "allete", "allon", "allura",
    "almira", "almo", "aloin", "alona", "aloys", "alric", "alroy", "alston",
    "altan", "altea", "alton", "altus", "alva", "alvar", "alvaro", "alvie",
    "alwin", "alwyn", "amabel", "amara", "amari", "ambre", "amby", "amei",
    "amell", "amena", "amery", "amias", "amiel", "amika", "amina", "amir",
    "amira", "amis", "amita", "amity", "amko", "amla", "amne", "amoli",
    "amon", "amora", "amori", "amory", "amro", "amsel", "amsha", "amta",
    "amyas", "amyla", "ancel", "ancil", "andie", "andin", "andis", "andro",
    "anela", "anek", "anet", "anett", "aney", "anfel", "anfri", "ange",
    "angell", "angus", "anik", "anika", "aniko", "anile", "anis", "anise",
    "anita", "anja", "anke", "anko", "ankur", "anla", "anli", "anlon",
    "anmol", "annar", "annis", "annot", "annya", "anona", "anora", "anos",
    "anouk", "anra", "anri", "ansgar", "ansie", "ansin", "anslow", "anson",
    "anta", "antar", "anthe", "antia", "antje", "antko", "antla", "antlo",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../keystrokes/files")
    parser.add_argument("--model_dir", default="D:/kdnlp_model")
    parser.add_argument("--processed_dir", default=None, help="Path to processed/ dir with train.csv to sample participants from")
    parser.add_argument("--n_sample", type=int, default=5000, help="Number of participants to sample from train.csv as candidate pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=10, help="Number of profiles to generate")
    parser.add_argument("--out", default="C:/Users/marco/AppData/Roaming/kdnlp/profiles.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "norm_stats.json") as f:
        stats = json.load(f)
    means = np.array(stats["means"], dtype=np.float32)
    stds = np.array(stats["stds"], dtype=np.float32)

    from lstm import KeystrokeEmbedder, TIMING_COLS
    import torch
    state = torch.load(model_dir / "embedder.pt", map_location="cpu", weights_only=True)
    hidden_size = state["lstm1.weight_ih_l0"].shape[0] // 4
    embed_dim = state["fc.weight"].shape[0]
    sess = KeystrokeEmbedder(input_size=len(TIMING_COLS), hidden_size=hidden_size, embed_dim=embed_dim)
    sess.load_state_dict(state)
    sess.eval()

    # Optionally restrict to a random sample of participants from train.csv
    eval_ids = None
    if args.processed_dir:
        import pandas as pd
        import random as rng
        rng.seed(args.seed)
        train_df = pd.read_csv(f"{args.processed_dir}/train.csv", encoding_errors="ignore")
        all_ids = train_df['participant_id'].unique().tolist()
        rng.shuffle(all_ids)
        eval_ids = set(str(u) for u in all_ids[:args.n_sample])
        print(f"Restricting to {len(eval_ids)} sampled participants from train.csv")

    data_path = Path(args.data_dir)
    files = sorted(data_path.glob("*_keystrokes.txt"))
    if eval_ids is not None:
        files = [f for f in files if f.stem.replace("_keystrokes", "") in eval_ids]
    print(f"Found {len(files)} participant files, picking {args.n}")

    if args.n > len(NAMES):
        print(f"Warning: only {len(NAMES)} unique names available, capping at {len(NAMES)}")
        args.n = len(NAMES)

    profiles = []
    for filepath in files:
        if len(profiles) >= args.n:
            break

        participant_id = filepath.stem.replace("_keystrokes", "")
        sessions = read_sessions(filepath)
        if len(sessions) < 2:
            continue

        # Merge all sessions into one sequence for embedding
        all_bigrams = []
        bigram_avgs = defaultdict(list)
        total_chars = 0
        total_intervals = 0
        dwell_sum = 0.0
        dwell_count = 0
        total_elapsed_ms = 0.0

        for ks in sessions.values():
            bgs = extract_bigrams(ks)
            all_bigrams.extend(bgs)
            total_chars += len(ks)
            for (a, b), iki, dwell, flight in bgs:
                bigram_avgs[(a, b)].append(iki)
                dwell_sum += dwell
                dwell_count += 1
            total_intervals += len(bgs)
            if len(ks) >= 2:
                total_elapsed_ms += ks[-1][0] - ks[0][0]

        if len(all_bigrams) < 200:
            continue

        emb = embed(all_bigrams, sess, means, stds)
        if emb is None:
            continue

        bigrams_list = [
            [a, b, sum(vs) / len(vs)]
            for (a, b), vs in bigram_avgs.items()
        ]
        bigram_counts_list = [
            [a, b, len(vs)]
            for (a, b), vs in bigram_avgs.items()
        ]

        avg_dwell = dwell_sum / dwell_count if dwell_count > 0 else 0.0
        wpm = (total_chars / 5.0) / (total_elapsed_ms / 60_000.0) if total_elapsed_ms > 0 else 0.0

        profile = {
            "name": NAMES[len(profiles)],
            "bigrams": bigrams_list,
            "bigram_counts": bigram_counts_list,
            "char_count": total_chars,
            "interval_count": total_intervals,
            "wpm": round(wpm, 1),
            "avg_dwell_ms": avg_dwell,
            "dwell_count": dwell_count,
            "embedding": emb,
        }
        profiles.append(profile)
        print(f"  [{len(profiles)}/{args.n}] {profile['name']}  bigrams={len(all_bigrams)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nWrote {len(profiles)} profiles to {out_path}")


if __name__ == "__main__":
    main()
