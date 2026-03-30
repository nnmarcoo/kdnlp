import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd

TIMING_COLS = ["iki_ms", "dwell_ms", "flight_ms"]


def build_profile(df):
    profile = {}
    for bg, bdf in df.groupby("bigram"):
        profile[bg] = {col: bdf[col].mean() for col in TIMING_COLS}
        profile[bg]["n"] = len(bdf)
    return profile


def compute_bigram_stats(profiles):
    vals = defaultdict(lambda: {col: [] for col in TIMING_COLS})
    for profile in profiles.values():
        for bg, feats in profile.items():
            for col in TIMING_COLS:
                vals[bg][col].append(feats[col])

    stats = {}
    for bg, col_vals in vals.items():
        stats[bg] = {}
        for col in TIMING_COLS:
            v = col_vals[col]
            std = np.std(v)
            stats[bg][col] = (np.mean(v), std if std > 0 else 1.0)
    return stats


def normalize_profile(profile, stats):
    normed = {}
    for bg, feats in profile.items():
        if bg not in stats:
            continue
        normed[bg] = {"n": feats["n"]}
        for col in TIMING_COLS:
            mean, std = stats[bg][col]
            normed[bg][col] = (feats[col] - mean) / std
    return normed


def compare_profiles(enrolled, test):
    shared = enrolled.keys() & test.keys()
    if not shared:
        return np.inf

    total_w, total_d = 0.0, 0.0
    for bg in shared:
        w = min(enrolled[bg]["n"], test[bg]["n"])
        for col in TIMING_COLS:
            total_d += w * (enrolled[bg][col] - test[bg][col]) ** 2
        total_w += w

    return total_d / total_w if total_w > 0 else np.inf


def run(data_dir, n_users_list, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    all_pids = sorted(train_df["participant_id"].unique())
    max_users = max(n_users_list)
    pool = sorted(random.sample(list(all_pids), min(max_users, len(all_pids))))

    train_df = train_df[train_df["participant_id"].isin(pool)]
    test_df = test_df[test_df["participant_id"].isin(pool)]
    print(f"Pool: {len(pool)} participants")
    print(f"Train bigrams: {len(train_df)}, Test bigrams: {len(test_df)}")

    print("Building per-user profiles...")
    profiles = {}
    for pid, udf in train_df.groupby("participant_id"):
        profiles[pid] = build_profile(udf)

    print("Z-normalizing per bigram across users...")
    stats = compute_bigram_stats(profiles)
    norm_profiles = {pid: normalize_profile(p, stats) for pid, p in profiles.items()}

    print("Building test samples...")
    test_samples = []
    for (pid, sid), sdf in test_df.groupby(["participant_id", "session_id"]):
        tp = build_profile(sdf)
        if tp:
            test_samples.append((pid, normalize_profile(tp, stats)))

    avg_profile = np.mean([len(p) for p in profiles.values()])
    avg_test = np.mean([len(tp) for _, tp in test_samples])
    print(f"Profiles: {len(profiles)}, Test samples: {len(test_samples)}")
    print(f"Avg bigrams per profile: {avg_profile:.0f}, per test: {avg_test:.0f}")

    print("\n" + "=" * 60)
    print(f"{'n_users':>8} {'Rank-1':>10} {'Top-5':>10} {'Random':>10}")
    print("=" * 60)

    for n_users in sorted(n_users_list):
        subset = set(pool[:n_users])
        sub_profiles = {p: norm_profiles[p] for p in subset if p in norm_profiles}
        sub_tests = [(p, tp) for p, tp in test_samples if p in sub_profiles]

        correct_1 = correct_5 = 0
        for true_pid, test_profile in sub_tests:
            ranked = sorted(
                ((compare_profiles(ep, test_profile), pid)
                 for pid, ep in sub_profiles.items()),
            )
            if ranked[0][1] == true_pid:
                correct_1 += 1
            if true_pid in {pid for _, pid in ranked[:5]}:
                correct_5 += 1

        n = len(sub_tests)
        print(f"{n_users:>8} {correct_1 / n:>9.1%} {correct_5 / n:>9.1%} "
              f"{1 / n_users:>9.1%}")

    print("=" * 60)
    print(f"Test samples evaluated: {len(test_samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./processed")
    parser.add_argument("--n_users", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.n_users, args.seed)
