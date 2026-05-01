# Baseline: weighted nearest-neighbor over z-normalized bigram timing profiles.
#
# For each user we build an enrollment profile by aggregating all of their
# training sessions: for every bigram they typed we store the mean IKI, dwell,
# and flight time along with the observation count. At test time each held-out
# session is compared against every enrolled user's profile using a weighted
# mean squared difference (more observations -> higher weight), and the closest user
# is returned as the prediction. We report rank-1 and top-5 accuracy.
#
# Why this outperforms baseline_svm.py:
#   The SVM trains on individual sessions, giving it only ~2-4 examples per
#   class (one per training session). With 500 classes and so few samples,
#   it can't learn reliable decision boundaries. This approach avoids that
#   problem entirely: every user's training sessions are collapsed into a
#   single rich profile before any comparison is made, so we're matching
#   against a stable, low-noise representation of each person's typing style.
#   Weighting by observation count further reduces noise by down-weighting
#   bigrams seen only once or twice. No boundary learning, no sparsity issues.

import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd

TIMING_COLS = ["iki_ms", "dwell_ms", "flight_ms"]


# Build a typing profile for one user: for each bigram they typed,
# average out the timing features and record how many times they typed it.
# The observation count is stored so compare_profiles can weight by reliability.
def build_profile(df):
    profile = {}
    for bg, bdf in df.groupby("bigram"):
        profile[bg] = {col: bdf[col].mean() for col in TIMING_COLS}
        profile[bg]["n"] = len(bdf)
    return profile


# For each bigram, collect the mean and std of each timing feature across
# all users. We need this so we can z-normalize later.
#
# Different bigrams have different raw timings - "th" is fast for everyone
# (~80ms) while "qz" is slow for everyone (~300ms). Without normalization,
# slow bigrams dominate the distance calculation even though a 20ms difference
# is equally meaningful on both. Z-normalizing puts every bigram on the same
# scale so the distance reflects typing style, not absolute typing speed.
def compute_bigram_stats(profiles):
    # Gather each user's per-bigram mean into lists, one list per (bigram, feature)
    vals = defaultdict(lambda: {col: [] for col in TIMING_COLS})
    for profile in profiles.values():
        for bg, feats in profile.items():
            for col in TIMING_COLS:
                vals[bg][col].append(feats[col])

    # Compute population (mean, std) for each (bigram, feature) pair
    stats = {}
    for bg, col_vals in vals.items():
        stats[bg] = {}
        for col in TIMING_COLS:
            v = col_vals[col]
            std = np.std(v)
            # If std is 0 (everyone typed it identically), use 1 to avoid dividing by zero
            stats[bg][col] = (np.mean(v), std if std > 0 else 1.0)
    return stats


# Z-normalize a profile: for each bigram timing subtract the population mean
# and divide by population std. The result is how many standard deviations
# this user's timing differs from average for that bigram - positive means
# slower, negative means faster.
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


# Weighted mean squared difference between an enrolled profile and a test profile,
# computed only over bigrams that appear in both. Weighting by the minimum
# observation count means that bigrams seen many times (reliable signal) count
# more than bigrams seen once or twice (noisy). Lower score = more similar.
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
    train_df = pd.read_csv(f"{data_dir}/train.csv", encoding_errors="ignore")
    test_df = pd.read_csv(f"{data_dir}/test.csv", encoding_errors="ignore")

    # Sample a pool up to the largest group size we'll test so that
    # smaller subsets are always strict subsets of larger ones.
    all_pids = sorted(train_df["participant_id"].unique())
    max_users = max(n_users_list)
    pool = sorted(random.sample(list(all_pids), min(max_users, len(all_pids))))

    train_df = train_df[train_df["participant_id"].isin(pool)]
    test_df = test_df[test_df["participant_id"].isin(pool)]
    print(f"Pool: {len(pool)} participants")
    print(f"Train bigrams: {len(train_df)}, Test bigrams: {len(test_df)}")

    # Aggregate all training sessions per user into one enrollment profile
    print("Building per-user profiles...")
    profiles = {}
    for pid, udf in train_df.groupby("participant_id"):
        profiles[pid] = build_profile(udf)

    # Compute population statistics then normalize every profile so distances
    # reflect typing style rather than raw speed differences between bigrams.
    print("Z-normalizing per bigram across users...")
    stats = compute_bigram_stats(profiles)
    norm_profiles = {pid: normalize_profile(p, stats) for pid, p in profiles.items()}

    # Build one test sample per held-out session, normalized with the same stats
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
            # Rank all enrolled users by distance; pick the nearest as rank-1
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
