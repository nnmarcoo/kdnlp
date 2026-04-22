# Baseline: multi-class SVM over top-50 bigram timing features.
#
# Each session becomes a fixed-length feature vector: for each of the 50
# most frequent bigrams in the training pool, we append the mean IKI, dwell,
# and flight time observed in that session (150 dimensions total). Bigrams
# that didn't appear in a session are imputed with the training population
# mean so they contribute zero signal after standardization, rather than
# pulling toward an artificial extreme. We then fit a one-vs-one RBF SVM
# and report rank-1 identification accuracy on the held-out test sessions.
#
# Why this underperforms baseline_nn.py:
#   The SVM treats every training session as an independent sample, so it
#   sees only ~2-4 examples per class (one per training session per user).
#   With 500 classes and so few samples each, the decision boundaries are
#   poorly constrained. The NN baseline sidesteps this by aggregating all
#   training sessions into a single rich enrollment profile per user before
#   comparing - more signal, less noise, no boundary-learning required.
#   The SVM also struggles with short test sessions: a session covering only
#   ~30 of the 50 bigrams leaves 20 features imputed to the mean, diluting
#   the discriminative signal the SVM was trained on.

import argparse
import random

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_BIGRAMS = 50
TIMING_COLS = ["iki_ms", "dwell_ms", "flight_ms"]


# One feature vector per (participant, session).
# For each of the top N bigrams, append mean IKI, dwell, and flight.
# Bigrams absent from the session are left as NaN so the imputer can
# fill them with the population mean (rather than zero, which after
# standardization reads as "unusually fast" rather than "no data").
def vectorize(df, top_bigrams):
    bg_set = set(top_bigrams)
    rows, labels = [], []
    for (pid, sid), sdf in df.groupby(["participant_id", "session_id"]):
        present = sdf[sdf["bigram"].isin(bg_set)].groupby("bigram")[TIMING_COLS].mean()
        vec = []
        for bg in top_bigrams:
            if bg in present.index:
                vec.extend(present.loc[bg, TIMING_COLS].tolist())
            else:
                # NaN so SimpleImputer can fill with training population mean
                vec.extend([np.nan] * len(TIMING_COLS))
        rows.append(vec)
        labels.append(pid)
    return np.array(rows, dtype=float), labels


def run(data_dir, n_users_list, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # Sample a pool up to the largest group size we'll test so that
    # smaller subsets are always strict subsets of larger ones.
    all_pids = sorted(train_df["participant_id"].unique())
    max_users = max(n_users_list)
    pool = sorted(random.sample(list(all_pids), min(max_users, len(all_pids))))
    print(f"Pool: {len(pool)} participants")

    # Select top N bigrams by total occurrence across the full training pool.
    # Using corpus-wide frequency ensures the chosen bigrams are well-covered
    # for most users, minimising imputed (missing) values in the feature vectors.
    pool_train = train_df[train_df["participant_id"].isin(pool)]
    top_bigrams = pool_train["bigram"].value_counts().head(N_BIGRAMS).index.tolist()
    print(f"Top {N_BIGRAMS} bigrams: {top_bigrams[0]!r} ... {top_bigrams[-1]!r}")

    print("\n" + "=" * 50)
    print(f"{'n_users':>8} {'Rank-1':>10} {'Random':>10}")
    print("=" * 50)

    for n_users in sorted(n_users_list):
        subset = set(pool[:n_users])

        X_train, y_train = vectorize(
            train_df[train_df["participant_id"].isin(subset)], top_bigrams
        )
        X_test, y_test = vectorize(
            test_df[test_df["participant_id"].isin(subset)], top_bigrams
        )

        # Fill missing bigrams with the training population mean so that
        # absent bigrams contribute zero signal after standardization.
        # Fit only on training data to avoid test leakage.
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Z-normalize each feature so that slow bigrams (high absolute ms)
        # don't dominate the SVM kernel over fast bigrams.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel="rbf", random_state=seed)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        print(f"{n_users:>8} {acc:>9.1%} {1 / n_users:>9.1%}")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./processed")
    parser.add_argument("--n_users", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.n_users, args.seed)
