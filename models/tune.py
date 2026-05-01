import argparse
import random
import time

import numpy as np
import torch

from lstm import load_data, train as lstm_train


STAGE1_SPACE = {
    "hidden_size":       [64, 128, 256],
    "dropout":           [0.3, 0.4, 0.5, 0.6],
    "recurrent_dropout": [0.0, 0.1, 0.2, 0.3],
    "lr":                [1e-4, 5e-4, 1e-3],
    "n_augments":        [1, 3, 5],
    "temperature":       [0.05, 0.07, 0.1, 0.2],
}

# Focused around stage 1 best: hidden=256, dropout=0.4, rec_dropout=0.3, lr=1e-3, aug=5, temp=0.2
STAGE2_SPACE = {
    "hidden_size":       [192, 256, 320],
    "dropout":           [0.3, 0.4, 0.5],
    "recurrent_dropout": [0.2, 0.3, 0.4],
    "lr":                [5e-4, 1e-3, 2e-3],
    "n_augments":        [5, 7, 10],
    "temperature":       [0.15, 0.2, 0.3],
}


def sample_configs(space, n, seed):
    seen = set()
    configs = []
    trial_seed = seed
    while len(configs) < n:
        random.seed(trial_seed)
        trial_seed += 1
        cfg = {k: random.choice(v) for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def run_trial(trial_num, total, config, args):
    print(f"\n{'='*60}")
    print(f"Trial {trial_num}/{total} | " + " | ".join(f"{k}={v}" for k, v in config.items()))
    print(f"{'='*60}")

    t0 = time.time()
    try:
        eer = lstm_train(
            data=args.data,
            epochs=args.epochs,
            p_users=args.p_users,
            k_samples=args.k_samples,
            hidden_size=config["hidden_size"],
            lr=config["lr"],
            dropout=config["dropout"],
            recurrent_dropout=config["recurrent_dropout"],
            n_augments=config["n_augments"],
            temperature=config["temperature"],
            seed=args.seed,
        )
    except Exception as e:
        print(f"Trial failed: {e}")
        return None

    elapsed = time.time() - t0
    print(f"Trial {trial_num}/{total} done in {elapsed:.0f}s | EER: {eer*100:.2f}%")
    return eer


def print_leaderboard(results, stage):
    results.sort()  # lower EER is better
    print(f"\n{'='*60}")
    print(f"STAGE {stage} LEADERBOARD ({len(results)} trials)")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'EER':<12} Config")
    print(f"{'-'*60}")
    for rank, (eer, cfg) in enumerate(results, 1):
        cfg_str = " | ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{rank:<6} {eer*100:<11.2f}% {cfg_str}")
    best_eer, best_cfg = results[0]
    print(f"\nBest config (EER: {best_eer*100:.2f}%):")
    for k, v in best_cfg.items():
        print(f"  --{k} {v}")


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.data = load_data(
        args.data_dir, args.n_train_users, args.n_eval_users,
        val_fraction=args.val_fraction, seed=args.seed,
    )

    space = STAGE1_SPACE if args.stage == 1 else STAGE2_SPACE
    configs = sample_configs(space, args.n_trials, args.seed)

    results = []
    for i, config in enumerate(configs, 1):
        eer = run_trial(i, len(configs), config, args)
        if eer is not None:
            results.append((eer, config))

    if not results:
        print("All trials failed.")
        return

    print_leaderboard(results, args.stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",           type=int,   default=1, choices=[1, 2])
    parser.add_argument("--data_dir",        default="./processed")
    parser.add_argument("--n_train_users",   type=int,   default=5000)
    parser.add_argument("--n_eval_users",    type=int,   default=200)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--p_users",         type=int,   default=64)
    parser.add_argument("--k_samples",       type=int,   default=8)
    parser.add_argument("--val_fraction",    type=float, default=0.15)
    parser.add_argument("--n_trials",        type=int,   default=20)
    args = parser.parse_args()
    run(args)
