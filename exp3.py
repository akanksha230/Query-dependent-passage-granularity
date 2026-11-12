# exp3.py — Experiment 3: Query + Retrieval Signals
#
# This is the "core proposed method" experiment.
# It follows route.py's query+ret_all ("qa") implementation exactly:
#   features = query features + candidate granularity one-hot
#             + retrieval features for the candidate granularity
#             + mean retrieval features across all granularities
#
# It tunes on TRAIN -> DEV only. It does NOT touch TEST.
#
# Run:
#   cd C:\Users\asus\Downloads\2026CSY7519
#   python exp3.py

import json
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

G = ["sent", "p128", "p256", "p512", "para", "section"]
BUDGETS = [256, 512, 1024]

# Same grid as route.py
_base = [
    dict(learning_rate=0.05, max_depth=3, max_iter=200),
    dict(learning_rate=0.05, max_depth=5, max_iter=200),
    dict(learning_rate=0.1, max_depth=3, max_iter=100),
    dict(learning_rate=0.1, max_depth=5, max_iter=100),
]
GRID = [dict(p, centered=c) for c in (False, True) for p in _base]


def load(split):
    with open(f"./cache_{split}.pkl", "rb") as f:
        return pickle.load(f)


def rows(c, B):
    """
    Query + retrieval ("qa") features from route.py.

    For each candidate granularity g:
      [one-hot(g), query features, retrieval features for g,
       mean retrieval features across all 6 granularities]
    """
    o = c["out"][B]
    qf = o["qf"]
    rf = o["rf"]
    Q = len(qf)

    mean_rf = rf.mean(1)
    Xs = []

    for gi in range(len(G)):
        oh = np.tile(np.eye(len(G))[gi], (Q, 1))
        parts = [
            oh,
            qf,
            rf[:, gi],
            mean_rf,
        ]
        Xs.append(np.hstack(parts))

    return np.vstack(Xs)


def fit_predict(ctr, cev, B, params):
    params = dict(params)
    centered = params.pop("centered")

    cv = ctr["out"][B]["cov"]
    if centered:
        cv = cv - cv.mean(1, keepdims=True)

    Xtr = rows(ctr, B)
    ytr = cv.T.ravel()

    model = HGB(
        min_samples_leaf=40,
        random_state=0,
        **params
    ).fit(Xtr, ytr)

    return model.predict(rows(cev, B)).reshape(len(G), -1).T


def evaluate(cev, B, pred):
    ch = pred.argmax(1)
    r = np.arange(len(ch))

    cov = cev["out"][B]["cov"][r, ch]
    hit = cev["out"][B]["hit"][r, ch]

    return ch, cov, hit


def main():
    print("=" * 80)
    print("EXPERIMENT 3 — QUERY + RETRIEVAL SIGNALS")
    print("=" * 80)
    print("Uses query features + candidate granularity + retrieval features.")
    print("Retrieval features include candidate-granularity signals and")
    print("mean signals across granularities.")
    print("Training: TRAIN -> DEV only. TEST is untouched.")
    print()

    ctr = load("train")
    cev = load("dev")

    results = {}

    for B in BUDGETS:
        print("=" * 80)
        print(f"BUDGET = {B}")
        print("=" * 80)

        cov = cev["out"][B]["cov"]
        hit = cev["out"][B]["hit"]

        # Fixed baselines
        fixed = cov.mean(0)

        print("Fixed-granularity dev coverage:")
        for gi, g in enumerate(G):
            marker = "  <-- BEST FIXED" if gi == int(fixed.argmax()) else ""
            print(f"  {g:8s} {fixed[gi]:.4f}{marker}")

        print()
        print("Query + retrieval grid search:")

        best = None

        for pi, params in enumerate(GRID):
            pred = fit_predict(ctr, cev, B, params)
            ch, routed_cov, routed_hit = evaluate(cev, B, pred)

            score = routed_cov.mean()

            print(
                f"  grid{pi}: dev_cov={score:.4f} | "
                f"centered={params['centered']} | "
                f"lr={params['learning_rate']} | "
                f"depth={params['max_depth']} | "
                f"iter={params['max_iter']}"
            )

            if best is None or score > best["coverage"]:
                best = {
                    "grid": pi,
                    "coverage": float(score),
                    "hit": float(routed_hit.mean()),
                    "params": params,
                    "choices": ch,
                }

        best_fixed = float(fixed.max())
        improvement = best["coverage"] - best_fixed

        print()
        print("BEST QUERY + RETRIEVAL RESULT")
        print(f"  grid              : {best['grid']}")
        print(f"  mean coverage     : {best['coverage']:.4f}")
        print(f"  mean hit recall   : {best['hit']:.4f}")
        print(f"  best fixed (para): {best_fixed:.4f}")
        print(f"  improvement       : {improvement:+.4f}")

        print()
        print("Router choices:")
        counts = np.bincount(best["choices"], minlength=len(G))
        for gi, g in enumerate(G):
            pct = 100.0 * counts[gi] / len(best["choices"])
            print(f"  {g:8s} {counts[gi]:4d} questions ({pct:5.1f}%)")

        results[str(B)] = {
            "best_grid": best["grid"],
            "coverage": best["coverage"],
            "hit_recall": best["hit"],
            "best_fixed": best_fixed,
            "improvement": float(improvement),
            "params": best["params"],
            "choice_counts": counts.tolist(),
        }

        print()

    with open("experiment3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Saved: experiment3_results.json")


if __name__ == "__main__":
    main()
