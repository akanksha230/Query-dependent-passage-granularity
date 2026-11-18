import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

G = ["sent", "p128", "p256", "p512", "para", "section"]
BUDGETS = [256, 512, 1024]

_base = [
    dict(learning_rate=0.05, max_depth=3, max_iter=200),
    dict(learning_rate=0.05, max_depth=5, max_iter=200),
    dict(learning_rate=0.1, max_depth=3, max_iter=100),
    dict(learning_rate=0.1, max_depth=5, max_iter=100),
]

GRID = [
    dict(p, centered=c)
    for c in (False, True)
    for p in _base
]


def load(split):
    with open(f"cache_{split}.pkl", "rb") as f:
        import pickle
        return pickle.load(f)


def rows(c, B):
    """
    Experiment 4:
    Query + retrieval features, but WITHOUT
    cross-granularity agreement.

    Retained retrieval features:
      0: top-1 score
      1: top-1 score / qidf
      2: score gap
      3: score entropy
      4: top result position
      5: top unit length
      6: number of strong results
      7: mean top-5 score / top-1

    Removed:
      8: mean cross-granularity Jaccard
      9: max cross-granularity Jaccard
    """

    o = c["out"][B]
    qf = o["qf"]
    rf = o["rf"]

    Q = len(qf)

    # Mean retrieval features across granularities,
    # but only for the first 8 features.
    mean_rf = rf[:, :, :8].mean(1)

    Xs = []

    for gi in range(len(G)):

        # One-hot encoding for candidate granularity
        oh = np.tile(
            np.eye(len(G))[gi],
            (Q, 1)
        )

        parts = [
            oh,
            qf,
            rf[:, gi, :8],
            mean_rf
        ]

        Xs.append(np.hstack(parts))

    return np.vstack(Xs)


def fit_predict(train, dev, B, params):

    params = dict(params)
    centered = params.pop("centered")

    # Training target = evidence coverage
    cv = train["out"][B]["cov"]

    if centered:
        cv = cv - cv.mean(
            1,
            keepdims=True
        )

    Xtr = rows(train, B)
    ytr = cv.T.ravel()

    model = HistGradientBoostingRegressor(
        min_samples_leaf=40,
        random_state=0,
        **params
    )

    model.fit(Xtr, ytr)

    pred = model.predict(
        rows(dev, B)
    )

    return pred.reshape(len(G), -1).T


def routed(dev, B, pred):

    choices = pred.argmax(1)
    r = np.arange(len(choices))

    cov = dev["out"][B]["cov"][r, choices]
    hit = dev["out"][B]["hit"][r, choices]

    return choices, cov, hit


def main():

    train = load("train")
    dev = load("dev")

    results = {}

    for B in BUDGETS:

        print("\n" + "=" * 70)
        print(f"BUDGET = {B}")
        print("=" * 70)

        # Fixed-granularity baseline
        fixed_cov = dev["out"][B]["cov"].mean(0)
        fixed_hit = dev["out"][B]["hit"].mean(0)

        print("\nFixed granularity:")
        for gi, g in enumerate(G):
            print(
                f"  {g:8s} "
                f"coverage={fixed_cov[gi]:.4f} "
                f"hit={fixed_hit[gi]:.4f}"
            )

        best_fixed_idx = fixed_cov.argmax()

        print(
            f"\nBest fixed: {G[best_fixed_idx]} "
            f"coverage={fixed_cov[best_fixed_idx]:.4f}"
        )

        best = None

        for pi, params in enumerate(GRID):

            pred = fit_predict(
                train,
                dev,
                B,
                params
            )

            choices, cov, hit = routed(
                dev,
                B,
                pred
            )

            score = cov.mean()

            print(
                f"Query+Ret-NoAgreement "
                f"grid{pi}: "
                f"coverage={score:.4f} "
                f"hit={hit.mean():.4f}"
            )

            if best is None or score > best["coverage"]:
                best = {
                    "grid": pi,
                    "coverage": float(score),
                    "hit": float(hit.mean()),
                    "choices": choices,
                    "params": params
                }

        choices = best["choices"]

        counts = np.bincount(
            choices,
            minlength=len(G)
        )

        print("\nBEST:")
        print(
            f"  grid={best['grid']}"
        )
        print(
            f"  coverage={best['coverage']:.4f}"
        )
        print(
            f"  hit={best['hit']:.4f}"
        )
        print(
            f"  improvement vs fixed="
            f"{best['coverage'] - fixed_cov[best_fixed_idx]:+.4f}"
        )

        print("\nGranularity choices:")

        for gi, g in enumerate(G):
            pct = counts[gi] / len(choices) * 100

            print(
                f"  {g:8s}: "
                f"{counts[gi]:4d} "
                f"({pct:5.1f}%)"
            )

        results[str(B)] = {
            "best_fixed": G[best_fixed_idx],
            "fixed_coverage": float(
                fixed_cov[best_fixed_idx]
            ),
            "fixed_hit": float(
                fixed_hit[best_fixed_idx]
            ),
            "best_grid": int(best["grid"]),
            "coverage": float(best["coverage"]),
            "hit": float(best["hit"]),
            "improvement_vs_fixed": float(
                best["coverage"]
                - fixed_cov[best_fixed_idx]
            ),
            "choices": {
                G[gi]: int(counts[gi])
                for gi in range(len(G))
            },
            "params": best["params"]
        }

    with open(
        "experiment4_results.json",
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("Experiment 4 complete.")
    print("Saved: experiment4_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()