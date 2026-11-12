import pickle, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

G = ["sent", "p128", "p256", "p512", "para", "section"]
BUDGETS = [256, 512, 1024]

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
    o = c["out"][B]
    qf = o["qf"]
    Q = len(qf)
    Xs = []
    for gi in range(len(G)):
        one_hot = np.tile(np.eye(len(G))[gi], (Q, 1))
        Xs.append(np.hstack([one_hot, qf]))  # QUERY ONLY: no rf
    return np.vstack(Xs)

def fit_predict(train, dev, B, params):
    params = dict(params)
    centered = params.pop("centered")
    cv = train["out"][B]["cov"]
    if centered:
        cv = cv - cv.mean(1, keepdims=True)

    model = HGB(min_samples_leaf=40, random_state=0, **params)
    model.fit(rows(train, B), cv.T.ravel())
    return model.predict(rows(dev, B)).reshape(len(G), -1).T

def evaluate(dev, B, pred):
    chosen = pred.argmax(axis=1)
    q = np.arange(len(chosen))
    return chosen, dev["out"][B]["cov"][q, chosen], dev["out"][B]["hit"][q, chosen]

def main():
    train, dev = load("train"), load("dev")
    print("=" * 80)
    print("EXPERIMENT 2 — QUERY-ONLY / MoG-STYLE ROUTER")
    print("=" * 80)
    print("Uses query features + candidate granularity ONLY. No retrieval features.\n")

    results = []
    for B in BUDGETS:
        fixed = dev["out"][B]["cov"].mean(0)
        bf_i = int(np.argmax(fixed))
        best_fixed = G[bf_i]
        best_fixed_score = float(fixed[bf_i])

        print("=" * 80)
        print(f"BUDGET = {B}")
        print("=" * 80)
        print("Fixed-granularity dev coverage:")
        for i, g in enumerate(G):
            print(f"  {g:<10} {fixed[i]:.4f}" + ("  <-- BEST FIXED" if i == bf_i else ""))

        best = None
        print("\nQuery-only grid search:")
        for pi, params in enumerate(GRID):
            pred = fit_predict(train, dev, B, params)
            chosen, cov, hit = evaluate(dev, B, pred)
            score = float(cov.mean())
            print(f"  grid{pi}: dev_cov={score:.4f} | centered={params['centered']} | "
                  f"lr={params['learning_rate']} | depth={params['max_depth']} | iter={params['max_iter']}")
            if best is None or score > best["score"]:
                best = {"score": score, "grid": pi, "chosen": chosen, "cov": cov, "hit": hit}

        counts = np.bincount(best["chosen"], minlength=len(G))
        improvement = best["score"] - best_fixed_score

        print("\nBEST QUERY-ONLY RESULT")
        print(f"  grid              : {best['grid']}")
        print(f"  mean coverage     : {best['score']:.4f}")
        print(f"  mean hit recall   : {best['hit'].mean():.4f}")
        print(f"  best fixed ({best_fixed}): {best_fixed_score:.4f}")
        print(f"  improvement       : {improvement:+.4f}")
        print("\nRouter choices:")
        for i, g in enumerate(G):
            print(f"  {g:<10} {counts[i]:>5} questions ({100*counts[i]/len(best['chosen']):5.1f}%)")

        results.append({
            "budget": B,
            "best_fixed": best_fixed,
            "best_fixed_coverage": best_fixed_score,
            "query_only_grid": best["grid"],
            "query_only_coverage": best["score"],
            "query_only_hit_recall": float(best["hit"].mean()),
            "improvement_vs_fixed": improvement,
            "choice_counts": counts.tolist()
        })

    with open("experiment2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: experiment2_results.json")

if __name__ == "__main__":
    main()
