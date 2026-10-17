"""
Routers + evaluation.  Usage:  python route.py dev     -> tunes on dev (train->dev), writes config.json
                               python route.py test    -> frozen config, train->test, run ONCE
Systems: fixed-g (best on dev), oracle, random, MoG-style query-only router, Query+Ret(all), Query+Ret(p256 only),
         Ret-only (ablation), Query+Ret minus agreement (ablation), coarse-to-fine descent (backoff).
Primary: budget B=512 whitespace words, utility = evidence coverage.  Others are secondary.
"""
import sys, json, pickle, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

G = ["sent", "p128", "p256", "p512", "para", "section"]
BUDGETS = [256, 512, 1024]
MODES = {"mog_style_query_only": "q", "query+ret_all": "qa", "query+ret_p256": "qp", "ret_only": "ra", "query+ret_no_agreement": "qn"}
_base = [dict(learning_rate=0.05, max_depth=3, max_iter=200), dict(learning_rate=0.05, max_depth=5, max_iter=200),
         dict(learning_rate=0.1, max_depth=3, max_iter=100), dict(learning_rate=0.1, max_depth=5, max_iter=100)]
GRID = [dict(p, centered=c) for c in (False, True) for p in _base]   # centered: target = utility minus per-query mean over granularities


def load(split): return pickle.load(open(f"./cache_{split}.pkl", "rb"))


def rows(c, mode, B):
    o = c["out"][B]; qf, rf = o["qf"], o["rf"]; Q = len(qf)
    mean_rf = rf.mean(1)
    Xs = []
    for gi in range(len(G)):
        oh = np.tile(np.eye(len(G))[gi], (Q, 1)); parts = [oh]
        if mode in ("q", "qa", "qp", "qn"): parts.append(qf)
        if mode in ("qa", "ra"): parts += [rf[:, gi], mean_rf]
        if mode == "qp": parts.append(rf[:, G.index("p256"), :8])
        if mode == "qn": parts += [rf[:, gi, :8], mean_rf[:, :8]]
        Xs.append(np.hstack(parts))
    return np.vstack(Xs)  # (G*Q, d), g-major


def fit_predict(ctr, cev, mode, B, params):
    params = dict(params); centered = params.pop("centered")
    cv = ctr["out"][B]["cov"]; cv = cv - cv.mean(1, keepdims=True) if centered else cv
    Xtr = rows(ctr, mode, B); ytr = cv.T.ravel()
    m = HGB(min_samples_leaf=40, random_state=0, **params).fit(Xtr, ytr)
    return m.predict(rows(cev, mode, B)).reshape(len(G), -1).T  # Q×G


def routed(cev, B, pred):
    ch = pred.argmax(1); r = np.arange(len(ch))
    return ch, cev["out"][B]["cov"][r, ch], cev["out"][B]["hit"][r, ch]


def boot_diff(a, b, pids, n=2000, seed=0):
    """paper-clustered paired bootstrap of mean(a-b)"""
    d = a - b; u, inv = np.unique(pids, return_inverse=True)
    s = np.bincount(inv, d); k = np.bincount(inv)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(u), (n, len(u)))
    est = s[idx].sum(1) / k[idx].sum(1)
    return d.mean(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def ci(a, pids, n=2000, seed=0):
    u, inv = np.unique(pids, return_inverse=True); s = np.bincount(inv, a); k = np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u))); est = s[idx].sum(1) / k[idx].sum(1)
    return np.percentile(est, 2.5), np.percentile(est, 97.5)


def main(phase):
    ctr = load("train")
    if phase == "dev":
        cev = load("dev"); cfg = {}
        for B in BUDGETS:
            fixed = cev["out"][B]["cov"].mean(0); cfg[str(B)] = {"best_fixed": G[int(fixed.argmax())]}
            for name, mode in MODES.items():
                best = None
                for pi, params in enumerate(GRID):
                    pred = fit_predict(ctr, cev, mode, B, params); sc = routed(cev, B, pred)[1].mean()
                    print(f"B={B} {name:24s} grid{pi} dev_cov={sc:.4f}")
                    if best is None or sc > best[0]: best = (sc, pi)
                cfg[str(B)][name] = {"grid": best[1], "dev_cov": best[0]}
        json.dump(cfg, open("./config.json", "w"), indent=1); print(json.dumps(cfg, indent=1)); return

    # ---- TEST: frozen config, run once ----
    cfg = json.load(open("./config.json")); cev = load("test"); meta = pd.DataFrame(cev["meta"]); pids = meta.pid.values
    results, detail = [], {}
    tim = pd.DataFrame(cev["timing"])[G] * 1000  # ms per query per granularity (BM25 scoring + ranking)
    for B in BUDGETS:
        cov, hit = cev["out"][B]["cov"], cev["out"][B]["hit"]; Q = len(cov); sysm = {}
        bf = G.index(cfg[str(B)]["best_fixed"])
        for gi, g in enumerate(G): sysm[f"fixed_{g}"] = (cov[:, gi], hit[:, gi], tim[g].values, 1)
        sysm["oracle"] = (cov.max(1), hit[np.arange(Q), cov.argmax(1)], np.full(Q, np.nan), np.nan)
        rr = np.random.default_rng(0).integers(0, len(G), Q)
        sysm["random_router"] = (cov[np.arange(Q), rr], hit[np.arange(Q), rr], tim.values[np.arange(Q), rr], 1)
        choices = {}
        for name, mode in MODES.items():
            pred = fit_predict(ctr, cev, mode, B, GRID[cfg[str(B)][name]["grid"]]); ch, c, h = routed(cev, B, pred); choices[name] = ch
            t = tim.values[np.arange(Q), ch].copy()
            if mode in ("qa", "ra", "qn"): t = tim.values.sum(1); nl = 6   # first pass over all 6 indexes
            elif mode == "qp": t = tim["p256"].values + np.where(ch == G.index("p256"), 0, tim.values[np.arange(Q), ch]); nl = 2
            else: nl = 1
            sysm[name] = (c, h, t, nl)
        for k in ["c2f_s1", "c2f_s2"]:
            a = cev["c2f"][B][k]; sysm[k] = (a[:, 0], a[:, 1], tim["section"].values + tim["sent"].values, 2)
        ref_q, ref_f = sysm["mog_style_query_only"][0], sysm[f"fixed_{G[bf]}"][0]
        for name, (c, h, t, nl) in sysm.items():
            lo, hi = ci(c, pids)
            row = dict(B=B, system=name, coverage=c.mean(), cov_lo=lo, cov_hi=hi, hit_recall=h.mean(), regret_vs_oracle=cov.max(1).mean() - c.mean(),
                       index_lookups=nl, mean_ms=np.nanmean(t) if not np.all(np.isnan(t)) else np.nan, p95_ms=np.nanpercentile(t, 95) if not np.all(np.isnan(t)) else np.nan)
            d, l, u = boot_diff(c, ref_q, pids); row.update(d_vs_mog=d, d_vs_mog_lo=l, d_vs_mog_hi=u)
            d, l, u = boot_diff(c, ref_f, pids); row.update(d_vs_bestfixed=d, d_vs_bf_lo=l, d_vs_bf_hi=u)
            if name in MODES:
                row.update(win_vs_mog=float((c > ref_q + 1e-9).mean()), loss_vs_mog=float((c < ref_q - 1e-9).mean()))
            results.append(row)
        detail[B] = {n: np.bincount(ch, minlength=6).tolist() for n, ch in choices.items()}
        detail[B]["oracle_best"] = np.bincount(cov.argmax(1), minlength=6).tolist()
        # answer-type slices (primary budget)
        if B == 512:
            sl = []
            for at in meta.atype.unique():
                m = (meta.atype == at).values
                for name in ["fixed_" + G[bf], "mog_style_query_only", "query+ret_all", "query+ret_p256", "oracle"]:
                    sl.append(dict(atype=at, n=int(m.sum()), system=name, coverage=sysm[name][0][m].mean()))
            pd.DataFrame(sl).to_csv("./results_slices_B512.csv", index=False)
    df = pd.DataFrame(results); df.to_csv("./results_test.csv", index=False)
    json.dump(dict(choices=detail, G=G, nq=len(meta), npapers=int(meta.pid.nunique()), best_fixed={b: cfg[str(b)]["best_fixed"] for b in BUDGETS}, nunits=cev["nunits"]),
              open("./results_detail.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df[df.B == 512].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
