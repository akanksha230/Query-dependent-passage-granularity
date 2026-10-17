"""
Pilot build: per-question, per-granularity BM25 retrieval over the *question's own paper* (QASPER),
filled to a matched word budget B, scored against gold evidence, plus router features.

Granularities (G): sent, p128, p256, p512, para, section
Budgets (B, whitespace words): 256, 512, 1024
Outputs: cache_{split}.pkl  (utilities, features, meta, C2F results, timing)
"""
import json, re, math, time, pickle, sys
from collections import Counter
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

DATA = "./data"
G = ["sent", "p128", "p256", "p512", "para", "section"]
BUDGETS = [256, 512, 1024]
K1, B_ = 1.2, 0.75
STOP = set(ENGLISH_STOP_WORDS)
NOISE = re.compile(r"^(bibref|figref|tabref|secref|inlineform|displayform)\d*$")
TOK = re.compile(r"[a-z0-9]+")
ABBR = {"e.g.", "i.e.", "et", "al.", "fig.", "figs.", "eq.", "vs.", "cf.", "etc.", "approx.", "no.", "inc.", "ca."}


def toks(text):
    return [t for t in TOK.findall(text.lower()) if t not in STOP and not NOISE.match(t) and len(t) > 1]


def split_sents(words):
    n, out, start = len(words), [], 0
    for i, w in enumerate(words):
        if re.search(r"[.!?][\"')\]]?$", w) and w.lower() not in ABBR and not re.fullmatch(r"[A-Z]\.", w):
            nxt = words[i + 1] if i + 1 < n else None
            if nxt is None or re.match(r"[A-Z0-9\"“(\[$]", nxt):
                out.append([start, i + 1]); start = i + 1
    if start < n: out.append([start, n])
    merged = []  # merge tiny fragments into the previous sentence
    for s, e in out:
        if merged and (e - s) < 4: merged[-1][1] = e
        else: merged.append([s, e])
    return [tuple(x) for x in merged]


class Paper:
    def __init__(self, pid, p, idf_fn):
        self.pid = pid
        paras, sec_of = [], []
        for si, sec in enumerate(p["full_text"]):
            for para in sec["paragraphs"]:
                if para and para.strip():
                    paras.append(para.strip()); sec_of.append(si)
        self.paras = paras
        words, pint = [], []
        for para in paras:
            w = para.split(); pint.append((len(words), len(words) + len(w))); words += w
        self.n = len(words); self.pint = pint
        # unit intervals per granularity
        U = {}
        sent = []; para_sent_sec = []
        for (s, e), si in zip(pint, sec_of):
            for a, b in split_sents(words[s:e]): sent.append((s + a, s + b))
        U["sent"] = sent
        for N, name in [(128, "p128"), (256, "p256"), (512, "p512")]:
            U[name] = [(i, min(i + N, self.n)) for i in range(0, self.n, N)]
        U["para"] = list(pint)
        secs = {}
        for (s, e), si in zip(pint, sec_of):
            a, b = secs.get(si, (s, e)); secs[si] = (min(a, s), max(b, e))
        U["section"] = [secs[k] for k in sorted(secs)]
        self.U = {g: (np.array([u[0] for u in U[g]], dtype=int), np.array([u[1] for u in U[g]], dtype=int)) for g in G}
        # sentence -> section index (for coarse-to-fine)
        ss = self.U["section"]
        self.sent_sec = np.searchsorted(ss[1], self.U["sent"][0], side="right")
        # term matrices
        vocab, rows, cols, vals = {}, [], [], []
        for wi, w in enumerate(words):
            for t in toks(w):
                j = vocab.setdefault(t, len(vocab)); rows.append(wi); cols.append(j); vals.append(1.0)
        V = max(len(vocab), 1)
        M = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, V))
        self.vocab = vocab
        self.idf = np.array([idf_fn(t) for t in sorted(vocab, key=vocab.get)]) if vocab else np.zeros(1)
        self.W = {}
        for g in G:
            s, e = self.U[g]
            uow = np.repeat(np.arange(len(s)), e - s)  # every granularity partitions the paper's words
            A = sp.csr_matrix((np.ones(self.n), (uow, np.arange(self.n))), shape=(len(s), self.n))
            C = (A @ M).tocoo()
            ln = (e - s).astype(float); avg = ln.mean()
            denom = C.data + K1 * (1 - B_ + B_ * ln[C.row] / avg)
            data = self.idf[C.col] * C.data * (K1 + 1) / denom
            self.W[g] = sp.csr_matrix((data, (C.row, C.col)), shape=C.shape)
        # evidence lookup
        self.p2i = {}
        for i, t in enumerate(paras): self.p2i.setdefault(t, i)

    def qterms(self, q):
        qt = list(dict.fromkeys(toks(q)))
        ids = [self.vocab[t] for t in qt if t in self.vocab]
        return qt, ids

    def scores(self, g, ids):
        W = self.W[g]
        if not ids: return np.zeros(W.shape[0])
        return np.asarray(W[:, ids].sum(axis=1)).ravel()

    def evidence_sets(self, qa):
        """list of annotator gold sets: each = (list of para idx). Float/table evidence dropped."""
        sets, atype = [], None
        for a in qa["answers"]:
            ans = a["answer"]
            if ans.get("unanswerable"): continue
            idxs = []
            for ev in ans.get("evidence", []):
                ev = ev.strip()
                if not ev or ev.startswith("FLOAT SELECTED"): continue
                if ev in self.p2i: idxs.append(self.p2i[ev])
                else:
                    hit = [i for i, t in enumerate(self.paras) if ev in t]
                    if hit: idxs.append(hit[0])
            if idxs:
                sets.append(sorted(set(idxs)))
                if atype is None:
                    atype = "yes_no" if ans.get("yes_no") is not None else ("extractive" if ans.get("extractive_spans") else "free_form")
        return sets, atype


def fill(order, s, e, B, mask):
    used = 0
    for u in order:
        rem = B - used
        if rem <= 0: break
        ln = e[u] - s[u]; take = min(ln, rem)
        mask[s[u]:s[u] + take] = True; used += take
    return used


def evaluate(mask, pint, sets):
    best_cov, best_hit = 0.0, 0.0
    for idxs in sets:
        tot = cov = hit = 0
        for i in idxs:
            a, b = pint[i]; tot += b - a; c = mask[a:b].sum(); cov += c; hit += c > 0
        best_cov = max(best_cov, cov / max(tot, 1)); best_hit = max(best_hit, hit / len(idxs))
    return best_cov, best_hit


def entropy_top(s, k=10):
    top = np.sort(s)[::-1][:k]
    if top.sum() <= 0: return math.log(max(len(top), 1))
    p = top / top.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


WH = ["what", "which", "how", "why", "where", "who", "when", "do", "does", "is", "are", "can", "did", "was", "were", "will", "could", "has", "have"]


def qfeats(q, qt, idf_of):
    ql = q.lower().split()
    f = [len(ql), len(qt)]
    idfs = [idf_of(t) for t in qt] or [0.0]
    f += [sum(idfs), float(np.mean(idfs)), max(idfs)]
    first = ql[0] if ql else ""
    f += [1.0 if first == w else 0.0 for w in WH]
    f += [float(any(c.isdigit() for c in q)), float(any(w[0].isupper() for w in q.split()[1:] if w)),
          float(" or " in q.lower()), float(" and " in q.lower()), float('"' in q or "'" in q)]
    return f


def main():
    t0 = time.time()
    train = json.load(open(f"{DATA}/qasper-train-v0.3.json"))
    # global df over train paragraphs (BM25 idf), LSA embedding fit on train only
    df, Np, sample = Counter(), 0, []
    for p in train.values():
        for sec in p["full_text"]:
            for para in sec["paragraphs"]:
                if para and para.strip():
                    Np += 1; df.update(set(toks(para)))
                    if Np % 4 == 0: sample.append(para)
    idf_of = lambda t: math.log(1 + (Np - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
    tf = TfidfVectorizer(tokenizer=toks, lowercase=False, token_pattern=None, min_df=3, max_features=40000, sublinear_tf=True)
    X = tf.fit_transform(sample); svd = TruncatedSVD(50, random_state=0).fit(X)
    print(f"idf/LSA fit: {Np} train paragraphs, {time.time()-t0:.0f}s", flush=True)
    pickle.dump({"tf": tf, "svd": svd}, open("./lsa.pkl", "wb"))

    for split in ["train", "dev", "test"]:
        data = train if split == "train" else json.load(open(f"{DATA}/qasper-{split}-v0.3.json"))
        out = {b: {"cov": [], "hit": [], "rf": [], "qf": [], "extra": []} for b in BUDGETS}
        meta, c2f, timing, nunits, stats = [], [], [], {g: 0 for g in G}, Counter()
        t1 = time.time()
        for pi, (pid, p) in enumerate(data.items()):
            if not any(para and para.strip() for sec in p['full_text'] for para in sec['paragraphs']):
                stats['empty_papers'] += 1; continue
            paper = Paper(pid, p, idf_of)
            for g in G: nunits[g] += len(paper.U[g][0])
            for qa in p["qas"]:
                stats["questions"] += 1
                sets, atype = paper.evidence_sets(qa)
                if not sets: stats["no_valid_evidence"] += 1; continue
                q = qa["question"]; qt, ids = paper.qterms(q)
                qidf = sum(idf_of(t) for t in qt) or 1.0
                sc, orders, tms = {}, {}, {}
                for g in G:
                    ts = time.perf_counter(); sc[g] = paper.scores(g, ids); orders[g] = np.argsort(-sc[g], kind="stable"); tms[g] = time.perf_counter() - ts
                qf = qfeats(q, qt, idf_of) + list(svd.transform(tf.transform([q]))[0])
                meta.append(dict(pid=pid, qid=qa["question_id"], atype=atype, n_words=paper.n, split=split))
                timing.append(tms)
                for B in BUDGETS:
                    masks, covs, hits, rf = {}, [], [], []
                    for g in G:
                        s, e = paper.U[g]; m = np.zeros(paper.n, bool); fill(orders[g], s, e, B, m); masks[g] = m
                        c, h = evaluate(m, paper.pint, sets); covs.append(c); hits.append(h)
                    for g in G:
                        s, e = paper.U[g]; o = orders[g]; x = sc[g][o]
                        s1 = x[0]; s2 = x[1] if len(x) > 1 else 0.0
                        jac = [ (masks[g] & masks[h]).sum() / max((masks[g] | masks[h]).sum(), 1) for h in G if h != g ]
                        rf.append([s1, s1 / qidf, (s1 - s2) / (s1 + 1e-9), entropy_top(sc[g]), s[o[0]] / paper.n, float(e[o[0]] - s[o[0]]),
                                   float((x > 0.5 * s1).sum()) if s1 > 0 else 0.0, float(x[:5].mean() / (s1 + 1e-9)),
                                   float(np.mean(jac)), float(np.max(jac))])
                    out[B]["cov"].append(covs); out[B]["hit"].append(hits); out[B]["rf"].append(rf); out[B]["qf"].append(qf)
                    # coarse-to-fine descent: top-k sections, then rank sentences within them
                    res = {}
                    ssn, sen = paper.U["sent"]
                    sec_rank = orders["section"]
                    for kname, kk in [("c2f_s1", 1), ("c2f_s2", 2)]:
                        m = np.zeros(paper.n, bool); used = 0; r = 0
                        while used < B and r < len(sec_rank):
                            secs = sec_rank[r:r + kk] if r == 0 else sec_rank[r:r + 1]
                            r += len(secs)
                            sel = np.where(np.isin(paper.sent_sec, secs))[0]
                            if len(sel) == 0: continue
                            o = sel[np.argsort(-sc["sent"][sel], kind="stable")]
                            used += fill(o, ssn, sen, B - used, m)
                        res[kname] = evaluate(m, paper.pint, sets)
                    c2f.append((B, res))
            if (pi + 1) % 100 == 0: print(f"  {split}: {pi+1}/{len(data)} papers {time.time()-t1:.0f}s", flush=True)
        for B in BUDGETS:
            for k in out[B]: out[B][k] = np.array(out[B][k], float)
        # reorganise c2f
        c2 = {B: {k: np.array([r[k] for b, r in c2f if b == B]) for k in ["c2f_s1", "c2f_s2"]} for B in BUDGETS}
        pickle.dump(dict(out=out, meta=meta, c2f=c2, timing=timing, nunits=nunits, stats=dict(stats), npapers=len(data)),
                    open(f"./cache_{split}.pkl", "wb"))
        print(f"{split} done: {len(meta)} usable questions of {stats['questions']} ({time.time()-t1:.0f}s); units={nunits}", flush=True)


if __name__ == "__main__":
    main()
