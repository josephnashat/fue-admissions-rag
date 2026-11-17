"""Statistics for Tables 7-9: mean ± SD, 95% CI (t), two-sided Wilcoxon signed-rank tests with Holm
correction and effect size r = |Z|/sqrt(n), paired t-test as a secondary check, refusal accuracy with
Wilson 95% CI and exact McNemar tests. Writes results.json and results.md."""
import os, json, math
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = ["B1", "B2", "B3", "P"]
ABL = ["P", "P-noT", "P-noG", "P-noA", "P-noR"]
METRICS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer relevance",
           "llm_context_precision_with_reference": "Context precision", "semantic_similarity": "Semantic similarity"}

S = {s: pd.read_csv(os.path.join(HERE, "scores", f"{s}.csv")).set_index("id")
     for s in sorted(set(MAIN + ABL)) if os.path.exists(os.path.join(HERE, "scores", f"{s}.csv"))}


def msd(x):
    x = pd.Series(x).dropna().astype(float); n = len(x)
    if n == 0:
        return None
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "ci": [m - h, m + h]}


def wilson(k, n):
    if n == 0:
        return [float("nan")] * 2
    z = 1.959964; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def refusal(df):
    un = df[df.answerable == 0]; an = df[df.answerable == 1]
    ok = ((un.declines == 1) & (un.unsupported_specifics == 0)).astype(int)
    return {"refusal_acc": ok.mean(), "refusal_ci": wilson(int(ok.sum()), len(ok)), "n_unanswerable": len(ok),
            "ok": ok, "false_refusal": an.declines.mean(), "hallucinated_specifics_unanswerable": un.unsupported_specifics.mean()}


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj.tolist()


def compare(a, b, col):
    j = pd.concat([S[a][col], S[b][col]], axis=1, keys=["a", "b"]).dropna()
    d = j.a - j.b
    if len(d) < 5 or (d == 0).all():
        return None
    w = stats.wilcoxon(j.a, j.b, zero_method="wilcox", method="approx")
    nz = int((d != 0).sum())
    z = abs(stats.norm.ppf(w.pvalue / 2))
    t = stats.ttest_rel(j.a, j.b)
    return {"n": len(d), "diff": d.mean(), "p": w.pvalue, "r": z / math.sqrt(nz), "t_p": t.pvalue}


def mcnemar(a_ok, b_ok):
    j = pd.concat([a_ok, b_ok], axis=1, keys=["a", "b"]).dropna()
    b01 = int(((j.a == 1) & (j.b == 0)).sum()); b10 = int(((j.a == 0) & (j.b == 1)).sum())
    p = stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"a_only": b01, "b_only": b10, "p": p}


out = {"systems": {}, "tests": {}, "ablation": {}}
for s, df in S.items():
    an = df[df.answerable == 1]; rf = refusal(df)
    out["systems"][s] = {**{m: msd(an[m]) if m in an else None for m in METRICS},
                         "llm_calls": msd(df.llm_calls), "latency_s": msd(df.latency_s),
                         "gen_usd_per_q": float(df.gen_usd.mean()),
                         **{k: v for k, v in rf.items() if k != "ok"},
                         "reuse_rate": float(df.reused.fillna(False).astype(bool).mean()) if "reused" in df else None}

if "P" in S:
    for m in METRICS:
        res = {b: compare("P", b, m) for b in ["B1", "B2", "B3"] if b in S and m in S[b]}
        res = {k: v for k, v in res.items() if v}
        adj = holm([v["p"] for v in res.values()]) if res else []
        for (k, v), pa in zip(res.items(), adj):
            v["p_holm"] = pa
        out["tests"][m] = res
    okP = refusal(S["P"])["ok"]
    out["tests"]["refusal_mcnemar"] = {b: mcnemar(okP, refusal(S[b])["ok"]) for b in ["B1", "B2", "B3"] if b in S}
    for a in ABL[1:]:
        if a in S:
            out["ablation"][a] = {m: compare("P", a, m) for m in METRICS if m in S[a]}

json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2, default=float)


def f(d, pct=False):
    if not d:
        return "n/a"
    k = 100 if pct else 1
    return f"{d['mean'] * k:.3f} ± {d['sd'] * k:.3f}" if not pct else f"{d['mean'] * k:.1f} ± {d['sd'] * k:.1f}"


lines = ["## Main comparison (answerable questions; refusal on unanswerable)", "",
         "| System | " + " | ".join(METRICS.values()) + " | Refusal acc. % [95% CI] | False refusal % | LLM calls | Latency s | $/q |",
         "|---" * (len(METRICS) + 6) + "|"]
for s in MAIN + [a for a in ABL[1:]]:
    if s not in out["systems"]:
        continue
    r = out["systems"][s]
    lines.append(f"| {s} | " + " | ".join(f(r[m]) for m in METRICS) +
                 f" | {100 * r['refusal_acc']:.1f} [{100 * r['refusal_ci'][0]:.1f}, {100 * r['refusal_ci'][1]:.1f}]"
                 f" | {100 * r['false_refusal']:.1f} | {f(r['llm_calls'])} | {f(r['latency_s'])} | {r['gen_usd_per_q']:.5f} |")
lines += ["", "## Proposed vs baselines (Wilcoxon, Holm-adjusted p, effect size r; paired-t p)", ""]
for m, res in out["tests"].items():
    if m == "refusal_mcnemar":
        for b, v in res.items():
            lines.append(f"- Refusal, P vs {b}: P-only correct {v['a_only']}, {b}-only correct {v['b_only']}, McNemar exact p = {v['p']:.4g}")
        continue
    for b, v in res.items():
        lines.append(f"- {METRICS[m]}, P vs {b}: Δ = {v['diff']:+.3f}, p_holm = {v['p_holm']:.4g}, r = {v['r']:.2f} (t-test p = {v['t_p']:.4g}, n = {v['n']})")
lines += ["", "## Ablations (P minus variant)", ""]
for a, res in out["ablation"].items():
    for m, v in res.items():
        if v:
            lines.append(f"- {a}, {METRICS[m]}: Δ(P − {a}) = {v['diff']:+.3f}, Wilcoxon p = {v['p']:.4g}, r = {v['r']:.2f}")
open(os.path.join(HERE, "results.md"), "w", encoding="utf8").write("\n".join(lines))
print("\n".join(lines))
