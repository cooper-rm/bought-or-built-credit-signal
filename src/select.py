import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss


def null_importance(make_gbm, X, y, k=20, n=80_000, seed=0, verbose=True):
    """Target-permutation (null) importance. Fit a GBM once on the real target for actual gain per
    feature, then k times on a *shuffled* target for a null gain distribution. Each feature is judged
    against ITSELF under a broken target (same distribution / cardinality / correlations), which
    avoids the distribution-mismatch bias of random-noise probes. Returns a frame of actual gain,
    null mean/95th/max, and a keep flag (actual gain > its own 95th-percentile null)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].reset_index(drop=True).to_numpy()

    def gains(target):
        return make_gbm().fit(Xs, target).booster_.feature_importance("gain")

    actual = pd.Series(gains(ys), index=Xs.columns)
    nulls = []
    for i in range(k):
        nulls.append(gains(rng.permutation(ys)))
        if verbose:
            print(f"null run {i + 1:2d}/{k}")
    null = pd.DataFrame(np.vstack(nulls), columns=Xs.columns)
    out = pd.DataFrame({"actual": actual, "null_mean": null.mean(),
                        "null_95": null.quantile(0.95), "null_max": null.max()})
    out["keep"] = out["actual"] > out["null_95"]
    return out.sort_values("actual", ascending=False)


def boruta(make_gbm, X, y, k=20, n=80_000, seed=0, alpha=0.05, verbose=True):
    """Boruta all-relevant selection: LightGBM gain vs. shadow features.

    Each round every feature is duplicated as a *shadow* (its values shuffled), a GBM is fit on
    the real+shadow matrix, and a real feature scores a **hit** when its gain beats the best
    shadow's. Over ``k`` rounds, hits ~ Binomial(k, 0.5) under 'no better than a shadow'; a
    feature is **kept** when its hits land in the upper ``alpha`` tail. Unlike a single-pass
    importance filter this keeps *all* features with real signal (redundant or not) — an
    all-relevant filter; pair it with a wrapper (``backward_stability``) to trim redundancy.
    Doubles the column count per fit, so use a light ``make_gbm`` (few trees)."""
    from scipy.stats import binom

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].reset_index(drop=True).to_numpy()
    cols = list(Xs.columns)
    shadow_cols = [f"shadow_{c}" for c in cols]
    hits = pd.Series(0, index=cols)
    for i in range(k):
        shadow = pd.DataFrame({s: rng.permutation(Xs[c].to_numpy()) for c, s in zip(cols, shadow_cols)})
        gain = make_gbm().fit(pd.concat([Xs, shadow], axis=1), ys).booster_.feature_importance("gain")
        gain = pd.Series(gain, index=cols + shadow_cols)
        hits += (gain[cols].to_numpy() > gain[shadow_cols].max()).astype(int)
        if verbose:
            conf = int((hits >= binom.ppf(1 - alpha, i + 1, 0.5)).sum())
            print(f"round {i + 1:2d}/{k}  confirmed so far: {conf}")
    out = pd.DataFrame({"hits": hits})
    out["keep"] = out["hits"] >= binom.ppf(1 - alpha, k, 0.5)
    return out.sort_values("hits", ascending=False)


def backward_stability(make_model, X, y, k_runs=10, n: int | None = 60_000, tol=1e-4, batch=15, n_jobs=1, seed=0, verbose=True):
    """K backward-elimination runs, each following a random feature drop-order (seed + r), scored on
    log-loss. Start from all features; drop a batch if removing it does not raise held-out log-loss
    by more than tol (else retry each singly). Returns per-feature survival frequency across runs
    (high = robustly kept). No permutation-importance step, so cheaper than ranked backward.
    Runs serially (n_jobs=1) by default so verbose per-batch progress prints live; set n_jobs=-1 to
    parallelise across runs (faster, but per-batch prints are then swallowed by the workers)."""
    cols = list(X.columns)

    def one_run(r):
        rng = np.random.RandomState(seed + r)
        size = len(y) if n is None else min(n, len(y))
        idx = rng.choice(len(y), size=size, replace=False)
        Xr, yr = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        tr, va = train_test_split(np.arange(len(yr)), test_size=0.30, stratify=yr, random_state=r)
        Xtr, Xva, ytr, yva = Xr.iloc[tr], Xr.iloc[va], yr.iloc[tr], yr.iloc[va]

        def prob(c):
            return make_model().fit(Xtr[c], ytr).predict_proba(Xva[c])[:, 1]

        cur = list(cols)
        best = log_loss(yva, prob(cur), labels=[0, 1])
        queue, i = list(rng.permutation(cols)), 0
        while i < len(queue):
            drop = [c for c in queue[i:i + batch] if c in cur]
            i += batch
            if not drop or len(cur) - len(drop) < 5:
                continue
            trial = [c for c in cur if c not in drop]
            l = log_loss(yva, prob(trial), labels=[0, 1])
            if l <= best + tol:
                cur, best = trial, min(best, l)
            else:
                for c in drop:
                    t = [x for x in cur if x != c]
                    l1 = log_loss(yva, prob(t), labels=[0, 1])
                    if l1 <= best + tol:
                        cur, best = t, min(best, l1)
            if verbose:
                print(f"  run {r} | batch {i // batch:3d} | kept {len(cur):4d} | logloss {best:.4f}")
        p = prob(cur)
        return cur, best, roc_auc_score(yva, p)

    runs = Parallel(n_jobs=n_jobs)(delayed(one_run)(r) for r in range(k_runs))
    freq = pd.Series(0.0, index=cols)
    for r, (cur, ll, auc) in enumerate(runs):
        print(f"run {r:2d}: kept {len(cur):4d} / {len(cols)} | AUC {auc:.4f} | logloss {ll:.4f}")
        for c in cur:
            freq[c] += 1
    return (freq / k_runs).sort_values(ascending=False)


def forward_stability(make_model, X, y, k_runs=20, n=25_000, tol=1e-4, max_feats=60, n_jobs=-1, seed=0):
    """Randomized-restart forward selection scored on log-loss (a smooth proper score; AUC is only
    the final eval metric). Each run draws a no-replacement row subsample and a shuffled feature
    order, then does a fixed-order forward pass: add a feature only if it drops held-out log-loss
    by >= tol, else keep it out. Returns per-feature selection frequency and mean entry rank across
    the k_runs, so order/sampling luck surfaces as a stability score. No threshold is applied."""
    cols = list(X.columns)

    def one_run(r):
        rng = np.random.RandomState(seed + r)
        idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
        Xr, yr = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        tr, va = train_test_split(np.arange(len(yr)), test_size=0.30, stratify=yr, random_state=r)
        Xtr, Xva, ytr, yva = Xr.iloc[tr], Xr.iloc[va], yr.iloc[tr], yr.iloc[va]
        best = log_loss(yva, np.full(len(yva), ytr.mean()), labels=[0, 1])
        best_auc, selected, entry = 0.5, [], {}
        for c in rng.permutation(cols):
            trial = selected + [c]
            m = make_model().fit(Xtr[trial], ytr)
            p = m.predict_proba(Xva[trial])[:, 1]
            loss = log_loss(yva, p, labels=[0, 1])
            if loss <= best - tol:
                selected.append(c)
                best, best_auc, entry[c] = loss, roc_auc_score(yva, p), len(selected)
                if max_feats is not None and len(selected) >= max_feats:
                    break
        return selected, entry, best_auc, best

    runs = Parallel(n_jobs=n_jobs)(delayed(one_run)(r) for r in range(k_runs))
    freq = pd.Series(0.0, index=cols)
    ranks = {c: [] for c in cols}
    for r, (selected, entry, auc, ll) in enumerate(runs):
        print(f"run {r:2d}: {len(selected):3d} feats | AUC {auc:.4f} | logloss {ll:.4f}")
        for c in selected:
            freq[c] += 1
        for c, rk in entry.items():
            ranks[c].append(rk)
    freq = (freq / k_runs).sort_values(ascending=False)
    order_rank = pd.Series({c: np.mean(v) if v else np.nan for c, v in ranks.items()})
    return freq, order_rank


def eliminate(make_model, X, y, n=100_000, tol=5e-4, batch=15):
    Xs, _, ys, _ = train_test_split(X, y, train_size=min(n, len(y)), stratify=y, random_state=0)
    Xs, ys = Xs.reset_index(drop=True), ys.reset_index(drop=True)
    tr, va = train_test_split(np.arange(len(ys)), test_size=0.30, stratify=ys, random_state=1)
    ytr, yva = ys.iloc[tr], ys.iloc[va]
    Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]

    def auc(cols):
        m = make_model().fit(Xtr[cols], ytr)
        return roc_auc_score(yva, m.predict_proba(Xva[cols])[:, 1])

    feats = list(X.columns)
    base = auc(feats)
    print(f"baseline {len(feats)} feats: {base:.4f}")
    mdl = make_model().fit(Xtr, ytr)
    perm = permutation_importance(mdl, Xva, yva, scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=-1)
    queue = list(pd.Series(perm.importances_mean, index=feats).sort_values().index)

    cur, best, i = list(feats), base, 0
    while i < len(queue):
        drop = [c for c in queue[i:i + batch] if c in cur]
        i += batch
        if not drop or len(cur) - len(drop) < 5:
            continue
        trial = [c for c in cur if c not in drop]
        a = auc(trial)
        if a >= best - tol:
            cur, best = trial, max(best, a)
            print(f"  drop {len(drop):2d} -> {len(cur):4d}  AUC {a:.4f}")
        else:
            for c in drop:
                t = [x for x in cur if x != c]
                a1 = auc(t)
                if a1 >= best - tol:
                    cur, best = t, max(best, a1)
            print(f"  batch hurt (AUC {a:.4f}); {len(cur):4d} remain")
    print(f"FINAL kept {len(cur)}/{len(feats)} | AUC {auc(cur):.4f} vs {base:.4f}")
    return cur
