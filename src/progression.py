import pyarrow.parquet as pq
import pandas as pd
from sklearn.metrics import roc_auc_score
from .data import INTERIM
from .models import oof

SOURCES = [
    ("application", ["application"]),
    ("+ bureau", ["bureau"]),
    ("+ previous_app", ["previous_application"]),
    ("+ pos_cash", ["pos_cash_balance"]),
    ("+ installments", ["installments_payments"]),
    ("+ credit_card", ["credit_card_balance"]),
    ("+ cross-table", ["cross"]),
    ("+ learned", ["kmeans", "pca", "autoencoder"]),
]

# Iteration-4 features landed inside earlier-iteration blocks (affordability in application,
# payment-timing in installments, new ratios in cross). They are peeled back out below so each
# project round shows only the lift it actually added, not iter-4 work double-counted early.
ITER4_APP = ["disposable_income", "disposable_income_ratio", "disposable_per_person",
             "annuity_per_person", "annuity_to_goods", "goods_to_income", "credit_markup",
             "credit_markup_ratio", "income_to_age", "income_to_employ"]
ITER4_INS = ["ins_dpd30_share", "ins_dpd60_share", "ins_early_share",
             "ins_months_since_last_late", "ins_late_amt_share"]
ITER4_CROSS = ["x_max_late_share_any", "x_mean_late_share_any", "x_total_overdue",
               "x_total_overdue_to_income", "x_all_annuity_to_income",
               "x_disposable_after_all_annuity", "x_income_to_total_obligations",
               "x_active_debt_to_income", "x_closed_debt_to_income", "x_refused_to_approved",
               "x_approved_credit_share", "x_total_debt_per_person", "x_all_obligation_per_person"]
ITER4 = set(ITER4_APP + ITER4_INS + ITER4_CROSS)
LEARNED = ["kmeans", "pca", "autoencoder"]


def is_ext(c):
    cl = c.lower()
    return "ext_source" in cl or cl.startswith("ext_calc_")


def _file_cols(f, universe):
    p = INTERIM / f"{f}.parquet"
    if not p.exists():
        return []
    return [c for c in pq.ParquetFile(p).schema.names
            if c in universe and not c.startswith("SK_ID") and not is_ext(c)]


def _source_steps(universe):
    """Cumulative data sources: add one relational block at a time, all raw features."""
    seen, internal, steps = set(), [], []
    for label, files in SOURCES:
        for f in files:
            for c in _file_cols(f, universe):
                if c not in seen:
                    seen.add(c)
                    internal.append(c)
        steps.append((label, list(internal)))
    return steps


# Iteration-2 features are the subset-split aggregations (bureau active/closed, prev approved/
# refused, pos & credit-card active/done, installments late/on-time). They carry these prefixes,
# so the first per-table pass (iter 1) can be separated from the split pass (iter 2).
ITER2_PREFIXES = ("bureau_active_", "bureau_closed_", "prev_appr_", "prev_ref_",
                  "pos_active_", "pos_done_", "cc_active_", "cc_done_", "ins_ontime_")
ITER2_INS_SUFFIX = ("_underpay_mean", "_payratio_mean", "_amt_mean", "_days_late_mean")


def _is_iter2(c):
    return c.startswith(ITER2_PREFIXES) or (c.startswith("ins_late_") and c.endswith(ITER2_INS_SUFFIX))


def _iteration_steps(universe):
    """Cumulative project rounds, labelled from the feature notebooks, so each iteration shows only
    the lift it added -- the diminishing-returns view. iter1 = first per-table pass, iter2 = the
    subset-split aggregations, iter3 = cross-table, iter4 = affordability/timing, iter5 = learned.
    Iter-4 features are peeled out of the earlier blocks so they only count in iter 4."""
    base = [c for c in _file_cols("application", universe) if c not in ITER4]
    for f in ["bureau", "previous_application", "pos_cash_balance", "credit_card_balance"]:
        base += _file_cols(f, universe)
    base += [c for c in _file_cols("installments_payments", universe) if c not in ITER4]
    i1 = [c for c in base if not _is_iter2(c)]
    i2 = i1 + [c for c in base if _is_iter2(c)]
    cross = [c for c in _file_cols("cross", universe) if c not in ITER4]
    iter4 = [c for c in ITER4 if c in universe]
    learned = []
    for f in LEARNED:
        learned += _file_cols(f, universe)
    i3 = i2 + cross
    i4 = i3 + iter4
    i5 = i4 + learned
    return [("iter1", i1), ("iter2", i2), ("iter3", i3), ("iter4", i4), ("iter5", i5)]


def _score(X, y, make_gbm, steps, axis, cv, built_only=False):
    """Score built (internal) at each step; also built+bought unless ``built_only``. The iteration
    axis is built-only -- it is about the diminishing returns of our own feature engineering, not
    the external gap, so it only needs the built AUC."""
    ext_cols = [c for c in X.columns if is_ext(c)]
    rows = []
    for label, internal in steps:
        ai = roc_auc_score(y, oof(make_gbm(), X[internal], y, cv=cv))
        if built_only:
            rows.append({"axis": axis, "step": label, "n_internal": len(internal),
                         "n_combined": None, "auc_internal": round(ai, 4),
                         "auc_combined": None, "ext_gap": None})
            print(f"[{axis:9s}] {label:28s} built {ai:.4f}")
        else:
            ac = roc_auc_score(y, oof(make_gbm(), X[internal + ext_cols], y, cv=cv))
            rows.append({"axis": axis, "step": label, "n_internal": len(internal),
                         "n_combined": len(internal + ext_cols), "auc_internal": round(ai, 4),
                         "auc_combined": round(ac, 4), "ext_gap": round(ac - ai, 4)})
            print(f"[{axis:9s}] {label:28s} internal {ai:.4f}  combined {ac:.4f}  gap {ac - ai:+.4f}")
    return rows


def run_progression(X, y, make_gbm, axes=("source", "iteration"), n=150_000, cv=3, seed=0):
    """Both progressions in one pass. The gap between the built and built+bought lines is the
    value of the purchased scores at that step; it should shrink as in-house work accumulates.
    Fixed untuned model, all raw features per step (no selection), so every gap is data, not
    tuning. For speed the run uses a row subsample (``n``) and fewer folds (``cv``) -- the gaps
    are robust to both; only absolute AUC shifts a hair. Selection and tuning on the full data
    add a final capstone on top (top of nb 16, and nb 21)."""
    if n and n < len(X):
        idx = y.sample(n=n, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    uni = set(X.columns)
    rows = []
    if "source" in axes:
        rows += _score(X, y, make_gbm, _source_steps(uni), "source", cv)
    if "iteration" in axes:
        rows += _score(X, y, make_gbm, _iteration_steps(uni), "iteration", cv, built_only=True)
    return pd.DataFrame(rows)
