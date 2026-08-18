"""Standard aggregation for the Home Credit child tables.

Every many-to-one table is collapsed to one row per ``key`` with the *same* recipe,
so the master file is modeled consistently and is uniform to explore.

Recipe:
- numeric columns  -> mean, max, min, sum, std
- categorical cols -> one-hot, then sum (count) and mean (share)
- a ``<prefix>_count`` of rows
- ``<prefix>_<col>_missing_rate`` for meaningfully-missing columns (missingness is signal)
- columns that are almost entirely missing are dropped before aggregating
- uniform naming: ``<prefix>_<column>_<agg>``
"""
import pandas as pd

NUM_AGGS = ["mean", "max", "min", "sum", "std"]


def aggregate(df, key, prefix, cat_cols=None, ignore=None, max_missing=0.95, missing_rate_min=0.10):
    """Collapse ``df`` to one row per ``key``, prefixing every feature with ``prefix``.

    Parameters
    ----------
    df : DataFrame           the child table
    key : str                grouping key (e.g. "SK_ID_CURR" or "SK_ID_BUREAU")
    prefix : str             short table tag (e.g. "bn", "bb", "prev")
    cat_cols : list[str]     categorical columns to one-hot then aggregate
    ignore : list[str]       numeric columns to exclude (e.g. other id columns)
    max_missing : float      drop numeric columns whose missing fraction exceeds this
    missing_rate_min : float emit a `_missing_rate` feature for columns at least this missing
    """
    ignore = set([key] + list(ignore or []))
    numeric = [c for c in df.select_dtypes("number").columns if c not in ignore]
    frac = df[numeric].isna().mean()
    num_cols = [c for c in numeric if frac[c] <= max_missing]
    g = df.groupby(key)

    num = g[num_cols].agg(NUM_AGGS)
    num.columns = [f"{prefix}_{c}_{a}" for c, a in num.columns]
    parts = [num, g.size().rename(f"{prefix}_count")]

    miss_cols = [c for c in num_cols if frac[c] >= missing_rate_min]
    if miss_cols:
        m = df[miss_cols].isna()
        m[key] = df[key].values
        rate = m.groupby(key).mean()
        rate.columns = [f"{prefix}_{c}_missing_rate" for c in rate.columns]
        parts.append(rate)

    if cat_cols:
        d = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype="int8")
        d[key] = df[key].values
        cat = d.groupby(key).agg(["sum", "mean"])
        cat.columns = [f"{prefix}_{c}_{a}" for c, a in cat.columns]
        parts.append(cat)

    return pd.concat(parts, axis=1).copy()
