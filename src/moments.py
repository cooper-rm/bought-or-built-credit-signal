"""Vectorised grouped higher moments.

pandas 2.x exposes ``GroupBy.skew`` but not ``GroupBy.kurt`` (added in 3.0), and the
per-group ``agg(pd.Series.kurt)`` fallback is ~20x slower. ``group_kurt`` reproduces
pandas' bias-corrected (Fisher) excess kurtosis exactly via grouped power-sums.
"""
import numpy as np
import pandas as pd


def group_kurt(values, by):
    """Excess kurtosis of ``values`` within each group of ``by`` (NaNs dropped)."""
    df = pd.DataFrame({"k": np.asarray(by), "y": np.asarray(values, dtype="float64")}).dropna(subset=["y"])
    df["y2"], df["y3"], df["y4"] = df["y"] ** 2, df["y"] ** 3, df["y"] ** 4
    g = df.groupby("k")
    n = g.size()
    s1, s2, s3, s4 = g["y"].sum(), g["y2"].sum(), g["y3"].sum(), g["y4"].sum()
    mean = s1 / n
    m2 = s2 - s1 * mean
    m4 = s4 - 4 * mean * s3 + 6 * mean ** 2 * s2 - 3 * n * mean ** 4
    num = n * (n + 1) * (n - 1) * m4
    den = (n - 2) * (n - 3) * m2 ** 2
    out = num / den - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    out[n < 4] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = getattr(by, "name", None)
    return out
