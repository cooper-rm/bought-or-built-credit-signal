from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

BLOCKS = ["bureau", "previous_application", "pos_cash_balance", "installments_payments", "credit_card_balance"]
DERIVED = ["ext", "kmeans", "pca", "autoencoder", "cross"]


def save_features(df, name):
    """Clean column names, downcast feature columns to float32, and write a parquet block.

    Feature blocks are stored as float32 parquet so ``load_master`` stays memory-light and
    ``load_selected`` can column-push-down to only the columns it needs. Key columns
    (``SK_ID_*``) and ``TARGET`` are left as-is; a keyed index is reset to a column first.
    """
    df = df.copy()
    if df.index.name is not None and df.index.name not in df.columns:
        df = df.reset_index()
    df.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)) for c in df.columns]
    feat = [c for c in df.columns if not c.startswith("SK_ID") and c != "TARGET"]
    df[feat] = df[feat].astype("float32")
    df.to_parquet(INTERIM / f"{name}.parquet", index=False)
    return df.shape


def load_master(all_rows=False):
    """Assemble the feature master. ``all_rows=True`` includes the test applicants (TARGET NaN)
    so the derived-feature notebooks can compute on train+test and stay leaderboard-ready."""
    labels = pd.read_csv(RAW / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
    if all_rows:
        test = pd.read_csv(RAW / "application_test.csv", usecols=["SK_ID_CURR"])
        test["TARGET"] = float("nan")
        labels = pd.concat([labels, test], ignore_index=True)
    master = labels.merge(pd.read_parquet(INTERIM / "application.parquet"), on="SK_ID_CURR", how="left")
    for name in BLOCKS:
        master = master.merge(pd.read_parquet(INTERIM / f"{name}.parquet"), on="SK_ID_CURR", how="left")
    for name in DERIVED:
        path = INTERIM / f"{name}.parquet"
        if path.exists():
            master = master.merge(pd.read_parquet(path), on="SK_ID_CURR", how="left")
    master = master.set_index("SK_ID_CURR")
    return master.drop(columns="TARGET"), master["TARGET"]


def load_selected(rebuild=False):
    """Selected features and the target, cached to ``model_matrix.parquet``.

    The cache auto-rebuilds when ``null_importance_features.csv`` is newer than it. The rebuild
    uses parquet **column pushdown** — it reads only the selected columns from each block, so it
    never materialises the full ~3.7k-column master (peak memory ~1 GB instead of ~5 GB).
    """
    import pyarrow.parquet as pq

    cache = INTERIM / "model_matrix.parquet"
    sel = INTERIM / "selected_features.csv"
    fresh = cache.exists() and (not sel.exists() or cache.stat().st_mtime >= sel.stat().st_mtime)
    if fresh and not rebuild:
        try:
            d = pd.read_parquet(cache)
            return d.drop(columns="TARGET"), d["TARGET"]
        except Exception:
            pass  # corrupt/partial cache (e.g. an interrupted write) -> fall through and rebuild

    kept = set(pd.read_csv(sel)["feature"])
    master = pd.read_csv(RAW / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"]).set_index("SK_ID_CURR")
    for name in ["application"] + BLOCKS + DERIVED:
        path = INTERIM / f"{name}.parquet"
        if not path.exists():
            continue
        cols = [c for c in pq.ParquetFile(path).schema.names if c in kept]
        if not cols:
            continue
        block = pd.read_parquet(path, columns=["SK_ID_CURR"] + cols).set_index("SK_ID_CURR")
        master = master.join(block, how="left")
    y = master.pop("TARGET")
    tmp = cache.with_name(cache.name + ".tmp")
    pd.concat([master, y.rename("TARGET")], axis=1).to_parquet(tmp)
    tmp.replace(cache)  # atomic swap: an interrupted write can never leave a corrupt cache
    return master, y
