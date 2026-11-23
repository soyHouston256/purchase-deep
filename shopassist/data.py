"""Preparación de datos a partir de Amazon Shopping Queries (ESCI), versión reducida, español.

Salidas en ``cfg.data_dir``:
- ``catalog.parquet``: product_id, title, description, bullet_point, brand, color.
- ``queries.parquet``: query_id, query, split  (train = desarrollo, test = evaluación).
- ``qrels.parquet``: query_id, product_id, esci_label, gain  (SOLO lo lee ``evaluation``;
  el agente y el recuperador nunca acceden a este archivo).

Estrategia (determinista con ``cfg.SEED``):
1. Filtrar ``small_version == 1`` y ``product_locale == 'es'``.
2. Descartar consultas sin ningún producto relevante (todas 'I'), cuyo nDCG no está definido.
3. Muestrear ``n_test_queries`` del split test y ``n_train_queries`` del split train.
4. Catálogo = TODOS los productos anotados de esas consultas + relleno aleatorio de otros
   productos 'es' (``fill_extra``, y lo que haga falta para llegar a ``catalog_size``).
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .textutil import clean_html

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"


def _paths(cfg: Config) -> dict:
    d = cfg.data_dir
    return {
        "raw_dir": os.path.join(d, "raw"),
        "examples_raw": os.path.join(d, "raw", EXAMPLES_FILE),
        "products_raw": os.path.join(d, "raw", PRODUCTS_FILE),
        "catalog": os.path.join(d, "catalog.parquet"),
        "queries": os.path.join(d, "queries.parquet"),
        "qrels": os.path.join(d, "qrels.parquet"),
    }


def download_esci(cfg: Config) -> dict:
    """Descarga (con caché) los parquet de ESCI desde GitHub (LFS vía redirección)."""
    p = _paths(cfg)
    os.makedirs(p["raw_dir"], exist_ok=True)
    for key, fname in (("examples_raw", EXAMPLES_FILE), ("products_raw", PRODUCTS_FILE)):
        dst = p[key]
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            continue
        url = cfg.esci_base_url + fname
        print(f"Descargando {url} ...")
        tmp = dst + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
    return p


def _split_counts(cfg: Config) -> tuple[int, int]:
    if cfg.n_queries is not None:
        n_test = max(int(round(cfg.n_queries * 0.7)), cfg.min_queries)
        n_train = max(cfg.n_queries - n_test, 1)
        return n_test, n_train
    return max(cfg.n_test_queries, cfg.min_queries), cfg.n_train_queries


def prepare_data(cfg: Optional[Config] = None, force: bool = False) -> dict:
    """Construye catalog/queries/qrels (si no existen) y devuelve sus rutas."""
    cfg = cfg or Config()
    p = _paths(cfg)
    if not force and all(os.path.exists(p[k]) for k in ("catalog", "queries", "qrels")):
        return p
    download_esci(cfg)
    rng = np.random.default_rng(cfg.SEED)

    ex = pd.read_parquet(
        p["examples_raw"],
        columns=["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"],
        filters=[("product_locale", "=", cfg.locale), ("small_version", "=", 1)],
    )
    ex["gain"] = ex["esci_label"].map(GAINS).astype(float)
    has_rel = ex.groupby("query_id")["gain"].max()
    ex = ex[ex["query_id"].isin(has_rel[has_rel > 0].index)]

    n_test, n_train = _split_counts(cfg)
    chosen = []
    for split, n in (("test", n_test), ("train", n_train)):
        qids = np.sort(ex.loc[ex["split"] == split, "query_id"].unique())
        n = min(n, len(qids))
        chosen.extend(rng.choice(qids, size=n, replace=False).tolist())
    ex = ex[ex["query_id"].isin(set(chosen))].copy()

    queries = (ex[["query_id", "query", "split"]].drop_duplicates("query_id")
               .sort_values(["split", "query_id"]).reset_index(drop=True))
    qrels = (ex[["query_id", "product_id", "esci_label", "gain"]]
             .drop_duplicates(["query_id", "product_id"]).reset_index(drop=True))

    prods = pd.read_parquet(p["products_raw"], filters=[("product_locale", "=", cfg.locale)])
    prods = prods.drop_duplicates("product_id")
    annotated = set(qrels["product_id"])
    missing = annotated - set(prods["product_id"])
    if missing:  # no debería ocurrir; se eliminan de qrels para mantener la consistencia
        qrels = qrels[~qrels["product_id"].isin(missing)]
    others = np.sort(prods.loc[~prods["product_id"].isin(annotated), "product_id"].to_numpy())
    n_fill = max(cfg.catalog_size - len(annotated), 0) + cfg.fill_extra
    n_fill = min(n_fill, len(others))
    fill = set(rng.choice(others, size=n_fill, replace=False).tolist()) if n_fill else set()
    keep = annotated | fill
    cat = prods[prods["product_id"].isin(keep)].rename(columns={
        "product_title": "title", "product_description": "description",
        "product_bullet_point": "bullet_point", "product_brand": "brand", "product_color": "color"})
    cat = cat[["product_id", "title", "description", "bullet_point", "brand", "color"]].copy()
    for c in ("title", "description", "bullet_point", "brand", "color"):
        cat[c] = cat[c].map(clean_html)
    cat = cat.sort_values("product_id").reset_index(drop=True)

    os.makedirs(cfg.data_dir, exist_ok=True)
    cat.to_parquet(p["catalog"], index=False)
    queries.to_parquet(p["queries"], index=False)
    qrels.to_parquet(p["qrels"], index=False)
    print(f"Catálogo: {len(cat)} productos | consultas: "
          f"{queries['split'].value_counts().to_dict()} | qrels: {len(qrels)}")
    return p


def load_data(cfg: Optional[Config] = None):
    """Devuelve ``(catalog_df, queries_df, qrels_df)``; prepara los datos si faltan.

    Nota: ``qrels_df`` es solo para evaluación. No debe pasarse al agente ni al recuperador.
    """
    cfg = cfg or Config()
    p = prepare_data(cfg)
    return pd.read_parquet(p["catalog"]), pd.read_parquet(p["queries"]), pd.read_parquet(p["qrels"])


def check_data(catalog_df, queries_df, qrels_df, cfg: Optional[Config] = None) -> dict:
    """Verificaciones de los requisitos del enunciado."""
    cfg = cfg or Config()
    cat_ids = set(catalog_df["product_id"])
    return {
        "catalog_size": len(catalog_df),
        "catalog_ok": len(catalog_df) >= cfg.catalog_size,
        "n_test_queries": int((queries_df["split"] == "test").sum()),
        "n_train_queries": int((queries_df["split"] == "train").sum()),
        "eval_queries_ok": int((queries_df["split"] == "test").sum()) >= cfg.min_queries,
        "all_annotated_in_catalog": bool(set(qrels_df["product_id"]) <= cat_ids),
        "catalog_has_no_labels": not any(c in catalog_df.columns for c in ("esci_label", "gain")),
    }
