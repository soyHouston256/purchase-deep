"""Evaluación de la recuperación: nDCG@10 sobre los candidatos anotados de cada consulta.

Ganancias: Exact=1, Substitute=0.1, Complement=0.01, Irrelevant=0 (columna ``gain`` de qrels).
DCG = sum_i gain_i / log2(i + 1) (i desde 1); IDCG = DCG del orden ideal de TODOS los
candidatos anotados. Este es el ÚNICO módulo que lee qrels (etiquetas ESCI).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import pandas as pd


def ndcg_at_k(ranked_ids: list, gains: dict, k: int = 10) -> float:
    """nDCG@k. ``gains``: {product_id: ganancia}; ids no juzgados cuentan como 0.

    Si ninguna ganancia es positiva (IDCG=0) devuelve ``nan`` (consulta no evaluable).
    """
    dcg = sum(float(gains.get(pid, 0.0)) / math.log2(i + 2) for i, pid in enumerate(ranked_ids[:k]))
    ideal = sorted((float(g) for g in gains.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return float("nan") if idcg <= 0 else dcg / idcg


def evaluate_retriever(retriever, queries_df: pd.DataFrame, qrels_df: pd.DataFrame,
                       split: Optional[str] = "test", k: int = 10, name: Optional[str] = None) -> dict:
    """Ordena los candidatos anotados de cada consulta con ``retriever.rank_candidates``.

    El recuperador solo recibe la consulta y la lista de ids (sin etiquetas).
    Devuelve ``{"name", "split", "mean", "n", "time_s", "per_query": DataFrame}``.
    """
    qs = queries_df if split is None else queries_df[queries_df["split"] == split]
    groups = {qid: dict(zip(g["product_id"], g["gain"])) for qid, g in qrels_df.groupby("query_id")}
    rows = []
    t0 = time.perf_counter()
    for qid, query in zip(qs["query_id"], qs["query"]):
        gains = groups.get(qid)
        if not gains:
            continue
        cands = sorted(gains.keys())  # orden neutral: no filtra información de relevancia
        ranked = retriever.rank_candidates(query, cands)
        score = ndcg_at_k(ranked, gains, k)
        n_exact = sum(1 for g in gains.values() if g >= 1.0)
        rows.append({"query_id": qid, "query": query, "ndcg": score, "n_candidates": len(cands),
                     "n_exact": n_exact, "top3": ranked[:3],
                     "top3_gains": [gains.get(p, 0.0) for p in ranked[:3]]})
    per_q = pd.DataFrame(rows)
    valid = per_q["ndcg"].dropna() if len(per_q) else pd.Series(dtype=float)
    return {"name": name or getattr(retriever, "name", type(retriever).__name__), "split": split,
            "mean": float(valid.mean()) if len(valid) else float("nan"), "n": int(len(valid)),
            "time_s": time.perf_counter() - t0, "per_query": per_q}


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianza bootstrap de la media."""
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(per_a: pd.DataFrame, per_b: pd.DataFrame) -> dict:
    """Diferencia pareada (B - A) por consulta: media, % de consultas que mejoran/empeoran."""
    m = per_a[["query_id", "ndcg"]].merge(per_b[["query_id", "ndcg"]], on="query_id", suffixes=("_a", "_b")).dropna()
    d = m["ndcg_b"] - m["ndcg_a"]
    return {"mean_diff": float(d.mean()), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
            "ties": int((d.abs() <= 1e-9).sum()), "ci95": bootstrap_ci(d.tolist())}


def tune_hybrid(cfg, catalog_df, queries_df, qrels_df, grid: Optional[list] = None, **build_kw) -> dict:
    """Ajusta la fusión SOLO con el split train (desarrollo). Devuelve el mejor conjunto."""
    from .retrieval import build_retriever
    grid = grid or ([{"fusion": "rrf", "k_rrf": k} for k in (10, 30, 60, 100)]
                    + [{"fusion": "linear", "alpha": a} for a in (0.3, 0.5, 0.7)])
    results = []
    for params in grid:
        r = build_retriever(cfg, catalog_df, "hybrid", **params, **build_kw)
        res = evaluate_retriever(r, queries_df, qrels_df, split="train")
        results.append({**params, "ndcg_train": res["mean"]})
    df = pd.DataFrame(results).sort_values("ndcg_train", ascending=False).reset_index(drop=True)
    best = {k: v for k, v in df.iloc[0].to_dict().items() if k != "ndcg_train" and not (isinstance(v, float) and math.isnan(v))}
    if "k_rrf" in best:
        best["k_rrf"] = int(best["k_rrf"])
    return {"best": best, "table": df}


EXPERIMENTS = [
    # (nombre, kind, descripción)
    ("Aleatorio", "random", "Cota inferior: orden aleatorio"),
    ("BM25 (baseline)", "bm25", "Línea base léxica: BM25 propio sobre título+marca+color+viñetas+descripción"),
    ("BM25 solo título", "bm25_title", "Ablación del texto indexado (solo título)"),
    ("Denso e5", "dense", "multilingual-e5-small, coseno"),
    ("Híbrido RRF", "hybrid", "BM25 + denso fusionados (parámetros ajustados en train)"),
    ("Híbrido + re-ranker", "hybrid_rerank", "Sistema completo: híbrido + cross-encoder mMARCO top-50"),
    ("BM25 + re-ranker (sin denso)", "bm25_rerank", "Ablación: sin componente denso"),
]


def run_retrieval_experiments(cfg, catalog_df=None, queries_df=None, qrels_df=None,
                              experiments: Optional[list] = None, tune: bool = True,
                              splits=("train", "test"), **build_kw) -> dict:
    """Ejecuta baseline, sistema y ablaciones. Devuelve ``{"table", "results", "tuning"}``.

    La ablación principal es *Híbrido + re-ranker* vs *Híbrido RRF* (quitar re-ranker) y
    vs *BM25 + re-ranker* (quitar el denso); además *BM25 solo título* (quitar texto).
    """
    from .retrieval import build_retriever
    if catalog_df is None or queries_df is None or qrels_df is None:
        from .data import load_data
        catalog_df, queries_df, qrels_df = load_data(cfg)
    experiments = experiments or EXPERIMENTS
    tuning = None
    params: dict = {}
    if tune and any(k.startswith("hybrid") for _, k, _ in experiments):
        tuning = tune_hybrid(cfg, catalog_df, queries_df, qrels_df, **build_kw)
        params = tuning["best"]
    rows, results = [], {}
    for name, kind, desc in experiments:
        r = build_retriever(cfg, catalog_df, kind, **(params if kind.startswith("hybrid") else {}), **build_kw)
        res_by_split = {s: evaluate_retriever(r, queries_df, qrels_df, split=s, name=name) for s in splits}
        results[name] = res_by_split
        row = {"sistema": name, "kind": kind, "descripcion": desc}
        for s, res in res_by_split.items():
            row[f"ndcg@10_{s}"] = round(res["mean"], 4)
            row[f"n_{s}"] = res["n"]
        if "test" in res_by_split:
            lo, hi = bootstrap_ci(res_by_split["test"]["per_query"]["ndcg"].tolist())
            row["ic95_test"] = f"[{lo:.3f}, {hi:.3f}]"
            row["s_por_consulta"] = round(res_by_split["test"]["time_s"] / max(res_by_split["test"]["n"], 1), 4)
        rows.append(row)
    return {"table": pd.DataFrame(rows), "results": results, "tuning": tuning, "params": params}


def success_analysis(per_query: pd.DataFrame, n: int = 5) -> dict:
    """Las ``n`` mejores y ``n`` peores consultas (por nDCG) para el análisis cualitativo."""
    df = per_query.dropna(subset=["ndcg"]).sort_values(["ndcg", "query_id"], ascending=[False, True])
    cols = [c for c in ("query_id", "query", "ndcg", "n_candidates", "n_exact", "top3_gains") if c in df.columns]
    return {"best": df.head(n)[cols].reset_index(drop=True),
            "worst": df.tail(n).iloc[::-1][cols].reset_index(drop=True)}
