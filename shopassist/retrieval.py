"""Recuperación de productos: BM25 propio, denso (e5), fusión híbrida (RRF) y re-ranking.

Interfaz común de todos los recuperadores:
- ``search(query, k=10, filters=None) -> list[dict]`` con product_id, score, title, brand,
  color (+ price/stock si hay una ``Store`` asociada). Filtros (post-recuperación, unidos
  con la tienda): ``max_price``, ``in_stock`` (bool o cantidad mínima), ``brand``, ``color``,
  ``exclude`` (lista de ids).
- ``rank_candidates(query, candidate_ids) -> list[product_id]``: ordena SOLO los candidatos
  dados (protocolo de nDCG sobre candidatos anotados).

Los recuperadores nunca reciben etiquetas ESCI: solo el texto del catálogo.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from .textutil import clean_html, norm

# ------------------------------------------------------------------ tokenización
SPANISH_STOPWORDS = set("""
a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde donde
dos e el ella ellas ellos en entre era es esa esas ese eso esos esta estas este esto estos fue ha hay la las le
les lo los mas me mi mis muy ni no nos o os otra otro para pero poco por porque que quien se sea ser si sin sobre
son su sus tambien te tiene tu tus un una unas uno unos y ya yo the and for with of to in on by x
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def light_stem(tok: str) -> str:
    """Stemming ligero para español: plurales y algunas terminaciones frecuentes."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    for suf in ("aciones", "acion", "mente"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    if tok.endswith("ces") and len(tok) > 5:
        return tok[:-3] + "z"
    if tok.endswith("es") and len(tok) > 5 and tok[-3] not in "aeiou":
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 4:
        tok = tok[:-1]
    if tok[-1] in "ao" and len(tok) > 4:  # neutraliza género: negro/negra
        tok = tok[:-1]
    return tok


def tokenize(text: str, stem: bool = True, stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(norm(text))
    if stopwords:
        toks = [t for t in toks if t not in SPANISH_STOPWORDS]
    if stem:
        toks = [light_stem(t) for t in toks]
    return toks


# Sinónimos de español latinoamericano/peninsular para la búsqueda del agente (no se usan en
# la evaluación nDCG, para que la línea base BM25 sea un BM25 estándar).
QUERY_SYNONYMS = {
    "audifono": ["auricular", "casco"], "auricular": ["audifono", "casco"], "casco": ["auricular"],
    "mouse": ["raton"], "raton": ["mouse"], "celular": ["movil", "smartphone"], "movil": ["smartphone", "telefono"],
    "computadora": ["ordenador", "portatil"], "laptop": ["portatil"], "notebook": ["portatil"],
    "parlante": ["altavoz"], "bocina": ["altavoz"], "altavoz": ["parlante"], "cuaderno": ["libreta"],
    "libreta": ["cuaderno"], "lapicero": ["boligrafo"], "pluma": ["boligrafo"], "playera": ["camiseta"],
    "remera": ["camiseta"], "polera": ["camiseta", "sudadera"], "lente": ["gafa"], "anteojo": ["gafa"],
    "cargador": ["adaptador"], "tomacorriente": ["enchufe"], "heladera": ["frigorifico"], "refrigerador": ["frigorifico"],
}


def expand_query(query: str) -> str:
    """Añade sinónimos (formas raíz) a la consulta: "audífonos" -> "... auricular casco"."""
    toks = tokenize(query)
    extra = [syn for t in toks for syn in QUERY_SYNONYMS.get(t, []) if syn not in toks]
    return query if not extra else query + " " + " ".join(dict.fromkeys(extra))


def product_text(row: dict, fields: str = "full", max_chars: int = 1200) -> str:
    """Texto indexable de un producto: título + marca + color + viñetas + descripción."""
    title = clean_html(row.get("title", ""))
    if fields == "title":
        return title
    parts = [title]
    for key in ("brand", "color"):
        v = clean_html(row.get(key, ""))
        if v:
            parts.append(v)
    for key in ("bullet_point", "description"):
        v = clean_html(row.get(key, ""))
        if v:
            parts.append(v)
    return " . ".join(parts)[:max_chars]


# ------------------------------------------------------------------ base
class BaseRetriever:
    name = "base"

    def __init__(self, catalog_df, fields: str = "full", max_chars: int = 1200):
        self.ids: list[str] = [str(x) for x in catalog_df["product_id"].tolist()]
        self.id2idx = {pid: i for i, pid in enumerate(self.ids)}
        recs = catalog_df.to_dict("records")
        self.meta = {str(r["product_id"]): {"title": clean_html(r.get("title", "")),
                                           "brand": clean_html(r.get("brand", "")),
                                           "color": clean_html(r.get("color", ""))} for r in recs}
        self.fields = fields
        self.texts = [product_text(r, fields, max_chars) for r in recs]
        self.store = None
        self.last_timing = 0.0

    def attach_store(self, store):
        self.store = store
        return self

    # --- a implementar
    def score_all(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def score_subset(self, query: str, idx: np.ndarray) -> np.ndarray:
        return self.score_all(query)[idx]

    # --- comunes
    def ranked_indices(self, query: str, depth: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        s = self.score_all(query)
        order = np.argsort(-s, kind="stable")
        if depth is not None:
            order = order[:depth]
        return order, s[order]

    def rank_candidates(self, query: str, candidate_ids: list) -> list:
        cands = [str(c) for c in candidate_ids if str(c) in self.id2idx]
        if not cands:
            return []
        idx = np.array([self.id2idx[c] for c in cands])
        s = self.score_subset(query, idx)
        order = np.argsort(-s, kind="stable")
        return [cands[i] for i in order]

    def _passes(self, pid: str, filters: dict) -> bool:
        if not filters:
            return True
        if pid in set(filters.get("exclude") or []):
            return False
        m = self.meta[pid]
        if filters.get("brand"):
            if norm(filters["brand"]) not in norm(m["brand"]) and norm(filters["brand"]) not in norm(m["title"]):
                return False
        if filters.get("color"):
            if not color_matches(filters["color"], m["color"] + " " + m["title"]):
                return False
        need_store = filters.get("max_price") is not None or filters.get("in_stock")
        if need_store and self.store is not None:
            p = self.store.get_product(pid)
            if p is None:
                return False
            if filters.get("max_price") is not None and p["price"] > float(filters["max_price"]) + 1e-9:
                return False
            ins = filters.get("in_stock")
            if ins:
                min_stock = int(ins) if not isinstance(ins, bool) else 1
                if p["stock"] < min_stock:
                    return False
        return True

    def search(self, query: str, k: int = 10, filters: Optional[dict] = None) -> list[dict]:
        t0 = time.perf_counter()
        order, scores = self.ranked_indices(query)
        out = []
        for i, sc in zip(order, scores):
            pid = self.ids[i]
            if not self._passes(pid, filters or {}):
                continue
            item = {"product_id": pid, "score": float(sc), **self.meta[pid]}
            if self.store is not None:
                p = self.store.get_product(pid)
                if p:
                    item.update(price=p["price"], stock=p["stock"])
            out.append(item)
            if len(out) >= k:
                break
        self.last_timing = time.perf_counter() - t0
        return out


_COLOR_SYNONYMS = {
    "negro": ["negr", "black", "noir"], "blanco": ["blanc", "white"], "rojo": ["roj", "red", "rouge"],
    "azul": ["azul", "blue"], "verde": ["verde", "green"], "amarillo": ["amarill", "yellow"],
    "gris": ["gris", "grey", "gray"], "rosa": ["rosa", "pink", "rosad"], "morado": ["morad", "purple", "lila", "violeta"],
    "naranja": ["naranj", "orange"], "plateado": ["platead", "plata", "silver"], "dorado": ["dorad", "oro", "gold"],
    "marron": ["marron", "brown", "cafe"], "beige": ["beige"], "transparente": ["transparent", "clear"],
    "multicolor": ["multicolor"],
}


def canonical_color(c: str) -> Optional[str]:
    c = norm(c)
    for canon, syns in _COLOR_SYNONYMS.items():
        if any(c.startswith(s) or c == s for s in syns):
            return canon
    return None


def color_matches(color: str, text: str) -> bool:
    canon = canonical_color(color) or norm(color)
    syns = _COLOR_SYNONYMS.get(canon, [norm(color)])
    words = re.findall(r"[a-z]+", norm(text))
    return any(w.startswith(s) for w in words for s in syns)


# ------------------------------------------------------------------ BM25
class BM25Retriever(BaseRetriever):
    """BM25 (Okapi) implementado desde cero con índice invertido y tokenizador español."""

    name = "bm25"

    def __init__(self, catalog_df, k1: float = 1.2, b: float = 0.75, fields: str = "full",
                 stem: bool = True, max_chars: int = 1200):
        super().__init__(catalog_df, fields, max_chars)
        self.k1, self.b, self.stem = k1, b, stem
        docs = [tokenize(t, stem=stem) for t in self.texts]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        postings: dict[str, list] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                postings[term].append((i, tf))
        n = len(docs)
        self.postings = {}
        self.idf = {}
        for term, plist in postings.items():
            arr = np.array(plist, dtype=np.int64)
            self.postings[term] = (arr[:, 0], arr[:, 1].astype(np.float32))
            df = len(plist)
            self.idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        self.norm_len = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))

    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term in set(tokenize(query, stem=self.stem)):
            if term not in self.postings:
                continue
            idx, tf = self.postings[term]
            scores[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm_len[idx])
        return scores


# ------------------------------------------------------------------ denso
class DenseRetriever(BaseRetriever):
    """Bi-encoder e5 (prefijos "query: "/"passage: "), producto interno sobre vectores
    normalizados (equivale a coseno). Las incrustaciones se cachean en .npy."""

    name = "dense"

    def __init__(self, catalog_df, model_name: str = "intfloat/multilingual-e5-small",
                 cache_dir: Optional[str] = "data", batch_size: int = 64, fields: str = "full",
                 max_chars: int = 1200, device: Optional[str] = None, model=None):
        super().__init__(catalog_df, fields, max_chars)
        self.model_name = model_name
        self._model = model
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.emb = self._load_or_encode()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # import perezoso
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _cache_path(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        h = hashlib.md5(("\n".join(self.texts) + self.model_name).encode()).hexdigest()[:12]
        safe = self.model_name.replace("/", "_")
        return os.path.join(self.cache_dir, f"emb_{safe}_{self.fields}_{h}.npy")

    def _load_or_encode(self) -> np.ndarray:
        path = self._cache_path()
        if path and os.path.exists(path):
            return np.load(path)
        emb = self.model.encode([f"passage: {t}" for t in self.texts], batch_size=self.batch_size,
                                normalize_embeddings=True, show_progress_bar=True,
                                convert_to_numpy=True).astype(np.float32)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.save(path, emb)
        return emb

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([f"query: {query}"], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)

    def score_all(self, query: str) -> np.ndarray:
        return self.emb @ self.encode_query(query)

    def score_subset(self, query: str, idx: np.ndarray) -> np.ndarray:
        return self.emb[idx] @ self.encode_query(query)


# ------------------------------------------------------------------ re-ranker
class CrossEncoderReranker:
    """Cross-encoder multilingüe (mMARCO) que puntúa pares (consulta, producto)."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                 max_length: int = 256, batch_size: int = 32, device: Optional[str] = None, model=None):
        self.model_name, self.max_length, self.batch_size, self.device = model_name, max_length, batch_size, device
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # import perezoso
            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                             show_progress_bar=False), dtype=np.float32)


# ------------------------------------------------------------------ híbrido
def rrf_fuse(rank_lists: list[list[int]], k: int = 60, weights: Optional[list[float]] = None) -> dict:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(rank_lists)
    fused: dict = defaultdict(float)
    for w, lst in zip(weights, rank_lists):
        for r, d in enumerate(lst):
            fused[d] += w / (k + r + 1)
    return fused


def _minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class HybridRetriever(BaseRetriever):
    """Fusión de BM25 + denso (``fusion='rrf'`` o ``'linear'`` con peso ``alpha`` para el
    denso) y re-ranking opcional del top-``rerank_top`` con un cross-encoder."""

    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: Optional[DenseRetriever] = None,
                 reranker: Optional[CrossEncoderReranker] = None, k_rrf: int = 60,
                 fusion: str = "rrf", alpha: float = 0.5, depth: int = 100, rerank_top: int = 50,
                 rerank_texts: Optional[list[str]] = None):
        self.bm25, self.dense, self.reranker = bm25, dense, reranker
        self.ids, self.id2idx, self.meta = bm25.ids, bm25.id2idx, bm25.meta
        self.texts = rerank_texts or bm25.texts
        self.k_rrf, self.fusion, self.alpha, self.depth, self.rerank_top = k_rrf, fusion, alpha, depth, rerank_top
        self.store = None
        self.last_timing = 0.0
        self.name = "hybrid" + ("+rerank" if reranker else "") + ("" if dense else "(sin denso)")

    def _fuse_scores(self, query: str, idx: Optional[np.ndarray]) -> np.ndarray:
        """Puntuación fusionada sobre ``idx`` (o todo el catálogo si None)."""
        comps = [self.bm25] + ([self.dense] if self.dense is not None else [])
        n = len(self.ids) if idx is None else len(idx)
        raw = [c.score_all(query) if idx is None else c.score_subset(query, idx) for c in comps]
        if len(raw) == 1:
            return raw[0]
        if self.fusion == "linear":
            return (1 - self.alpha) * _minmax(raw[0]) + self.alpha * _minmax(raw[1])
        fused = np.zeros(n, dtype=np.float64)
        for s in raw:
            order = np.argsort(-s, kind="stable")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1)
            fused += 1.0 / (self.k_rrf + ranks)
        return fused

    def _rerank(self, query: str, idx: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Re-ordena los ``rerank_top`` primeros con el cross-encoder; el resto queda detrás."""
        order = np.argsort(-base, kind="stable")
        top = order[: self.rerank_top]
        ce = self.reranker.score(query, [self.texts[idx[i]] for i in top])
        new = np.full(len(idx), -1e9, dtype=np.float64)
        new[top] = ce
        rest = order[self.rerank_top:]
        new[rest] = -1e6 - np.arange(len(rest))  # conserva el orden de la fusión
        return new

    def score_all(self, query: str) -> np.ndarray:
        return self._fuse_scores(query, None)

    def ranked_indices(self, query: str, depth: Optional[int] = None):
        s = self.score_all(query)
        order = np.argsort(-s, kind="stable")
        if self.reranker is not None:
            top = order[: max(self.rerank_top, depth or 0)]
            rs = self._rerank(query, top, s[top])
            o2 = np.argsort(-rs, kind="stable")
            head = top[o2]
            order = np.concatenate([head, order[len(top):]])
            scores = np.concatenate([rs[o2], s[order[len(top):]] - 1e6])
            return (order[:depth], scores[:depth]) if depth else (order, scores)
        return (order[:depth], s[order][:depth]) if depth else (order, s[order])

    def score_subset(self, query: str, idx: np.ndarray) -> np.ndarray:
        base = self._fuse_scores(query, idx)
        if self.reranker is not None:
            return self._rerank(query, idx, base)
        return base


class RerankOnly(BaseRetriever):
    """Para ablación: re-ranker aplicado sobre el orden de un recuperador base."""

    def __init__(self, base: BaseRetriever, reranker: CrossEncoderReranker, rerank_top: int = 50):
        self.base, self.reranker, self.rerank_top = base, reranker, rerank_top
        self.ids, self.id2idx, self.meta, self.texts = base.ids, base.id2idx, base.meta, base.texts
        self.store, self.last_timing = None, 0.0
        self.name = base.name + "+rerank"

    def score_all(self, query):
        s = self.base.score_all(query).astype(np.float64)
        order = np.argsort(-s, kind="stable")[: self.rerank_top]
        s = s - 1e6
        s[order] = self.reranker.score(query, [self.texts[i] for i in order])
        return s

    def score_subset(self, query, idx):
        s = self.base.score_subset(query, idx).astype(np.float64)
        order = np.argsort(-s, kind="stable")[: self.rerank_top]
        s = s - 1e6
        s[order] = self.reranker.score(query, [self.texts[idx[i]] for i in order])
        return s


class RandomRetriever(BaseRetriever):
    """Cota inferior: orden aleatorio determinista por consulta."""

    name = "random"

    def __init__(self, catalog_df, seed: int = 42):
        super().__init__(catalog_df, "title")
        self.seed = seed

    def score_all(self, query):
        h = int(hashlib.md5(f"{self.seed}{query}".encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h).random(len(self.ids))


# ------------------------------------------------------------------ fábrica
_CACHE: dict = {}


def build_retriever(cfg, catalog_df, kind: str = "hybrid", store=None, **kw):
    """Construye un recuperador: ``bm25`` | ``bm25_title`` | ``dense`` | ``dense_title`` |
    ``hybrid`` | ``hybrid_rerank`` | ``bm25_rerank`` | ``random``.

    Los componentes (BM25, denso, cross-encoder) se reutilizan entre llamadas para no
    recargar modelos en Colab. ``embed_model`` / ``rerank_model`` permiten inyectar modelos
    ya cargados (o falsos, en pruebas). Otros kwargs: ``k_rrf``, ``fusion`` ('rrf'|'linear'),
    ``alpha``, ``k1``, ``b``.
    """
    key_cat = (id(catalog_df), len(catalog_df))
    max_chars = getattr(cfg, "text_max_chars", 1200)

    def bm25(fields="full"):
        k = ("bm25", fields, key_cat, kw.get("k1", 1.2), kw.get("b", 0.75))
        if k not in _CACHE:
            _CACHE[k] = BM25Retriever(catalog_df, k1=kw.get("k1", 1.2), b=kw.get("b", 0.75),
                                      fields=fields, max_chars=max_chars)
        return _CACHE[k]

    def dense(fields="full"):
        k = ("dense", fields, key_cat, cfg.embed_name, id(kw.get("embed_model")))
        if k not in _CACHE:
            shared = kw.get("embed_model") or next(
                (v._model for kk, v in _CACHE.items() if kk[0] == "dense" and kk[3] == cfg.embed_name
                 and v._model is not None), None)
            _CACHE[k] = DenseRetriever(catalog_df, cfg.embed_name, cache_dir=cfg.data_dir,
                                       batch_size=cfg.embed_batch_size, fields=fields,
                                       max_chars=max_chars, model=shared)
        return _CACHE[k]

    def reranker():
        k = ("rerank", cfg.reranker_name, id(kw.get("rerank_model")))
        if k not in _CACHE:
            _CACHE[k] = CrossEncoderReranker(cfg.reranker_name, model=kw.get("rerank_model"))
        return _CACHE[k]

    k_rrf = kw.get("k_rrf", cfg.k_rrf)
    fusion = kw.get("fusion", "rrf")
    alpha = kw.get("alpha", 0.5)
    if kind == "bm25":
        r = bm25()
    elif kind == "bm25_title":
        r = bm25("title")
    elif kind == "dense":
        r = dense()
    elif kind == "dense_title":
        r = dense("title")
    elif kind == "hybrid":
        r = HybridRetriever(bm25(), dense(), None, k_rrf=k_rrf, fusion=fusion, alpha=alpha)
    elif kind == "hybrid_rerank":
        r = HybridRetriever(bm25(), dense(), reranker(), k_rrf=k_rrf, fusion=fusion, alpha=alpha,
                            rerank_top=cfg.rerank_top)
    elif kind == "bm25_rerank":
        r = RerankOnly(bm25(), reranker(), rerank_top=cfg.rerank_top)
    elif kind == "random":
        r = RandomRetriever(catalog_df, cfg.SEED)
    else:
        raise ValueError(f"Tipo de recuperador desconocido: {kind}")
    if store is not None:
        r.attach_store(store)
    return r
