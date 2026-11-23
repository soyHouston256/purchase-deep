import math

import pandas as pd
import pytest

from shopassist.evaluation import evaluate_retriever, ndcg_at_k, success_analysis
from shopassist.retrieval import BM25Retriever, HybridRetriever, light_stem, rrf_fuse, tokenize
from shopassist.store import Store
from tests.fixtures import make_toy_catalog


def test_ndcg_hand_example():
    gains = {"a": 0.0, "b": 1.0, "c": 0.1}
    dcg = 1 / math.log2(3) + 0.1 / math.log2(4)
    idcg = 1 + 0.1 / math.log2(3)
    assert ndcg_at_k(["a", "b", "c"], gains) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["b", "c", "a"], gains) == pytest.approx(1.0)
    assert math.isnan(ndcg_at_k(["a"], {"a": 0.0}))
    # k corta la lista
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 1}, k=1) == 0.0


def test_tokenizer_spanish():
    assert tokenize("Audífonos NEGROS para el móvil") == tokenize("audifono negro movil")
    assert light_stem("cargadores") == light_stem("cargador")


def test_rrf():
    f = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    assert f["a"] == pytest.approx(f["b"])


@pytest.fixture
def toy():
    cat = make_toy_catalog()
    store = Store(":memory:", cat)
    return cat, store, BM25Retriever(cat).attach_store(store)


def test_bm25_search_and_filters(toy):
    cat, store, r = toy
    res = r.search("audífonos bluetooth", k=5)
    assert res and all(x["product_id"].startswith(("AUD", "ALT")) for x in res[:3])
    res = r.search("cargador usb-c", k=10, filters={"max_price": 30, "in_stock": True})
    assert all(x["price"] <= 30 and x["stock"] > 0 for x in res)
    res = r.search("funda iphone", k=10, filters={"color": "rojo"})
    assert res and all("rojo" in x["color"] for x in res)
    res = r.search("mouse", k=10, filters={"brand": "logitech"})
    assert res and all(x["brand"] == "Logitech" for x in res)


def test_rank_candidates_and_evaluate(toy):
    cat, store, r = toy
    cands = ["PIL001", "AUD001", "CAB001", "AUD003"]
    ranked = r.rank_candidates("audífonos inalámbricos", cands)
    assert set(ranked) == set(cands) and ranked[0].startswith("AUD")
    queries = pd.DataFrame({"query_id": [1, 2], "query": ["audífonos bluetooth", "pilas alcalinas"],
                            "split": ["test", "test"]})
    qrels = pd.DataFrame({"query_id": [1, 1, 1, 2, 2], "product_id": ["AUD001", "CAB001", "PIL001", "PIL001", "MOU001"],
                          "esci_label": ["E", "C", "I", "E", "I"], "gain": [1.0, 0.01, 0.0, 1.0, 0.0]})
    out = evaluate_retriever(r, queries, qrels, split="test")
    assert out["n"] == 2 and out["mean"] == pytest.approx(1.0)
    sa = success_analysis(out["per_query"], n=1)
    assert len(sa["best"]) == 1 and len(sa["worst"]) == 1


def test_hybrid_without_dense_equals_bm25_order(toy):
    cat, store, r = toy
    h = HybridRetriever(r, None)
    assert [x["product_id"] for x in h.search("teclado mecánico", 5)] == [x["product_id"] for x in r.search("teclado mecánico", 5)]


def test_retriever_never_sees_labels(toy):
    cat, store, r = toy
    assert not any(c in cat.columns for c in ("esci_label", "gain"))


# ---------------------------------------------------------------- modelos falsos (sin torch)
class FakeEncoder:
    """Bolsa de palabras con hashing: imita SentenceTransformer.encode."""

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t.split(": ", 1)[-1]):
                out[i, hash(tok) % 256] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class FakeCross:
    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [len(set(tokenize(q)) & set(tokenize(t))) for q, t in pairs]


def test_dense_hybrid_rerank_with_fake_models(tmp_path):
    from shopassist.config import Config
    from shopassist.evaluation import run_retrieval_experiments
    from shopassist.retrieval import build_retriever

    cat = make_toy_catalog()
    cfg = Config(data_dir=str(tmp_path))
    kw = {"embed_model": FakeEncoder(), "rerank_model": FakeCross()}
    for kind in ("dense", "hybrid", "hybrid_rerank", "bm25_rerank", "dense_title"):
        r = build_retriever(cfg, cat, kind, **kw)
        res = r.search("audífonos bluetooth", k=5)
        assert len(res) == 5 and res[0]["product_id"].startswith(("AUD", "ALT"))
        ranked = r.rank_candidates("pilas alcalinas", ["AUD001", "PIL001", "CAB001"])
        assert ranked[0] == "PIL001"
    assert list(tmp_path.glob("emb_*.npy"))  # caché de incrustaciones
    queries = pd.DataFrame({"query_id": [1, 2, 3], "query": ["audífonos bluetooth", "pilas alcalinas", "mochila portátil"],
                            "split": ["train", "test", "test"]})
    qrels = pd.DataFrame({"query_id": [1, 1, 2, 2, 3, 3], "product_id": ["AUD001", "PIL001", "PIL001", "MOU001", "MOC001", "CUA001"],
                          "esci_label": ["E", "I", "E", "I", "E", "I"], "gain": [1.0, 0, 1.0, 0, 1.0, 0]})
    out = run_retrieval_experiments(cfg, cat, queries, qrels, **kw)
    t = out["table"]
    assert {"BM25 (baseline)", "Híbrido + re-ranker", "BM25 + re-ranker (sin denso)"} <= set(t["sistema"])
    assert out["tuning"] is not None and "fusion" in out["params"]
