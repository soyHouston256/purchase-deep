"""Pruebas ligeras de voz/evaluación/escenarios (sin modelos pesados).

Los escenarios de texto se ejecutan sobre el catálogo de juguete con ``llm=None``
(política basada en reglas). Se exige que corran sin excepciones, que no haya
violaciones de restricciones y que los escenarios de control (aclaración,
confirmación sin carrito, idempotencia) pasen.
"""
import json

import pytest

from shopassist.latency import StageTimer, latency_summary
from shopassist.voice import text_overlaps_reference
from shopassist.voice_eval import asr_error_analysis, extract_numbers, wer_cer, words_to_digits


# ------------------------------------------------------------------ voz (puro)
def test_words_to_digits_and_numbers():
    assert words_to_digits("quiero veinticinco botellas") == "quiero 25 botellas"
    assert words_to_digits("ciento cincuenta euros") == "150 euros"
    assert words_to_digits("diecinueve con noventa") == "19.90"
    n = extract_numbers("Quiero tres pilas con un presupuesto de cuarenta euros")
    assert n == {"amounts": [40.0], "quantities": [3.0]}


def test_asr_error_analysis_detects_amount_error():
    r = asr_error_analysis("presupuesto de cuarenta euros, dos mochilas",
                           "presupuesto de catorce euros, dos mochilas")
    assert r["amount_errors"] == 1 and r["qty_errors"] == 0 and r["product_errors"] == 0
    assert wer_cer("dos fundas", "2 fundas")["wer"] == 0.0


def test_reference_guard():
    ref = "Esta mañana salí temprano a caminar por el parque y el cielo estaba despejado"
    assert text_overlaps_reference(ref, ref)
    assert text_overlaps_reference("Hola. Esta mañana salí temprano a caminar por el parque.", ref)
    assert not text_overlaps_reference("Tu carrito tiene dos unidades.", ref)


def test_stage_timer_and_summary():
    t = StageTimer()
    with t("asr_s"):
        pass
    t.add("tts_s", 0.5)
    recs = [{"timings": {**t.timings, "total_s": 1.0}, "warm": False},
            {"timings": {"asr_s": 0.1, "tts_s": 0.2, "total_s": 0.4}, "warm": True}]
    s = latency_summary(recs, load_timings=[{"model": "asr", "load_s": 3.0}])
    assert {"primer_turno", "caliente", "todos", "carga_asr"} <= set(s["fase"])


# ------------------------------------------------------------------ escenarios
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    core = pytest.importorskip("shopassist.agent")
    pytest.importorskip("tests.fixtures")
    from shopassist.config import Config
    from shopassist.retrieval import build_retriever
    from shopassist.store import Store
    from tests.fixtures import make_toy_catalog

    cfg = Config()
    cat = make_toy_catalog()
    store = Store(":memory:", cat, seed=cfg.SEED)
    retriever = build_retriever(cfg, cat, kind="bm25", store=store)

    def factory():
        return core.ShoppingAgent(store, retriever, llm=None, cart_id="c1", cfg=cfg)

    return cfg, store, factory, tmp_path_factory.mktemp("results")


def test_load_scenarios_schema():
    from shopassist.scenarios import load_scenarios

    scs = load_scenarios()
    assert len(scs) >= 20
    voice = [s for s in scs if s["voice"]]
    assert len(voice) == 6
    assert sum(len(s["turns"]) > 1 for s in voice) >= 3
    for s in voice:
        assert all(t.get("audio", "").startswith("voice/scenarios/v") for t in s["turns"])


def test_text_scenarios_run(env):
    from shopassist.scenarios import load_scenarios, run_all, scenario_metrics

    cfg, store, factory, out = env
    scs = load_scenarios()
    df, results = run_all(scs, factory, store, input_mode="text", out_dir=str(out), seed=cfg.SEED)
    assert len(df) == len(scs)
    assert (df["error"] == "").all()
    m = scenario_metrics(df)
    print(df[["id", "ok", "fallidos"]].to_string())
    print(m)
    assert m["violaciones_restricciones"] == 0, df[[c for c in df.columns if c.startswith("v_")]].sum()
    for r in results:  # persistencia por escenario
        d = out / "text" / r["id"]
        assert (d / "result.json").exists() and (d / "final_store.json").exists()
        json.loads((d / "ops.json").read_text(encoding="utf-8"))
    assert m["pct_correctos"] >= 85, df[~df["ok"]][["id", "fallidos"]].to_string()
    ok = dict(zip(df["id"], df["ok"]))
    for sid in ("t07_referencia_ambigua", "t08_solicitud_vaga", "t11_confirmar_sin_cotizacion",
                "t15_confirmacion_repetida"):
        assert ok[sid], df.set_index("id").loc[sid, "fallidos"]


def test_gold_transcript_voice_scenarios(env):
    from shopassist.scenarios import load_scenarios, run_all, scenario_metrics

    cfg, store, factory, out = env
    df, _ = run_all(load_scenarios(), factory, store, input_mode="gold_transcript", out_dir=str(out), seed=cfg.SEED)
    assert len(df) == 6
    print(df[["id", "ok", "fallidos"]].to_string())
    assert scenario_metrics(df)["violaciones_restricciones"] == 0


def test_restore_reproducible_between_scenarios(env):
    from shopassist.scenarios import all_stock, load_scenarios, run_scenario

    cfg, store, factory, out = env
    store.restore(cfg.SEED)
    s0 = all_stock(store)
    sc = next(s for s in load_scenarios() if s["id"] == "t14_stock_cambia_antes_confirmar")
    run_scenario(sc, factory, store, "text", out_dir=None, seed=cfg.SEED)
    store.restore(cfg.SEED)
    assert all_stock(store) == s0
