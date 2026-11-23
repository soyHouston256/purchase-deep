"""Funciones auxiliares del notebook.

Viven aquí para que las celdas del notebook muestren solo lo importante
(qué se hace y qué resultado sale) y no los detalles de medición o formato.
"""
import gc
import hashlib
import json
import os
import random
import subprocess
import time

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # sin torch: el notebook sigue funcionando en modo reglas
    torch = None


def set_seeds(seed: int) -> None:
    """Fija las semillas de random, numpy y torch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def memory_report(tag: str = "") -> dict:
    """Imprime la VRAM asignada/reservada por PyTorch y la RAM del sistema."""
    out = {"tag": tag}
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        out["vram_alloc_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        out["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
        out["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    try:
        out["ram"] = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.splitlines()[1:2]
    except FileNotFoundError:  # `free` no existe fuera de Linux
        out["ram"] = None
    print(out)
    return out


def free_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def store_hash(store) -> str:
    """Huella del estado de la tienda (stock, precios y pedidos)."""
    pids = store.product_ids()
    inv = store.inventory(pids)
    prices = {p: store.get_price(p) for p in pids}
    payload = json.dumps([inv, prices, store.snapshot()["orders"]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def show_turn(agent, user: str, source: str = "text"):
    """Ejecuta un turno del agente e imprime la conversación y las herramientas usadas."""
    turn = agent.handle(user, source=source)
    print(f"👤 {user}\n🤖 {turn.reply}")
    print("   herramientas:", [(o["name"], o.get("ok")) for o in turn.ops],
          "| tiempos:", {k: round(v, 2) for k, v in turn.timings.items()})
    return turn


def load_voice_and_llm(cfg, use_llm: bool = True, asr_size: str = "small",
                       ref_wav: str = "voice/referencia.wav", ref_txt: str = "voice/referencia.txt"):
    """Carga LLM, ASR y TTS midiendo el arranque en frío.

    Devuelve ``(llm, asr, tts, cold_df)``. ``tts`` es ``None`` si falta la grabación de referencia.
    """
    from shopassist.latency import load_models
    from shopassist.llm import LocalLLM
    from shopassist.voice import ASR, TTS

    asr = ASR(asr_size)
    tts = TTS(ref_wav, ref_txt, language="es")

    def _load_tts():
        if not os.path.exists(ref_wav):
            print(f"⚠ Falta {ref_wav}: grábala en A3 y vuelve a ejecutar esta celda")
            return None
        tts.load()
        return tts

    models, cold = load_models({
        "llm": lambda: LocalLLM(cfg) if use_llm else None,
        "asr": lambda: (asr.load(), asr)[1],
        "tts": _load_tts,
    })
    llm = models["llm"]
    if hasattr(llm, "load"):  # LocalLLM descarga y carga los pesos en load()
        t0 = time.perf_counter()
        llm.load()
        cold[0]["load_s"] += time.perf_counter() - t0
    tts = models["tts"]
    cold_df = pd.DataFrame(cold)
    if tts is not None and tts.latents_time_s is not None:
        cold_df = pd.concat([cold_df, pd.DataFrame([{"model": "tts_latentes_ref", "load_s": tts.latents_time_s}])])
    return llm, asr, tts, cold_df


def _to_markdown(obj) -> str:
    if isinstance(obj, pd.DataFrame):
        return obj.to_markdown(index=False) if obj.index.name is None else obj.reset_index().to_markdown(index=False)
    if isinstance(obj, pd.Series):
        return obj.to_frame().to_markdown()
    if isinstance(obj, dict):
        return pd.Series(obj).to_frame("valor").to_markdown()
    return str(obj)


def write_report(report: dict, path: str = "results/REPORTE_auto.md") -> None:
    """Vuelca todas las tablas del dict ``report`` a un Markdown."""
    lines = ["# Reporte automático – Laboratorio 4", "", f"Generado: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for k, v in report.items():
        lines += [f"## {k}", "", _to_markdown(v), ""]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(path, "escrito")
