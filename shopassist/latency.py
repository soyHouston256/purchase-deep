"""Medición de latencia por etapa: ASR -> agente (recuperación, LLM, herramientas) -> TTS.

El tiempo total se mide desde que termina la grabación (el audio ya está en
disco) hasta que el wav de respuesta está escrito y puede reproducirse.
La carga inicial de modelos (arranque en frío) se registra aparte de los turnos.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

STAGES = ["asr_s", "retrieval_s", "llm_s", "tools_s", "tts_s"]


class StageTimer:
    """Acumulador de tiempos por etapa.

    >>> t = StageTimer()
    >>> with t("asr_s"):
    ...     pass
    >>> "asr_s" in t.timings
    True
    """

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.timings[stage] = self.timings.get(stage, 0.0) + time.perf_counter() - t0

    def add(self, stage: str, seconds: float) -> None:
        self.timings[stage] = self.timings.get(stage, 0.0) + float(seconds or 0.0)


def _vram_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**3
    except Exception:
        pass
    return None


def load_models(loaders: Dict[str, Callable[[], object]]) -> tuple:
    """Carga modelos en frío midiendo tiempo y VRAM después de cada uno.

    ``loaders``: dict nombre -> función sin argumentos que devuelve el modelo.
    Devuelve ``(modelos: dict, tiempos: list[dict])``.
    """
    models, rows = {}, []
    for name, fn in loaders.items():
        t0 = time.perf_counter()
        models[name] = fn()
        rows.append({"model": name, "load_s": time.perf_counter() - t0, "vram_gb_after": _vram_gb()})
    return models, rows


class VoicePipeline:
    """Turno completo por voz. ``tts`` puede ser ``None`` (sólo texto)."""

    def __init__(self, asr, agent, tts=None, out_dir: str = "results/voice_turns") -> None:
        self.asr, self.agent, self.tts = asr, agent, tts
        self.out_dir = Path(out_dir)
        self.history: List[dict] = []
        self.load_timings: List[dict] = []

    def load_models(self) -> List[dict]:
        """Arranque en frío de ASR y TTS (el LLM se carga fuera y se mide igual)."""
        loaders = {}
        if self.asr is not None and hasattr(self.asr, "load"):
            loaders["asr"] = self.asr.load
        if self.tts is not None and hasattr(self.tts, "load"):
            loaders["tts"] = self.tts.load
        _, rows = load_models(loaders)
        self.load_timings.extend(rows)
        return rows

    def run_turn(self, audio_path: str, turn_id: Optional[str] = None, transcript: Optional[str] = None) -> dict:
        """Ejecuta ASR -> agente -> TTS. Si ``transcript`` se da, se omite el ASR
        (útil para la repetición con transcripción correcta)."""
        timer = StageTimer()
        t_start = time.perf_counter()
        info = {}
        if transcript is None:
            with timer("asr_s"):
                transcript, info = self.asr.transcribe(audio_path)
        else:
            timer.add("asr_s", 0.0)
        t_agent = time.perf_counter()
        turn = self.agent.handle(transcript, source="voice")
        agent_s = time.perf_counter() - t_agent
        at = dict(getattr(turn, "timings", {}) or {})
        for k in ("retrieval_s", "llm_s", "tools_s"):
            timer.add(k, at.get(k, 0.0))
        # tiempo del agente no atribuido (lógica de control, NLU, memoria)
        timer.add("agent_other_s", max(0.0, agent_s - sum(at.get(k, 0.0) for k in ("retrieval_s", "llm_s", "tools_s"))))
        audio_out = None
        if self.tts is not None and turn.reply:
            tid = turn_id or f"turn_{len(self.history):03d}"
            out = self.out_dir / f"{tid}_reply.wav"
            with timer("tts_s"):
                audio_out = self.tts.synthesize(turn.reply, str(out))
        else:
            timer.add("tts_s", 0.0)
        total = time.perf_counter() - t_start
        timings = {k: timer.timings.get(k, 0.0) for k in STAGES + ["agent_other_s"]}
        timings["total_s"] = total
        rec = {
            "turn_id": turn_id,
            "audio_in": str(audio_path),
            "transcript": transcript,
            "asr_info": info,
            "reply": turn.reply,
            "ops": getattr(turn, "ops", []),
            "audio_out": audio_out,
            "timings": timings,
            "warm": len(self.history) > 0,
        }
        self.history.append(rec)
        return rec


def latency_summary(turn_records: List[dict], load_timings: Optional[List[dict]] = None):
    """DataFrame con media/mediana/p90 por etapa, separando el primer turno
    (frío, incluye *warm-up* de kernels) de los siguientes (calientes).
    Si se pasan ``load_timings`` se añaden como filas ``carga_<modelo>``."""
    import pandas as pd

    rows = []
    for i, r in enumerate(turn_records):
        t = dict(r.get("timings", {}))
        warm = r.get("warm", i > 0)
        t["fase"] = "caliente" if warm else "primer_turno"
        rows.append(t)
    df = pd.DataFrame(rows)
    out = []
    cols = [c for c in STAGES + ["agent_other_s", "total_s"] if c in df.columns]
    if not df.empty:
        for fase, g in list(df.groupby("fase")) + [("todos", df)]:
            for stat, fn in (("media", "mean"), ("mediana", "median"), ("p90", lambda x: x.quantile(0.9))):
                vals = g[cols].agg(fn)
                out.append({"fase": fase, "estadistico": stat, "n": len(g), **vals.to_dict()})
    summ = pd.DataFrame(out)
    if load_timings:
        lt = pd.DataFrame(load_timings)
        lt = lt.rename(columns={"load_s": "total_s"})
        lt["fase"] = "carga_" + lt["model"].astype(str)
        lt["estadistico"] = "valor"
        lt["n"] = 1
        summ = pd.concat([summ, lt.drop(columns=["model"])], ignore_index=True)
    return summ


def plot_latency(summary, ax=None, stat: str = "media"):
    """Barras apiladas por etapa (media) para primer turno vs calientes."""
    import matplotlib.pyplot as plt

    d = summary[(summary["estadistico"] == stat) & summary["fase"].isin(["primer_turno", "caliente"])]
    cols = [c for c in STAGES + ["agent_other_s"] if c in d.columns]
    d = d.set_index("fase")[cols]
    ax = d.plot(kind="bar", stacked=True, ax=ax, figsize=(7, 4))
    ax.set_ylabel("segundos")
    ax.set_title(f"Latencia por etapa ({stat}): fin de grabación → audio reproducible")
    plt.tight_layout()
    return ax
