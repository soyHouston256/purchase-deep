"""Demo web local de ShopAssist: catálogo + compra por voz (FastAPI).

Flujo de un turno por voz (el mismo pipeline cronometrado del notebook, :class:`VoicePipeline`):
navegador (MediaRecorder webm/ogg) -> ``POST /api/voice`` -> ffmpeg a wav 16 kHz mono ->
faster-whisper (ASR) -> ``ShoppingAgent`` (recuperación + LLM en Ollama + herramientas de la
tienda) -> XTTS-v2 con la voz clonada (TTS) -> wav servido en ``/audio/...``.

TTS por frases (streaming): la respuesta se parte en trozos (:func:`shopassist.voice.split_tts_chunks`);
el endpoint devuelve el texto y el PRIMER trozo de audio en cuanto está listo, y los demás se sintetizan
en segundo plano (sin soltar el candado de modelos) mientras el navegador reproduce; la página pide cada
trozo con ``/api/tts_chunk`` (espera larga hasta que exista). Se mide el "tiempo hasta la primera voz".

Categorías (:mod:`shopassist.categories`) e imágenes: el catálogo ESCI no trae ni categoría ni imagen;
la categoría se infiere (reglas + kNN sobre las incrustaciones e5) y las imágenes se generan OFFLINE con
``scripts/generate_product_images.py`` (SD-Turbo); aquí solo se sirven las ya cacheadas
(``webapp/product_images/``) y para el resto se muestra el icono de la categoría.

Los modelos se cargan UNA vez al arrancar (en un hilo, para que ``/api/health`` informe del
progreso) y se usan de a un turno por vez (``asyncio.Lock`` + ``asyncio.to_thread``: el bucle de
eventos no se bloquea). Configuración por variables de entorno (ver ``Settings``).

Ejecutar desde la raíz del repo::

    python -m webapp            # o: uvicorn webapp.server:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import unicodedata
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("COQUI_TOS_AGREED", "1")          # XTTS-v2: licencia CPML (uso no comercial)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    llm: str = field(default_factory=lambda: _env("SHOP_LLM", "ollama"))             # ollama | rules
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:3b"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST_URL", "http://localhost:11434"))
    retriever: str = field(default_factory=lambda: _env("SHOP_RETRIEVER", "hybrid_rerank"))
    asr_size: str = field(default_factory=lambda: _env("ASR_SIZE", "small"))
    asr_device: str = field(default_factory=lambda: _env("ASR_DEVICE", "cpu"))       # CTranslate2: cpu en Mac
    asr_compute: str = field(default_factory=lambda: _env("ASR_COMPUTE", "int8"))
    tts: bool = field(default_factory=lambda: _env("SHOP_TTS", "1") not in ("0", "false", "no"))
    tts_device: str = field(default_factory=lambda: _env("TTS_DEVICE", "cpu"))       # cpu | mps | auto
    ref_wav: str = field(default_factory=lambda: _env("REF_WAV", str(ROOT / "voice/referencia.wav")))
    ref_txt: str = field(default_factory=lambda: _env("REF_TXT", str(ROOT / "voice/referencia.txt")))
    out_dir: str = field(default_factory=lambda: _env("WEB_OUT_DIR", str(ROOT / "results/web_demo")))
    tts_engine: str = field(default_factory=lambda: _env("TTS_ENGINE", "xtts"))     # xtts (voz clonada) | fast (Piper)
    tts_max_chars: int = field(default_factory=lambda: int(_env("TTS_MAX_CHARS", "0")))  # 0 = respuesta entera
    piper_voice: str = field(default_factory=lambda: _env("PIPER_VOICE", ""))
    torch_threads: int = field(default_factory=lambda: int(_env("TORCH_THREADS", "0")))  # 0 = núcleos de rendimiento
    ollama_keep_alive: str = field(default_factory=lambda: _env("OLLAMA_KEEP_ALIVE", "2h"))
    images_dir: str = field(default_factory=lambda: _env("PRODUCT_IMAGES_DIR", str(ROOT / "webapp/product_images")))
    warmup: bool = field(default_factory=lambda: _env("SHOP_WARMUP", "1") not in ("0", "false", "no"))


# ---------------------------------------------------------------------------- estado global
class Models:
    """Modelos y tienda compartidos por todas las sesiones (se cargan una sola vez)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.cfg = None
        self.store = None
        self.retriever = None
        self.llm = None
        self.asr = None
        self.tts = None
        self.catalog: list = []          # fichas ligeras para la grilla (sin precio/stock)
        self.cats: dict = {}             # product_id -> {"category", "icon", "source"}
        self.ready = False
        self.stage = "pendiente"
        self.errors: list = []
        self.info: dict = {}             # por modelo: nombre, dispositivo, tiempo de carga
        self.ffmpeg = shutil.which("ffmpeg")

    # -------------------------------------------------------------- carga
    def _step(self, name: str, fn):
        self.stage = f"cargando {name}"
        t0 = time.perf_counter()
        try:
            extra = fn() or {}
            self.info[name] = {"ok": True, "load_s": round(time.perf_counter() - t0, 2), **extra}
        except Exception as e:  # el servidor sigue arriba e informa del error en /api/health
            self.info[name] = {"ok": False, "load_s": round(time.perf_counter() - t0, 2),
                               "error": f"{type(e).__name__}: {e}"}
            self.errors.append(f"{name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        print(f"[webapp] {name}: {self.info[name]}", flush=True)

    def load_all(self) -> None:
        self._step("hilos", self._set_threads)
        self._step("tienda", self._load_store)
        if self.store is None:
            self.stage = "error"
            return
        self._step("recuperador", self._load_retriever)
        self._step("categorías", self._load_categories)
        self._step("llm", self._load_llm)
        self._step("asr", self._load_asr)
        if self.s.tts:
            self._step("tts", self._load_tts)
        else:
            self.info["tts"] = {"ok": False, "disabled": True}
        if self.s.warmup:
            self._step("calentamiento", self._warmup)
        self.stage = "listo"
        self.ready = True

    def _set_threads(self):
        """Hilos de PyTorch = núcleos de rendimiento (8 en un M1 Pro; los 2 de eficiencia solo frenan)."""
        import torch
        from shopassist.voice import performance_cores

        n = self.s.torch_threads or performance_cores()
        torch.set_num_threads(n)
        return {"torch_threads": torch.get_num_threads()}

    def _load_store(self):
        from shopassist.config import Config
        from shopassist.data import load_data
        from shopassist.notebook_utils import set_seeds
        from shopassist.store import Store

        cfg = Config()
        cfg.data_dir = str(ROOT / cfg.data_dir)
        set_seeds(cfg.SEED)
        catalog_df, _, _ = load_data(cfg)             # qrels NO se usan aquí
        self.cfg = cfg
        self.catalog_df = catalog_df
        self.store = Store(cfg.db_path, catalog_df, seed=cfg.SEED, currency=cfg.currency)
        return {"productos": len(self.store.product_ids()), "semilla": cfg.SEED, "moneda": cfg.currency}

    def _load_categories(self):
        """Categoría de cada producto (reglas + kNN sobre las incrustaciones e5 del recuperador) y
        orden de la grilla: primero los productos con imagen generada, alternando categorías."""
        from shopassist.categories import classify_catalog

        dense = getattr(self.retriever, "dense", None)
        emb = getattr(dense, "emb", None)
        if emb is not None and len(emb) != len(self.catalog_df):
            emb = None
        self.cats = classify_catalog(self.catalog_df, emb, cache_dir=self.cfg.data_dir)
        imgs = {p.stem for p in Path(self.s.images_dir).glob("*.webp")} if Path(self.s.images_dir).exists() else set()
        self.catalog = []
        seen_cat: dict = {}
        for pid in self.store.product_ids():
            p = self.store.catalog_text(pid)
            c = self.cats.get(pid, {"category": "Otros"})["category"]
            k = seen_cat.get(c, 0)
            seen_cat[c] = k + 1
            self.catalog.append({"product_id": pid, "_img": pid in imgs, "_rr": k,
                                 "_norm": _norm(f"{p.get('title', '')} {p.get('brand', '')} {c} {pid}")})
        self.catalog.sort(key=lambda c: (not c["_img"], c["_rr"]))
        n_src: dict = {}
        for v in self.cats.values():
            n_src[v["source"]] = n_src.get(v["source"], 0) + 1
        return {"fuente": n_src, "imagenes": len(imgs), "kNN": emb is not None}

    def _load_retriever(self):
        from shopassist.retrieval import build_retriever

        self.retriever = build_retriever(self.cfg, self.catalog_df, self.s.retriever, store=self.store)
        self.retriever.search("calentar", k=1)        # fuerza la carga del codificador y el re-ranker
        return {"tipo": self.s.retriever, "embeddings": self.cfg.embed_name, "reranker": self.cfg.reranker_name}

    def _load_llm(self):
        if self.s.llm == "rules":
            self.llm = None
            return {"modelo": "política por reglas (sin LLM)"}
        from shopassist.llm import OllamaLLM

        llm = OllamaLLM(self.s.ollama_model, host=self.s.ollama_host, seed=self.cfg.SEED,
                        keep_alive=self.s.ollama_keep_alive)
        if not llm.available():
            raise RuntimeError(f"Ollama no responde en {self.s.ollama_host} o falta el modelo "
                               f"(ejecuta: ollama pull {self.s.ollama_model}). Se usa la política por reglas.")
        llm.load()
        self.llm = llm
        return {"modelo": f"ollama:{self.s.ollama_model}", "host": self.s.ollama_host,
                "keep_alive": self.s.ollama_keep_alive}

    def _load_asr(self):
        from shopassist.voice import ASR

        self.asr = ASR(self.s.asr_size, device=self.s.asr_device, compute_type=self.s.asr_compute)
        self.asr.load()
        return {"modelo": f"faster-whisper {self.s.asr_size}", "device": self.asr.device,
                "compute_type": self.asr.compute_type}

    def _load_tts(self):
        from shopassist.voice import TTS, PiperTTS

        if self.s.tts_engine == "fast":
            tts = PiperTTS(self.s.piper_voice or None, cache_dir=str(ROOT / "data/piper"))
            tts.load()
            tts.synthesize_chunk("Prueba de voz.", str(Path(self.s.out_dir) / "_tts_probe.wav"))
            self.tts = tts
            return {"modelo": "Piper es_ES-davefx-medium (rápido, voz NO clonada)", "device": "cpu",
                    "engine": "fast"}
        if not Path(self.s.ref_wav).exists():
            raise FileNotFoundError(f"Falta la voz de referencia {self.s.ref_wav}; se responde solo con texto.")
        devices = ["mps", "cpu"] if self.s.tts_device == "auto" else [self.s.tts_device]
        last = None
        for dev in devices:
            try:
                tts = TTS(self.s.ref_wav, self.s.ref_txt, language="es", device=dev,
                          max_chars=self.s.tts_max_chars or None)
                tts.normalize = True               # importes y nº de pedido legibles en voz alta
                tts.load()                         # incluye los latentes del hablante (una sola vez)
                # prueba real en este dispositivo (en MPS algunas operaciones fallan recién al sintetizar)
                tts.synthesize_chunk("Prueba de voz.", str(Path(self.s.out_dir) / "_tts_probe.wav"))
                self.tts = tts
                return {"modelo": "XTTS-v2 (voz clonada)", "device": dev, "referencia": self.s.ref_wav,
                        "latentes_s": round(tts.latents_time_s or 0.0, 2), "engine": "xtts"}
            except Exception as e:  # noqa: PERF203
                last = e
                print(f"[webapp] XTTS en {dev} falló: {type(e).__name__}: {e}", flush=True)
        raise last  # type: ignore[misc]

    def _warmup(self):
        """Un turno de calentamiento (kernels, cachés) para que el primer turno de la demo sea "caliente"."""
        from shopassist.agent import ShoppingAgent

        out = {}
        t0 = time.perf_counter()
        ag = ShoppingAgent(self.store, self.retriever, llm=self.llm, cart_id="_warmup", cfg=self.cfg)
        ag.handle("Hola, ¿qué puedes hacer?", source="text")
        ag.reset()
        out["agente_s"] = round(time.perf_counter() - t0, 2)
        # el primer uso del re-ranker/codificador con textos de distinta longitud es lento (asignación
        # de memoria, kernels): unas búsquedas genéricas (no son las de la demo y no hay caché de
        # resultados) dejan la recuperación "caliente"
        t0 = time.perf_counter()
        for q in ("lámpara de escritorio", "zapatillas de running para hombre", "cargador usb rápido"):
            self.retriever.search(q, k=5, filters={"max_price": 100.0})
        out["recuperacion_s"] = round(time.perf_counter() - t0, 2)
        if self.tts is not None:
            t0 = time.perf_counter()
            self.tts.synthesize_chunk("Hola, esto es un calentamiento de la síntesis.",
                                      str(Path(self.s.out_dir) / "_tts_warm.wav"))
            out["tts_s"] = round(time.perf_counter() - t0, 2)
        if self.asr is not None:
            probe = Path(self.s.out_dir) / "_tts_probe.wav"
            if probe.exists():
                t0 = time.perf_counter()
                self.asr.transcribe(str(probe))
                out["asr_s"] = round(time.perf_counter() - t0, 2)
        return out

    def health(self) -> dict:
        return {"ready": self.ready, "stage": self.stage, "errors": self.errors, "models": self.info,
                "ffmpeg": self.ffmpeg, "settings": {k: v for k, v in self.s.__dict__.items()}}


class TTSJob:
    """Síntesis por trozos de UNA respuesta. El trozo 0 se sintetiza antes de responder; el resto en
    segundo plano. Cada trozo tiene un ``asyncio.Event`` que ``/api/tts_chunk`` espera."""

    def __init__(self, sess: "WebSession", tid: str, chunks: list):
        self.sess, self.tid = sess, tid
        self.token = uuid.uuid4().hex[:8]          # URLs únicas (tras Reiniciar se repiten los tid)
        self.chunks = [{"i": i, "text": c, "ready": False, "url": None, "synth_s": None, "duration_s": None}
                       for i, c in enumerate(chunks)]
        self.events = [asyncio.Event() for _ in chunks]
        self.done = asyncio.Event()
        self.error: Optional[str] = None
        self.timings: dict = {}

    def synth(self, i: int) -> None:
        """Sintetiza el trozo ``i`` (en un hilo: bloquea)."""
        name = f"{self.tid}_reply_{i:02d}.wav"
        info = M.tts.synthesize_chunk(self.chunks[i]["text"], str(self.sess.dir / name))
        self.chunks[i].update(ready=True, url=f"/audio/{self.sess.sid}/{name}?v={self.token}",
                              synth_s=round(info["synth_s"], 3), duration_s=round(info["duration_s"], 3))

    def public(self) -> dict:
        return {"turn_id": self.tid, "n": len(self.chunks), "done": self.done.is_set(), "error": self.error,
                "chunks": [{k: c[k] for k in ("i", "text", "ready", "url", "synth_s", "duration_s")}
                           for c in self.chunks], "timings": self.timings}


class WebSession:
    """Una pestaña del navegador: su agente, su carrito y su pipeline cronometrado."""

    def __init__(self, sid: str, m: Models):
        from shopassist.agent import ShoppingAgent
        from shopassist.latency import VoicePipeline

        self.sid = sid
        self.cart_id = f"web_{sid}"
        self.agent = ShoppingAgent(m.store, m.retriever, llm=m.llm, cart_id=self.cart_id, cfg=m.cfg)
        self.dir = Path(m.s.out_dir) / sid
        self.pipe = VoicePipeline(m.asr, self.agent, m.tts, out_dir=str(self.dir))
        self.jobs: dict = {}
        self.n = 0

    def reset(self):
        self.agent.reset()
        self.pipe.history.clear()
        self.jobs.clear()
        self.n = 0


SETTINGS = Settings()
M = Models(SETTINGS)
SESSIONS: dict = {}
MODEL_LOCK = asyncio.Lock()          # un turno a la vez sobre los modelos (incluida la síntesis de fondo)


# ---------------------------------------------------------------------------- utilidades
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _image_url(pid: str) -> Optional[str]:
    if re.fullmatch(r"[A-Za-z0-9]{6,20}", pid) and (Path(SETTINGS.images_dir) / f"{pid}.webp").exists():
        return f"/img/{pid}.webp"
    return None


def _card(pid: str) -> Optional[dict]:
    info = M.store.get_product(pid)
    if info is None:
        return None
    c = M.cats.get(pid) or {"category": "Otros", "icon": "🛍️"}
    return {"product_id": pid, "title": info["title"], "brand": info["brand"], "color": info["color"],
            "price": info["price"], "currency": info["currency"], "stock": info["stock"],
            "category": c["category"], "emoji": c["icon"], "image": _image_url(pid)}


def _state(sess: Optional[WebSession]) -> dict:
    from shopassist.session_view import session_view

    if sess is None:
        return {"cart": {"lines": [], "n_items": 0, "total": 0.0}, "orders": [], "stock": [],
                "last_results": [], "results": [], "currency": M.cfg.currency if M.cfg else "EUR"}
    v = session_view(M.store, sess.cart_id, sess.agent)
    v["results"] = [c for c in (_card(p) for p in v["last_results"][:8]) if c]
    v["currency"] = M.cfg.currency
    return v


def _session(sid: Optional[str], create: bool = True) -> Optional[WebSession]:
    if not sid or not re.fullmatch(r"[A-Za-z0-9_-]{4,40}", sid):
        raise HTTPException(400, "session inválida")
    if sid not in SESSIONS and create:
        SESSIONS[sid] = WebSession(sid, M)
    return SESSIONS.get(sid)


def _require_ready():
    if not M.ready:
        raise HTTPException(503, f"Modelos aún cargando ({M.stage}). Mira /api/health.")


def _jsonable(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def _turn_payload(sess: WebSession, rec: dict, kind: str, timings: dict, job: Optional[TTSJob]) -> dict:
    ops = [{"name": o["name"], "ok": bool(o.get("ok")), "by": o.get("by"), "args": o.get("args", {})}
           for o in rec.get("ops", [])]
    asr_info = {k: v for k, v in (rec.get("asr_info") or {}).items() if k in ("duration_s", "language_probability")}
    return _jsonable({"turn_id": rec["turn_id"], "kind": kind, "transcript": rec["transcript"],
                      "reply": rec["reply"], "ops": ops, "timings": {k: round(float(v), 3) for k, v in timings.items()},
                      "audio_url": job.chunks[0]["url"] if job and job.chunks and job.chunks[0]["ready"] else None,
                      "tts": job.public() if job else None, "tts_engine": getattr(M.tts, "engine", None),
                      "asr_info": asr_info, "clarification": any(o["name"] == "ask_clarification" for o in ops),
                      "state": _state(sess)})


async def _finish_tts(job: TTSJob, base: dict, t_turn0: float) -> None:
    """Sintetiza los trozos 1..n-1 y suelta el candado de modelos al terminar (o al fallar)."""
    try:
        for i in range(1, len(job.chunks)):
            await asyncio.to_thread(job.synth, i)
            job.events[i].set()
    except Exception as e:  # el turno ya se respondió: se informa en /api/turn_timings
        job.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        for ev in job.events:
            ev.set()
    finally:
        tts_s = sum(c["synth_s"] or 0.0 for c in job.chunks)
        job.timings = {**base, "tts_s": round(tts_s, 3),
                       "total_s": round(time.perf_counter() - t_turn0, 3),
                       "audio_s": round(sum(c["duration_s"] or 0.0 for c in job.chunks), 3)}
        job.done.set()
        MODEL_LOCK.release()


async def _run_turn(sess: WebSession, kind: str, tid: str, t_req: float, wav: Optional[str] = None,
                    text: Optional[str] = None, src: Optional[str] = None, use_tts: bool = True) -> dict:
    """Turno completo con el candado YA adquirido. Devuelve el payload tras el primer trozo de voz;
    si hay más trozos, el candado se suelta en :func:`_finish_tts`."""
    handed_off = False
    t0 = time.perf_counter()                        # inicio real del trabajo (tras la cola)
    try:
        decode_s = 0.0
        if src is not None:
            await asyncio.to_thread(_to_wav, src, wav)
            decode_s = time.perf_counter() - t0
        if text is not None:
            rec = await asyncio.to_thread(sess.pipe.run_turn, "(texto)", tid, text, False)
        else:
            rec = await asyncio.to_thread(sess.pipe.run_turn, wav, tid, None, False)
        st = rec["timings"]
        base = {k: st.get(k, 0.0) for k in ("asr_s", "retrieval_s", "llm_s", "tools_s", "agent_other_s")}
        base["decode_s"] = decode_s
        base["cola_s"] = t0 - t_req
        base["agent_s"] = base["retrieval_s"] + base["llm_s"] + base["tools_s"] + base["agent_other_s"]
        job = None
        if use_tts and M.tts is not None and rec["reply"]:
            chunks = M.tts.split(rec["reply"])
            if chunks:
                job = TTSJob(sess, tid, chunks)
                sess.jobs[tid] = job
                await asyncio.to_thread(job.synth, 0)
                job.events[0].set()
        t_first = time.perf_counter() - t0
        timings = dict(base)
        if job is not None:
            timings["tts_first_s"] = job.chunks[0]["synth_s"]
            timings["ttfa_s"] = t_first                 # fin de grabación -> primer wav listo
            timings["tts_s"] = job.chunks[0]["synth_s"]  # se completa en /api/turn_timings
        else:
            timings["tts_first_s"] = timings["tts_s"] = 0.0
            timings["ttfa_s"] = None
        timings["total_s"] = t_first
        timings["server_s"] = t_first
        rec["timings"].update({k: v for k, v in timings.items() if v is not None})
        payload = _turn_payload(sess, rec, kind, {k: v for k, v in timings.items() if v is not None}, job)
        if job is not None and len(job.chunks) > 1:
            asyncio.create_task(_finish_tts(job, base, t0))
            handed_off = True
        elif job is not None:
            job.timings = {**base, "tts_s": job.chunks[0]["synth_s"], "total_s": round(t_first, 3),
                           "audio_s": job.chunks[0]["duration_s"]}
            job.done.set()
        return payload
    finally:
        if not handed_off:
            MODEL_LOCK.release()


# ---------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(SETTINGS.out_dir).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=M.load_all, name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="ShopAssist · demo web local", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return M.health()


@app.get("/api/products")
async def products(q: str = "", offset: int = 0, limit: int = 24, ids: str = "", in_stock: bool = False):
    _require_ready()
    limit = max(1, min(limit, 96))
    if ids:
        items = [c for c in (_card(p) for p in ids.split(",")[:96]) if c]
        return {"total": len(items), "offset": 0, "items": items}
    toks = _norm(q).split()
    rows = [p for p in M.catalog if all(t in p["_norm"] for t in toks)] if toks else M.catalog
    if in_stock:
        inv = M.store.inventory([p["product_id"] for p in rows])
        rows = [p for p in rows if inv.get(p["product_id"], 0) > 0]
    page = rows[offset: offset + limit]
    return {"total": len(rows), "offset": offset, "items": [c for c in (_card(p["product_id"]) for p in page) if c]}


@app.get("/api/state")
async def state(session: str):
    _require_ready()
    return _jsonable(_state(_session(session, create=False)))


class TextIn(BaseModel):
    session: str
    text: str
    tts: bool = True


@app.post("/api/text")
async def text_turn(body: TextIn):
    _require_ready()
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "texto vacío")
    t_req = time.perf_counter()
    sess = _session(body.session)
    await MODEL_LOCK.acquire()                     # lo suelta _run_turn (o _finish_tts)
    try:
        sess.n += 1
        tid = f"t{sess.n:02d}"
        sess.dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        MODEL_LOCK.release()
        raise
    try:
        return await _run_turn(sess, "texto", tid, t_req, text=text, use_tts=body.tts)
    except Exception as e:
        sess.n -= 1
        traceback.print_exc()
        raise HTTPException(500, f"Falló el turno: {type(e).__name__}: {e}")


@app.post("/api/voice")
async def voice_turn(session: str = Form(...), audio: UploadFile = File(...)):
    _require_ready()
    if M.asr is None:
        raise HTTPException(503, "El ASR no está cargado (mira /api/health).")
    t_req = time.perf_counter()
    raw = await audio.read()
    if len(raw) < 200:
        raise HTTPException(400, "audio vacío o demasiado corto")
    sess = _session(session)
    await MODEL_LOCK.acquire()                     # lo suelta _run_turn (o _finish_tts)
    try:
        sess.n += 1
        tid = f"t{sess.n:02d}"
        sess.dir.mkdir(parents=True, exist_ok=True)
        ext = Path(audio.filename or "").suffix.lower() or _ext_from_type(audio.content_type)
        src = sess.dir / f"{tid}_in_raw{ext}"
        src.write_bytes(raw)                          # se conserva la grabación original
        wav = sess.dir / f"{tid}_in.wav"
    except Exception:
        MODEL_LOCK.release()
        raise
    try:
        return await _run_turn(sess, "voz", tid, t_req, wav=str(wav), src=str(src))
    except HTTPException:
        sess.n -= 1
        raise
    except Exception as e:
        sess.n -= 1
        traceback.print_exc()
        raise HTTPException(500, f"Falló el turno: {type(e).__name__}: {e}")


def _job(session: str, turn: str) -> TTSJob:
    sess = _session(session, create=False)
    if sess is None or turn not in sess.jobs:
        raise HTTPException(404, "turno sin síntesis")
    return sess.jobs[turn]


@app.get("/api/tts_chunk")
async def tts_chunk(session: str, turn: str, i: int):
    """Espera (hasta 3 min) a que el trozo ``i`` esté sintetizado y devuelve su URL y tiempos."""
    job = _job(session, turn)
    if not 0 <= i < len(job.chunks):
        raise HTTPException(404, "trozo inexistente")
    try:
        await asyncio.wait_for(job.events[i].wait(), timeout=180)
    except asyncio.TimeoutError:
        raise HTTPException(504, "la síntesis tarda demasiado")
    if not job.chunks[i]["ready"]:
        raise HTTPException(500, f"falló la síntesis: {job.error}")
    return _jsonable({**job.chunks[i], "n": len(job.chunks)})


@app.get("/api/turn_timings")
async def turn_timings(session: str, turn: str):
    """Tiempos finales del turno (cuando ya se sintetizaron todos los trozos)."""
    job = _job(session, turn)
    try:
        await asyncio.wait_for(job.done.wait(), timeout=300)
    except asyncio.TimeoutError:
        raise HTTPException(504, "la síntesis tarda demasiado")
    return _jsonable(job.public())


def _ext_from_type(ct: Optional[str]) -> str:
    ct = (ct or "").lower()
    for k, e in (("webm", ".webm"), ("ogg", ".ogg"), ("mp4", ".m4a"), ("mpeg", ".mp3"), ("wav", ".wav")):
        if k in ct:
            return e
    return ".webm"


def _to_wav(src: str, dst: str) -> None:
    if M.ffmpeg is None:
        raise HTTPException(500, "Falta ffmpeg: instálalo con `brew install ffmpeg` y reinicia el servidor.")
    from shopassist.voice import to_wav_16k

    to_wav_16k(src, dst)


class ResetIn(BaseModel):
    session: str
    restore_store: bool = False


@app.post("/api/reset")
async def reset(body: ResetIn):
    _require_ready()
    async with MODEL_LOCK:
        sess = _session(body.session, create=False)
        if body.restore_store:
            M.store.restore(M.cfg.SEED)
        if sess is not None:
            sess.reset()
        return _jsonable({"ok": True, "restored": body.restore_store, "seed": M.cfg.SEED, "state": _state(sess)})


@app.get("/audio/{sid}/{name}")
async def audio_file(sid: str, name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,40}", sid) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.wav", name):
        raise HTTPException(404)
    path = Path(SETTINGS.out_dir) / sid / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.get("/img/{name}")
async def product_image(name: str):
    """Imagen generada offline (``scripts/generate_product_images.py``); 404 -> la página usa el icono."""
    if not re.fullmatch(r"[A-Za-z0-9]{6,20}\.webp", name):
        raise HTTPException(404)
    path = Path(SETTINGS.images_dir) / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.exception_handler(HTTPException)
async def _http_exc(request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
