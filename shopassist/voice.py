"""Voz: reconocimiento (ASR), síntesis con clonación (TTS) y grabación en Colab.

Selección de modelos (T4 16 GB, junto al LLM y el índice):

* ASR: ``faster-whisper`` (CTranslate2). Modelo por defecto ``small``
  (~244M parámetros, ~1 GB de VRAM en float16, RTF ~0.05-0.1 en T4). Es el mejor
  compromiso para español leído con micrófono de laptop: ``medium`` (~2.5 GB, ~2x
  más lento) reduce el WER pocos puntos; ``large-v3-turbo`` (~1.6 GB, 4 capas de
  decodificador) ofrece calidad cercana a large-v3 con latencia parecida a
  ``medium``. Se usa ``compute_type=float16`` en GPU e ``int8`` en CPU,
  ``language="es"`` fijo (evita la detección de idioma y errores de idioma) y
  ``beam_size=5``.
* TTS: Coqui XTTS-v2 (``coqui-tts``, ``tts_models/multilingual/multi-dataset/xtts_v2``),
  clonación *zero-shot* a partir de 6-30 s de audio de referencia, soporta español
  nativamente. ~2 GB de VRAM en float32. Los *speaker latents* se calculan una sola
  vez (``get_conditioning_latents``) y se reutilizan en cada turno, lo que elimina
  ~0.5-1 s de cada síntesis. Licencia: Coqui Public Model License (CPML), sólo uso
  no comercial: adecuada para un laboratorio académico.
  Alternativa documentada: F5-TTS con *checkpoint* en español
  (p.ej. ``jpgallegoar/F5-Spanish``), MIT/CC-BY-NC, requiere también la
  transcripción de la referencia; no se usa por defecto por mayor latencia
  (difusión con ~32 pasos de flujo).

Todas las importaciones pesadas son perezosas para que el paquete pueda
importarse y probarse sin GPU.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional, Tuple, Union

DEFAULT_REFERENCE_WAV = "voice/referencia.wav"
DEFAULT_REFERENCE_TXT = "voice/referencia.txt"
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def _has_cuda() -> bool:
    try:
        import torch  # noqa: WPS433

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------
def _patch_av_open() -> None:
    """faster-whisper 1.1 llama ``av.open(..., metadata_errors=...)``, opción que PyAV >= 14 quitó."""
    try:
        import av
    except ImportError:
        return
    if getattr(av.open, "_compat", False):
        return
    _orig = av.open

    def _open(*args, metadata_errors=None, **kwargs):
        return _orig(*args, **kwargs)

    _open._compat = True
    av.open = _open


class ASR:
    """Reconocimiento de voz en español con faster-whisper.

    ``model_size``: "small" (por defecto), "medium" o "large-v3-turbo".
    ``transcribe`` acepta una ruta de audio o un arreglo numpy float32 a 16 kHz y
    devuelve ``(texto, info)`` donde ``info`` incluye idioma, duración y tiempo.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "es",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.language = language
        self.beam_size = beam_size
        # Un prompt inicial corto orienta la ortografía de números y moneda sin
        # usar consultas de evaluación.
        self.initial_prompt = initial_prompt or (
            "Asistente de compras. Quiero comprar dos unidades, presupuesto de 50 euros."
        )
        self._model = None
        self.load_time_s: Optional[float] = None

    def load(self) -> float:
        """Carga el modelo (arranque en frío). Devuelve segundos de carga."""
        if self._model is None:
            from faster_whisper import WhisperModel  # import perezoso

            _patch_av_open()
            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            self.load_time_s = time.perf_counter() - t0
        return self.load_time_s or 0.0

    def transcribe(self, audio: Union[str, os.PathLike, Any]) -> Tuple[str, dict]:
        self.load()
        if isinstance(audio, (str, os.PathLike)):
            audio = str(audio)
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        text = " ".join(s.text.strip() for s in segments).strip()  # consume el generador
        elapsed = time.perf_counter() - t0
        meta = {
            "language": getattr(info, "language", self.language),
            "language_probability": getattr(info, "language_probability", None),
            "duration_s": getattr(info, "duration", None),
            "asr_s": elapsed,
            "model": self.model_size,
        }
        return text, meta


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def _norm_tokens(s: str) -> list:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


def text_overlaps_reference(text: str, reference_text: str, n: int = 6) -> bool:
    """True si ``text`` comparte un n-grama de ``n`` palabras con la referencia
    o es idéntico. Se usa para garantizar que las respuestas generadas sean
    distintas del texto grabado como referencia (requisito del PDF)."""
    a, b = _norm_tokens(text), _norm_tokens(reference_text)
    if not a or not b:
        return False
    if a == b:
        return True
    grams = {tuple(b[i : i + n]) for i in range(max(0, len(b) - n + 1))}
    return any(tuple(a[i : i + n]) in grams for i in range(max(0, len(a) - n + 1)))


class TTS:
    """Síntesis en español con clonación de voz (Coqui XTTS-v2).

    ``reference_wav``: grabación propia/autorizada (6-30 s, mono, sin ruido).
    ``reference_txt``: transcripción de esa grabación; se usa como guarda para no
    sintetizar textos iguales a la referencia.
    """

    def __init__(
        self,
        reference_wav: str = DEFAULT_REFERENCE_WAV,
        reference_txt: Optional[str] = DEFAULT_REFERENCE_TXT,
        language: str = "es",
        device: Optional[str] = None,
        model_name: str = XTTS_MODEL_NAME,
        temperature: float = 0.65,
        speed: float = 1.05,
        max_chars: int = 400,
    ) -> None:
        self.reference_wav = reference_wav
        self.reference_txt = reference_txt
        self.language = language
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.model_name = model_name
        self.temperature = temperature
        self.speed = speed
        self.max_chars = max_chars
        self._tts = None
        self._model = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.sample_rate = 24000
        self.load_time_s: Optional[float] = None
        self.latents_time_s: Optional[float] = None
        self.reference_text = ""
        if reference_txt and Path(reference_txt).exists():
            self.reference_text = Path(reference_txt).read_text(encoding="utf-8").strip()

    def load(self) -> float:
        """Carga XTTS-v2 y precalcula los latentes del hablante."""
        if self._model is None:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")  # acepta CPML (uso no comercial)
            from TTS.api import TTS as CoquiTTS  # import perezoso (paquete coqui-tts)

            t0 = time.perf_counter()
            self._tts = CoquiTTS(self.model_name).to(self.device)
            self._model = self._tts.synthesizer.tts_model
            self.load_time_s = time.perf_counter() - t0
            self.compute_latents()
        return self.load_time_s or 0.0

    def compute_latents(self, reference_wav: Optional[str] = None) -> float:
        ref = reference_wav or self.reference_wav
        if not Path(ref).exists():
            raise FileNotFoundError(
                f"No existe la grabación de referencia {ref}. Grábala con record_audio_colab()."
            )
        t0 = time.perf_counter()
        self.gpt_cond_latent, self.speaker_embedding = self._model.get_conditioning_latents(
            audio_path=[ref], gpt_cond_len=30, max_ref_length=30
        )
        self.latents_time_s = time.perf_counter() - t0
        return self.latents_time_s

    def _check_text(self, text: str) -> None:
        if self.reference_text and text_overlaps_reference(text, self.reference_text):
            raise ValueError(
                "El texto a sintetizar coincide con la transcripción de referencia; "
                "el laboratorio exige textos distintos a los de la referencia."
            )

    def _split(self, text: str) -> list:
        parts = re.split(r"(?<=[\.\!\?;:])\s+", text.strip())
        chunks, cur = [], ""
        for p in parts:
            if len(cur) + len(p) + 1 <= 220:
                cur = f"{cur} {p}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        return chunks or [text]

    def synthesize(self, text: str, out_path: str, use_default_speaker: Optional[str] = None) -> str:
        """Sintetiza ``text`` a ``out_path`` (wav 24 kHz). ``use_default_speaker``
        permite generar con un hablante de fábrica de XTTS (línea base de similitud)."""
        import numpy as np
        import soundfile as sf

        self.load()
        self._check_text(text)
        text = text if len(text) <= self.max_chars else text[: self.max_chars].rsplit(" ", 1)[0] + "."
        if use_default_speaker:
            sm = self._model.speaker_manager
            d = sm.speakers[use_default_speaker]
            gpt, spk = d["gpt_cond_latent"], d["speaker_embedding"]
        else:
            gpt, spk = self.gpt_cond_latent, self.speaker_embedding
        wavs = []
        for chunk in self._split(text):
            out = self._model.inference(
                chunk,
                self.language,
                gpt,
                spk,
                temperature=self.temperature,
                speed=self.speed,
                enable_text_splitting=False,
            )
            w = out["wav"]
            w = w.cpu().numpy() if hasattr(w, "cpu") else np.asarray(w)
            wavs.append(w.squeeze())
            wavs.append(np.zeros(int(0.15 * self.sample_rate), dtype=np.float32))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, np.concatenate(wavs), self.sample_rate)
        return out_path

    def default_speakers(self) -> list:
        self.load()
        try:
            return list(self._model.speaker_manager.speakers.keys())
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Grabación / carga de audio en Colab
# ---------------------------------------------------------------------------
_RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time));
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = e => resolve(e.srcElement.result);
  reader.readAsDataURL(blob);
});
var record = time => new Promise(async resolve => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = e => chunks.push(e.data);
  const div = document.createElement('div');
  div.innerHTML = '<b style="color:red">&#9679; Grabando... habla ahora</b>';
  document.body.appendChild(div);
  recorder.start();
  await sleep(time);
  recorder.onstop = async () => {
    const blob = new Blob(chunks, { type: 'audio/webm' });
    const text = await b2text(blob);
    stream.getTracks().forEach(t => t.stop());
    div.remove();
    resolve(text);
  };
  recorder.stop();
});
"""


def to_wav_16k(src: str, out_path: str) -> str:
    """Convierte cualquier audio a wav mono 16 kHz con ffmpeg."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", out_path],
        check=True,
    )
    return out_path


def record_audio_colab(seconds: float, out_path: str) -> str:
    """Graba ``seconds`` segundos desde el micrófono del navegador (Colab).

    Usa MediaRecorder (webm/opus) vía ``google.colab.output.eval_js`` y convierte
    a wav mono 16 kHz con ffmpeg. Devuelve ``out_path``.
    """
    import base64

    from google.colab.output import eval_js  # sólo disponible en Colab
    from IPython.display import Javascript, display

    display(Javascript(_RECORD_JS))
    data_url = eval_js(f"record({int(seconds * 1000)})")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    webm = str(Path(out_path).with_suffix(".webm"))
    Path(webm).parent.mkdir(parents=True, exist_ok=True)
    Path(webm).write_bytes(raw)
    to_wav_16k(webm, out_path)
    return out_path


def upload_audio_colab(out_path: str) -> str:
    """Alternativa a la grabación: sube un archivo (wav/mp3/m4a/webm/ogg) y lo
    convierte a wav mono 16 kHz en ``out_path``."""
    from google.colab import files  # sólo disponible en Colab

    up = files.upload()
    if not up:
        raise RuntimeError("No se subió ningún archivo")
    name, content = next(iter(up.items()))
    tmp = Path(out_path).with_name("_upload_" + Path(name).name)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)
    to_wav_16k(str(tmp), out_path)
    return out_path


def audio_duration_s(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return float("nan")
