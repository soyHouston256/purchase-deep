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


def performance_cores() -> int:
    """Núcleos de rendimiento (Apple Silicon: ``hw.perflevel0.physicalcpu``); si no, núcleos físicos."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True,
                             timeout=2)
        n = int(out.stdout.strip())
        if n > 0:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


_ORDER_RE = re.compile(r"\bO0*(\d+)\b")
_MONEY_RE = re.compile(r"(\d+)[\.,](\d{2})\s*(?:EUR|€|euros?)\b", re.IGNORECASE)


def speech_text(text: str) -> str:
    """Normaliza la respuesta para leerla en voz alta (no cambia el texto mostrado):
    importes ``9,98 EUR`` -> ``9 euros con 98``, ``O00001`` -> ``número 1``, ``2 ×`` -> ``2 por``,
    sin comillas angulares ni viñetas."""
    t = _MONEY_RE.sub(lambda m: f"{int(m.group(1))} euros" + (f" con {int(m.group(2))}" if int(m.group(2)) else ""), text)
    t = re.sub(r"\bEUR\b", "euros", t)
    t = _ORDER_RE.sub(lambda m: f"número {int(m.group(1))}", t)
    t = t.replace("×", " por ").replace("«", "").replace("»", "").replace("\"", "")
    t = re.sub(r"^\s*[-•*]\s*", "", t, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", t).strip()


def split_tts_chunks(text: str, first_min: int = 25, min_chars: int = 70, max_chars: int = 200) -> list:
    """Parte la respuesta en trozos para la síntesis por frases (streaming).

    El primer trozo es la primera frase (al menos ``first_min`` caracteres): sale antes y fija el
    "tiempo hasta la primera voz". Los siguientes agrupan frases hasta ``min_chars``-``max_chars``
    (XTTS tiende a divagar con frases muy cortas y cada llamada tiene un coste fijo). Las frases
    más largas que ``max_chars`` se cortan por comas o espacios. No se pierde texto.
    """
    text = (text or "").strip()
    if not text:
        return []
    sents = [s for s in re.split(r"(?<=[\.\!\?;:…])\s+", text) if s.strip()]
    pieces = []
    for s in sents:
        while len(s) > max_chars:
            cut = s.rfind(", ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            pieces.append(s)
    chunks, cur = [], ""
    for p in pieces:
        need = first_min if not chunks else min_chars
        if not cur:
            cur = p
        elif len(cur) < need and len(cur) + 1 + len(p) <= max_chars:
            cur = f"{cur} {p}"
        else:
            chunks.append(cur)
            cur = p
    if cur:
        # un último trozo muy corto se une al anterior si cabe (evita divagaciones de XTTS)
        if chunks and len(cur) < 25 and len(chunks[-1]) + 1 + len(cur) <= max_chars + 40:
            chunks[-1] = f"{chunks[-1]} {cur}"
        else:
            chunks.append(cur)
    return chunks


def xtts_text(chunk: str) -> str:
    """Texto que se pasa a XTTS: sin ¿¡ ni puntuación final, y las pausas internas como comas.

    Medido aquí (3 frases x 4 semillas): con la puntuación final XTTS-v2 a veces no emite el fin de
    secuencia y "balbucea" hasta 30 s por cada 100 caracteres; sin ella, como mucho 9.4 s (lo normal
    son 5-7 s). La entonación de pregunta se pierde un poco, pero el texto mostrado no cambia."""
    t = re.sub(r"[¿¡«»\"]", "", chunk)
    t = re.sub(r"\s*[\.\?!;:…]+\s+", ", ", t)
    return t.strip().rstrip(".?!;:…,").strip()


def trim_trailing_silence(w, sr: int, frame_s: float = 0.02, keep_s: float = 0.15, rel_db: float = -38.0):
    """Recorta el silencio/ruido muy débil del final (XTTS a veces deja segundos de cola)."""
    import numpy as np

    n = int(frame_s * sr)
    if len(w) < 4 * n:
        return w
    frames = len(w) // n
    rms = np.sqrt(np.mean(w[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    thr = rms.max() * (10 ** (rel_db / 20))
    loud = np.nonzero(rms > thr)[0]
    if len(loud) == 0:
        return w
    end = min(len(w), (loud[-1] + 1) * n + int(keep_s * sr))
    return w[:end]


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
        max_chars: Optional[int] = None,
        num_threads: Optional[int] = None,
        tokens_per_char: float = 2.2,
    ) -> None:
        self.reference_wav = reference_wav
        self.reference_txt = reference_txt
        self.language = language
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.model_name = model_name
        self.temperature = temperature
        self.speed = speed
        self.max_chars = max_chars                 # None = sin recorte (se sintetiza la respuesta entera)
        self.num_threads = num_threads
        # tope de códigos de audio del GPT por carácter: ~1.8 en habla normal; el tope corta las
        # divagaciones ("babble") de XTTS, que a veces añade varios segundos de ruido tras frases cortas
        self.tokens_per_char = tokens_per_char
        self.engine = "xtts"
        self.normalize = False                     # la demo web lo activa (importes, nº de pedido)
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

            if self.num_threads:
                import torch

                torch.set_num_threads(int(self.num_threads))
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

    def split(self, text: str) -> list:
        """Trozos de texto en el orden en que se sintetizan (normalizado para voz si ``normalize``;
        por defecto no, para no alterar la evaluación WER/CER del notebook)."""
        text = speech_text(text) if self.normalize else text.strip()
        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0] + "."
        return split_tts_chunks(text)

    def _speaker(self, use_default_speaker: Optional[str] = None):
        if use_default_speaker:
            d = self._model.speaker_manager.speakers[use_default_speaker]
            return d["gpt_cond_latent"], d["speaker_embedding"]
        return self.gpt_cond_latent, self.speaker_embedding

    def _infer(self, chunk: str, gpt, spk):
        import numpy as np

        chunk = xtts_text(chunk)
        cap = int(30 + self.tokens_per_char * len(chunk))
        out = self._model.inference(
            chunk,
            self.language,
            gpt,
            spk,
            temperature=self.temperature,
            speed=self.speed,
            enable_text_splitting=False,
            max_new_tokens=cap,
        )
        w = out["wav"]
        w = w.cpu().numpy() if hasattr(w, "cpu") else np.asarray(w)
        return trim_trailing_silence(w.squeeze().astype("float32"), self.sample_rate)

    def synthesize_chunk(self, chunk: str, out_path: str, use_default_speaker: Optional[str] = None) -> dict:
        """Sintetiza UN trozo (ver :meth:`split`) a ``out_path``. Devuelve tiempos y duración."""
        import numpy as np
        import soundfile as sf

        self.load()
        self._check_text(chunk)
        gpt, spk = self._speaker(use_default_speaker)
        t0 = time.perf_counter()
        w = self._infer(chunk, gpt, spk)
        w = np.concatenate([w, np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)])
        synth_s = time.perf_counter() - t0
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, w, self.sample_rate)
        return {"path": out_path, "synth_s": synth_s, "duration_s": len(w) / self.sample_rate, "text": chunk}

    def synthesize(self, text: str, out_path: str, use_default_speaker: Optional[str] = None) -> str:
        """Sintetiza ``text`` completo a ``out_path`` (wav 24 kHz), trozo a trozo y sin recortar
        (salvo ``max_chars``). ``use_default_speaker`` permite generar con un hablante de fábrica
        de XTTS (línea base de similitud)."""
        import numpy as np
        import soundfile as sf

        self.load()
        self._check_text(text)
        gpt, spk = self._speaker(use_default_speaker)
        wavs = []
        for chunk in self.split(text):
            wavs.append(self._infer(chunk, gpt, spk))
            wavs.append(np.zeros(int(0.15 * self.sample_rate), dtype=np.float32))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, np.concatenate(wavs) if wavs else np.zeros(1, dtype=np.float32), self.sample_rate)
        return out_path

    def default_speakers(self) -> list:
        self.load()
        try:
            return list(self._model.speaker_manager.speakers.keys())
        except Exception:
            return []


class PiperTTS:
    """Modo rápido (``TTS_ENGINE=fast`` en la demo web): Piper (VITS en ONNX Runtime, CPU).

    Voz española de fábrica (``es_ES-davefx-medium``, ~60 MB), NO la voz clonada: es ~50x más
    rápida que XTTS-v2 en CPU (≈0.15 s para 5 s de audio) a cambio de perder la clonación.
    Misma interfaz que :class:`TTS` (``load``/``split``/``synthesize_chunk``/``synthesize``).
    """

    DEFAULT_VOICE = "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"

    def __init__(self, voice_path: Optional[str] = None, cache_dir: str = "data/piper",
                 length_scale: float = 0.95) -> None:
        self.voice_path = voice_path or str(Path(cache_dir) / self.DEFAULT_VOICE)
        self.cache_dir = cache_dir
        self.length_scale = length_scale
        self.engine = "piper"
        self.device = "cpu"
        self.sample_rate = 22050
        self._voice = None
        self.load_time_s: Optional[float] = None
        self.latents_time_s = 0.0

    def load(self) -> float:
        if self._voice is None:
            from piper import PiperVoice  # paquete piper-tts

            t0 = time.perf_counter()
            if not Path(self.voice_path).exists():
                from huggingface_hub import hf_hub_download

                for suf in ("", ".json"):
                    hf_hub_download("rhasspy/piper-voices", self.DEFAULT_VOICE + suf, local_dir=self.cache_dir)
            self._voice = PiperVoice.load(self.voice_path)
            self.sample_rate = int(self._voice.config.sample_rate)
            self.load_time_s = time.perf_counter() - t0
        return self.load_time_s or 0.0

    def split(self, text: str) -> list:
        return split_tts_chunks(speech_text(text))

    def _wav(self, text: str):
        import numpy as np
        from piper import SynthesisConfig

        parts = [c.audio_float_array for c in self._voice.synthesize(
            text, syn_config=SynthesisConfig(length_scale=self.length_scale))]
        return np.concatenate(parts).astype("float32") if parts else np.zeros(1, dtype="float32")

    def synthesize_chunk(self, chunk: str, out_path: str, **_) -> dict:
        import numpy as np
        import soundfile as sf

        self.load()
        t0 = time.perf_counter()
        w = np.concatenate([self._wav(chunk), np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)])
        synth_s = time.perf_counter() - t0
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, w, self.sample_rate)
        return {"path": out_path, "synth_s": synth_s, "duration_s": len(w) / self.sample_rate, "text": chunk}

    def synthesize(self, text: str, out_path: str, **_) -> str:
        import numpy as np
        import soundfile as sf

        self.load()
        w = self._wav(speech_text(text))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, w if len(w) else np.zeros(1, dtype=np.float32), self.sample_rate)
        return out_path


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
