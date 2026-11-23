"""Evaluación de voz.

* Inteligibilidad: WER/CER de Whisper sobre el audio sintetizado vs. el texto
  que se quiso decir (``jiwer``).
* Semejanza con la voz de referencia: coseno entre *embeddings* de hablante
  ECAPA-TDNN (``speechbrain/spkrec-ecapa-voxceleb``). Línea base: el mismo texto
  sintetizado con un hablante de fábrica de XTTS (otra voz) y, como cota
  superior, la similitud entre dos mitades de la propia referencia.
* Análisis de errores del ASR: WER y errores específicos en cantidades,
  importes y productos comparando la transcripción correcta (gold) y la del ASR.

Las dependencias pesadas (speechbrain, torch) se importan de forma perezosa.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Normalización y WER/CER
# ---------------------------------------------------------------------------


def normalize_text(s: str) -> str:
    """Minúsculas, sin tildes ni puntuación (conserva dígitos, ',' y '.' decimales)."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"[^a-z0-9. ]+", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def wer_cer(reference: str, hypothesis: str, numbers_as_digits: bool = True) -> Dict[str, float]:
    """WER y CER tras normalizar. Si ``numbers_as_digits`` convierte los números
    escritos en palabras a dígitos en ambos lados ("dos" == "2")."""
    import jiwer

    r, h = normalize_text(reference), normalize_text(hypothesis)
    if numbers_as_digits:
        r, h = words_to_digits(r), words_to_digits(h)
    if not r:
        return {"wer": 0.0 if not h else 1.0, "cer": 0.0 if not h else 1.0}
    return {"wer": float(jiwer.wer(r, h or "")), "cer": float(jiwer.cer(r, h or ""))}


# ---------------------------------------------------------------------------
# Números en español (palabras -> valor)
# ---------------------------------------------------------------------------
_UNITS = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21,
    "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29,
}
_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
         "ochenta": 80, "noventa": 90}
_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200,
             "trescientos": 300, "trescientas": 300, "cuatrocientos": 400,
             "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
             "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
             "setecientas": 700, "ochocientos": 800, "ochocientas": 800,
             "novecientos": 900, "novecientas": 900}
_SPECIAL = {"docena": 12, "media docena": 6, "par": 2}
_NUM_WORDS = set(_UNITS) | set(_TENS) | set(_HUNDREDS) | {"mil", "y"}

# "un/una" sólo cuenta como cantidad si va seguido de sustantivo; para no
# inflar falsos positivos en la comparación ambos lados se tratan igual.


def _parse_number_words(tokens: Sequence[str]) -> Optional[int]:
    total, cur, seen = 0, 0, False
    for t in tokens:
        if t == "y":
            continue
        if t in _UNITS:
            cur += _UNITS[t]
        elif t in _TENS:
            cur += _TENS[t]
        elif t in _HUNDREDS:
            cur += _HUNDREDS[t]
        elif t == "mil":
            total += (cur or 1) * 1000
            cur = 0
        else:
            return None
        seen = True
    return total + cur if seen else None


def words_to_digits(text: str) -> str:
    """Reemplaza secuencias de números en palabras por dígitos.
    Maneja decimales hablados "diecinueve con noventa" -> 19.90.
    """
    toks = normalize_text(text).split()
    out: List[str] = []
    i = 0
    while i < len(toks):
        if toks[i] in _NUM_WORDS and toks[i] != "y":
            j = i
            while j < len(toks) and toks[j] in _NUM_WORDS:
                j += 1
            while j > i and toks[j - 1] == "y":  # "y" final no forma parte del número
                j -= 1
            val = _parse_number_words(toks[i:j])
            if val is not None:
                # decimales: "<n> con <m>" / "<n> punto <m>"
                if j + 1 < len(toks) and toks[j] in ("con", "punto", "coma"):
                    k = j + 1
                    while k < len(toks) and toks[k] in _NUM_WORDS:
                        k += 1
                    dec = _parse_number_words(toks[j + 1 : k])
                    if dec is not None and k > j + 1:
                        out.append(f"{val}.{dec:02d}" if dec < 100 else f"{val}.{dec}")
                        i = k
                        continue
                out.append(str(val))
                i = j
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


_CURRENCY = {"euros", "euro", "eur", "soles", "sol", "dolares", "dolar", "usd", "pesos",
             "centimos", "presupuesto", "maximo", "gastar", "precio", "cuesta", "pagar"}
_QTY_HINT = {"unidades", "unidad", "piezas", "paquetes", "cajas", "pares", "cantidad"}


def extract_numbers(text: str) -> Dict[str, List[float]]:
    """Extrae números y los clasifica en ``amounts`` (importes: próximos a una
    palabra de moneda/presupuesto) y ``quantities`` (resto)."""
    raw = (text or "").replace("€", " euros ").replace("$", " dolares ").replace("S/", " soles ")
    # "un/una/uno" se tratan como artículos (no como cantidad) para evitar falsos
    # importes del tipo "un presupuesto de ..."; los dígitos "1" sí cuentan.
    norm = " ".join(t for t in normalize_text(raw).split() if t not in ("un", "una", "uno"))
    toks = words_to_digits(norm).split()
    amounts, quantities = [], []
    for i, t in enumerate(toks):
        if not re.fullmatch(r"\d+(\.\d+)?", t):
            continue
        v = float(t)
        ctx = set(toks[max(0, i - 3) : i]) | set(toks[i + 1 : i + 3])
        if ctx & _CURRENCY and not (set(toks[i + 1 : i + 2]) & _QTY_HINT):
            amounts.append(v)
        else:
            quantities.append(v)
    # "media docena", "un par de"
    low = normalize_text(raw)
    for k, v in _SPECIAL.items():
        if re.search(rf"\b{k}\b", low) and not (k == "docena" and "media docena" in low):
            quantities.append(float(v))
    return {"amounts": amounts, "quantities": quantities}


# ---------------------------------------------------------------------------
# Palabras clave de producto
# ---------------------------------------------------------------------------
_STOP = set(
    """a al algo algun alguna alguno ahora ante antes asi bien cada como con cual cuales
    cuanto cuantos de del desde donde dos e el ella ellos en entonces era es esa ese eso
    esta este esto estos favor fue hay la las le lo los mas me mejor mi mis muy nada ni no
    nos o otra otro para pero poco por porque que quiero quisiera quiere se sea si sin sobre
    solo su sus tambien tengo tiene todo todos tu un una unas uno unos y ya yo
    comprar compra compro anade anadir agrega agregar agregalo pon poner cambia cambiar
    quita quitar confirma confirmo confirmar busca buscar buscame necesito dame muestrame
    carrito pedido cotizacion presupuesto maximo euros euro soles dolares unidades unidad
    gracias hola vale ok okay correcto cuesta precio mejor barato barata economico
    primero primera segundo segunda tercero ultimo esa esos esas mejor oye por favor""".split()
)


def product_keywords(text: str) -> set:
    toks = words_to_digits(text).split()
    return {t for t in toks if len(t) > 2 and t not in _STOP and t not in _NUM_WORDS
            and not re.fullmatch(r"\d+(\.\d+)?", t)}


def _multiset_diff(a: Iterable[float], b: Iterable[float]) -> int:
    from collections import Counter

    ca, cb = Counter(round(x, 2) for x in a), Counter(round(x, 2) for x in b)
    # una sustitución (40 -> 14) cuenta como un solo error
    return max(sum((ca - cb).values()), sum((cb - ca).values()))


def asr_error_analysis(gold: str, hyp: str) -> dict:
    """Compara transcripción correcta vs. ASR. Devuelve WER/CER y conteos de
    errores en cantidades, importes y productos (palabras clave)."""
    m = wer_cer(gold, hyp)
    ng, nh = extract_numbers(gold), extract_numbers(hyp)
    kg, kh = product_keywords(gold), product_keywords(hyp)
    return {
        **m,
        "gold": gold,
        "hyp": hyp,
        "qty_gold": ng["quantities"], "qty_hyp": nh["quantities"],
        "amount_gold": ng["amounts"], "amount_hyp": nh["amounts"],
        "qty_errors": _multiset_diff(ng["quantities"], nh["quantities"]),
        "amount_errors": _multiset_diff(ng["amounts"], nh["amounts"]),
        "product_missing": sorted(kg - kh),
        "product_extra": sorted(kh - kg),
        "product_errors": len(kg - kh),
        "n_qty": len(ng["quantities"]),
        "n_amount": len(ng["amounts"]),
        "n_product_kw": len(kg),
    }


def asr_error_table(pairs: Sequence[tuple]):
    """``pairs``: lista de (id, gold, hyp). DataFrame por enunciado + fila total."""
    import pandas as pd

    rows = [{"id": i, **asr_error_analysis(g, h)} for i, g, h in pairs]
    df = pd.DataFrame(rows)
    return df


def summarize_asr_errors(df) -> dict:
    if df is None or len(df) == 0:
        return {}
    import jiwer

    golds = [words_to_digits(normalize_text(x)) for x in df["gold"]]
    hyps = [words_to_digits(normalize_text(x)) for x in df["hyp"]]
    return {
        "enunciados": int(len(df)),
        "wer_corpus": float(jiwer.wer(golds, hyps)),
        "cer_corpus": float(jiwer.cer(golds, hyps)),
        "errores_cantidad": int(df["qty_errors"].sum()),
        "cantidades_total": int(df["n_qty"].sum()),
        "errores_importe": int(df["amount_errors"].sum()),
        "importes_total": int(df["n_amount"].sum()),
        "errores_producto": int(df["product_errors"].sum()),
        "palabras_producto_total": int(df["n_product_kw"].sum()),
    }


# ---------------------------------------------------------------------------
# Semejanza de hablante (ECAPA)
# ---------------------------------------------------------------------------
class SpeakerEncoder:
    """Embeddings de hablante con SpeechBrain ECAPA-TDNN (VoxCeleb, 192-d)."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", device: Optional[str] = None,
                 savedir: str = "pretrained_models/spkrec-ecapa-voxceleb") -> None:
        self.source, self.savedir = source, savedir
        self.device = device
        self._clf = None

    def load(self):
        if self._clf is None:
            import torch

            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # speechbrain < 1.0
                from speechbrain.pretrained import EncoderClassifier
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._clf = EncoderClassifier.from_hparams(
                source=self.source, savedir=self.savedir, run_opts={"device": dev}
            )
        return self._clf

    @staticmethod
    def _load_wav(path: str, start_s: float = 0.0, end_s: Optional[float] = None):
        import numpy as np
        import soundfile as sf
        import torch

        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        a = int(start_s * sr)
        b = int(end_s * sr) if end_s else len(wav)
        wav = torch.from_numpy(np.ascontiguousarray(wav[a:b]))
        if sr != 16000:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav

    def embed(self, path: str, start_s: float = 0.0, end_s: Optional[float] = None):
        import torch

        clf = self.load()
        wav = self._load_wav(path, start_s, end_s).unsqueeze(0)
        with torch.no_grad():
            e = clf.encode_batch(wav.to(clf.device)).squeeze()
        return torch.nn.functional.normalize(e, dim=0).cpu()

    def similarity(self, a: str, b: str) -> float:
        return float((self.embed(a) * self.embed(b)).sum())


def evaluate_voice_quality(tts, asr, texts: Sequence[str], out_dir: str = "results/voice_quality",
                           reference_wav: Optional[str] = None, baseline_speaker: Optional[str] = None,
                           encoder: Optional[SpeakerEncoder] = None):
    """Sintetiza ``texts`` con la voz clonada (y con un hablante de fábrica como
    línea base), transcribe con Whisper y calcula WER/CER y similitud coseno.

    Devuelve ``(detalle_df, resumen_df)``.
    """
    import pandas as pd

    from .voice import audio_duration_s

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = reference_wav or tts.reference_wav
    enc = encoder or SpeakerEncoder()
    ref_emb = enc.embed(ref)
    if baseline_speaker is None:
        spk = tts.default_speakers()
        baseline_speaker = spk[0] if spk else None
    rows = []
    for i, text in enumerate(texts):
        variants = [("clonada", None)] + ([("base_" + baseline_speaker, baseline_speaker)] if baseline_speaker else [])
        for name, spk in variants:
            path = str(out / f"q{i:02d}_{'clon' if spk is None else 'base'}.wav")
            tts.synthesize(text, path, use_default_speaker=spk)
            hyp, _ = asr.transcribe(path)
            m = wer_cer(text, hyp)
            sim = float((enc.embed(path) * ref_emb).sum())
            rows.append({"id": i, "voz": name, "texto": text, "asr": hyp, **m,
                         "sim_ref": sim, "dur_s": audio_duration_s(path), "wav": path})
    det = pd.DataFrame(rows)
    # cota superior: similitud entre dos mitades de la propia referencia
    dur = audio_duration_s(ref)
    upper = None
    if dur == dur and dur > 4:
        e1, e2 = enc.embed(ref, 0, dur / 2), enc.embed(ref, dur / 2, None)
        upper = float((e1 * e2).sum())
    summ = det.groupby("voz")[["wer", "cer", "sim_ref"]].agg(["mean", "std"]).round(3)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    if upper is not None:
        summ = pd.concat([summ, pd.DataFrame([{"voz": "referencia_mitad_vs_mitad", "sim_ref_mean": round(upper, 3)}])],
                         ignore_index=True)
    return det, summ
