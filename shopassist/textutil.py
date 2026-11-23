"""Utilidades de texto compartidas (normalización, acentos, formato de precios)."""

from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Elimina tildes y diacríticos (á->a, ñ->n) para comparar de forma robusta."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def norm(text) -> str:
    """Minúsculas + sin acentos + espacios colapsados."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", strip_accents(str(text)).lower()).strip()


def clean_html(text) -> str:
    """Quita etiquetas HTML y entidades de descripciones del catálogo."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            import math
            if isinstance(text, float) and math.isnan(text):
                return ""
        except Exception:  # pragma: no cover
            pass
        text = str(text)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


CURRENCY_WORD = {"EUR": "euros", "USD": "dólares", "PEN": "soles"}


def fmt_price(value, currency: str = "EUR") -> str:
    """Formatea un importe al estilo español: 1234.5 -> '1234,50 euros'."""
    if value is None:
        return "desconocido"
    s = f"{float(value):.2f}".replace(".", ",")
    return f"{s} {CURRENCY_WORD.get(currency, currency)}"
