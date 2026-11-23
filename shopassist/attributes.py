"""Extracción de atributos VERIFICABLES en el texto del producto.

Solo se informa un atributo si aparece literalmente en el título, viñetas, descripción o
campos estructurados (marca, color). En caso contrario el valor es ``"desconocido"``:
el agente nunca debe completar atributos con conocimiento propio del LLM.
"""

from __future__ import annotations

import re
from typing import Optional

from .textutil import norm

UNKNOWN = "desconocido"
NUM = r"(\d+(?:[.,]\d+)?)"

ATTRIBUTE_KEYS = ["marca", "color", "capacidad", "bateria", "potencia", "peso", "tamano",
                  "dimensiones", "longitud", "conectividad", "material", "garantia", "voltaje"]

ALIASES = {
    "brand": "marca", "marca": "marca",
    "color": "color", "colour": "color",
    "capacidad": "capacidad", "almacenamiento": "capacidad", "memoria": "capacidad", "capacity": "capacidad",
    "bateria": "bateria", "autonomia": "bateria", "mah": "bateria", "battery": "bateria",
    "potencia": "potencia", "vatios": "potencia", "watts": "potencia", "wattage": "potencia",
    "peso": "peso", "weight": "peso",
    "tamano": "tamano", "pantalla": "tamano", "pulgadas": "tamano", "size": "tamano", "talla": "tamano",
    "dimensiones": "dimensiones", "medidas": "dimensiones",
    "longitud": "longitud", "largo": "longitud", "cable": "longitud", "length": "longitud",
    "conectividad": "conectividad", "conexion": "conectividad", "bluetooth": "conectividad",
    "usb": "conectividad", "usb-c": "conectividad", "inalambrico": "conectividad", "wifi": "conectividad",
    "connectivity": "conectividad",
    "material": "material",
    "garantia": "garantia", "warranty": "garantia",
    "voltaje": "voltaje", "voltios": "voltaje", "voltage": "voltaje",
}

_MATERIALS = ["acero inoxidable", "aluminio", "silicona", "cuero", "piel sintetica", "piel", "algodon",
              "poliester", "nailon", "nylon", "plastico", "abs", "tpu", "policarbonato", "madera",
              "bambu", "vidrio templado", "vidrio", "cristal", "ceramica", "metal", "goma", "lana", "acrilico"]
_CONNECTIVITY = [
    (r"bluetooth\s*(\d(?:\.\d)?)", lambda m: f"Bluetooth {m.group(1)}"),
    (r"\bbluetooth\b", lambda m: "Bluetooth"),
    (r"\busb[\s-]?(tipo[\s-]?)?c\b|\btype[\s-]?c\b", lambda m: "USB-C"),
    (r"\bmicro[\s-]?usb\b", lambda m: "micro-USB"),
    (r"\busb\s*3(\.\d)?\b", lambda m: "USB 3"),
    (r"\busb\b", lambda m: "USB"),
    (r"\bwi-?fi\b", lambda m: "WiFi"),
    (r"\bnfc\b", lambda m: "NFC"),
    (r"\b(jack|minijack)\b|3[.,]5\s?mm", lambda m: "jack 3,5 mm"),
    (r"\blightning\b", lambda m: "Lightning"),
    (r"\bhdmi\b", lambda m: "HDMI"),
    (r"\binalambric[oa]s?\b|\bwireless\b", lambda m: "inalámbrico"),
]
_COLORS = ["negro", "negra", "blanco", "blanca", "rojo", "roja", "azul", "verde", "amarillo", "amarilla", "gris",
           "rosa", "morado", "morada", "naranja", "plateado", "plateada", "plata", "dorado", "dorada", "marron",
           "beige", "transparente", "multicolor", "black", "white", "red", "blue", "green", "grey", "gray",
           "pink", "silver", "gold"]


def product_fulltext(product: dict) -> str:
    return " . ".join(str(product.get(k) or "") for k in ("title", "bullet_point", "description"))


def _first(pattern: str, text: str, fmt=None) -> Optional[str]:
    m = re.search(pattern, text)
    if not m:
        return None
    return fmt(m) if fmt else m.group(0)


def _num(s: str) -> str:
    return s.replace(".", ",")


def extract_attributes(product: dict) -> dict:
    """Devuelve ``{atributo: valor | "desconocido"}`` para ``ATTRIBUTE_KEYS``."""
    t = norm(product_fulltext(product))
    out = {k: UNKNOWN for k in ATTRIBUTE_KEYS}
    brand = (product.get("brand") or "").strip()
    if brand and norm(brand) not in ("desconocido", "nan", "none", ""):
        out["marca"] = brand
    color = (product.get("color") or "").strip()
    if color and norm(color) not in ("desconocido", "nan", "none", ""):
        out["color"] = color
    else:
        m = re.search(r"\b(" + "|".join(_COLORS) + r")s?\b", norm(product.get("title") or ""))
        if m:
            out["color"] = m.group(1)
    v = _first(NUM + r"\s?(tb|gb)\b", t, lambda m: f"{_num(m.group(1))} {m.group(2).upper()}")
    if v:
        out["capacidad"] = v
    v = _first(r"(\d{3,6})\s?mah\b", t, lambda m: f"{m.group(1)} mAh")
    if v:
        out["bateria"] = v
    else:
        v = _first(r"(\d{1,3})\s?(horas|h)\s+de\s+(bateria|autonomia|reproduccion|uso)", t,
                   lambda m: f"{m.group(1)} horas de {m.group(3)}")
        if v:
            out["bateria"] = v
    v = _first(NUM + r"\s?(w|vatios|watts?)\b", t, lambda m: f"{_num(m.group(1))} W")
    if v:
        out["potencia"] = v
    m = re.search(r"\bpeso\b[^0-9]{0,20}" + NUM + r"\s?(kg|g|gr|gramos|kilos?)\b", t) or \
        re.search(NUM + r"\s?(kg|kilos?|gramos|gr)\b", t)
    if m:
        unit = {"gramos": "g", "gr": "g", "kilo": "kg", "kilos": "kg"}.get(m.group(2), m.group(2))
        out["peso"] = f"{_num(m.group(1))} {unit}"
    v = _first(NUM + r"\s?(pulgadas|\"|”|'')", t, lambda m: f"{_num(m.group(1))} pulgadas")
    if v:
        out["tamano"] = v
    else:
        v = _first(r"\btalla\s+([a-z0-9]{1,4})\b", t, lambda m: f"talla {m.group(1).upper()}")
        if v:
            out["tamano"] = v
    v = _first(NUM + r"\s?x\s?" + NUM + r"(?:\s?x\s?" + NUM + r")?\s?(cm|mm|m)\b", t,
               lambda m: " x ".join(_num(g) for g in m.groups()[:3] if g) + f" {m.groups()[3]}")
    if v:
        out["dimensiones"] = v
    v = _first(r"(?:cable|longitud|largo)[^0-9]{0,25}" + NUM + r"\s?(m|metros?|cm)\b", t,
               lambda m: f"{_num(m.group(1))} {'m' if m.group(2).startswith('m') else 'cm'}")
    if v:
        out["longitud"] = v
    conns = []
    for pat, fmt in _CONNECTIVITY:
        mm = re.search(pat, t)
        if mm:
            val = fmt(mm)
            if not any(val.split()[0] in c for c in conns):
                conns.append(val)
    if conns:
        out["conectividad"] = ", ".join(conns)
    mats = [m for m in _MATERIALS if re.search(r"\b" + m + r"\b", t)]
    if mats:
        dedup = [m for m in mats if not any(m != o and m in o for o in mats)]
        out["material"] = ", ".join(dedup[:3])
    v = _first(r"garantia\s+(?:de\s+)?(\d+)\s+(anos?|meses)", t, lambda m: f"{m.group(1)} {m.group(2).replace('anos', 'años').replace('ano', 'año')}")
    if v:
        out["garantia"] = v
    v = _first(NUM + r"\s?(v|voltios)\b", t, lambda m: f"{_num(m.group(1))} V")
    if v:
        out["voltaje"] = v
    return out


def canonical_attr(attr: str) -> Optional[str]:
    a = norm(attr).replace("ñ", "n")
    return ALIASES.get(a)


def answer_attribute(product: dict, attr: str) -> str:
    """Valor verificable del atributo pedido o ``"desconocido"``.

    Para atributos no previstos se busca la palabra en el texto y se devuelve el fragmento
    literal (sustento textual); si no aparece, ``"desconocido"``.
    """
    key = canonical_attr(attr)
    if key is not None:
        return extract_attributes(product).get(key, UNKNOWN)
    word = norm(attr)
    if not word:
        return UNKNOWN
    text = product_fulltext(product)
    nt = norm(text)
    i = nt.find(word)
    if i < 0:
        # todas las palabras de contenido deben aparecer (p. ej. "cancelacion ruido")
        words = [w for w in re.findall(r"[a-z0-9]+", word) if len(w) > 2 and w not in ("tiene", "para", "con")]
        if not words or not all(re.search(r"\b" + re.escape(w[:max(4, len(w) - 2)]), nt) for w in words):
            return UNKNOWN
        i = re.search(r"\b" + re.escape(words[0][:max(4, len(words[0]) - 2)]), nt).start()
    start = max(nt.rfind(".", 0, i) + 1, i - 60)
    end_dot = nt.find(".", i)
    end = min(end_dot if end_dot > 0 else len(nt), i + 100)
    return "según la ficha: \"" + text[start:end].strip() + "\""
