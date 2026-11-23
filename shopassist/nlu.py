"""NLU determinista en español (red de seguridad y validador del LLM).

Extrae, sin modelos: números escritos con palabras ("dos", "veinte", "una docena",
"ciento cincuenta"), cantidades, presupuesto ("presupuesto de 100 euros", "máximo 50",
"no más de"), precio máximo por unidad ("de menos de 30 euros"), intención de
confirmar / negar, correcciones ("no dije trece, dije tres", "cambia la cantidad a 3",
"mejor 2"), referencias ("el segundo", "eso", "el más barato"), color y marca.

Todo trabaja sobre texto normalizado (minúsculas, sin tildes) en el que los números con
palabras se reemplazan por dígitos (``normalize_numbers``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from .textutil import norm

# ------------------------------------------------------------------ números
UNITS = {
    "cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiuno": 21, "veintiun": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
HUNDREDS = {
    "cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300, "trescientas": 300,
    "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500, "quinientas": 500, "seiscientos": 600,
    "seiscientas": 600, "setecientos": 700, "setecientas": 700, "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}
ARTICLE_LIKE = {"un", "una"}  # solo cuentan como número en contextos de cantidad
_NUMBER_WORDS = set(UNITS) | set(TENS) | set(HUNDREDS) | {"mil"}
_QTY_TRIGGER_BEFORE = {"solo", "mejor", "sean", "sea", "seran", "a", "pon", "ponme", "quiero", "dame", "cambia",
                       "anade", "anademe", "agrega", "agregame", "mete", "incluye", "necesito", "llevo", "compra",
                       "comprame", "pideme", "queria", "quise", "dije",
                       "unicamente", "nada", "mas", "menos"}

_TOKEN_SPLIT = re.compile(r"(\s+|[,;:!?¿¡()])")


def _parse_number_tokens(toks: list[str]) -> Optional[int]:
    total, current, seen = 0, 0, False
    for i, t in enumerate(toks):
        if t == "y":
            continue
        if t in UNITS:
            current += UNITS[t]
        elif t in TENS:
            current += TENS[t]
        elif t in HUNDREDS:
            current += HUNDREDS[t]
        elif t == "mil":
            current = max(current, 1) * 1000
            total += current
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


def normalize_numbers(text: str) -> str:
    """Normaliza (minúsculas, sin tildes, €->euros) y convierte números en palabras a dígitos.

    >>> normalize_numbers("Quiero dos cargadores y una docena de pilas, máximo cien euros")
    'quiero 2 cargadores y 12 de pilas, maximo 100 euros'
    """
    t = norm(text).replace("€", " euros ")
    t = re.sub(r"(\d)\s*euros", r"\1 euros", t)
    t = re.sub(r"\s+", " ", t).strip()
    # expresiones fijas
    t = re.sub(r"\bmedia docena\b", "6", t)
    t = re.sub(r"\b(?:una|1) docena\b", "12", t)
    t = re.sub(r"\b(\w+) docenas\b", lambda m: str((_parse_number_tokens([m.group(1)]) or (int(m.group(1)) if m.group(1).isdigit() else 1)) * 12) if (m.group(1) in _NUMBER_WORDS or m.group(1).isdigit()) else m.group(0), t)
    t = re.sub(r"\bun par\b", "2", t)
    t = re.sub(r"\b(\d+)\s+mil\b", lambda m: str(int(m.group(1)) * 1000), t)
    # thousands "1.000" -> 1000 ; decimales con coma "49,99" -> 49.99
    t = re.sub(r"\b(\d{1,3})\.(\d{3})\b(?![.,]\d)", r"\1\2", t)
    t = re.sub(r"\b(\d+),(\d{1,2})\b", r"\1.\2", t)
    parts = _TOKEN_SPLIT.split(t)
    out: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok in _NUMBER_WORDS:
            # acumula una secuencia de palabras numéricas (permitiendo "y" y espacios)
            seq = [tok]
            j = i + 1
            last = i
            while j < len(parts):
                p = parts[j]
                if p.isspace() or p == "":
                    j += 1
                    continue
                if p in _NUMBER_WORDS and not (p in ARTICLE_LIKE and seq[-1] not in TENS):
                    seq.append(p)
                    last = j
                    j += 1
                    continue
                if p == "y" and seq[-1] in TENS:
                    # "treinta y cinco": solo si sigue una unidad
                    k = j + 1
                    while k < len(parts) and (parts[k].isspace() or parts[k] == ""):
                        k += 1
                    if k < len(parts) and parts[k] in UNITS and UNITS[parts[k]] < 10:
                        seq.append("y")
                        j = k
                        continue
                break
            if len(seq) == 1 and tok in ARTICLE_LIKE:
                prev = next((p for p in reversed(out) if p.strip() and not p.isspace()), "")
                if prev not in _QTY_TRIGGER_BEFORE:
                    out.append(tok)
                    i += 1
                    continue
            val = _parse_number_tokens(seq)
            out.append(str(val))
            i = last + 1
            continue
        out.append(tok)
        i += 1
    t = "".join(out)
    t = re.sub(r"\bcada 1\b", "cada uno", t)
    # "50 con 99 euros" -> 50.99
    t = re.sub(r"\b(\d+) (?:euros )?con (\d{1,2})\b", lambda m: f"{m.group(1)}.{int(m.group(2)):02d}", t)
    return re.sub(r"\s+", " ", t).strip()


NUM = r"(\d+(?:\.\d+)?)"
CURR = r"(?:\s*(?:euros?|eur|soles|dolares|pesos))"
_SPEC_UNITS = r"(?:gb|tb|mb|mah|w|v|cm|mm|m|metros?|pulgadas|hz|kg|g|gr|gramos|kilos?|%|piezas? de|p|k|mp|ml|l|litros?|horas?|anos?|meses|x\d)"


@dataclass
class Item:
    """Un producto pedido en la frase (p. ej. "2 cargadores usb-c")."""
    qty: Optional[int]
    query: str
    raw: str


@dataclass
class NLUResult:
    text: str
    normalized: str
    intents: list = field(default_factory=list)
    quantity: Optional[int] = None
    budget: Optional[float] = None
    max_unit_price: Optional[float] = None
    color: Optional[str] = None
    brand: Optional[str] = None
    ordinal: Optional[int] = None          # 0-based; -1 = último
    superlative: Optional[str] = None      # "cheapest" | "best"
    deictic: bool = False                  # "eso", "ese", "lo mismo"
    correction: Optional[tuple] = None     # (viejo|None, nuevo)
    items: list = field(default_factory=list)
    product_query: str = ""
    attribute_question: Optional[str] = None
    amount: Optional[float] = None         # importe suelto ("200 euros"), sin contexto
    ordinals: list = field(default_factory=list)  # todos los ordinales ("el primero y el segundo")
    is_question: bool = False

    def has(self, intent: str) -> bool:
        return intent in self.intents

    @property
    def explicit_confirmation(self) -> bool:
        """Confirmación explícita: afirma, no niega y no pide cambios en la misma frase."""
        blocking = {"deny", "change_qty", "correction", "remove", "change_pref", "buy"}
        return "confirm" in self.intents and not (blocking & set(self.intents)) and self.budget is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["items"] = [asdict(i) for i in self.items]
        d["explicit_confirmation"] = self.explicit_confirmation
        return d


# ------------------------------------------------------------------ presupuesto / precio
_BUDGET_PATTERNS = [
    r"presupuesto\s+(?:a|hasta|en)\s+" + NUM,
    r"presupuesto\s+(?:total\s+)?(?:maximo\s+)?(?:es\s+|sera\s+|seria\s+)?(?:de\s+)?(?:hasta\s+)?(?:unos\s+)?(?:como\s+maximo\s+)?" + NUM,
    r"(?:no\s+(?:quiero\s+)?(?:gastar|pagar)\s+mas\s+de|no\s+mas\s+de|como\s+(?:mucho|maximo)|maximo(?:\s+de)?|hasta\s+un\s+maximo\s+de|tope\s+de|limite\s+de)\s+" + NUM + r"(?!\s*" + _SPEC_UNITS + r"\b)",
    r"(?:tengo|dispongo\s+de|puedo\s+gastar|quiero\s+gastar|gastar(?:\s+hasta)?|hasta|en\s+total)\s+" + NUM + CURR,
    NUM + CURR + r"\s+(?:como\s+(?:mucho|maximo)|en\s+total|de\s+presupuesto|maximo)",
]
_UNIT_PRICE_PATTERNS = [
    r"(?:de\s+|que\s+(?:cuesten?|valgan?)\s+)?(?:menos\s+de|por\s+debajo\s+de|inferior(?:es)?\s+a|que\s+no\s+pasen?\s+de)\s+" + NUM + CURR + r"?",
    NUM + CURR + r"?\s+(?:cada\s+un[oa]|por\s+unidad|la\s+unidad|c/u)",
    r"(?:maximo|como\s+mucho|hasta)\s+" + NUM + CURR + r"?\s+(?:cada\s+un[oa]|por\s+unidad|la\s+unidad)",
]


def extract_money(t: str) -> tuple[Optional[float], Optional[float], list]:
    """Devuelve (presupuesto, precio_max_unitario, spans usados) sobre texto normalizado."""
    spans = []
    unit = None
    for pat in _UNIT_PRICE_PATTERNS:
        m = re.search(pat, t)
        if m and not re.search(r"presupuesto|en total", t[max(0, m.start() - 25): m.end() + 10]):
            unit = float(m.group(1))
            spans.append(m.span())
            break
    budget = None
    for pat in _BUDGET_PATTERNS:
        for m in re.finditer(pat, t):
            if any(s <= m.start(1) < e for s, e in spans):
                continue
            tail = t[m.end(): m.end() + 20]
            if re.match(r"\s*(?:cada|por unidad|la unidad)", tail):
                continue
            budget = float(m.group(1))
            spans.append(m.span())
            break
        if budget is not None:
            break
    return budget, unit, spans


# ------------------------------------------------------------------ intenciones
_CONFIRM_START = re.compile(
    r"^(?:si|vale|ok|okay|okey|dale|perfecto|correcto|exacto|de acuerdo|adelante|claro|venga|bueno,? si|"
    r"confirmo|confirmado|confirmar|confirma|procede|hazlo|eso es|esta bien|me parece bien|afirmativo)\b"
    r"(?!\s+(?:hay|tiene|tienes|es|son|puedes|cuesta|existe|quedan?))")
_CONFIRM_ANY = re.compile(
    r"\b(?:confirmo|confirmar (?:el |la )?(?:pedido|compra)|confirma (?:el |la )?(?:pedido|compra)|"
    r"realiza(?:r)? el pedido|haz el pedido|hagan el pedido|procede con|si,? confirmo|adelante con|"
    r"compralo|compralos|tramita|registra el pedido)\b")
_DENY_START = re.compile(r"^(?:no(?!\s+(?:mas|quiero gastar|quiero pagar|gastar|pagar|supere|pase|me importa))|nop|negativo|"
                         r"cancela|cancelar|espera|todavia no|aun no|mejor no)\b")
_DENY_ANY = re.compile(r"\b(?:no confirmo|cancela(?:r)?(?: el pedido| la compra)?|no quiero (?:comprar|confirmar)|no lo compres)\b")
_CORRECTION = [
    re.compile(r"no\s+dije\s+(.+?)[,.]?\s+(?:sino|dije|era)\s+(.+)$"),
    re.compile(r"no\s+(?:es|son|era|eran)\s+(.+?)[,.]?\s+(?:sino|es|son|eran?)\s+(.+)$"),
    re.compile(r"\bno\s+(\d+)\s*,?\s*(?:sino\s+)?(\d+)\b"),
    re.compile(r"(?:quise decir|queria decir|me referia a|lo correcto es|corrijo:?|dije)\s+(.+)$"),
]
_CHANGE_QTY = [
    re.compile(r"(?:cambia|cambiar|modifica|pon|ajusta|deja)\s+(?:la\s+)?cantidad\s+(?:de\s+.+?\s+)?(?:a|en|por)\s+(\d+)"),
    re.compile(r"\b(?:mejor|prefiero|que\s+sean|que\s+sea|ahora|solo|solamente|unicamente)\s+(?:solo\s+)?(\d+)\b(?!\s*" + _SPEC_UNITS + r"\b)(?!" + CURR + r")"),
    re.compile(r"\ben\s+(?:vez|lugar)\s+de\s+\d+\s*,?\s*(?:\w+\s+)?(\d+)\b"),
    re.compile(r"\b(\d+)\s+en\s+(?:vez|lugar)\s+de\s+\d+\b"),
    re.compile(r"\b(?:quiero|dame|pon|ponme|necesito)\s+(\d+)\s+(?:mas|menos)?\s*(?:de\s+(?:ese|esos|esa|esas|ellos|ellas))\b"),
    re.compile(r"\b(?:sube|baja|aumenta|reduce)\s+(?:la\s+cantidad\s+)?(?:a|hasta)\s+(\d+)"),
]
_REMOVE = re.compile(r"\b(?:quita|quitar|quitame|elimina|eliminar|borra|borrar|saca|sacar|retira|ya no quiero)\b")
_VIEW_CART = re.compile(r"\b(?:carrito|que llevo|que tengo|mi pedido actual|resumen)\b")
_ADD = re.compile(r"\b(?:anade|anademe|agrega|agregame|mete|incluye|pon|ponme|dame|quiero|quisiera|me llevo|compra|"
                  r"comprame|llevame|pideme|necesito)\b")
_STOCK_Q = re.compile(r"\b(?:cuant[ao]s?\s+(?:unidades\s+)?(?:quedan|queda|hay|tienes|tienen|teneis)|stock|existencias|"
                      r"disponibles?|disponibilidad|quedan)\b")
_QUOTE = re.compile(r"\b(?:cotiza|cotizar|cotizacion|presupuestame|cuanto (?:seria|es|sale|me sale|cuesta todo|costaria)|"
                    r"total|precio final|cuanto pago)\b")
_COMPARE = re.compile(r"\b(?:compara|comparar|comparame|comparacion|diferencia|diferencias|cual es mejor|cual me recomiendas|versus|vs)\b")
_DETAILS = re.compile(r"\b(?:detalles|caracteristicas|especificaciones|ficha|informacion|mas info|cuentame)\b")
_BUY = re.compile(r"\b(?:quiero|quisiera|queria|necesito|busco|buscame|comprar|compra|dame|ponme|anade|agrega|anademe|"
                  r"me gustaria|me llevo|llevame|pideme|encargame|consigueme|recomiendame|muestrame|ensename|tienes|hay)\b")
_SEARCH_ONLY = re.compile(r"\b(?:busco|buscame|muestrame|ensename|que .* tienes|tienes|hay|opciones|recomiendame)\b")
_CHEAPEST = re.compile(r"\b(?:el|la|los|las)?\s*mas\s+barat[oa]s?\b|\bel de menor precio\b|\bmas economic[oa]\b")
_BEST = re.compile(r"\b(?:el|la)\s+mejor\b(?!\s+\d)")
_ORDINALS = {"primer": 0, "primero": 0, "primera": 0, "segundo": 1, "segunda": 1, "tercer": 2, "tercero": 2,
             "tercera": 2, "cuarto": 3, "cuarta": 3, "quinto": 4, "quinta": 4, "ultimo": -1, "ultima": -1}
_ORD_RE = re.compile(r"\b(?:el|la|lo|opcion|numero|producto)?\s*(primer[oa]?|segund[oa]|tercer[oa]?|cuart[oa]|quint[oa]|ultim[oa])\b")
_ORD_NUM_RE = re.compile(r"\b(?:opcion|numero|producto|el|la)\s+(?:numero\s+)?(\d)\b(?!\s*(?:" + _SPEC_UNITS + r"|" + CURR + r"))")
_DEICTIC = re.compile(r"\b(?:eso|ese|esa|esos|esas|esto|este|esta|aquel|aquella|aquello|lo mismo|el mismo|la misma)\b")
_ATTR_Q = re.compile(r"\b(?:que|cual|cuanta|cuanto|de que|tiene|es|son|lleva|incluye)\b.*\b(color|marca|capacidad|bateria|autonomia|"
                     r"potencia|peso|tamano|pantalla|dimensiones|medidas|longitud|largo|conectividad|bluetooth|usb-c|usb|"
                     r"inalambrico|wifi|material|garantia|voltaje|resistente al agua|impermeable|talla)\b")

COLOR_WORDS = {
    "negro": "negro", "negra": "negro", "negros": "negro", "negras": "negro", "blanco": "blanco", "blanca": "blanco",
    "blancos": "blanco", "blancas": "blanco", "rojo": "rojo", "roja": "rojo", "rojos": "rojo", "rojas": "rojo",
    "azul": "azul", "azules": "azul", "verde": "verde", "verdes": "verde", "amarillo": "amarillo", "amarilla": "amarillo",
    "gris": "gris", "grises": "gris", "rosa": "rosa", "rosas": "rosa", "morado": "morado", "morada": "morado",
    "naranja": "naranja", "plateado": "plateado", "plateada": "plateado", "plata": "plateado", "dorado": "dorado",
    "dorada": "dorado", "marron": "marron", "beige": "beige", "transparente": "transparente",
}

_FILLER = set("""
hola buenas buenos dias tardes noches por favor gracias quiero quisiera queria necesito busco buscame comprar compra
dame ponme anade agrega anademe me gustaria llevo llevame pideme encargame consigueme recomiendame muestrame ensename
tienes hay algun alguna algunos algunas un una unos unas el la los las de del para mi con y que me puedes podrias
tambien ademas otro otra otros otras porfa porfavor seria favor tu ti mas barato barata baratos baratas eso ese esa esos esas
esto este esta unidades unidad uds piezas presupuesto maximo euros euro eur total cada uno como mucho no gastar pagar
hasta menos debajo tengo dispongo puedo en vale ok si bueno pues entonces ahora solo color marca mejor prefiero cambia
cambialo cambiala cambiar opcion lo le les se a o tipo y comprame comprarme poner pon confirmo confirma confirmar
al carrito mete incluye agregame tiene tienen lleva trae viene sube subir aumenta baja bajar quedan queda existencias disponible disponibles
cancela cancelar dije dijiste cantidad sean sea vez lugar pero es son quita quitar quitame elimina eliminar saca sacar
borra borrar retira compara comparar comparame opciones primero primera segundo segunda tercero tercera cuarto cuarta
quinto quinta ultimo ultima primer tercer stock dimelo dime cual cuales cuanto cuanta cuesta cuestan precio quiere
favor perfecto correcto exacto adelante claro venga hazlo procede ya entonces sino era eran corrijo quise decir
queria referia mismo misma igual ver carrito resumen detalles caracteristicas informacion cuentame anadir agregar
""".split())


def _strip_spans(t: str, spans: list) -> str:
    for s, e in sorted(spans, reverse=True):
        t = t[:s] + " " + t[e:]
    return re.sub(r"\s+", " ", t).strip()


def clean_product_query(seg: str, qty_span: Optional[tuple] = None) -> str:
    """Quita verbos de compra, la cantidad e importes; conserva el núcleo del producto
    (incluidos números de especificación como "iphone 13" o "128 gb")."""
    if qty_span:
        seg = seg[: qty_span[0]] + " " + seg[qty_span[1]:]
    seg = re.sub(r"\b\d+(?:\.\d+)?" + CURR, " ", seg)
    seg = re.sub(r"\b\d+\s+(?:unidades|uds|unidad|piezas)\b", " ", seg)
    toks = re.findall(r"[a-z0-9][a-z0-9\-\.]*", seg)
    toks = [t for t in toks if t not in _FILLER]
    if all(re.fullmatch(r"[\d.\-]+", t) for t in toks):
        return ""
    return " ".join(toks).strip(" .")


_ITEM_SPLIT = re.compile(r"\s*(?:,|\by\b|\bademas\b|\btambien\b|\bmas\b)\s+(?=(?:\d+|un|una|unos|unas)\s+[a-z])")
_QTY_IN_SEG = re.compile(r"(?:^|\b(?:quiero|quisiera|necesito|comprar|compra|dame|ponme|anade|agrega|llevo|pideme|"
                         r"busco|y|solo|mejor|tambien|ademas|me|unos|unas)\s+)(\d+)\s+(?!" + _SPEC_UNITS + r"\b)(?!" + CURR.strip() + r")([a-z])")


def _segment_qty(seg: str) -> tuple[Optional[int], Optional[tuple]]:
    m = re.search(r"\b(\d+)\s+(?:unidades|uds|unidad|piezas)\b", seg)
    if m:
        return int(m.group(1)), None
    m = _QTY_IN_SEG.search(seg)
    if m:
        v = int(m.group(1))
        return (v, m.span(1)) if 0 < v <= 1000 else (None, None)
    return None, None


def parse(text: str, known_brands: Optional[set] = None) -> NLUResult:
    """Analiza una intervención del usuario."""
    t = normalize_numbers(text)
    r = NLUResult(text=text, normalized=t)
    r.is_question = "?" in text or bool(re.match(r"^(?:que|cual|cuanto|cuanta|como|tiene|tienes|es|hay|donde)\b", t))
    budget, unit, money_spans = extract_money(t)
    r.budget, r.max_unit_price = budget, unit
    m = re.search(NUM + CURR, t)
    if m:
        r.amount = float(m.group(1))
    intents = []
    # correcciones
    for pat in _CORRECTION:
        m = pat.search(t)
        if m:
            if m.lastindex == 2:
                r.correction = (m.group(1).strip(" .,"), m.group(2).strip(" .,"))
            else:
                r.correction = (None, m.group(1).strip(" .,"))
            intents.append("correction")
            break
    # cambio de cantidad
    for pat in _CHANGE_QTY:
        m = pat.search(t)
        if m and not any(s <= m.start(1) < e for s, e in money_spans):
            r.quantity = int(m.group(1))
            intents.append("change_qty")
            break
    no_money = _strip_spans(t, money_spans)
    # confirmación / negación
    ts = t.lstrip("¡ ")
    if _DENY_START.search(ts) or _DENY_ANY.search(t):
        intents.append("deny")
    if (_CONFIRM_START.search(ts) or _CONFIRM_ANY.search(t)) and not _DENY_ANY.search(t):
        intents.append("confirm")
    if _REMOVE.search(t):
        intents.append("remove")
    if _VIEW_CART.search(t) and not (_ADD.search(t) or _REMOVE.search(t)):
        intents.append("view_cart")
    if _ADD.search(t):
        intents.append("add")
    if r.is_question and _STOCK_Q.search(t):
        intents.append("stock_query")
    if _QUOTE.search(t) and "presupuesto de" not in t:
        intents.append("quote")
    if _COMPARE.search(t):
        intents.append("compare")
    if _DETAILS.search(t):
        intents.append("details")
    m = _ATTR_Q.search(t)
    if m and r.is_question:
        r.attribute_question = m.group(1)
        intents.append("attribute")
    # referencias
    if _CHEAPEST.search(t):
        r.superlative = "cheapest"
    elif _BEST.search(t):
        r.superlative = "best"
    m = _ORD_RE.search(t)
    if m:
        r.ordinal = _ORDINALS.get(m.group(1), _ORDINALS.get(m.group(1).rstrip("oa"), None))
        if r.ordinal is None:
            for k, v in _ORDINALS.items():
                if m.group(1).startswith(k):
                    r.ordinal = v
                    break
    else:
        m = _ORD_NUM_RE.search(no_money)
        if m and "change_qty" not in intents and 1 <= int(m.group(1)) <= 9 and \
                re.search(r"\b(?:opcion|numero|producto)\b", no_money):
            r.ordinal = int(m.group(1)) - 1
    r.deictic = bool(_DEICTIC.search(t))
    # color / marca
    for w in re.findall(r"[a-z]+", no_money):
        if w in COLOR_WORDS:
            r.color = COLOR_WORDS[w]
            break
    m = re.search(r"\b(?:de\s+)?(?:la\s+)?marca\s+([a-z0-9][a-z0-9\-]+)", t)
    if m:
        r.brand = m.group(1)
    elif known_brands:
        for w in re.findall(r"[a-z0-9\-]{3,}", no_money):
            if w in known_brands:
                r.brand = w
                break
    # productos / cantidades
    segs = [s for s in _ITEM_SPLIT.split(no_money) if s.strip()]
    items, seg_qtys = [], []
    for s in segs:
        q, span = _segment_qty(s)
        if q is not None:
            seg_qtys.append(q)
        pq = clean_product_query(s, span)
        if pq:
            items.append(Item(qty=q, query=pq, raw=s))
    r.items = items
    r.product_query = " ".join(i.query for i in items)
    if r.quantity is None:
        qs = [i.qty for i in items if i.qty is not None] or seg_qtys
        if qs:
            r.quantity = qs[0]
    r.ordinals = [(_ORDINALS.get(m.group(1)) if m.group(1) in _ORDINALS else
                   next((v for k, v in _ORDINALS.items() if m.group(1).startswith(k)), None))
                  for m in _ORD_RE.finditer(t)]
    r.ordinals = [o for o in r.ordinals if o is not None]
    if items and (_BUY.search(t) or not intents) and "change_qty" not in intents:
        intents.append("buy")
        if _SEARCH_ONLY.search(t) and not re.search(r"\b(?:quiero|quisiera|necesito|comprar|dame|ponme|anade|agrega|me llevo)\b", t):
            intents.append("search_only")
    if ("buy" not in intents and (r.color or r.brand)) and not items and "change_qty" not in intents:
        intents.append("change_pref")
    elif r.color and items and re.search(r"\b(?:mejor|prefiero|cambia|cambialo|cambiala|en vez|en lugar)\b", t) and \
            all(set(i.query.split()) <= {r.color, "color", *(COLOR_WORDS.keys())} | ({r.brand} if r.brand else set()) for i in items):
        intents = [i for i in intents if i != "buy"] + ["change_pref"]
        r.items, r.product_query = [], ""
    # pregunta sobre un atributo no previsto de un producto referenciado: "¿el primero tiene X?"
    if r.is_question and "attribute" not in intents and "stock_query" not in intents and "add" not in intents \
            and (r.ordinal is not None or r.deictic) and r.product_query and \
            re.search(r"\b(?:tiene|tienen|es|son|lleva|incluye|trae|viene|sirve|funciona)\b", t):
        r.attribute_question = r.product_query
        intents = [i for i in intents if i not in ("buy", "search_only")] + ["attribute"]
        r.items, r.product_query = [], ""
    if budget is not None and "set_budget" not in intents:
        intents.append("set_budget")
    if unit is not None:
        intents.append("set_unit_price")
    r.intents = intents
    return r


def apply_correction(original: str, correction: tuple) -> str:
    """Aplica una corrección ("no dije X, dije Y") sobre una frase normalizada."""
    old, new = correction
    orig = normalize_numbers(original)
    new_n = normalize_numbers(new)
    old_n = normalize_numbers(old) if old else None
    if old_n and re.search(r"\b" + re.escape(old_n) + r"\b", orig):
        return re.sub(r"\b" + re.escape(old_n) + r"\b", new_n, orig, count=1)
    new_nums = re.findall(r"\d+(?:\.\d+)?", new_n)
    if new_nums and len(new_n.split()) <= 3:
        # sustituye la cantidad o el importe según el contexto
        if re.search(r"euros?|presupuesto", new_n):
            b, _, sp = extract_money(orig)
            if b is not None:
                s, e = sp[-1]
                seg = orig[s:e]
                return orig[:s] + re.sub(r"\d+(?:\.\d+)?", new_nums[0], seg, count=1) + orig[e:]
        p = parse(orig)
        q = p.quantity
        if q is not None:
            return re.sub(r"\b" + str(q) + r"\b", new_nums[0], orig, count=1)
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", orig)
        if nums:
            return re.sub(r"\b" + re.escape(nums[0]) + r"\b", new_nums[0], orig, count=1)
    if old_n is None and len(new_n.split()) > 3:
        return new_n
    return orig if not old_n else orig.replace(old_n, new_n)
