"""Demo EXTRA: el mismo agente comprando en la tienda real de merch (soyHouston256/store).

NO forma parte de la evaluación oficial (que sigue siendo ESCI + ``Store`` simulada, según
el PDF del laboratorio). Este módulo solo añade un backend alternativo con la MISMA interfaz
pública que :class:`shopassist.store.Store`, de modo que ``ShoppingAgent``, las herramientas
y ``scenarios.run_all`` funcionan sin cambios.

API pública usada (Express + Mongoose, ``api/src/routes``):

* ``GET  {base}/api/products``       -> lista de ``ProductDTO`` publicados (ordenados por nombre).
* ``GET  {base}/api/products/:id``   -> un ``ProductDTO`` (404 si no existe o no está publicado).
* ``POST {base}/api/orders``         -> crea un pedido REAL. Cuerpo validado con zod::

      {"user": {"dni": str, "name": str, "phone": str},          # obligatorios, no vacíos
       "items": [{"productId": str, "quantity": int >= 1,
                  "cut"?: "hombre"|"mujer", "size"?: str, "color"?: str, "logoPosition"?: str}]}

  Respuesta 201 ``OrderDTO`` ``{id, user, items, total, status, createdAt}``; el servidor
  recalcula precios y total (cualquier ``total`` del cliente se descarta). Errores con el
  sobre ``{"error": {"code", "message", "details"?}}`` (400 VALIDATION si un producto está
  agotado / no publicado o el corte no es válido).

Diseño:

* Catálogo: se descarga UNA vez (``refresh_catalog`` lo vuelve a sincronizar) y se convierte
  a un DataFrame con el mismo esquema que el catálogo ESCI (product_id, title, description,
  bullet_point, brand, color) para que ``build_retriever`` funcione igual.
* Precios: los de la API (moneda configurable, PEN por defecto).
* Stock: la API no tiene stock numérico, así que se simula en la "sombra" SQLite heredada de
  ``Store`` (semilla reproducible, ``restore``; ``soldOut`` -> 0). Carritos, cotizaciones,
  idempotencia y la revalidación atómica de ``Store.place_order`` se reutilizan tal cual.
* Antes de confirmar se vuelve a consultar cada producto en la API: si ahora está agotado o
  cambió el precio se actualiza la sombra y ``Store.place_order`` rechaza el pedido con el
  problema correspondiente (el agente propone alternativa o re-cotiza).
* Variantes (talla, color, corte): la API las acepta como opcionales, pero la tienda las
  necesita para preparar el pedido. Se guardan por línea de carrito, se congelan en la
  cotización y la cotización es inválida (problema ``variant``) si falta alguna con más de
  una opción; así el agente pregunta.
* Pedidos: por defecto **DRY-RUN** (se registra localmente y se guarda el payload que se
  enviaría). Solo con ``allow_real_orders=True`` Y datos del cliente se hace el POST real.
  Nunca se hace POST dos veces para la misma cotización: la tabla ``remote_orders`` (que
  ``restore`` NO borra) guarda la correspondencia clave -> pedido remoto.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from .nlu import COLOR_WORDS, _FILLER
from .retrieval import canonical_color
from .store import Store, StoreError, synth_stock

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "tienda_real_fixture.json"

# Tallas globales por corte (copia de api/src/models/sizes.ts: los polos ya no usan su `sizes`).
SIZES_BY_CUT = {"hombre": ["S", "M", "L", "XL", "XXL"], "mujer": ["XS", "S", "M", "L", "XL"]}
CUTS = ("hombre", "mujer")
TYPE_LABEL = {"polo": "Polo", "taza": "Taza", "mousepad": "Mousepad"}
TYPE_SYNONYMS = {
    "polo": "polo (camiseta, polera, playera) de algodón",
    "taza": "taza (mug) de cerámica",
    "mousepad": "mousepad (alfombrilla para ratón o mouse pad)",
}
LOGO_LABEL = {"pocket": "bolsillo", "chest": "pecho", "back": "espalda", "front-back": "frente y espalda"}
# Palabras que, solas, describen un TIPO de producto (consulta demasiado genérica en esta tienda).
_GENERIC_WORDS = {"polo", "polos", "camiseta", "camisetas", "polera", "poleras", "playera", "playeras", "taza",
                  "tazas", "mug", "mugs", "mousepad", "mousepads", "alfombrilla", "merch", "producto", "productos"}

# Colores de referencia (RGB) -> nombre canónico en español (coincide con retrieval._COLOR_SYNONYMS).
_PALETTE = [
    ("negro", (0, 0, 0)), ("negro", (30, 35, 40)), ("blanco", (255, 255, 255)), ("blanco", (245, 247, 250)),
    ("gris", (128, 128, 128)), ("gris", (95, 95, 90)), ("gris", (74, 97, 112)), ("gris", (203, 213, 225)),
    ("azul", (0, 0, 255)), ("azul", (172, 203, 221)), ("azul", (147, 197, 253)), ("azul", (15, 40, 120)),
    ("verde", (0, 160, 0)), ("verde", (208, 255, 237)), ("verde", (167, 243, 208)), ("verde", (163, 195, 189)),
    ("rojo", (220, 20, 20)), ("amarillo", (253, 230, 138)), ("amarillo", (255, 220, 0)),
    ("beige", (227, 219, 198)), ("rosa", (255, 182, 193)), ("morado", (128, 0, 128)), ("naranja", (255, 140, 0)),
    ("marron", (120, 70, 30)),
]

_SIZE_WORDS = {"xxl": "XXL", "xl": "XL", "xs": "XS", "s": "S", "m": "M", "l": "L", "pequena": "S", "chica": "S",
               "mediana": "M", "grande": "L", "extra grande": "XL"}
_SIZE_RE = re.compile(r"\btalla\s+(extra grande|xxl|xl|xs|s|m|l|pequena|chica|mediana|grande)\b")
_CUT_RE = re.compile(r"\b(?:corte|modelo|para|de)\s+(hombre|mujer|dama|caballero|varon)\b")
_CUT_WORDS = {"hombre": "hombre", "caballero": "hombre", "varon": "hombre", "mujer": "mujer", "dama": "mujer"}


class RemoteStoreError(StoreError):
    """Fallo de red o respuesta inesperada de la API de la tienda real."""


def hex_to_color_name(value: str) -> str:
    """'#000' -> 'negro', '#ACCBDD' -> 'azul'. Si no es hexadecimal se normaliza el nombre."""
    v = (value or "").strip()
    m = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", v)
    if not m:
        return canonical_color(v) or v.lower()
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return min(_PALETTE, key=lambda p: sum((a - b) ** 2 for a, b in zip(rgb, p[1])))[0]


def load_fixture_products(path: Optional[str] = None) -> list[dict]:
    """Copia local (offline) con la forma exacta de ``GET /api/products`` (derivada de products.json)."""
    with open(path or FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_base_url(base_url: Optional[str] = None, cfg=None) -> Optional[str]:
    """Argumento explícito > ``cfg.remote_store_url`` > variable de entorno ``STORE_API_URL``."""
    url = base_url or getattr(cfg, "remote_store_url", None) or os.environ.get("STORE_API_URL") or None
    if not url:
        return None
    url = url.strip().rstrip("/")
    return url[:-4] if url.endswith("/api") else url


# ====================================================================== transporte HTTP
class HttpTransport:
    """Transporte real con ``urllib`` (sin dependencias). ``request`` -> ``(status, body_json)``."""

    def __init__(self, timeout: float = 10.0, user_agent: str = "shopassist-demo/1.0"):
        self.timeout = timeout
        self.user_agent = user_agent

    def request(self, method: str, url: str, payload: Optional[dict] = None) -> tuple[int, Any]:
        import urllib.error
        import urllib.request

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers={
            "Accept": "application/json", "Content-Type": "application/json", "User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as e:  # 4xx/5xx: el servidor respondió
            status, raw = e.code, e.read()
        except (urllib.error.URLError, OSError) as e:
            raise RemoteStoreError(f"No se pudo contactar con {url}: {e}") from e
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            body = {"raw": raw.decode("utf-8", "replace")[:500]}
        return status, body


class FakeTransport:
    """API falsa en memoria que imita las rutas públicas (pruebas y demo offline).

    Valida el cuerpo de ``POST /api/orders`` como el zod del servidor y registra cada
    llamada en ``calls``; ``set_product`` simula cambios en la tienda (precio, ``soldOut``).
    """

    def __init__(self, products: Optional[list] = None, fail_post: bool = False):
        self.products = {p["id"]: dict(p) for p in (products if products is not None else load_fixture_products())}
        self.calls: list[dict] = []
        self.orders: list[dict] = []
        self.fail_post = fail_post

    @property
    def posts(self) -> list[dict]:
        return [c for c in self.calls if c["method"] == "POST"]

    def set_product(self, pid: str, **changes):
        self.products[pid].update(changes)

    @staticmethod
    def _err(status: int, code: str, message: str, details=None):
        return status, {"error": {"code": code, "message": message, **({"details": details} if details else {})}}

    def request(self, method: str, url: str, payload: Optional[dict] = None) -> tuple[int, Any]:
        method = method.upper()
        self.calls.append({"method": method, "url": url, "payload": json.loads(json.dumps(payload))
                           if payload is not None else None})
        path = re.sub(r"^https?://[^/]+", "", url).split("?")[0].rstrip("/")
        if method == "GET" and path == "/api/products":
            pubs = [p for p in self.products.values() if p.get("published", True)]
            return 200, sorted((dict(p) for p in pubs), key=lambda p: p["name"])
        m = re.fullmatch(r"/api/products/([^/]+)", path)
        if method == "GET" and m:
            key = m.group(1)
            p = next((p for p in self.products.values() if p.get("published", True)
                      and (p["id"] == key or p.get("slug") == key.lower())), None)
            return (200, dict(p)) if p else self._err(404, "NOT_FOUND", "Product not found")
        if method == "POST" and path == "/api/orders":
            if self.fail_post:
                return self._err(500, "INTERNAL", "Internal server error")
            return self._create_order(payload or {})
        return self._err(404, "NOT_FOUND", "Resource not found")

    def _create_order(self, body: dict):
        user, items, details = body.get("user") or {}, body.get("items"), []
        for f in ("dni", "name", "phone"):
            if not isinstance(user.get(f), str) or not user[f].strip():
                details.append({"field": f"user.{f}", "message": "Required"})
        if not isinstance(items, list) or not items:
            details.append({"field": "items", "message": "Array must contain at least 1 element(s)"})
            items = []
        for i, it in enumerate(items):
            if not isinstance(it.get("productId"), str) or not it["productId"]:
                details.append({"field": f"items.{i}.productId", "message": "Required"})
            q = it.get("quantity")
            if isinstance(q, bool) or not isinstance(q, int) or q < 1:
                details.append({"field": f"items.{i}.quantity", "message": "Invalid quantity"})
            if "cut" in it and it["cut"] not in CUTS:
                details.append({"field": f"items.{i}.cut", "message": "Invalid enum value"})
        if details:
            return self._err(400, "VALIDATION", "Invalid request body", details)
        total, snap = 0.0, []
        for i, it in enumerate(items):
            p = self.products.get(it["productId"])
            if p is None or not p.get("published", True):
                return self._err(400, "VALIDATION", "Order references an unavailable product",
                                 [{"field": f"items.{i}.productId", "message": "Product is not available"}])
            if p.get("soldOut"):
                return self._err(400, "VALIDATION", "Order references a sold-out product",
                                 [{"field": f"items.{i}.productId", "message": "Product is sold out"}])
            total += p["price"] * it["quantity"]
            snap.append({"productId": p["id"], "name": p["name"], "type": p["type"], "price": p["price"],
                         "quantity": it["quantity"], **{k: it[k] for k in ("cut", "size", "color", "logoPosition")
                                                         if k in it}})
        order = {"id": uuid.uuid4().hex[:8].upper(), "user": user, "items": snap, "total": total,
                 "status": "pendiente", "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())}
        self.orders.append(order)
        return 201, order


# ====================================================================== adaptador
def _type_of(p: dict) -> str:
    return p.get("type") or "polo"


def _title(p: dict) -> str:
    label = TYPE_LABEL.get(_type_of(p), _type_of(p).capitalize())
    name = (p.get("name") or "").strip()
    return name if name.lower().startswith(label.lower()) else f"{label} {name}"


def _theme(p: dict) -> str:
    label = TYPE_LABEL.get(_type_of(p), "")
    name = (p.get("name") or "").strip()
    return name[len(label):].strip() if label and name.lower().startswith(label.lower()) else name


def _cuts_for(p: dict) -> list[str]:
    """Igual que ``cutsFor`` del servidor: solo los polos tienen corte (por defecto 'hombre')."""
    if _type_of(p) != "polo":
        return []
    valid = [c for c in (p.get("cuts") or []) if c in CUTS]
    return list(dict.fromkeys(valid)) or ["hombre"]


def _sizes_for(p: dict, cut: Optional[str] = None) -> list[str]:
    if _type_of(p) == "polo":
        cuts = [cut] if cut in CUTS else _cuts_for(p)
        return list(dict.fromkeys(s for c in cuts for s in SIZES_BY_CUT[c]))
    return list(p.get("sizes") or [])


def _colors_for(p: dict) -> list[dict]:
    out, seen = [], set()
    for raw in p.get("colors") or []:
        name = hex_to_color_name(raw)
        if name not in seen:  # dos tonos que se llaman igual: se ofrece el primero
            seen.add(name)
            out.append({"name": name, "value": raw})
    return out


def products_to_catalog(products: list[dict], brand: str = "DevHaus") -> pd.DataFrame:
    """ProductDTO[] -> DataFrame con el esquema del catálogo ESCI (texto en español)."""
    rows = []
    for p in products:
        t = _type_of(p)
        colors = ", ".join(c["name"] for c in _colors_for(p))
        sizes = ", ".join(_sizes_for(p))
        cuts = ", ".join(_cuts_for(p))
        logos = ", ".join(LOGO_LABEL.get(x, x) for x in (p.get("logoPositions") or []))
        desc = f"{TYPE_SYNONYMS.get(t, t)} de la tienda {brand} con diseño de {_theme(p)}."
        bullets = [f"Colores disponibles: {colors}" if colors else "",
                   f"Tallas: {sizes}" if sizes else "", f"Corte: {cuts}" if cuts else "",
                   f"Posición del logo: {logos}" if logos else "", "Agotado" if p.get("soldOut") else ""]
        rows.append({"product_id": str(p["id"]), "title": _title(p), "description": desc,
                     "bullet_point": ". ".join(b for b in bullets if b), "brand": brand, "color": colors,
                     "product_type": t})
    return pd.DataFrame(rows, columns=["product_id", "title", "description", "bullet_point", "brand", "color",
                                       "product_type"])


class RemoteStoreAdapter(Store):
    """Backend sobre la API pública de soyHouston256/store con la interfaz de ``Store``.

    ``RemoteStoreAdapter(base_url, transport=None, db_path=':memory:', seed=42, currency='PEN',
    allow_real_orders=False, customer=None)``. Sin ``allow_real_orders=True`` y ``customer``
    completo NUNCA se crea un pedido real (modo dry-run).
    """

    prefer_same_type = True   # el agente no sustituye un polo agotado por una taza

    def __init__(self, base_url: Optional[str] = None, transport=None, db_path: str = ":memory:",
                 seed: int = 42, currency: str = "PEN", allow_real_orders: bool = False,
                 customer: Optional[dict] = None, brand: str = "DevHaus", products: Optional[list] = None):
        self.base_url = resolve_base_url(base_url) or ("http://fake-store.local" if transport is not None else None)
        if self.base_url is None:
            raise RemoteStoreError("Falta la URL de la API: pasa base_url, Config.remote_store_url o STORE_API_URL")
        self.transport = transport or HttpTransport()
        self.allow_real_orders = bool(allow_real_orders)
        self.customer = customer
        self.brand = brand
        self._epoch = uuid.uuid4().hex[:8]
        self._api: dict[str, dict] = {}
        api_products = products if products is not None else self._fetch_products()
        self._api = {str(p["id"]): dict(p) for p in api_products}
        self.catalog_df = products_to_catalog(list(self._api.values()), brand)
        super().__init__(db_path, self.catalog_df, seed=seed, currency=currency, zero_stock_frac=0.0)

    @classmethod
    def from_config(cls, cfg, transport=None, **kw) -> "RemoteStoreAdapter":
        kw.setdefault("currency", getattr(cfg, "remote_store_currency", "PEN"))
        kw.setdefault("seed", cfg.SEED)
        return cls(resolve_base_url(kw.pop("base_url", None), cfg), transport=transport, **kw)

    # ------------------------------------------------------------------ HTTP
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _call(self, method: str, path: str, payload: Optional[dict] = None) -> tuple[int, Any]:
        return self.transport.request(method, self._url(path), payload)

    def _fetch_products(self) -> list[dict]:
        status, body = self._call("GET", "/api/products")
        if status != 200 or not isinstance(body, list):
            raise RemoteStoreError(f"GET /api/products respondió {status}: {str(body)[:200]}")
        for p in body:
            if not all(k in p for k in ("id", "name", "price")):
                raise RemoteStoreError(f"Producto con forma inesperada: {str(p)[:200]}")
        return [p for p in body if p.get("published", True)]

    def fetch_product(self, pid: str) -> Optional[dict]:
        """Ficha actual en la API (None si ya no existe o no está publicada)."""
        status, body = self._call("GET", f"/api/products/{pid}")
        if status == 404:
            return None
        if status != 200 or not isinstance(body, dict):
            raise RemoteStoreError(f"GET /api/products/{pid} respondió {status}: {str(body)[:200]}")
        return body

    # ------------------------------------------------------------------ esquema / restore
    def _create_schema(self):
        super()._create_schema()
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS line_variants(
                cart_id TEXT, product_id TEXT, size TEXT, color TEXT, cut TEXT,
                PRIMARY KEY(cart_id, product_id));
            CREATE TABLE IF NOT EXISTS remote_orders(
                remote_key TEXT PRIMARY KEY, quote_id TEXT, local_order_id TEXT, mode TEXT, status TEXT,
                endpoint TEXT, payload_json TEXT, response_json TEXT, remote_order_id TEXT, detail TEXT,
                created_at REAL);
            """
        )

    def _initial_price(self, pid: str, p: dict) -> float:
        return float(self._api[pid]["price"])

    def _initial_stock(self, pid: str, p: dict) -> int:
        # stock simulado y reproducible; un producto agotado en la tienda real empieza en 0
        return 0 if self._api[pid].get("soldOut") else synth_stock(pid, self.seed, 0.0)

    def restore(self, seed: Optional[int] = None):
        """Como ``Store.restore`` (sobre la última copia del catálogo) + variantes vacías.

        ``remote_orders`` NO se borra: es el registro de efectos reales (o simulados) en la
        tienda. Cada restore abre una nueva "época" para que los ids de cotización reiniciados
        (Q00001...) no choquen con pedidos de una ejecución anterior.
        """
        super().restore(seed)
        with self._lock:
            self.conn.execute("DELETE FROM line_variants")
        self._epoch = uuid.uuid4().hex[:8]

    def refresh_catalog(self) -> dict:
        """Vuelve a leer la API y sincroniza precio y agotados de los productos ya indexados."""
        fresh = {str(p["id"]): p for p in self._fetch_products()}
        changes = {"price": [], "sold_out": [], "missing": [], "new_ignored": sorted(set(fresh) - set(self._api))}
        for pid in list(self._api):
            p = fresh.get(pid)
            if p is None:
                changes["missing"].append(pid)
                self.set_stock(pid, 0)
                continue
            self._sync_one(pid, p, changes)
        return changes

    def _sync_one(self, pid: str, p: Optional[dict], changes: Optional[dict] = None):
        if p is None:
            if self.get_stock(pid):
                self.set_stock(pid, 0)
            return
        self._api[pid] = dict(p)
        if abs(float(p["price"]) - (self.get_price(pid) or 0.0)) > 1e-9:
            self.set_price(pid, float(p["price"]))
            if changes is not None:
                changes["price"].append(pid)
        if p.get("soldOut") and self.get_stock(pid):
            self.set_stock(pid, 0)
            if changes is not None:
                changes["sold_out"].append(pid)

    # ------------------------------------------------------------------ productos
    def get_product(self, pid) -> Optional[dict]:
        p = super().get_product(pid)
        if p is not None:
            api = self._api.get(str(pid), {})
            p.update(product_type=_type_of(api), options=self.variant_options(pid), sold_out=bool(api.get("soldOut")))
        return p

    def variant_options(self, pid, cut: Optional[str] = None) -> dict:
        api = self._api.get(str(pid), {})
        return {"cut": _cuts_for(api), "size": _sizes_for(api, cut), "color": [c["name"] for c in _colors_for(api)]}

    def is_generic_query(self, query: str) -> bool:
        """"polo", "unas tazas negras": solo nombran un TIPO de producto de esta tienda."""
        toks = [t for t in re.findall(r"[a-z0-9]+", _norm_text(query)) if t not in _FILLER]
        rest = [t for t in toks if t not in _GENERIC_WORDS and t not in COLOR_WORDS and t != "talla"
                and t not in _SIZE_WORDS and t not in _CUT_WORDS and not t.isdigit()]
        return bool(toks) and not rest

    # ------------------------------------------------------------------ variantes
    @staticmethod
    def extract_variants(text: str) -> tuple[dict, str, bool]:
        """Extrae talla/corte/color de una frase -> ``(variantes, resto, solo_variantes)``."""
        t = _norm_text(text)
        found = {}
        m = _SIZE_RE.search(t)
        if m:
            found["size"] = _SIZE_WORDS[m.group(1)]
            t = t[:m.start()] + " " + t[m.end():]
        m = _CUT_RE.search(t)
        if m:
            found["cut"] = _CUT_WORDS[m.group(1)]
            t = t[:m.start()] + " " + t[m.end():]
        colors = [COLOR_WORDS[w] for w in re.findall(r"[a-z]+", t) if w in COLOR_WORDS]
        if colors:
            found["color"] = colors[0]
        rest = re.sub(r"\s+", " ", t).strip()
        leftover = [w for w in re.findall(r"[a-z0-9]+", rest)
                    if w not in _FILLER and w not in COLOR_WORDS and w not in {"talla", "corte", "color", "modelo"}]
        return found, rest, bool(found) and not leftover

    def _variants_rows(self, cart_id: str) -> dict:
        rows = self.conn.execute("SELECT product_id, size, color, cut FROM line_variants WHERE cart_id=?",
                                 (cart_id,)).fetchall()
        return {r["product_id"]: {k: r[k] for k in ("size", "color", "cut") if r[k]} for r in rows}

    def get_line_variant(self, cart_id: str, pid) -> dict:
        return self._variants_rows(cart_id).get(str(pid), {})

    def set_line_variant(self, cart_id: str, pid, size=None, color=None, cut=None) -> dict:
        """Fija la variante de una línea; valida contra las opciones del producto."""
        pid = str(pid)
        cur = self.get_line_variant(cart_id, pid)
        new = dict(cur)
        if cut is not None:
            c = _CUT_WORDS.get(_norm_text(cut), _norm_text(cut))
            if c not in _cuts_for(self._api.get(pid, {})):
                raise StoreError(f"Corte '{cut}' no disponible para este producto")
            new["cut"] = c
            if new.get("size") and new["size"] not in _sizes_for(self._api[pid], c):
                new.pop("size")
        if size is not None:
            s = _SIZE_WORDS.get(_norm_text(size), str(size).strip().upper())
            if s not in _sizes_for(self._api.get(pid, {}), new.get("cut")):
                raise StoreError(f"Talla '{size}' no disponible para este producto")
            new["size"] = s
        if color is not None:
            raw = self._color_value(pid, color)
            if raw is None:
                raise StoreError(f"Color '{color}' no disponible para este producto")
            new["color"] = raw
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO line_variants(cart_id,product_id,size,color,cut) VALUES(?,?,?,?,?)",
                (cart_id, pid, new.get("size"), new.get("color"), new.get("cut")))
        return new

    def _color_value(self, pid: str, color: str) -> Optional[str]:
        opts = _colors_for(self._api.get(pid, {}))
        c = (color or "").strip()
        for o in opts:  # valor exacto de la API (hex) o nombre en español
            if c.lower() == str(o["value"]).lower():
                return o["value"]
        canon = canonical_color(c) or _norm_text(c)
        return next((o["value"] for o in opts if o["name"] == canon), None)

    def apply_preferences(self, cart_id: str, prefs: dict, pids: Optional[Iterable] = None,
                          overwrite_last: bool = False) -> list:
        """Rellena talla/color/corte que falten en las líneas del carrito con las preferencias.

        Solo se aplican valores válidos para cada producto. Con ``overwrite_last`` (el usuario
        acaba de decir solo "talla L"), si ninguna línea tenía ese campo pendiente se
        sobrescribe en la última línea añadida. Devuelve la lista de cambios.
        """
        cart = [i["product_id"] for i in self.get_cart(cart_id)]
        targets = [p for p in cart if pids is None or p in {str(x) for x in pids}]
        changes = []
        order = ("cut", "size", "color")  # el corte primero: condiciona las tallas
        for field in order:
            val = (prefs or {}).get(field)
            if not val:
                continue
            filled = False
            for pid in targets:
                if self.get_line_variant(cart_id, pid).get(field):
                    continue
                if self._try_set(cart_id, pid, field, val):
                    changes.append((pid, field, val))
                    filled = True
            if overwrite_last and not filled and targets:
                pid = targets[-1]
                if self._try_set(cart_id, pid, field, val):
                    changes.append((pid, field, val))
        return changes

    def _try_set(self, cart_id, pid, field, val) -> bool:
        try:
            self.set_line_variant(cart_id, pid, **{field: val})
            return True
        except StoreError:
            return False

    def resolved_variant(self, cart_id: str, pid) -> tuple[dict, list]:
        """Variante de la línea con autocompletado de campos de una sola opción -> (variante, faltantes)."""
        pid = str(pid)
        api = self._api.get(pid, {})
        v = dict(self.get_line_variant(cart_id, pid))
        missing = []
        cuts = _cuts_for(api)
        if cuts and not v.get("cut"):
            if len(cuts) == 1:
                v["cut"] = cuts[0]
            else:
                missing.append("cut")
        sizes = _sizes_for(api, v.get("cut"))
        if sizes and not v.get("size"):
            if len(sizes) == 1:
                v["size"] = sizes[0]
            else:
                missing.append("size")
        colors = _colors_for(api)
        if colors and not v.get("color"):
            if len(colors) == 1:
                v["color"] = colors[0]["value"]
            else:
                missing.append("color")
        return v, missing

    def variant_label(self, v: dict) -> str:
        parts = []
        if v.get("size"):
            parts.append(f"talla {v['size']}")
        if v.get("color"):
            parts.append(hex_to_color_name(v["color"]))
        if v.get("cut"):
            parts.append(f"corte {v['cut']}")
        return ", ".join(parts)

    # ------------------------------------------------------------------ carrito
    def remove_from_cart(self, cart_id: str, pid) -> list[dict]:
        with self._lock:
            self.conn.execute("DELETE FROM line_variants WHERE cart_id=? AND product_id=?", (cart_id, str(pid)))
        return super().remove_from_cart(cart_id, pid)

    def clear_cart(self, cart_id: str):
        with self._lock:
            self.conn.execute("DELETE FROM line_variants WHERE cart_id=?", (cart_id,))
        super().clear_cart(cart_id)

    # ------------------------------------------------------------------ cotización
    def create_quote(self, cart_id: str, budget: Optional[float] = None) -> dict:
        """``Store.create_quote`` + variantes congeladas; inválida si falta talla/color/corte."""
        _names = {"size": "talla", "color": "color", "cut": "corte"}
        with self._lock:
            q = super().create_quote(cart_id, budget)
            extra = []
            for it in q["items"]:
                v, missing = self.resolved_variant(cart_id, it["product_id"])
                it["variant"] = v
                it["variant_label"] = self.variant_label(v)
                if missing:
                    opts = self.variant_options(it["product_id"], v.get("cut"))
                    need = " y ".join(f"{_names[m]} ({', '.join(opts[m])})" for m in missing)
                    extra.append({"type": "variant", "product_id": it["product_id"], "missing": missing,
                                  "options": {m: opts[m] for m in missing},
                                  "detail": f"falta elegir {need} de {it['title']}"})
            if extra:
                q["problems"] = q["problems"] + extra
                q["valid"] = False
            self.conn.execute("UPDATE quotes SET items_json=?, valid=?, problems_json=? WHERE quote_id=?",
                              (json.dumps(q["items"]), int(q["valid"]), json.dumps(q["problems"]), q["quote_id"]))
            return q

    # ------------------------------------------------------------------ pedido
    def build_order_payload(self, quote: dict, customer: Optional[dict] = None) -> dict:
        """Cuerpo EXACTO de ``POST /api/orders`` (OrderCreateSchema del servidor)."""
        cust = customer or self.customer or {}
        user = {f: str(cust.get(f) or f"<{f}>").strip() for f in ("dni", "name", "phone")}
        items = []
        for it in quote["items"]:
            v = it.get("variant") or {}
            item = {"productId": it["product_id"], "quantity": int(it["qty"])}
            for k in ("cut", "size", "color"):
                if v.get(k):
                    item[k] = v[k]
            items.append(item)
        return {"user": user, "items": items}

    def _customer_ok(self) -> bool:
        c = self.customer or {}
        return all(isinstance(c.get(f), str) and c[f].strip() for f in ("dni", "name", "phone"))

    def _remote_row(self, where: str, arg) -> Optional[dict]:
        r = self.conn.execute(f"SELECT * FROM remote_orders WHERE {where}=? ORDER BY created_at DESC",
                              (arg,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["payload"] = json.loads(d.pop("payload_json") or "null")
        d["response"] = json.loads(d.pop("response_json") or "null")
        return d

    def remote_orders(self) -> list[dict]:
        keys = [r[0] for r in self.conn.execute("SELECT remote_key FROM remote_orders ORDER BY created_at")]
        return [self._remote_row("remote_key", k) for k in keys]

    def _order_dict(self, order_id: str) -> Optional[dict]:
        o = super()._order_dict(order_id)
        if o is not None:
            r = self._remote_row("local_order_id", order_id)
            if r is not None:
                o["remote"] = {k: r[k] for k in ("mode", "status", "endpoint", "remote_order_id", "detail", "payload")}
        return o

    def _revalidate_remote(self, q: dict) -> list:
        """Relee cada producto de la cotización en la API y actualiza la sombra local."""
        for it in q["items"]:
            try:
                p = self.fetch_product(it["product_id"])
            except RemoteStoreError as e:
                return [{"type": "remote", "detail": f"no pude verificar el producto en la tienda real ({e})"}]
            self._sync_one(it["product_id"], p)
        return []

    def place_order(self, quote_id: str, idempotency_key: Optional[str] = None,
                    _fail_after_first_line: bool = False) -> dict:
        """Revalida contra la API -> ``Store.place_order`` (atómico, idempotente) -> dry-run o POST."""
        key = idempotency_key or quote_id
        with self._lock:
            ex = self.conn.execute("SELECT order_id FROM orders WHERE quote_id=? OR idempotency_key=?",
                                   (quote_id, key)).fetchone()
            q = self.get_quote(quote_id)
            if ex is not None or q is None or not q["valid"]:
                return super().place_order(quote_id, key, _fail_after_first_line)  # duplicado o error
            problems = self._revalidate_remote(q)
            if problems:
                return {"ok": False, "duplicate": False, "problems": problems}
            payload = self.build_order_payload(q)
            res = super().place_order(quote_id, key, _fail_after_first_line)
            if not res["ok"] or res["duplicate"]:
                return res
            order_id = res["order"]["order_id"]
            remote = self._submit(f"{self._epoch}:{key}", quote_id, order_id, payload)
            if remote["status"] == "fallido":
                self._compensate(order_id, q)
                return {"ok": False, "duplicate": False,
                        "problems": [{"type": "remote", "detail": f"la tienda real rechazó el pedido: {remote['detail']}"}]}
            pids = [it["product_id"] for it in q["items"]]
            self.conn.executemany("DELETE FROM line_variants WHERE cart_id=? AND product_id=?",
                                  [(q["cart_id"], p) for p in pids])
            res["order"] = self._order_dict(order_id)
            return res

    def _submit(self, rkey: str, quote_id: str, order_id: str, payload: dict) -> dict:
        """Dry-run o POST real. Nunca hace POST dos veces para la misma clave."""
        prev = self._remote_row("remote_key", rkey)
        if prev is not None:
            return prev
        real = self.allow_real_orders and self._customer_ok()
        mode = "real" if real else "dry_run"
        endpoint = self._url("/api/orders")
        self.conn.execute("INSERT INTO remote_orders VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (rkey, quote_id, order_id, mode, "pendiente", endpoint, json.dumps(payload),
                           None, None, None, time.time()))
        if not real:
            why = ("allow_real_orders=False" if not self.allow_real_orders
                   else "faltan datos del cliente (dni, name, phone)")
            status, resp, remote_id, detail = "dry_run", None, None, f"no se envió ({why})"
        else:
            try:
                code, body = self._call("POST", "/api/orders", payload)
            except RemoteStoreError as e:
                # sin respuesta: el pedido pudo crearse o no -> no se reintenta ni se revierte
                code, body = None, None
                status, resp, remote_id = "incierto", None, None
                detail = f"sin respuesta de la tienda; verifica antes de reintentar ({e})"
            else:
                resp = body
                if code == 201 and isinstance(body, dict) and body.get("id"):
                    status, remote_id, detail = "enviado", str(body["id"]), f"total remoto {body.get('total')}"
                else:
                    err = (body or {}).get("error", {}) if isinstance(body, dict) else {}
                    status, remote_id = "fallido", None
                    detail = f"HTTP {code} {err.get('code', '')}: {err.get('message', body)} {err.get('details', '')}".strip()
        self.conn.execute("UPDATE remote_orders SET status=?, response_json=?, remote_order_id=?, detail=? "
                          "WHERE remote_key=?", (status, json.dumps(resp), remote_id, detail, rkey))
        return self._remote_row("remote_key", rkey)

    def _compensate(self, order_id: str, q: dict):
        """Deshace el pedido local si la tienda real lo rechazó (stock y carrito como antes)."""
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            for it in q["items"]:
                c.execute("UPDATE products SET stock = stock + ? WHERE product_id=?", (it["qty"], it["product_id"]))
                seq = self._next_seq("cart_seq")
                c.execute("INSERT OR IGNORE INTO carts(cart_id,product_id,qty,added_seq) VALUES(?,?,?,?)",
                          (q["cart_id"], it["product_id"], it["qty"], seq))
            c.execute("DELETE FROM order_lines WHERE order_id=?", (order_id,))
            c.execute("UPDATE remote_orders SET local_order_id=NULL WHERE local_order_id=?", (order_id,))
            c.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
            self._bump_stock_version()
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["remote_orders"] = [{k: r[k] for k in ("remote_key", "quote_id", "local_order_id", "mode", "status",
                                                    "remote_order_id", "detail", "payload")}
                                 for r in self.remote_orders()]
        snap["base_url"] = self.base_url
        return snap


def _norm_text(text: str) -> str:
    from .textutil import norm
    return norm(text)


# ====================================================================== helper de demo
def build_remote_demo(cfg=None, base_url: Optional[str] = None, transport=None, retriever_kind: str = "bm25",
                      llm=None, bm25_b: float = 0.3, **adapter_kw) -> dict:
    """Tienda real (o copia offline si no hay URL) + recuperador + fábrica de agentes.

    Devuelve ``{"cfg", "store", "retriever", "agent_factory", "offline"}``. El ``cfg``
    devuelto es una copia con ``currency`` = moneda de la tienda (PEN).
    """
    import dataclasses

    from .agent import ShoppingAgent
    from .config import Config
    from .retrieval import build_retriever

    cfg = cfg or Config()
    url = resolve_base_url(base_url, cfg)
    offline = url is None and transport is None
    if offline:
        transport = FakeTransport(load_fixture_products())
    store = RemoteStoreAdapter.from_config(cfg, transport=transport, base_url=url, **adapter_kw)
    cfg2 = dataclasses.replace(cfg, currency=store.currency)
    if retriever_kind == "bm25":
        # Índice propio (sin la caché de build_retriever, que se indexa por id() del DataFrame).
        # b=0.3: las fichas son plantillas cuya longitud depende del nº de tallas/colores, no de
        # la relevancia; con b=0.75 "polo docker" devolvía antes la taza (ficha más corta).
        from .retrieval import BM25Retriever
        retriever = BM25Retriever(store.catalog_df, b=bm25_b, max_chars=getattr(cfg2, "text_max_chars", 1200))
        retriever.attach_store(store)
    else:
        retriever = build_retriever(cfg2, store.catalog_df, kind=retriever_kind, store=store)

    def agent_factory(cart_id: str = "c1"):
        return ShoppingAgent(store, retriever, llm=llm, cart_id=cart_id, cfg=cfg2)

    return {"cfg": cfg2, "store": store, "retriever": retriever, "agent_factory": agent_factory,
            "offline": offline}
