"""Tienda simulada sobre SQLite: precios, existencias, carritos, cotizaciones y pedidos.

Principios de diseño:
- Todos los datos ficticios (precio, stock) se derivan de un hash determinista del
  ``product_id`` y la semilla, por lo que ``restore(seed)`` reproduce exactamente el
  mismo estado inicial.
- ``place_order`` se ejecuta dentro de UNA transacción ``BEGIN IMMEDIATE``: revalida
  stock, precio y presupuesto; si algo falla hace ``ROLLBACK`` completo (sin cambios
  parciales). La clave de idempotencia (por defecto el ``quote_id``) impide pedidos
  duplicados: repetir la confirmación devuelve el pedido existente con ``duplicate=True``.
- Las etiquetas ESCI NO se cargan aquí: la tienda solo conoce el catálogo.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from typing import Iterable, Optional

from .textutil import clean_html, norm, truncate

# Heurística de rango de precios por categoría (mediana en EUR) según palabras del título.
_PRICE_HEURISTICS = [
    (r"\b(portatil|laptop|ordenador|televisor|smart ?tv|monitor|frigorifico|lavadora|patinete|bicicleta)\b", 320.0),
    (r"\b(movil|smartphone|tablet|consola|impresora|cafetera|aspiradora|silla)\b", 140.0),
    (r"\b(reloj|smartwatch|altavoz|camara|auriculares|audifonos|cascos|teclado|disco duro|ssd|router|zapatillas)\b", 45.0),
    (r"\b(raton|mouse|cargador|bateria|memoria|pendrive|usb|lampara|libro|juego)\b", 22.0),
    (r"\b(mochila|bolso|maleta)\b", 30.0),
    (r"\b(cable|funda|protector|carcasa|adaptador|pegatina|llavero|pila|pilas|calcetines)\b", 11.0),
    (r"\b(botella|termo|taza|vaso|fiambrera)\b", 9.0),
    (r"\b(cuaderno|libreta|boligrafo|lapiz|lapices|rotulador|agenda)\b", 4.0),
]
_DEFAULT_MEDIAN = 25.0
_SIGMA = 0.55


def _u(seed: int, pid: str, salt: str) -> float:
    """Uniforme (0,1) determinista a partir de un hash (no depende del orden de inserción)."""
    h = hashlib.sha256(f"{seed}|{salt}|{pid}".encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2**64


def synth_price(pid: str, title: str, seed: int) -> float:
    """Precio lognormal plausible, redondeado a ,99."""
    t = norm(title)
    median, first = _DEFAULT_MEDIAN, len(t) + 1
    for pat, med in _PRICE_HEURISTICS:  # la palabra de categoría que aparece primero manda
        m = re.search(pat, t)
        if m and m.start() < first:
            median, first = med, m.start()
    u1, u2 = _u(seed, pid, "p1"), _u(seed, pid, "p2")
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)  # Box-Muller
    p = median * math.exp(_SIGMA * z)
    p = min(max(p, 2.0), 2500.0)
    return round(math.floor(p) + 0.99, 2)


def synth_stock(pid: str, seed: int, zero_frac: float = 0.10) -> int:
    """Stock entero; ~``zero_frac`` de los productos agotados."""
    if _u(seed, pid, "s0") < zero_frac:
        return 0
    return 1 + int(_u(seed, pid, "s1") * 40)


class StoreError(Exception):
    pass


class Store:
    """Tienda simulada. ``Store(db_path, catalog_df)``; ``db_path`` puede ser ':memory:'."""

    def __init__(self, db_path: str = ":memory:", catalog_df=None, seed: int = 42,
                 currency: str = "EUR", zero_stock_frac: float = 0.10):
        self.db_path = db_path
        self.currency = currency
        self.zero_stock_frac = zero_stock_frac
        self.seed = seed
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._catalog: dict[str, dict] = {}
        self._touched: set[str] = set()
        self._create_schema()
        if catalog_df is not None:
            self._load_catalog(catalog_df)
            self.restore(seed)

    # ------------------------------------------------------------------ esquema
    def _create_schema(self):
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS products(
                product_id TEXT PRIMARY KEY, title TEXT, brand TEXT, color TEXT,
                price REAL NOT NULL, stock INTEGER NOT NULL CHECK(stock >= 0));
            CREATE TABLE IF NOT EXISTS carts(
                cart_id TEXT, product_id TEXT REFERENCES products(product_id),
                qty INTEGER NOT NULL CHECK(qty > 0), added_seq INTEGER,
                PRIMARY KEY(cart_id, product_id));
            CREATE TABLE IF NOT EXISTS quotes(
                quote_id TEXT PRIMARY KEY, cart_id TEXT, budget REAL, total REAL,
                items_json TEXT, valid INTEGER, problems_json TEXT, stock_version INTEGER,
                created_at REAL);
            CREATE TABLE IF NOT EXISTS orders(
                order_id TEXT PRIMARY KEY, quote_id TEXT UNIQUE, idempotency_key TEXT UNIQUE,
                cart_id TEXT, total REAL, created_at REAL, status TEXT);
            CREATE TABLE IF NOT EXISTS order_lines(
                order_id TEXT REFERENCES orders(order_id), product_id TEXT, qty INTEGER,
                unit_price REAL, PRIMARY KEY(order_id, product_id));
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            """
        )

    def _load_catalog(self, df):
        cols = df.columns
        for r in df.itertuples(index=False):
            d = r._asdict()
            pid = str(d["product_id"])
            self._catalog[pid] = {
                "product_id": pid,
                "title": clean_html(d.get("title", "")),
                "brand": clean_html(d.get("brand", "")) if "brand" in cols else "",
                "color": clean_html(d.get("color", "")) if "color" in cols else "",
                "description": clean_html(d.get("description", "")) if "description" in cols else "",
                "bullet_point": clean_html(d.get("bullet_point", "")) if "bullet_point" in cols else "",
            }

    # ---------------------------------------------------------------- meta
    def _meta_get(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def _meta_set(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))

    def _next_seq(self, key: str) -> int:
        n = int(self._meta_get(key, 0)) + 1
        self._meta_set(key, n)
        return n

    @property
    def stock_version(self) -> int:
        """Contador que aumenta cada vez que cambia algún stock."""
        return int(self._meta_get("stock_version", 0))

    def _bump_stock_version(self):
        self._meta_set("stock_version", self.stock_version + 1)

    # ---------------------------------------------------------------- restore
    def restore(self, seed: Optional[int] = None):
        """Regenera precios/stock con la semilla y vacía carritos, cotizaciones y pedidos."""
        if seed is not None:
            self.seed = seed
        with self._lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                for t in ("order_lines", "orders", "quotes", "carts", "products", "meta"):
                    c.execute(f"DELETE FROM {t}")
                rows = [
                    (pid, p["title"], p["brand"], p["color"],
                     self._initial_price(pid, p), self._initial_stock(pid, p))
                    for pid, p in self._catalog.items()
                ]
                c.executemany("INSERT INTO products VALUES(?,?,?,?,?,?)", rows)
                self._meta_set("stock_version", 0)
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
            self._touched = set()

    def _initial_price(self, pid: str, p: dict) -> float:
        """Precio inicial de ``restore`` (sintético; las subclases pueden usar precios reales)."""
        return synth_price(pid, p["title"], self.seed)

    def _initial_stock(self, pid: str, p: dict) -> int:
        """Stock inicial de ``restore`` (sintético y determinista con la semilla)."""
        return synth_stock(pid, self.seed, self.zero_stock_frac)

    # ---------------------------------------------------------------- productos
    def has_product(self, pid) -> bool:
        return str(pid) in self._catalog

    def product_ids(self) -> list[str]:
        return list(self._catalog.keys())

    def catalog_text(self, pid) -> dict:
        return self._catalog.get(str(pid), {})

    def get_product(self, pid) -> Optional[dict]:
        """Ficha del producto con precio y stock actuales (o None si no existe)."""
        pid = str(pid)
        row = self.conn.execute("SELECT price, stock FROM products WHERE product_id=?", (pid,)).fetchone()
        if row is None:
            return None
        p = self._catalog[pid]
        return {
            "product_id": pid,
            "title": p["title"],
            "brand": p["brand"] or "desconocido",
            "color": p["color"] or "desconocido",
            "price": row["price"],
            "currency": self.currency,
            "stock": row["stock"],
            "description": truncate(p["description"], 600),
            "bullet_point": truncate(p["bullet_point"], 600),
        }

    def get_price(self, pid) -> Optional[float]:
        row = self.conn.execute("SELECT price FROM products WHERE product_id=?", (str(pid),)).fetchone()
        return None if row is None else row[0]

    def get_stock(self, pid) -> Optional[int]:
        row = self.conn.execute("SELECT stock FROM products WHERE product_id=?", (str(pid),)).fetchone()
        return None if row is None else row[0]

    def set_stock(self, pid, n: int):
        """Fija el stock (para pruebas/escenarios: simula ventas de otros clientes)."""
        if int(n) < 0:
            raise StoreError("El stock no puede ser negativo")
        with self._lock:
            cur = self.conn.execute("UPDATE products SET stock=? WHERE product_id=?", (int(n), str(pid)))
            if cur.rowcount == 0:
                raise StoreError(f"Producto inexistente: {pid}")
            self._bump_stock_version()
            self._touched.add(str(pid))

    def set_price(self, pid, price: float):
        """Cambia el precio (para escenarios de cambio de precio)."""
        with self._lock:
            cur = self.conn.execute("UPDATE products SET price=? WHERE product_id=?", (float(price), str(pid)))
            if cur.rowcount == 0:
                raise StoreError(f"Producto inexistente: {pid}")
            self._touched.add(str(pid))

    def inventory(self, pids: Iterable) -> dict:
        """Stock actual por producto: ``{product_id: stock}`` (None si no existe)."""
        return {str(p): self.get_stock(p) for p in pids}

    # ---------------------------------------------------------------- carrito
    @staticmethod
    def _check_qty(qty) -> int:
        if isinstance(qty, bool) or not isinstance(qty, (int, float)) or int(qty) != qty or int(qty) <= 0:
            raise StoreError(f"Cantidad inválida: {qty}")
        return int(qty)

    def add_to_cart(self, cart_id: str, pid, qty: int = 1) -> list[dict]:
        qty = self._check_qty(qty)
        pid = str(pid)
        if not self.has_product(pid):
            raise StoreError(f"Producto inexistente: {pid}")
        with self._lock:
            seq = self._next_seq("cart_seq")
            self.conn.execute(
                "INSERT INTO carts(cart_id,product_id,qty,added_seq) VALUES(?,?,?,?) "
                "ON CONFLICT(cart_id,product_id) DO UPDATE SET qty=qty+excluded.qty",
                (cart_id, pid, qty, seq),
            )
        return self.get_cart(cart_id)

    def set_cart_qty(self, cart_id: str, pid, qty: int) -> list[dict]:
        pid = str(pid)
        if qty == 0:
            return self.remove_from_cart(cart_id, pid)
        qty = self._check_qty(qty)
        with self._lock:
            cur = self.conn.execute("UPDATE carts SET qty=? WHERE cart_id=? AND product_id=?", (qty, cart_id, pid))
            if cur.rowcount == 0:
                raise StoreError(f"El producto {pid} no está en el carrito")
        return self.get_cart(cart_id)

    def remove_from_cart(self, cart_id: str, pid) -> list[dict]:
        with self._lock:
            self.conn.execute("DELETE FROM carts WHERE cart_id=? AND product_id=?", (cart_id, str(pid)))
        return self.get_cart(cart_id)

    def clear_cart(self, cart_id: str):
        with self._lock:
            self.conn.execute("DELETE FROM carts WHERE cart_id=?", (cart_id,))

    def get_cart(self, cart_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT c.product_id, c.qty, p.title, p.brand, p.price, p.stock FROM carts c "
            "JOIN products p USING(product_id) WHERE c.cart_id=? ORDER BY c.added_seq",
            (cart_id,),
        ).fetchall()
        return [
            {"product_id": r["product_id"], "title": r["title"], "brand": r["brand"] or "desconocido",
             "qty": r["qty"], "unit_price": r["price"], "line_total": round(r["price"] * r["qty"], 2),
             "stock": r["stock"]}
            for r in rows
        ]

    def cart_total(self, cart_id: str) -> float:
        return round(sum(i["line_total"] for i in self.get_cart(cart_id)), 2)

    # ---------------------------------------------------------------- cotización
    def create_quote(self, cart_id: str, budget: Optional[float] = None) -> dict:
        """Congela precios y stock actuales del carrito y valida stock y presupuesto."""
        with self._lock:
            items = self.get_cart(cart_id)
            problems = []
            if not items:
                problems.append({"type": "empty_cart", "detail": "El carrito está vacío"})
            for it in items:
                if it["stock"] < it["qty"]:
                    problems.append({"type": "stock", "product_id": it["product_id"],
                                     "requested": it["qty"], "available": it["stock"],
                                     "detail": f"Stock insuficiente para {it['title'][:60]} "
                                               f"(pedido {it['qty']}, disponible {it['stock']})"})
            total = round(sum(i["line_total"] for i in items), 2)
            if budget is not None and total > float(budget) + 1e-9:
                problems.append({"type": "budget", "total": total, "budget": float(budget),
                                 "detail": f"El total {total:.2f} supera el presupuesto {float(budget):.2f}"})
            n = self._next_seq("quote_seq")
            quote_id = f"Q{n:05d}"
            quote = {
                "quote_id": quote_id, "cart_id": cart_id,
                "items": [{"product_id": i["product_id"], "title": i["title"], "qty": i["qty"],
                           "unit_price": i["unit_price"], "line_total": i["line_total"],
                           "stock_at_quote": i["stock"]} for i in items],
                "subtotal": total, "total": total, "currency": self.currency,
                "budget": None if budget is None else float(budget),
                "valid": not problems, "problems": problems,
                "stock_version": self.stock_version, "created_at": time.time(),
            }
            self.conn.execute(
                "INSERT INTO quotes VALUES(?,?,?,?,?,?,?,?,?)",
                (quote_id, cart_id, quote["budget"], total, json.dumps(quote["items"]),
                 int(quote["valid"]), json.dumps(problems), quote["stock_version"], quote["created_at"]),
            )
            return quote

    def get_quote(self, quote_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM quotes WHERE quote_id=?", (quote_id,)).fetchone()
        if r is None:
            return None
        return {"quote_id": r["quote_id"], "cart_id": r["cart_id"], "budget": r["budget"],
                "total": r["total"], "items": json.loads(r["items_json"]), "valid": bool(r["valid"]),
                "problems": json.loads(r["problems_json"]), "stock_version": r["stock_version"]}

    # ---------------------------------------------------------------- pedido
    def _order_dict(self, order_id: str) -> Optional[dict]:
        o = self.conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        if o is None:
            return None
        lines = self.conn.execute(
            "SELECT product_id, qty, unit_price FROM order_lines WHERE order_id=? ORDER BY product_id",
            (order_id,)).fetchall()
        return {"order_id": o["order_id"], "quote_id": o["quote_id"], "idempotency_key": o["idempotency_key"],
                "cart_id": o["cart_id"], "total": o["total"], "status": o["status"],
                "currency": self.currency,
                "lines": [{"product_id": l["product_id"], "qty": l["qty"], "unit_price": l["unit_price"],
                           "title": self._catalog.get(l["product_id"], {}).get("title", "")} for l in lines]}

    def place_order(self, quote_id: str, idempotency_key: Optional[str] = None,
                    _fail_after_first_line: bool = False) -> dict:
        """Registra el pedido de una cotización de forma atómica e idempotente.

        Devuelve ``{"ok": True, "order": {...}, "duplicate": bool}`` o
        ``{"ok": False, "problems": [...], "duplicate": False}``.
        ``_fail_after_first_line`` solo sirve para probar la atomicidad (simula un fallo
        a mitad de la transacción).
        """
        key = idempotency_key or quote_id
        with self._lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                ex = c.execute("SELECT order_id FROM orders WHERE quote_id=? OR idempotency_key=?",
                               (quote_id, key)).fetchone()
                if ex is not None:
                    c.execute("COMMIT")
                    return {"ok": True, "duplicate": True, "order": self._order_dict(ex[0]), "problems": []}
                q = self.get_quote(quote_id)
                if q is None:
                    c.execute("ROLLBACK")
                    return {"ok": False, "duplicate": False,
                            "problems": [{"type": "quote", "detail": f"Cotización inexistente: {quote_id}"}]}
                problems = []
                if not q["valid"]:
                    problems.append({"type": "quote", "detail": "La cotización no es válida",
                                     "quote_problems": q["problems"]})
                total = 0.0
                for it in q["items"]:
                    row = c.execute("SELECT price, stock FROM products WHERE product_id=?",
                                    (it["product_id"],)).fetchone()
                    if row is None:
                        problems.append({"type": "missing", "product_id": it["product_id"],
                                         "detail": "Producto ya no existe"})
                        continue
                    if row["stock"] < it["qty"]:
                        problems.append({"type": "stock", "product_id": it["product_id"],
                                         "requested": it["qty"], "available": row["stock"],
                                         "detail": f"Stock insuficiente para {it['title'][:60]}: "
                                                   f"pedido {it['qty']}, disponible {row['stock']}"})
                    if abs(row["price"] - it["unit_price"]) > 1e-9:
                        problems.append({"type": "price", "product_id": it["product_id"],
                                         "quoted": it["unit_price"], "current": row["price"],
                                         "detail": f"El precio cambió de {it['unit_price']:.2f} a {row['price']:.2f}"})
                    total += row["price"] * it["qty"]
                total = round(total, 2)
                if q["budget"] is not None and total > q["budget"] + 1e-9:
                    problems.append({"type": "budget", "total": total, "budget": q["budget"],
                                     "detail": f"El total {total:.2f} supera el presupuesto {q['budget']:.2f}"})
                if problems:
                    c.execute("ROLLBACK")
                    return {"ok": False, "duplicate": False, "problems": problems}
                n = self._next_seq("order_seq")
                order_id = f"O{n:05d}"
                c.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?)",
                          (order_id, quote_id, key, q["cart_id"], total, time.time(), "confirmado"))
                for i, it in enumerate(q["items"]):
                    c.execute("INSERT INTO order_lines VALUES(?,?,?,?)",
                              (order_id, it["product_id"], it["qty"], it["unit_price"]))
                    cur = c.execute("UPDATE products SET stock = stock - ? WHERE product_id=? AND stock >= ?",
                                    (it["qty"], it["product_id"], it["qty"]))
                    if cur.rowcount != 1:
                        raise StoreError("Conflicto de stock durante la confirmación")
                    if _fail_after_first_line and i == 0:
                        raise StoreError("Fallo simulado a mitad de la transacción")
                c.execute("DELETE FROM carts WHERE cart_id=?", (q["cart_id"],))
                self._bump_stock_version()
                c.execute("COMMIT")
                self._touched.update(it["product_id"] for it in q["items"])
                return {"ok": True, "duplicate": False, "order": self._order_dict(order_id), "problems": []}
            except Exception as e:
                if c.in_transaction:
                    c.execute("ROLLBACK")
                return {"ok": False, "duplicate": False,
                        "problems": [{"type": "error", "detail": f"Transacción revertida: {e}"}]}

    def list_orders(self) -> list[dict]:
        ids = [r[0] for r in self.conn.execute("SELECT order_id FROM orders ORDER BY order_id")]
        return [self._order_dict(i) for i in ids]

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """Estado serializable (JSON): carritos, cotizaciones, pedidos y stock tocado."""
        carts: dict[str, list] = {}
        for r in self.conn.execute("SELECT cart_id, product_id, qty FROM carts ORDER BY cart_id, added_seq"):
            carts.setdefault(r["cart_id"], []).append({"product_id": r["product_id"], "qty": r["qty"]})
        touched = set(self._touched)
        for r in self.conn.execute("SELECT product_id FROM carts UNION SELECT product_id FROM order_lines"):
            touched.add(r[0])
        quotes = [{"quote_id": r["quote_id"], "total": r["total"], "valid": bool(r["valid"])}
                  for r in self.conn.execute("SELECT quote_id,total,valid FROM quotes ORDER BY quote_id")]
        return {
            "seed": self.seed,
            "stock_version": self.stock_version,
            "carts": carts,
            "quotes": quotes,
            "orders": self.list_orders(),
            "stock": {p: self.get_stock(p) for p in sorted(touched)},
        }
