"""Herramientas del agente con validaciones en código.

Cada herramienta tiene nombre, descripción en español, esquema de argumentos (tipo JSON
Schema) y una función que devuelve ``{"ok": bool, "data": ..., "error": str | None}``.

Validaciones comunes (no delegadas al LLM):
- cantidades enteras en 1..``cfg.max_qty``;
- el producto existe y su id apareció antes en resultados de esta conversación (anclaje);
- stock suficiente y presupuesto respetado al añadir/modificar;
- ``confirm_order`` solo si el usuario confirmó explícitamente en ESTA intervención y hay
  una cotización válida (idempotencia con clave = quote_id).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .attributes import UNKNOWN, answer_attribute, extract_attributes
from .textutil import truncate


@dataclass
class ToolContext:
    store: Any
    retriever: Any
    state: Any
    cfg: Any
    timings: dict = field(default_factory=lambda: {"tools_s": 0.0, "retrieval_s": 0.0})
    user_confirmed: bool = False   # confirmación explícita en la intervención actual


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def ok(data=None) -> dict:
    return {"ok": True, "data": data, "error": None}


def fail(error: str, data=None) -> dict:
    return {"ok": False, "data": data, "error": error}


# ------------------------------------------------------------------ validación de argumentos
def validate_args(schema: dict, args: dict) -> tuple[Optional[dict], Optional[str]]:
    """Valida/coacciona argumentos según un esquema simple tipo JSON Schema."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None, "Los argumentos deben ser un objeto JSON"
    props = schema.get("properties", {})
    clean = {}
    for name in schema.get("required", []):
        if name not in args or args[name] in (None, ""):
            return None, f"Falta el argumento obligatorio '{name}'"
    for name, val in args.items():
        if name not in props:
            continue  # se ignoran argumentos desconocidos
        spec = props[name]
        typ = spec.get("type")
        if val is None:
            continue
        try:
            if typ == "integer":
                if isinstance(val, bool):
                    raise ValueError
                f = float(val)
                if f != int(f):
                    return None, f"'{name}' debe ser un número entero (recibido {val})"
                val = int(f)
            elif typ == "number":
                if isinstance(val, bool):
                    raise ValueError
                val = float(str(val).replace(",", "."))
            elif typ == "string":
                val = str(val).strip()
            elif typ == "array":
                if isinstance(val, str):
                    val = [v.strip() for v in val.split(",") if v.strip()]
                val = [str(v) for v in list(val)]
            elif typ == "boolean":
                val = val if isinstance(val, bool) else str(val).lower() in ("true", "1", "si", "sí")
        except (TypeError, ValueError):
            return None, f"'{name}' tiene un tipo inválido (se esperaba {typ})"
        if "minimum" in spec and val < spec["minimum"]:
            return None, f"'{name}' debe ser >= {spec['minimum']}"
        if "maximum" in spec and val > spec["maximum"]:
            return None, f"'{name}' debe ser <= {spec['maximum']}"
        if "enum" in spec and val not in spec["enum"]:
            return None, f"'{name}' debe ser uno de {spec['enum']}"
        clean[name] = val
    for name, spec in props.items():
        if name not in clean and "default" in spec:
            clean[name] = spec["default"]
    return clean, None


# ------------------------------------------------------------------ helpers
def _grounded(ctx: ToolContext, pid) -> Optional[str]:
    pid = str(pid)
    if not ctx.store.has_product(pid):
        return f"El producto '{pid}' no existe en el catálogo"
    if pid not in ctx.state.seen_product_ids:
        return (f"El id '{pid}' no apareció en resultados de búsqueda de esta conversación; "
                f"busca primero con search_products")
    return None


def _brief(p: dict, snippet_len: int = 160) -> dict:
    snip = p.get("bullet_point") or p.get("description") or ""
    return {"product_id": p["product_id"], "title": truncate(p["title"], 110), "brand": p.get("brand") or UNKNOWN,
            "color": p.get("color") or UNKNOWN, "price": p["price"], "stock": p["stock"],
            "snippet": truncate(snip, snippet_len)}


def _cart_view(ctx: ToolContext) -> dict:
    items = ctx.store.get_cart(ctx.state.cart_id)
    total = round(sum(i["line_total"] for i in items), 2)
    b = ctx.state.budget
    return {"items": items, "total": total, "budget": b,
            "remaining": None if b is None else round(b - total, 2), "currency": ctx.state.currency}


def _search(ctx: ToolContext, query: str, k: int, filters: dict) -> list[dict]:
    t0 = time.perf_counter()
    res = ctx.retriever.search(query, k=k, filters=filters)
    ctx.timings["retrieval_s"] += time.perf_counter() - t0
    return res


def _budget_left_excluding(ctx: ToolContext, pid: Optional[str]) -> Optional[float]:
    if ctx.state.budget is None:
        return None
    others = sum(i["line_total"] for i in ctx.store.get_cart(ctx.state.cart_id) if i["product_id"] != pid)
    return round(ctx.state.budget - others, 2)


def _sync_variants(ctx: ToolContext, pids=None) -> list:
    """Tiendas con variantes (talla/color/corte, p. ej. la tienda real): aplica las
    preferencias a las líneas del carrito que no las tengan. La ``Store`` simulada no las
    usa (no tiene ``apply_preferences``) y no cambia nada."""
    fn = getattr(ctx.store, "apply_preferences", None)
    if fn is None:
        return []
    return fn(ctx.state.cart_id, ctx.state.preferences, pids=pids)


def _upsert_item(ctx: ToolContext, pid: str, qty: int, status: str, need: Optional[str] = None):
    st = ctx.state
    it = st.find_item(product_id=pid)
    if it is None and need is not None:
        it = next((i for i in st.requested_items if i["need"] == need and i["status"] in ("pendiente", "sin_stock")), None)
    if it is None:
        p = ctx.store.get_product(pid)
        it = {"need": need or (p["title"][:60] if p else pid), "qty": qty, "product_id": pid, "status": status}
        st.requested_items.append(it)
    else:
        it.update(product_id=pid, qty=qty, status=status)
    return it


# ------------------------------------------------------------------ herramientas
def search_products(ctx: ToolContext, query: str, max_price: float = None, brand: str = None,
                    color: str = None, k: int = 5, in_stock: bool = False):
    prefs = ctx.state.preferences
    max_price = max_price if max_price is not None else prefs.get("max_unit_price")
    brand = brand or prefs.get("brand")
    filters = {"max_price": max_price, "brand": brand, "color": color, "in_stock": in_stock}
    filters = {k_: v for k_, v in filters.items() if v not in (None, False, "")}
    from .retrieval import expand_query
    q_exp = expand_query(query)
    res = _search(ctx, q_exp, k, filters)
    relaxed = []
    for key in ("color", "brand"):
        if not res and key in filters:
            filters.pop(key)
            relaxed.append(key)
            res = _search(ctx, q_exp, k, filters)
    out = []
    for r in res:
        p = ctx.store.get_product(r["product_id"])
        b = _brief(p)
        b["score"] = round(float(r.get("score", 0.0)), 4)
        out.append(b)
    ctx.state.remember_products([o["product_id"] for o in out], key=query)
    return ok({"query": query, "filters": filters, "relaxed_filters": relaxed, "results": out,
               "n": len(out)})


def get_product_details(ctx: ToolContext, product_id: str):
    err = _grounded(ctx, product_id)
    if err:
        return fail(err)
    p = ctx.store.get_product(product_id)
    return ok({**_brief(p, 400), "attributes": extract_attributes(p),
               "description": truncate(p.get("description", ""), 400)})


def compare_products(ctx: ToolContext, product_ids: list, attributes: list = None):
    if not product_ids or len(product_ids) < 2:
        return fail("Se necesitan al menos 2 productos para comparar")
    rows = []
    for pid in product_ids[:5]:
        err = _grounded(ctx, pid)
        if err:
            return fail(err)
        p = ctx.store.get_product(pid)
        attrs = extract_attributes(p)
        if attributes:
            attrs = {a: answer_attribute(p, a) for a in attributes}
        rows.append({"product_id": pid, "title": truncate(p["title"], 90), "price": p["price"],
                     "stock": p["stock"], "attributes": attrs})
    cheapest = min(rows, key=lambda r: r["price"])["product_id"]
    return ok({"products": rows, "cheapest": cheapest,
               "note": "Los atributos 'desconocido' no constan en la ficha del producto."})


def check_stock(ctx: ToolContext, product_id: str, qty: int = 1):
    err = _grounded(ctx, product_id)
    if err:
        return fail(err)
    s = ctx.store.get_stock(product_id)
    return ok({"product_id": product_id, "available": s, "requested": qty, "enough": s >= qty})


def add_to_cart(ctx: ToolContext, product_id: str, qty: int = 1, need: str = None):
    err = _grounded(ctx, product_id)
    if err:
        return fail(err)
    p = ctx.store.get_product(product_id)
    in_cart = next((i["qty"] for i in ctx.store.get_cart(ctx.state.cart_id) if i["product_id"] == product_id), 0)
    new_qty = in_cart + qty
    if new_qty > ctx.cfg.max_qty:
        return fail(f"La cantidad total ({new_qty}) supera el máximo permitido ({ctx.cfg.max_qty})")
    if p["stock"] < new_qty:
        return fail(f"Stock insuficiente: disponible {p['stock']}, solicitado {new_qty}",
                    {"reason": "stock", "product_id": product_id, "available": p["stock"], "requested": new_qty})
    left = _budget_left_excluding(ctx, product_id)
    if left is not None and p["price"] * new_qty > left + 1e-9:
        return fail(f"Excede el presupuesto: {new_qty} x {p['price']:.2f} = {p['price'] * new_qty:.2f} > disponible {left:.2f}",
                    {"reason": "budget", "product_id": product_id, "line_total": round(p["price"] * new_qty, 2),
                     "remaining": left, "max_qty_affordable": int(max(left, 0) // p["price"])})
    ctx.store.add_to_cart(ctx.state.cart_id, product_id, qty)
    _sync_variants(ctx, [product_id])
    _upsert_item(ctx, product_id, new_qty, "en_carrito", need)
    ctx.state.invalidate_quote("carrito modificado")
    return ok({"added": {"product_id": product_id, "title": truncate(p["title"], 90), "qty": qty,
                         "unit_price": p["price"]}, "cart": _cart_view(ctx)})


def update_cart_quantity(ctx: ToolContext, product_id: str, qty: int):
    err = _grounded(ctx, product_id)
    if err:
        return fail(err)
    cart = ctx.store.get_cart(ctx.state.cart_id)
    line = next((i for i in cart if i["product_id"] == product_id), None)
    if line is None:
        return fail("Ese producto no está en el carrito")
    p = ctx.store.get_product(product_id)
    if p["stock"] < qty:
        return fail(f"Stock insuficiente: disponible {p['stock']}, solicitado {qty}",
                    {"reason": "stock", "product_id": product_id, "available": p["stock"], "requested": qty,
                     "kept_qty": line["qty"]})
    left = _budget_left_excluding(ctx, product_id)
    if left is not None and p["price"] * qty > left + 1e-9:
        return fail(f"Con {qty} unidades el total excede el presupuesto ({p['price'] * qty:.2f} > {left:.2f} disponible)",
                    {"reason": "budget", "product_id": product_id, "kept_qty": line["qty"],
                     "max_qty_affordable": int(max(left, 0) // p["price"]), "remaining": left})
    ctx.store.set_cart_qty(ctx.state.cart_id, product_id, qty)
    _upsert_item(ctx, product_id, qty, "en_carrito")
    ctx.state.invalidate_quote("cantidad modificada")
    return ok({"product_id": product_id, "old_qty": line["qty"], "qty": qty, "cart": _cart_view(ctx)})


def remove_from_cart(ctx: ToolContext, product_id: str):
    cart = ctx.store.get_cart(ctx.state.cart_id)
    if not any(i["product_id"] == product_id for i in cart):
        return fail("Ese producto no está en el carrito")
    ctx.store.remove_from_cart(ctx.state.cart_id, product_id)
    ctx.state.requested_items = [i for i in ctx.state.requested_items if i.get("product_id") != product_id]
    ctx.state.invalidate_quote("producto eliminado")
    return ok({"removed": product_id, "cart": _cart_view(ctx)})


def view_cart(ctx: ToolContext):
    return ok(_cart_view(ctx))


def set_budget(ctx: ToolContext, amount: float):
    if amount <= 0:
        return fail("El presupuesto debe ser positivo")
    ctx.state.budget = round(float(amount), 2)
    ctx.state.invalidate_quote("presupuesto modificado")
    view = _cart_view(ctx)
    return ok({"budget": ctx.state.budget, "cart_total": view["total"],
               "cart_exceeds_budget": view["total"] > ctx.state.budget + 1e-9})


_PREF_KEYS = ("brand", "color", "max_unit_price", "other", "size", "cut")


def set_preference(ctx: ToolContext, key: str, value: str):
    if key not in _PREF_KEYS:
        return fail(f"Preferencia desconocida '{key}'. Usa una de {list(_PREF_KEYS)}")
    if key == "max_unit_price":
        try:
            value = float(str(value).replace(",", "."))
        except ValueError:
            return fail("max_unit_price debe ser numérico")
        if value <= 0:
            return fail("max_unit_price debe ser positivo")
    if key == "other":
        ctx.state.preferences.setdefault("other", []).append(str(value))
    else:
        ctx.state.preferences[key] = value
    if key in ("color", "size", "cut") and _sync_variants(ctx):
        ctx.state.invalidate_quote("variante modificada")
    return ok({"preferences": ctx.state.preferences})


def create_quote(ctx: ToolContext):
    q = ctx.store.create_quote(ctx.state.cart_id, ctx.state.budget)
    ctx.state.last_quote = q
    ctx.state.pending_confirmation = bool(q["valid"])
    return ok(q) if q["valid"] else fail("La cotización tiene problemas: " +
                                         "; ".join(p["detail"] for p in q["problems"]), q)


def confirm_order(ctx: ToolContext):
    st = ctx.state
    q = st.last_quote
    if not ctx.user_confirmed:
        return fail("El usuario no ha confirmado explícitamente en esta intervención. Pide confirmación.")
    if q is None:
        return fail("No hay cotización. Llama primero a create_quote y pide confirmación.")
    already = st.last_order_quote_id == q["quote_id"]
    if not already and (not st.pending_confirmation or not q.get("valid")):
        return fail("La cotización no está vigente (cambió el carrito o el presupuesto). Genera una nueva.")
    res = ctx.store.place_order(q["quote_id"], idempotency_key=q["quote_id"])
    if res["ok"]:
        order = res["order"]
        st.last_order_id = order["order_id"]
        st.last_order_quote_id = q["quote_id"]
        if order["order_id"] not in st.confirmed_order_ids:
            st.confirmed_order_ids.append(order["order_id"])
        st.pending_confirmation = False
        for it in st.requested_items:
            if it["status"] == "en_carrito":
                it["status"] = "pedido"
        return ok({"order": order, "duplicate": res["duplicate"]})
    st.pending_confirmation = False
    st.last_quote = dict(q, valid=False, invalidated="falló la confirmación")
    return fail("No se pudo registrar el pedido: " + "; ".join(p["detail"] for p in res["problems"]),
                {"problems": res["problems"]})


def find_alternative(ctx: ToolContext, product_id: str, qty: int = None, query: str = None):
    err = _grounded(ctx, product_id)
    if err:
        return fail(err)
    orig = ctx.store.get_product(product_id)
    if qty is None:
        line = next((i for i in ctx.store.get_cart(ctx.state.cart_id) if i["product_id"] == product_id), None)
        it = ctx.state.find_item(product_id=product_id)
        qty = line["qty"] if line else (it["qty"] if it else 1)
    left = _budget_left_excluding(ctx, product_id)
    max_price = None if left is None else left / max(qty, 1)
    mup = ctx.state.preferences.get("max_unit_price")
    if mup is not None:
        max_price = mup if max_price is None else min(max_price, mup)
    it = ctx.state.find_item(product_id=product_id)
    from_title = not query and not (it and it.get("need"))
    query = query or (it["need"] if it and it.get("need") else "") or " ".join(orig["title"].split()[:8])
    filters = {"in_stock": max(qty, 1), "exclude": [product_id]}
    if max_price is not None:
        filters["max_price"] = max_price
    from .retrieval import QUERY_SYNONYMS, tokenize
    res = _search(ctx, query, 10, filters)
    # la alternativa debe ser del mismo tipo de producto: comparte alguno de los primeros
    # términos de la consulta (o un sinónimo) con el título
    key = [t for t in tokenize(query) if not t.isdigit()][: 2 if from_title else 3]
    key_set = set(key) | {s for t in key for s in QUERY_SYNONYMS.get(t, [])}
    res = [r for r in res if not key_set or key_set & set(tokenize(r["title"]))][:3]
    alts = [_brief(ctx.store.get_product(r["product_id"])) for r in res]
    ctx.state.remember_products([a["product_id"] for a in alts])
    return ok({"original": {"product_id": product_id, "title": truncate(orig["title"], 90),
                            "stock": orig["stock"]}, "qty": qty, "query": query,
               "max_unit_price": None if max_price is None else round(max_price, 2),
               "alternatives": alts, "found": bool(alts)})


def ask_clarification(ctx: ToolContext, question: str):
    ctx.state.pending_clarification = question
    return ok({"question": question})


# ------------------------------------------------------------------ registro
def _obj(props: dict, required=()) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


def build_tools(cfg) -> dict[str, Tool]:
    maxq = getattr(cfg, "max_qty", 50)
    qty = {"type": "integer", "minimum": 1, "maximum": maxq, "description": f"cantidad entera 1..{maxq}"}
    pid = {"type": "string", "description": "product_id obtenido de resultados previos"}
    tools = [
        Tool("search_products", "Busca productos en el catálogo (RAG). Devuelve id, título, marca, color, precio y stock.",
             _obj({"query": {"type": "string"}, "max_price": {"type": "number", "minimum": 0},
                   "brand": {"type": "string"}, "color": {"type": "string"},
                   "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": getattr(cfg, "search_k", 5)}},
                  ["query"]), search_products),
        Tool("get_product_details", "Ficha de un producto y sus atributos verificables ('desconocido' si no constan).",
             _obj({"product_id": pid}, ["product_id"]), get_product_details),
        Tool("compare_products", "Compara 2-5 productos por precio, stock y atributos.",
             _obj({"product_ids": {"type": "array"}, "attributes": {"type": "array"}}, ["product_ids"]), compare_products),
        Tool("check_stock", "Consulta existencias de un producto para una cantidad.",
             _obj({"product_id": pid, "qty": dict(qty, default=1)}, ["product_id"]), check_stock),
        Tool("add_to_cart", "Añade un producto al carrito (valida stock y presupuesto).",
             _obj({"product_id": pid, "qty": dict(qty, default=1), "need": {"type": "string"}}, ["product_id"]), add_to_cart),
        Tool("update_cart_quantity", "Cambia la cantidad de un producto del carrito (valida stock y presupuesto).",
             _obj({"product_id": pid, "qty": qty}, ["product_id", "qty"]), update_cart_quantity),
        Tool("remove_from_cart", "Quita un producto del carrito.", _obj({"product_id": pid}, ["product_id"]), remove_from_cart),
        Tool("view_cart", "Muestra el carrito, total y presupuesto restante.", _obj({}), view_cart),
        Tool("set_budget", "Fija el presupuesto total de la compra.",
             _obj({"amount": {"type": "number", "minimum": 0.01}}, ["amount"]), set_budget),
        Tool("set_preference", "Guarda una preferencia (brand, color, max_unit_price, other; size=talla y cut=corte "
             "para tiendas con variantes).",
             _obj({"key": {"type": "string", "enum": list(_PREF_KEYS)}, "value": {"type": "string"}}, ["key", "value"]),
             set_preference),
        Tool("create_quote", "Genera la cotización del carrito (valida stock y presupuesto). Obligatoria antes de confirmar.",
             _obj({}), create_quote),
        Tool("confirm_order", "Registra el pedido de la última cotización. SOLO tras confirmación explícita del usuario.",
             _obj({}), confirm_order),
        Tool("find_alternative", "Busca alternativas similares con stock y dentro del presupuesto.",
             _obj({"product_id": pid, "qty": qty, "query": {"type": "string", "description": "necesidad del usuario"}},
                  ["product_id"]), find_alternative),
        Tool("ask_clarification", "Pregunta al usuario cuando falta información o hay ambigüedad relevante.",
             _obj({"question": {"type": "string"}}, ["question"]), ask_clarification),
    ]
    return {t.name: t for t in tools}


def execute_tool(tools: dict, ctx: ToolContext, name: str, args: dict) -> dict:
    """Valida y ejecuta una herramienta; nunca lanza excepciones hacia el agente."""
    t0 = time.perf_counter()
    try:
        tool = tools.get(name)
        if tool is None:
            return fail(f"Herramienta desconocida '{name}'. Disponibles: {sorted(tools)}")
        clean, err = validate_args(tool.parameters, args or {})
        if err:
            return fail(err)
        return tool.fn(ctx, **clean)
    except Exception as e:  # pragma: no cover - defensa
        return fail(f"Error interno en {name}: {type(e).__name__}: {e}")
    finally:
        ctx.timings["tools_s"] += time.perf_counter() - t0
