"""Vista del estado de una sesión de demo (carrito, pedidos, stock relevante) en dicts JSON.

La comparten la demo Gradio (:mod:`shopassist.demo_ui`) y la demo web FastAPI (``webapp``):
todo se lee de la tienda en cada llamada, así que refleja el estado real tras cada turno.
"""
from __future__ import annotations

from typing import Iterable, Optional


def short(title: str, n: int = 60) -> str:
    title = " ".join((title or "").split())
    return title if len(title) <= n else title[: n - 1] + "…"


def cart_view(store, cart_id: str) -> dict:
    """``{"lines": [...], "n_items": int, "total": float}`` con el stock actual de cada línea."""
    lines = store.get_cart(cart_id)
    return {"lines": lines, "n_items": sum(l["qty"] for l in lines),
            "total": round(sum(l["line_total"] for l in lines), 2)}


def session_orders(store, cart_id: str) -> list:
    """Pedidos de este carrito, del más reciente al más antiguo."""
    return [o for o in reversed(store.list_orders()) if o.get("cart_id") == cart_id]


def relevant_pids(cart_lines: list, orders: list, last_results: Optional[Iterable] = None, n_results: int = 5) -> list:
    """Productos relevantes para el panel de stock: carrito, líneas de pedidos y últimos resultados."""
    pids: list = []
    for p in ([l["product_id"] for l in cart_lines]
              + [l["product_id"] for o in orders for l in o["lines"]]
              + list(last_results or [])[:n_results]):
        if p not in pids:
            pids.append(p)
    return pids


def stock_view(store, pids: Iterable) -> list:
    out = []
    for p in pids:
        info = store.get_product(p)
        if info is not None:
            out.append({"product_id": p, "title": info["title"], "price": info["price"], "stock": info["stock"]})
    return out


def session_view(store, cart_id: str, agent=None) -> dict:
    """Estado completo de la sesión: carrito, pedidos, stock relevante y últimos resultados del agente."""
    cart = cart_view(store, cart_id)
    orders = session_orders(store, cart_id)
    last = list(getattr(getattr(agent, "state", None), "last_result_list", []) or [])
    return {"cart": cart, "orders": orders, "last_results": last,
            "stock": stock_view(store, relevant_pids(cart["lines"], orders, last))}
