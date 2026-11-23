"""Estado de la conversación (memoria del agente)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ConversationState:
    """Memoria explícita de la conversación; se resume en cada prompt del LLM.

    - ``requested_items``: necesidades del usuario ``{need, qty, product_id, status}`` con
      status en {"pendiente", "en_carrito", "sin_stock", "pedido"}.
    - ``seen_product_ids``: ids que aparecieron en resultados de herramientas; el agente
      solo puede operar sobre ellos (anclaje: no puede inventar ids).
    - ``transcript_pending`` / ``pending_utterance``: para entrada por voz, el agente repite
      lo que entendió antes de ejecutar una compra (corrección de transcripción).
    - ``pending_alternative``: alternativa propuesta tras un fallo de stock/precio.
    """

    cart_id: str = "c1"
    currency: str = "EUR"
    history: list = field(default_factory=list)
    budget: Optional[float] = None
    preferences: dict = field(default_factory=lambda: {"brand": None, "color": None,
                                                        "max_unit_price": None, "other": []})
    requested_items: list = field(default_factory=list)
    last_results: dict = field(default_factory=dict)
    last_result_list: list = field(default_factory=list)
    seen_product_ids: list = field(default_factory=list)
    last_quote: Optional[dict] = None
    pending_confirmation: bool = False
    pending_clarification: Optional[str] = None
    pending_alternative: Optional[dict] = None
    pending_action: Optional[dict] = None   # acción a completar tras una aclaración
    last_order_id: Optional[str] = None
    last_order_quote_id: Optional[str] = None
    confirmed_order_ids: list = field(default_factory=list)
    transcript_pending: bool = False
    pending_utterance: Optional[str] = None
    turn_index: int = 0

    # ------------------------------------------------------------------ helpers
    def remember_products(self, pids, key: Optional[str] = None):
        pids = [str(p) for p in pids]
        for p in pids:
            if p not in self.seen_product_ids:
                self.seen_product_ids.append(p)
        if key is not None:
            self.last_results[key] = pids
        if pids:
            self.last_result_list = pids

    def find_item(self, product_id=None, need=None) -> Optional[dict]:
        for it in self.requested_items:
            if product_id is not None and it.get("product_id") == str(product_id):
                return it
        if need is not None:
            for it in self.requested_items:
                if it.get("need") == need:
                    return it
        return None

    def invalidate_quote(self, reason: str = ""):
        if self.last_quote is not None and self.last_quote.get("quote_id") != self.last_order_quote_id:
            self.last_quote = dict(self.last_quote, valid=False, invalidated=reason or True)
        self.pending_confirmation = False

    def summary(self) -> str:
        """Resumen compacto en español para el prompt del LLM."""
        prefs = {k: v for k, v in self.preferences.items() if v}
        items = [f"{i['qty']}x {i['need']} -> {i.get('product_id') or '?'} ({i['status']})" for i in self.requested_items]
        q = self.last_quote
        quote = (f"{q['quote_id']} total={q['total']:.2f} válida={q.get('valid')}" if q else "ninguna")
        return (f"presupuesto={self.budget if self.budget is not None else 'no indicado'} {self.currency}; "
                f"preferencias={prefs or 'ninguna'}; pedidos_del_usuario={items or 'ninguno'}; "
                f"cotización={quote}; esperando_confirmación={self.pending_confirmation}; "
                f"alternativa_pendiente={bool(self.pending_alternative)}; último_pedido={self.last_order_id}; "
                f"últimos_resultados={self.last_result_list[:5]}")

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))
