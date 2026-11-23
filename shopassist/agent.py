"""Agente de compras: controlador propio estilo ReAct + guardas deterministas.

Flujo de ``handle(text, source)``:
1. NLU determinista (``nlu.parse``): cantidades, presupuesto, confirmación, correcciones.
2. Voz: si la frase contiene cantidades o importes relevantes para una compra, el agente
   repite lo que entendió ("Entendí: ... ¿es correcto?") antes de ejecutar nada; el usuario
   puede confirmar o corregir ("no, dije tres").
3. Guardas en código (no delegadas al LLM): actualización de presupuesto/preferencias,
   confirmación explícita -> ``confirm_order`` (idempotente), fallo de stock al confirmar ->
   ``find_alternative`` automático, aceptación de alternativas, cotización invalidada al
   cambiar el carrito.
4. Política: ``LLMPolicy`` (el LLM elige herramientas emitiendo UN objeto JSON por paso,
   máximo ``cfg.max_steps`` pasos) o ``RuleBasedPolicy`` (determinista; sin LLM, usada en
   pruebas/CI y como ablación "sin LLM").
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .attributes import UNKNOWN, answer_attribute, canonical_attr
from .config import Config
from .llm import parse_llm_json
from .nlu import COLOR_WORDS, NLUResult, apply_correction, parse
from .retrieval import color_matches, tokenize
from .state import ConversationState
from .textutil import fmt_price, norm, truncate
from .tools import ToolContext, build_tools, execute_tool


@dataclass
class AgentTurn:
    reply: str
    ops: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    needs_clarification: bool = False
    timings: dict = field(default_factory=dict)
    source: str = "text"
    policy: str = "rules"
    nlu: dict = field(default_factory=dict)


def short_title(title: str, n_words: int = 9, n_chars: int = 70) -> str:
    words = (title or "").split()
    s = " ".join(words[:n_words])
    return truncate(s, n_chars)


class ShoppingAgent:
    """Agente conversacional. ``llm=None`` usa la política por reglas."""

    def __init__(self, store, retriever, llm=None, cart_id: str = "c1", cfg: Optional[Config] = None):
        self.store = store
        self.retriever = retriever
        if getattr(retriever, "store", None) is None and hasattr(retriever, "attach_store"):
            retriever.attach_store(store)
        self.llm = llm
        self.cfg = cfg or Config()
        self.cart_id = cart_id
        self.tools = build_tools(self.cfg)
        self.known_brands = self._known_brands()
        self.state = ConversationState(cart_id=cart_id, currency=self.cfg.currency)
        self.ctx = ToolContext(store, retriever, self.state, self.cfg)
        self.rules = RuleBasedPolicy(self)
        self.llm_policy = LLMPolicy(self) if llm is not None else None

    # ------------------------------------------------------------------ utilidades
    def _known_brands(self) -> set:
        cat = getattr(self.store, "_catalog", {})
        brand_count: dict = {}
        for p in cat.values():
            b = norm(p.get("brand", ""))
            if re.fullmatch(r"[a-z0-9\-]{4,}", b or ""):
                brand_count[b] = brand_count.get(b, 0) + 1
        title_count: dict = {}
        for p in cat.values():
            for w in set(re.findall(r"[a-z0-9\-]{4,}", norm(p.get("title", "")))):
                if w in brand_count:
                    title_count[w] = title_count.get(w, 0) + 1
        return {b for b, c in brand_count.items()
                if c >= 0.5 * title_count.get(b, 0) and b not in COLOR_WORDS and b not in {"generic", "generico", "unbranded"}}

    def reset(self):
        """Nueva conversación con la misma tienda (vacía el carrito de este cliente)."""
        self.store.clear_cart(self.cart_id)
        self.state = ConversationState(cart_id=self.cart_id, currency=self.cfg.currency)
        self.ctx = ToolContext(self.store, self.retriever, self.state, self.cfg)

    def call(self, name: str, args: Optional[dict] = None, ops: Optional[list] = None, by: str = "rules") -> dict:
        res = execute_tool(self.tools, self.ctx, name, args or {})
        if ops is not None:
            ops.append({"name": name, "args": dict(args or {}), "result": res, "ok": bool(res.get("ok")), "by": by})
        return res

    def price(self, v) -> str:
        return fmt_price(v, self.cfg.currency)

    def cart(self) -> list:
        return self.store.get_cart(self.cart_id)

    # ------------------------------------------------------------------ API principal
    def handle(self, text: str, source: str = "text") -> AgentTurn:
        t0 = time.perf_counter()
        self.ctx.timings = {"tools_s": 0.0, "retrieval_s": 0.0}
        self.ctx.user_confirmed = False
        self._llm_s = 0.0
        st = self.state
        st.turn_index += 1
        st.history.append({"role": "user", "text": text, "source": source})
        nlu = parse(text, self.known_brands)
        ops: list = []
        policy = "llm" if self.llm_policy is not None else "rules"
        try:
            reply, clar, used = self._dispatch(text, nlu, source, ops)
            policy = used or policy
        except Exception as e:  # defensa: nunca romper la conversación
            reply, clar = f"Lo siento, ocurrió un error interno ({type(e).__name__}). ¿Puedes repetirlo?", True
        st.pending_clarification = reply if clar else None
        st.history.append({"role": "assistant", "text": reply})
        tim = self.ctx.timings
        timings = {"llm_s": round(self._llm_s, 4), "retrieval_s": round(tim["retrieval_s"], 4),
                   "tools_s": round(max(tim["tools_s"] - tim["retrieval_s"], 0.0), 4),
                   "total_s": round(time.perf_counter() - t0, 4)}
        sd = st.to_dict()
        sd["cart"] = self.call("view_cart")["data"]
        return AgentTurn(reply=reply, ops=ops, state=sd, needs_clarification=bool(clar), timings=timings,
                         source=source, policy=policy, nlu=nlu.to_dict())

    # ------------------------------------------------------------------ despacho
    def _dispatch(self, text: str, nlu: NLUResult, source: str, ops: list):
        st = self.state
        if st.transcript_pending:
            pending = st.pending_utterance or ""
            st.transcript_pending, st.pending_utterance = False, None
            if nlu.correction or ("deny" in nlu.intents and (nlu.quantity is not None or nlu.amount is not None or nlu.items)):
                corr = nlu.correction or (None, text)
                if corr[1] and "deny" in nlu.intents and not nlu.correction:
                    corr = (None, re.sub(r"^\s*no\b[,.]?\s*", "", nlu.normalized))
                fixed = apply_correction(pending, corr)
                fixed_nlu = parse(fixed, self.known_brands)
                if source == "voice" and self._needs_echo(fixed_nlu):
                    # se vuelve a verificar la frase corregida antes de ejecutar la compra
                    st.transcript_pending, st.pending_utterance = True, fixed_nlu.normalized
                    return "Corregido. " + self._echo(fixed_nlu), False, "rules"
                reply, clar, used = self._process(fixed_nlu, ops)
                return f"Corregido: «{fixed}». " + reply, clar, used
            if "confirm" in nlu.intents and "deny" not in nlu.intents:
                return self._process(parse(pending, self.known_brands), ops)
            if "deny" in nlu.intents and not nlu.items:
                return "De acuerdo, no ejecuto nada. ¿Qué quieres comprar exactamente?", True, "rules"
            # en otro caso la nueva frase reemplaza a la pendiente
        if source == "voice" and self._needs_echo(nlu):
            st.transcript_pending, st.pending_utterance = True, nlu.normalized
            return self._echo(nlu), False, "rules"
        return self._process(nlu, ops)

    @staticmethod
    def _needs_echo(nlu: NLUResult) -> bool:
        """Solo se verifica la transcripción si la frase modificaría el carrito con una
        cantidad o importe (comprar N, añadir N del primero, cambiar cantidad)."""
        it = set(nlu.intents)
        if "search_only" in it or nlu.explicit_confirmation:
            return False
        has_ref = nlu.ordinal is not None or nlu.superlative is not None or nlu.deictic
        modifies = ("buy" in it) or ("change_qty" in it) or ("add" in it and has_ref)
        has_number = nlu.quantity is not None or (("buy" in it) and (nlu.budget is not None or nlu.max_unit_price is not None))
        return modifies and has_number

    def _echo(self, nlu: NLUResult) -> str:
        parts = []
        if "change_qty" in nlu.intents and nlu.quantity is not None and not nlu.items:
            parts.append(f"cambiar la cantidad a {nlu.quantity}")
        elif not nlu.items and nlu.quantity is not None:
            ref = ("la opción " + str(nlu.ordinal + 1)) if nlu.ordinal is not None and nlu.ordinal >= 0 else \
                ("la más barata" if nlu.superlative == "cheapest" else "el producto indicado")
            parts.append(f"añadir {nlu.quantity} unidad{'es' if nlu.quantity != 1 else ''} de {ref}")
        for it in nlu.items:
            q = it.qty if it.qty is not None else (nlu.quantity if len(nlu.items) == 1 and nlu.quantity else 1)
            parts.append(f"comprar {q} × «{it.query}»")
        if nlu.budget is not None:
            parts.append(f"presupuesto máximo de {self.price(nlu.budget)}")
        if nlu.max_unit_price is not None:
            parts.append(f"precio máximo por unidad de {self.price(nlu.max_unit_price)}")
        return ("Entendí: " + ", ".join(parts) + ". ¿Es correcto? Responde «sí» o corrígeme "
                "(por ejemplo: «no, dije dos»).")

    def _process(self, nlu: NLUResult, ops: list):
        st = self.state
        pre = self._apply_slots(nlu, ops)
        if self._apply_variants(nlu, ops) and not nlu.explicit_confirmation and self.cart():
            # el usuario solo indicó talla/color/corte (respuesta a la aclaración): re-cotizar
            txt, clar = self.quote_and_ask(ops)
            return pre + "Anotado. " + txt, clar, "rules"
        if st.pending_alternative and {"change_qty", "remove", "buy", "change_pref", "correction"} & set(nlu.intents):
            st.pending_alternative = None  # el usuario cambió de idea: se descarta la alternativa
        # --- guardas deterministas de confirmación
        if nlu.explicit_confirmation:
            r, c = self.rules.handle_confirmation(nlu, ops)
            return pre + r, c, "rules"
        if "deny" in nlu.intents and not (set(nlu.intents) - {"deny"}) and not nlu.items:
            if st.pending_alternative:
                st.pending_alternative = None
                return pre + "De acuerdo, no hago el cambio. ¿Prefieres otro producto o quitarlo del carrito?", True, "rules"
            if st.pending_confirmation:
                st.pending_confirmation = False
                return pre + "De acuerdo, no registro el pedido. ¿Qué quieres cambiar?", True, "rules"
        if self.llm_policy is not None:
            r, c = self.llm_policy.run(nlu, ops)
            return pre + r, c, "llm"
        r, c = self.rules.run(nlu, ops)
        return pre + r, c, "rules"

    def _apply_slots(self, nlu: NLUResult, ops: list) -> str:
        """Actualiza presupuesto/preferencias detectados por la NLU (en código)."""
        st, msgs = self.state, []
        budget = nlu.budget
        if budget is None and nlu.amount is not None and not nlu.items and nlu.max_unit_price is None \
                and st.pending_clarification and "presupuesto" in norm(st.pending_clarification):
            budget = nlu.amount
        if budget is not None and budget != st.budget:
            res = self.call("set_budget", {"amount": budget}, ops, by="guard")
            if res["ok"] and res["data"]["cart_exceeds_budget"]:
                msgs.append(f"Ojo: con el nuevo presupuesto ({self.price(budget)}) el carrito actual lo supera. ")
        if nlu.max_unit_price is not None and nlu.max_unit_price != st.preferences.get("max_unit_price"):
            self.call("set_preference", {"key": "max_unit_price", "value": nlu.max_unit_price}, ops, by="guard")
        if nlu.brand and ("change_pref" in nlu.intents or re.search(r"\bmarca\b", nlu.normalized)):
            self.call("set_preference", {"key": "brand", "value": nlu.brand}, ops, by="guard")
        return "".join(msgs)

    def _apply_variants(self, nlu: NLUResult, ops: list) -> bool:
        """Solo en tiendas con variantes (``store.extract_variants``, p. ej. la tienda real):
        guarda talla/color/corte como preferencias y los quita de la consulta de búsqueda.
        Devuelve True si la frase solo contenía variantes. Con ``Store`` no hace nada."""
        ex = getattr(self.store, "extract_variants", None)
        if ex is None:
            return False
        found, _rest, only = ex(nlu.normalized)
        if not found:
            return False
        for k, v in found.items():
            if self.state.preferences.get(k) != v:
                self.call("set_preference", {"key": k, "value": v}, ops, by="guard")
        for it in nlu.items:
            it.query = ex(it.query)[1] or it.query
        if only and self.cart():
            if self.store.apply_preferences(self.cart_id, self.state.preferences, overwrite_last=True):
                self.state.invalidate_quote("variante modificada")
        return only

    @staticmethod
    def remote_note(order: dict) -> str:
        """Aviso del modo de envío en tiendas remotas (dry-run / pedido real)."""
        r = order.get("remote") if isinstance(order, dict) else None
        if not r:
            return ""
        if r.get("mode") == "dry_run":
            return (" Modo simulación (dry-run): NO se envió a la tienda real; "
                    "guardé el payload que se enviaría.")
        if r.get("status") == "enviado":
            return f" Pedido creado en la tienda real con el código {r.get('remote_order_id')}."
        return f" Atención: estado en la tienda real «{r.get('status')}» ({r.get('detail')})."

    # ------------------------------------------------------------------ textos
    def fmt_product(self, p: dict, with_stock: bool = False) -> str:
        brand = p.get("brand") or UNKNOWN
        color = p.get("color") or UNKNOWN
        extra = f"marca {brand}" + (f", color {color}" if color != UNKNOWN else "")
        s = f"{short_title(p['title'])} ({extra}) a {self.price(p['price'])}"
        if with_stock:
            s += " (agotado)" if p.get("stock", 0) <= 0 else f" ({p['stock']} en stock)"
        return s

    def fmt_quote(self, q: dict) -> str:
        lines = [f"{i['qty']} × {short_title(i['title'], 7, 55)}"
                 + (f" ({i['variant_label']})" if i.get("variant_label") else "")
                 + f" ({self.price(i['unit_price'])} c/u)" for i in q["items"]]
        s = f"Cotización {q['quote_id']}: " + "; ".join(lines) + f". Total: {self.price(q['total'])}"
        if q.get("budget") is not None:
            s += f" (presupuesto {self.price(q['budget'])})"
        return s + "."

    def quote_and_ask(self, ops: list, by: str = "rules") -> tuple[str, bool]:
        res = self.call("create_quote", {}, ops, by=by)
        q = res["data"]
        if res["ok"]:
            return self.fmt_quote(q) + " ¿Confirmas el pedido?", False
        probs = q.get("problems", []) if q else []
        txt = "; ".join(p["detail"] for p in probs) or res["error"]
        if probs and all(p.get("type") == "variant" for p in probs):
            return f"Para preparar el pedido me falta un dato: {txt}. ¿Cuál prefieres?", True
        return f"No puedo cotizar todavía: {txt}. ¿Qué quieres ajustar?", True


# ====================================================================== política por reglas
class RuleBasedPolicy:
    """Política determinista (sin LLM). Cubre compra, aclaración, cambios, agotados,
    comparación, atributos y confirmación."""

    def __init__(self, agent: ShoppingAgent):
        self.a = agent

    @property
    def st(self) -> ConversationState:
        return self.a.state

    # ---------------------------------------------------------- confirmación (guarda)
    def handle_confirmation(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        if st.pending_alternative:
            return self.accept_alternative(ops)
        q = st.last_quote
        if q is not None and (st.pending_confirmation or st.last_order_quote_id == q["quote_id"]):
            a.ctx.user_confirmed = True
            res = a.call("confirm_order", {}, ops, by="guard")
            a.ctx.user_confirmed = False
            if res["ok"]:
                o = res["data"]["order"]
                note = a.remote_note(o)
                if res["data"]["duplicate"]:
                    return (f"El pedido {o['order_id']} ya estaba registrado por {a.price(o['total'])}; "
                            f"no se ha duplicado.{note}"), False
                return (f"¡Listo! Pedido {o['order_id']} registrado por {a.price(o['total'])}. "
                        f"Inventario actualizado.{note} ¿Necesitas algo más?"), False
            return self.on_confirm_failure(res, ops)
        if a.cart():
            return a.quote_and_ask(ops)
        return "¿Qué deseas confirmar? Todavía no hay ninguna compra en curso.", True

    def on_confirm_failure(self, res: dict, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        probs = (res.get("data") or {}).get("problems", [])
        stock_p = [p for p in probs if p.get("type") == "stock"]
        price_p = [p for p in probs if p.get("type") == "price"]
        if stock_p:
            p = stock_p[0]
            prod = a.store.get_product(p["product_id"])
            msg = (f"No pude registrar el pedido: el stock de {short_title(prod['title'])} cambió "
                   f"(pedías {p['requested']}, ahora hay {p['available']}). ")
            alt = a.call("find_alternative", {"product_id": p["product_id"], "qty": p["requested"]}, ops, by="guard")
            alts = (alt.get("data") or {}).get("alternatives", [])
            if alts:
                best = alts[0]
                st.pending_alternative = {"original": p["product_id"], "alternative": best["product_id"],
                                          "qty": p["requested"], "reason": "stock"}
                msg += f"Te propongo como alternativa {a.fmt_product(best, True)}. ¿Lo cambio y te preparo una nueva cotización?"
                if p["available"] > 0:
                    msg += f" También puedo reducir la cantidad a {p['available']}."
                return msg, True
            if p["available"] > 0:
                return msg + (f"No encontré alternativas con stock dentro del presupuesto. "
                              f"¿Quieres reducir la cantidad a {p['available']}?"), True
            return msg + ("No encontré ninguna alternativa con stock dentro del presupuesto, así que no hay "
                          "una solución para ese producto. ¿Quieres quitarlo del carrito?"), True
        if price_p:
            txt, clar = a.quote_and_ask(ops)
            return "El precio cambió desde la cotización. " + txt, clar
        txt = "; ".join(p.get("detail", "") for p in probs) or res.get("error", "")
        return f"No pude registrar el pedido: {txt}. ¿Qué quieres ajustar?", True

    def accept_alternative(self, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        alt = st.pending_alternative
        st.pending_alternative = None
        if any(i["product_id"] == alt["original"] for i in a.cart()):
            a.call("remove_from_cart", {"product_id": alt["original"]}, ops)
        res = a.call("add_to_cart", {"product_id": alt["alternative"], "qty": alt["qty"]}, ops)
        if not res["ok"]:
            return f"No pude añadir la alternativa: {res['error']}. ¿Qué prefieres hacer?", True
        txt, clar = a.quote_and_ask(ops)
        return "Hecho, cambié el producto. " + txt, clar

    # ---------------------------------------------------------- resolución de referencias
    def cart_target(self, nlu: NLUResult):
        """Devuelve (línea_del_carrito | None, ambiguo: bool)."""
        cart = self.a.cart()
        if not cart:
            return None, False
        if len(cart) == 1:
            return cart[0], False
        if nlu.items:
            q = set(tokenize(" ".join(i.query for i in nlu.items)))
            scored = sorted(((len(q & set(tokenize(c["title"]))), k) for k, c in enumerate(cart)), reverse=True)
            if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return cart[scored[0][1]], False
        if nlu.ordinal is not None and -len(cart) <= nlu.ordinal < len(cart) and re.search(r"carrito", nlu.normalized):
            return cart[nlu.ordinal], False
        return None, True

    def pick_from_results(self, nlu: NLUResult, pids: list) -> Optional[str]:
        if not pids:
            return None
        if nlu.ordinal is not None:
            return pids[nlu.ordinal] if -len(pids) <= nlu.ordinal < len(pids) else None
        if nlu.superlative == "cheapest":
            prods = [self.a.store.get_product(p) for p in pids]
            prods = [p for p in prods if p["stock"] > 0] or prods
            return min(prods, key=lambda p: p["price"])["product_id"]
        if nlu.superlative == "best":
            return pids[0]
        return None

    def ask_choice(self, pids: list, ops: list, intro: str = "") -> tuple[str, bool]:
        a = self.a
        opts = "; ".join(f"{k + 1}) {a.fmt_product(a.store.get_product(p), True)}" for k, p in enumerate(pids[:5]))
        q = (intro + f"Opciones: {opts}. ¿Cuál prefieres (por ejemplo «la opción 2» o «el más barato») "
             f"y cuántas unidades?")
        a.call("ask_clarification", {"question": q}, ops)
        return q, True

    # ---------------------------------------------------------- bucle principal
    def run(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        intents = set(nlu.intents)
        pend = st.pending_action

        # Respuesta a una aclaración previa sobre qué línea cambiar
        if pend and pend.get("type") == "change_qty" and (nlu.items or nlu.ordinal is not None):
            st.pending_action = None
            line, amb = self.cart_target(nlu)
            if line:
                return self.change_qty(line, pend["qty"], ops)
        st.pending_action = None

        if nlu.correction and "change_qty" not in intents:
            new = nlu.correction[1]
            m = re.search(r"\b(\d+)\b", new)
            if m and a.cart() and not re.search(r"euros?", new):
                nlu.quantity = int(m.group(1))
                intents.add("change_qty")
        if "remove" in intents:
            line, amb = self.cart_target(nlu)
            if amb:
                return self.ask_cart_line("quitar", ops)
            if not line:
                return "Tu carrito está vacío; no hay nada que quitar.", False
            a.call("remove_from_cart", {"product_id": line["product_id"]}, ops)
            msg = f"Quité {short_title(line['title'])} del carrito. "
            if a.cart():
                txt, clar = a.quote_and_ask(ops)
                return msg + txt, clar
            return msg + "El carrito quedó vacío. ¿Buscamos otra cosa?", False
        if "change_qty" in intents and nlu.quantity is not None and not ("buy" in intents and not a.cart()):
            line, amb = self.cart_target(nlu)
            if amb:
                st.pending_action = {"type": "change_qty", "qty": nlu.quantity}
                return self.ask_cart_line(f"cambiar a {nlu.quantity} unidades", ops)
            if line:
                return self.change_qty(line, nlu.quantity, ops)
            if st.last_result_list:
                return self.select_and_add(st.last_result_list[0] if nlu.ordinal is None else
                                           self.pick_from_results(nlu, st.last_result_list), nlu.quantity, ops)
            return "¿De qué producto quieres esa cantidad? Aún no hay nada en el carrito.", True
        if "change_pref" in intents and not nlu.items:
            return self.change_preference(nlu, ops)
        if "view_cart" in intents and not nlu.items:
            v = a.call("view_cart", {}, ops)["data"]
            if not v["items"]:
                return "Tu carrito está vacío.", False
            lines = "; ".join(f"{i['qty']} × {short_title(i['title'])} ({a.price(i['line_total'])})" for i in v["items"])
            extra = "" if v["budget"] is None else f" Presupuesto: {a.price(v['budget'])}, te quedan {a.price(v['remaining'])}."
            return f"En tu carrito: {lines}. Total: {a.price(v['total'])}.{extra}", False
        if "stock_query" in intents:
            return self.stock_query(nlu, ops)
        if nlu.quantity == 0 and ({"add", "buy", "change_qty"} & intents):
            q = "La cantidad debe ser al menos 1 unidad. ¿Cuántas unidades quieres?"
            a.call("ask_clarification", {"question": q}, ops)
            return q, True
        if "compare" in intents:
            return self.compare(nlu, ops)
        if "attribute" in intents or ("details" in intents and not ("buy" in intents and nlu.items and nlu.ordinal is None)):
            return self.answer_attribute_q(nlu, ops)
        if "compare" in intents:
            return self.compare(nlu, ops)
        # referencias sin producto explícito: "el segundo", "el más barato", "eso"
        has_ref = nlu.ordinal is not None or nlu.superlative is not None or nlu.deictic
        if has_ref and not nlu.items:
            pid = self.pick_from_results(nlu, st.last_result_list)
            if pid is None and nlu.deictic and len(st.last_result_list) == 1:
                pid = st.last_result_list[0]
            if pid is None:
                if st.last_result_list:
                    return self.ask_choice(st.last_result_list, ops, "No tengo claro a cuál te refieres. ")
                q = ("¿A qué producto te refieres? Todavía no te he mostrado opciones. "
                     "Dime qué buscas (por ejemplo «audífonos bluetooth») y, si quieres, cantidad y presupuesto.")
                a.call("ask_clarification", {"question": q}, ops)
                return q, True
            return self.select_and_add(pid, nlu.quantity if nlu.quantity is not None else 1, ops)
        if "buy" in intents and nlu.items:
            return self.buy(nlu, ops)
        if "quote" in intents:
            if not a.cart():
                return "Tu carrito está vacío, así que no hay nada que cotizar. ¿Qué te gustaría comprar?", True
            return a.quote_and_ask(ops)
        if intents & {"set_budget", "set_unit_price"}:
            msg = []
            if nlu.budget is not None:
                msg.append(f"Presupuesto fijado en {a.price(st.budget)}.")
            if nlu.max_unit_price is not None:
                msg.append(f"Buscaré productos de menos de {a.price(nlu.max_unit_price)} por unidad.")
            if a.cart():
                txt, clar = a.quote_and_ask(ops)
                return " ".join(msg) + " " + txt, clar
            return " ".join(msg) + " ¿Qué producto buscas?", True
        if "confirm" in intents and not a.cart() and not st.last_quote:
            return "¿Qué deseas confirmar? Todavía no hay ninguna compra en curso.", True
        if "deny" in intents:
            return "De acuerdo. ¿Qué te gustaría hacer?", True
        q = "No estoy seguro de qué necesitas. ¿Qué producto buscas y cuántas unidades?"
        a.call("ask_clarification", {"question": q}, ops)
        return q, True

    # ---------------------------------------------------------- acciones
    def ask_cart_line(self, what: str, ops: list) -> tuple[str, bool]:
        a = self.a
        opts = "; ".join(f"{k + 1}) {short_title(i['title'])} ({i['qty']} uds.)" for k, i in enumerate(a.cart()))
        q = f"Tienes varios productos en el carrito: {opts}. ¿Cuál quieres {what}?"
        a.call("ask_clarification", {"question": q}, ops)
        return q, True

    def change_qty(self, line: dict, qty: int, ops: list) -> tuple[str, bool]:
        a = self.a
        if qty == line["qty"]:
            txt, clar = a.quote_and_ask(ops)
            return f"Ya tenías {qty} unidades de {short_title(line['title'])}. " + txt, clar
        res = a.call("update_cart_quantity", {"product_id": line["product_id"], "qty": qty}, ops)
        if not res["ok"]:
            d = res.get("data") or {}
            if d.get("reason") == "budget":
                return (f"No puedo cambiar a {qty}: {res['error']}. Mantengo {line['qty']} unidades; "
                        f"con tu presupuesto caben como máximo {d.get('max_qty_affordable')}. ¿Qué prefieres?"), True
            if d.get("reason") == "stock":
                return (f"No puedo cambiar a {qty}: solo hay {d.get('available')} en stock. "
                        f"Mantengo {line['qty']} unidades. ¿Qué prefieres?"), True
            return f"No pude cambiar la cantidad: {res['error']}", True
        txt, clar = a.quote_and_ask(ops)
        budget = f" y mantengo tu presupuesto de {a.price(self.st.budget)}" if self.st.budget is not None else ""
        return f"Cambié la cantidad de {short_title(line['title'])} de {line['qty']} a {qty}{budget}. " + txt, clar

    def change_preference(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        if nlu.color:
            a.call("set_preference", {"key": "color", "value": nlu.color}, ops)
        line, amb = self.cart_target(nlu)
        if amb:
            line = None
            last = next((i for i in reversed(st.requested_items) if i["status"] == "en_carrito"), None)
            if last:
                line = next((c for c in a.cart() if c["product_id"] == last["product_id"]), None)
        if line is None:
            what = f"color {nlu.color}" if nlu.color else f"marca {nlu.brand}"
            last_q = next(reversed(st.last_results), None) if st.last_results else None
            if last_q:
                base = " ".join(w for w in last_q.split() if w not in COLOR_WORDS)
                args = {"query": " ".join(x for x in (base, nlu.color or "", nlu.brand or "") if x), "k": 5}
                if nlu.color:
                    args["color"] = nlu.color
                if nlu.brand:
                    args["brand"] = nlu.brand
                if st.budget is not None:
                    args["max_price"] = st.budget
                res = a.call("search_products", args, ops)
                pids = [r["product_id"] for r in res["data"]["results"]]
                if pids:
                    return self.ask_choice(pids, ops, f"Anotado: prefieres {what}. Para «{base}» tengo: ")
            return f"Anotado: prefieres {what}. ¿Qué producto buscas?", True
        item = st.find_item(product_id=line["product_id"]) or {"need": short_title(line["title"], 5), "qty": line["qty"]}
        base = " ".join(w for w in item["need"].split() if w not in COLOR_WORDS)
        query = " ".join(x for x in (base, nlu.color or "", nlu.brand or "") if x)
        left = a.call("view_cart", {})["data"]
        budget_left = None if st.budget is None else st.budget - (left["total"] - line["line_total"])
        args = {"query": query, "k": 10}
        if nlu.color:
            args["color"] = nlu.color
        if nlu.brand:
            args["brand"] = nlu.brand
        res = a.call("search_products", args, ops)
        cands = [r for r in res["data"]["results"] if r["product_id"] != line["product_id"]
                 and r["stock"] >= line["qty"]
                 and (budget_left is None or r["price"] * line["qty"] <= budget_left + 1e-9)
                 and (not nlu.color or color_matches(nlu.color, f"{r['color']} {r['title']}"))
                 and (not nlu.brand or norm(nlu.brand) in norm(r["brand"] + " " + r["title"]))]
        what = f"color {nlu.color}" if nlu.color else f"marca {nlu.brand}"
        if not cands:
            return (f"No encontré {base} en {what} con stock para {line['qty']} unidades dentro del presupuesto. "
                    f"Mantengo {short_title(line['title'])}. ¿Quieres otra opción?"), True
        new = cands[0]
        a.call("remove_from_cart", {"product_id": line["product_id"]}, ops)
        r2 = a.call("add_to_cart", {"product_id": new["product_id"], "qty": line["qty"], "need": base}, ops)
        if not r2["ok"]:
            a.call("add_to_cart", {"product_id": line["product_id"], "qty": line["qty"]}, ops)
            return f"No pude hacer el cambio: {r2['error']}. Mantengo el producto anterior.", True
        txt, clar = a.quote_and_ask(ops)
        keep = f"Mantengo la cantidad ({line['qty']})" + (f" y el presupuesto ({a.price(st.budget)})" if st.budget else "")
        return f"Cambié a {a.fmt_product(new)} en {what}. {keep}. " + txt, clar

    def select_and_add(self, pid: Optional[str], qty: int, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        if pid is None:
            return self.ask_choice(st.last_result_list, ops, "No encontré esa opción. ")
        res = a.call("add_to_cart", {"product_id": pid, "qty": qty}, ops)
        p = a.store.get_product(pid)
        if not res["ok"]:
            d = res.get("data") or {}
            if d.get("reason") == "stock":
                return self.propose_alternative(pid, qty, ops, f"{short_title(p['title'])} no tiene stock suficiente "
                                                               f"(hay {d.get('available')}). ")
            return f"No pude añadirlo: {res['error']}. ¿Quieres otra opción o cambiar la cantidad?", True
        txt, clar = a.quote_and_ask(ops)
        return f"Añadí {qty} × {a.fmt_product(p)}. " + txt, clar

    def propose_alternative(self, pid: str, qty: int, ops: list, intro: str, query: str = None) -> tuple[str, bool]:
        a, st = self.a, self.st
        args = {"product_id": pid, "qty": qty, **({"query": query} if query else {})}
        alt = a.call("find_alternative", args, ops)
        alts = (alt.get("data") or {}).get("alternatives", [])
        if not alts:
            return intro + "No encontré alternativas con stock dentro de tus restricciones; no hay una solución por ahora.", True
        st.pending_alternative = {"original": pid, "alternative": alts[0]["product_id"], "qty": qty, "reason": "stock"}
        return intro + f"Como alternativa tengo {a.fmt_product(alts[0], True)}. ¿La añado?", True

    def buy(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        msgs, added, clar = [], 0, False
        search_only = "search_only" in nlu.intents
        for it in nlu.items:
            qty = it.qty if it.qty is not None else (nlu.quantity if len(nlu.items) == 1 and nlu.quantity else 1)
            if qty > a.cfg.max_qty:
                msgs.append(f"La cantidad máxima por producto es {a.cfg.max_qty}; no puedo pedir {qty} de «{it.query}».")
                clar = True
                continue
            args = {"query": it.query, "k": 5}
            if nlu.color:
                args["color"] = nlu.color
            cart_total0 = sum(i["line_total"] for i in a.cart())
            left0 = None if st.budget is None else st.budget - cart_total0
            if left0 is not None:
                args["max_price"] = max(left0, 0.0)
            res = a.call("search_products", args, ops)
            results = res["data"]["results"]
            if not results and left0 is not None:
                args.pop("max_price")
                res = a.call("search_products", args, ops)
                over = res["data"]["results"]
                if over:
                    cheap = min(over, key=lambda r: r["price"])
                    msgs.append(f"No hay opciones de «{it.query}» dentro de tu presupuesto restante ({a.price(left0)}); "
                                f"la más barata es {a.fmt_product(cheap, True)}. ¿Quieres subir el presupuesto o "
                                f"buscar otra cosa?")
                    clar = True
                    continue
            if not results:
                msgs.append(f"No hay productos para «{it.query}» en el catálogo. ¿Puedes describirlo de otra forma?")
                clar = True
                continue
            if not self._noun_matches(it.query, results):
                r, c = self.ask_choice([r["product_id"] for r in results], ops,
                                       f"No hay «{it.query}» exactamente en el catálogo; lo más parecido que encontré: ")
                msgs.append(r.replace("Opciones: ", ""))
                clar = True
                continue
            generic = getattr(a.store, "is_generic_query", None)  # solo tiendas con tipos de producto
            vague = self._too_vague(it.query) or (generic is not None and generic(it.query))
            if search_only or (len(nlu.items) == 1 and nlu.ordinal is None and nlu.superlative is None
                               and vague):
                r, c = self.ask_choice([r["product_id"] for r in results], ops,
                                       f"Para «{it.query}» encontré varias opciones. ")
                msgs.append(r)
                clar = True
                continue
            cart_total = sum(i["line_total"] for i in a.cart())
            left = None if st.budget is None else st.budget - cart_total
            ordered = results
            if nlu.superlative == "cheapest":
                ordered = sorted(results, key=lambda r: r["price"])
            if nlu.ordinal is not None and -len(results) <= nlu.ordinal < len(results):
                ordered = [results[nlu.ordinal]]
            cands = [r for r in ordered if r["stock"] >= qty and (left is None or r["price"] * qty <= left + 1e-9)]
            if getattr(a.store, "prefer_same_type", False):
                # p. ej. tienda real: si el polo pedido está agotado no se sustituye por una taza
                cands = [r for r in cands if self._noun_matches(it.query, [r])] or cands
            if not cands:
                in_stock = [r for r in ordered if r["stock"] >= qty]
                if in_stock and left is not None:
                    cheap = min(in_stock, key=lambda r: r["price"])
                    msgs.append(f"Ninguna opción de «{it.query}» cabe en el presupuesto restante ({a.price(left)}) para "
                                f"{qty} unidades; la más barata es {a.fmt_product(cheap)}. ¿Quieres subir el presupuesto "
                                f"o reducir la cantidad?")
                else:
                    top = ordered[0]
                    alt = self.propose_alternative(top["product_id"], qty, ops,
                                                   f"{short_title(top['title'])} está agotado o sin stock suficiente. ",
                                                   query=it.query)
                    msgs.append(alt[0])
                clar = True
                continue
            chosen = cands[0]
            if chosen["product_id"] != ordered[0]["product_id"] and ordered[0]["stock"] < qty:
                msgs.append(f"La opción más relevante ({short_title(ordered[0]['title'])}) está agotada o no tiene "
                            f"{qty} unidades; te propongo una alternativa.")
            r = a.call("add_to_cart", {"product_id": chosen["product_id"], "qty": qty, "need": it.query}, ops)
            if not r["ok"]:
                msgs.append(f"No pude añadir {short_title(chosen['title'])}: {r['error']}.")
                clar = True
                continue
            added += 1
            others = [x for x in results if x["product_id"] != chosen["product_id"]
                      and self._noun_matches(it.query, [x])][:2]
            alt_txt = (" Otras opciones: " + "; ".join(a.fmt_product(x, True) for x in others) + ".") if others else ""
            msgs.append(f"Para «{it.query}» elegí {qty} × {a.fmt_product(chosen)}.{alt_txt}")
        if added and not clar:
            txt, c = a.quote_and_ask(ops)
            msgs.append(txt)
            clar = c
        elif added:
            msgs.append("Cuando resolvamos lo pendiente te preparo la cotización.")
        return " ".join(msgs), clar

    @staticmethod
    def _noun_matches(query: str, results: list) -> bool:
        """Comprueba que el sustantivo principal de la consulta (o un sinónimo) aparezca en
        algún título recuperado; si no, se pide aclaración en lugar de comprar algo distinto."""
        from .retrieval import QUERY_SYNONYMS
        toks = tokenize(query)
        if not toks:
            return True
        head = toks[0]
        if head.isdigit() or len(head) <= 2:
            return True
        names = {head, *QUERY_SYNONYMS.get(head, [])}
        for r in results:
            title = set(tokenize(r["title"]))
            if names & title or any(t.startswith(n) or n.startswith(t) for t in title for n in names if len(t) > 3):
                return True
        return False

    @staticmethod
    def _too_vague(query: str) -> bool:
        """Una sola palabra muy genérica ("algo", "cosas", "regalo") requiere aclaración."""
        return norm(query) in {"algo", "cosa", "cosas", "regalo", "producto", "productos", "articulo", "lo", "esto"}

    def answer_attribute_q(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        pid = self.pick_from_results(nlu, st.last_result_list) if nlu.ordinal is not None else None
        if pid is None:
            line, amb = self.cart_target(nlu)
            pid = line["product_id"] if line else (st.last_result_list[0] if len(st.last_result_list) == 1 else None)
            if pid is None and not amb and st.last_result_list and nlu.ordinal is None and not a.cart():
                pid = None
        if pid is None:
            q = "¿De qué producto quieres saberlo? Indícame la opción (por ejemplo «el primero»)."
            a.call("ask_clarification", {"question": q}, ops)
            return q, True
        res = a.call("get_product_details", {"product_id": pid}, ops)
        if not res["ok"]:
            return f"No pude consultar el producto: {res['error']}", True
        p = a.store.get_product(pid)
        attr = nlu.attribute_question
        if not attr:
            at = res["data"]["attributes"]
            known = {k: v for k, v in at.items() if v != UNKNOWN}
            facts = ", ".join(f"{k}: {v}" for k, v in list(known.items())[:6]) or "sin atributos verificables"
            return f"{a.fmt_product(p, True)}. Según la ficha: {facts}. Lo demás es desconocido.", False
        val = answer_attribute(p, attr)
        if val == UNKNOWN:
            return (f"La ficha de {short_title(p['title'])} no indica {attr}, así que es desconocido. "
                    f"No puedo confirmarlo."), False
        return f"Según la ficha de {short_title(p['title'])}, {attr}: {val}.", False

    def stock_query(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        pid = self.pick_from_results(nlu, st.last_result_list)
        if pid is None:
            line, _ = self.cart_target(nlu)
            pid = line["product_id"] if line else (st.last_result_list[0] if len(st.last_result_list) == 1 else None)
        if pid is None:
            q = "¿De qué producto quieres saber las existencias? Indícame la opción (por ejemplo «el primero»)."
            a.call("ask_clarification", {"question": q}, ops)
            return q, True
        qty = nlu.quantity if nlu.quantity else 1
        res = a.call("check_stock", {"product_id": pid, "qty": qty}, ops)
        if not res["ok"]:
            return f"No pude consultar el stock: {res['error']}", True
        d = res["data"]
        p = a.store.get_product(pid)
        if d["available"] <= 0:
            return self.propose_alternative(pid, qty, ops, f"{short_title(p['title'])} está agotado. ")
        return f"De {short_title(p['title'])} quedan {d['available']} unidades en stock.", False

    def compare(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a, st = self.a, self.st
        pids = st.last_result_list[:3]
        sel = [st.last_result_list[o] for o in getattr(nlu, "ordinals", [])
               if -len(st.last_result_list) <= o < len(st.last_result_list)]
        if len(sel) >= 2:
            pids = list(dict.fromkeys(sel))
        if len(pids) < 2:
            pids = [c["product_id"] for c in a.cart()][:3]
        if len(pids) < 2:
            q = "¿Qué productos quieres comparar? Primero dime qué buscas."
            a.call("ask_clarification", {"question": q}, ops)
            return q, True
        attrs = [nlu.attribute_question] if nlu.attribute_question else None
        res = a.call("compare_products", {"product_ids": pids, **({"attributes": attrs} if attrs else {})}, ops)
        if not res["ok"]:
            return f"No pude comparar: {res['error']}", True
        parts = []
        for k, r in enumerate(res["data"]["products"]):
            at = r["attributes"]
            keys = attrs or ["marca", "conectividad", "bateria", "capacidad", "material"]
            facts = ", ".join(f"{x}: {at.get(x, UNKNOWN)}" for x in keys)
            stock = "agotado" if r["stock"] <= 0 else f"{r['stock']} en stock"
            parts.append(f"{k + 1}) {short_title(r['title'], 7)} – {a.price(r['price'])}, {stock}; {facts}")
        cheapest = res["data"]["cheapest"]
        idx = pids.index(cheapest) + 1
        return ("Comparación: " + ". ".join(parts) + f". La más barata es la opción {idx}. "
                "Los atributos «desconocido» no aparecen en la ficha. ¿Cuál quieres?"), True


# ====================================================================== política con LLM
SYSTEM_PROMPT = """Eres un asistente de compras por voz de una tienda online. Hablas SIEMPRE en español, con frases breves (la respuesta se leerá en voz alta).

En cada paso respondes con UN ÚNICO objeto JSON y nada más:
- Para usar una herramienta: {{"thought": "razonamiento breve", "action": "<nombre_herramienta>", "args": {{...}}}}
- Para responder al usuario: {{"thought": "...", "action": "respond", "message": "texto para el usuario"}}

Herramientas disponibles (nombre: descripción; argumentos):
{tools}

Reglas obligatorias:
1. Nunca inventes productos, ids, precios, stock ni atributos: usa solo datos de las OBSERVACIONES. Si un atributo no aparece, di que es "desconocido".
2. Solo puedes usar product_id que aparecieron en resultados de herramientas de esta conversación.
3. Si falta información o hay una ambigüedad relevante (p. ej. "quiero eso" sin contexto), usa ask_clarification.
4. Para comprar: search_products -> add_to_cart (con la cantidad pedida) -> create_quote -> respond con el total y pregunta "¿Confirmas el pedido?". Nunca llames confirm_order: la confirmación la gestiona el sistema cuando el usuario dice "sí, confirmo".
5. Respeta el presupuesto y las cantidades del ESTADO. Si el usuario cambia una cantidad, usa update_cart_quantity y vuelve a cotizar; conserva el resto de restricciones.
6. Si un producto está agotado o no cabe en el presupuesto, usa find_alternative y propón la alternativa.
7. Máximo {max_steps} pasos por turno. Responde en cuanto tengas la información.

ESTADO ACTUAL: {state}
CARRITO: {cart}
"""


class LLMPolicy:
    """El LLM decide qué herramientas usar (un JSON por paso); el código valida y ejecuta."""

    def __init__(self, agent: ShoppingAgent):
        self.a = agent

    def _tools_text(self) -> str:
        lines = []
        for t in self.a.tools.values():
            if t.name == "confirm_order":
                continue
            props = t.parameters.get("properties", {})
            req = set(t.parameters.get("required", []))
            args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type')}" for k, v in props.items())
            lines.append(f"- {t.name}: {t.description} ({args})")
        return "\n".join(lines)

    def _messages(self, nlu: NLUResult) -> list:
        a = self.a
        cart = a.call("view_cart")["data"]
        cart_txt = "; ".join(f"{i['qty']}x {i['product_id']} {short_title(i['title'], 6)} {i['unit_price']:.2f}"
                             for i in cart["items"]) or "vacío"
        sys = SYSTEM_PROMPT.format(tools=self._tools_text(), max_steps=a.cfg.max_steps,
                                   state=a.state.summary(), cart=cart_txt + f" (total {cart['total']:.2f})")
        msgs = [{"role": "system", "content": sys}]
        hist = a.state.history[:-1][-8:]
        for h in hist:
            msgs.append({"role": "user" if h["role"] == "user" else "assistant", "content": h["text"]})
        hints = {k: v for k, v in {"cantidad": nlu.quantity, "presupuesto": nlu.budget,
                                   "precio_max_unidad": nlu.max_unit_price, "color": nlu.color,
                                   "productos": [i.query for i in nlu.items] or None,
                                   "intenciones": nlu.intents or None, "ordinal": nlu.ordinal,
                                   "superlativo": nlu.superlative}.items() if v is not None}
        msgs.append({"role": "user", "content": f"{nlu.text}\n[Análisis automático: {json.dumps(hints, ensure_ascii=False)}]"})
        return msgs

    @staticmethod
    def _compact(res: dict, limit: int = 1800) -> str:
        s = json.dumps(res, ensure_ascii=False, default=str)
        return s if len(s) <= limit else s[:limit] + "…"

    def _gen(self, msgs) -> str:
        t0 = time.perf_counter()
        out = self.a.llm.generate(msgs, max_new_tokens=self.a.cfg.max_new_tokens, temperature=0.0)
        self.a._llm_s += time.perf_counter() - t0
        return out

    def run(self, nlu: NLUResult, ops: list) -> tuple[str, bool]:
        a = self.a
        msgs = self._messages(nlu)
        failures = 0
        n_ops0 = len(ops)
        for _ in range(a.cfg.max_steps):
            out = self._gen(msgs)
            obj = parse_llm_json(out)
            if obj is None or not isinstance(obj.get("action"), str):
                failures += 1
                if failures >= 2:
                    break
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content": "Formato inválido. Responde SOLO con un objeto JSON con las claves "
                                                     "thought, action y args (o message si action es respond)."}]
                continue
            action = obj["action"].strip()
            if action in ("respond", "final", "answer", "responder"):
                msg = str(obj.get("message") or obj.get("args", {}).get("message") or "").strip()
                return self._finalize(msg, ops[n_ops0:]), False
            args = obj.get("args") or {}
            if action == "ask_clarification":
                q = str(args.get("question") or obj.get("message") or "¿Puedes darme más detalles?")
                a.call("ask_clarification", {"question": q}, ops, by="llm")
                return q, True
            res = a.call(action, args, ops, by="llm")
            msgs += [{"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                     {"role": "user", "content": f"OBSERVACIÓN ({action}): {self._compact(res)}"}]
        if len(ops) == n_ops0:
            # el LLM no produjo nada útil: respaldo determinista
            return a.rules.run(nlu, ops)
        return self._finalize("", ops[n_ops0:]), False

    def _observed_numbers(self, ops: list) -> set:
        nums = set()

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                nums.add(round(float(x), 2))
        for op in ops:
            walk(op["result"])
        st = self.a.state
        if st.budget is not None:
            nums.add(round(st.budget, 2))
        for i in self.a.cart():
            nums.update({round(i["unit_price"], 2), round(i["line_total"], 2)})
        return nums

    def _finalize(self, msg: str, turn_ops: list) -> str:
        """Verificación de anclaje: los importes del mensaje deben provenir de observaciones."""
        a = self.a
        observed = self._observed_numbers(turn_ops)
        mentioned = [float(m.replace(",", ".")) for m in re.findall(r"(\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|eur)\b", norm(msg))]
        grounded = all(any(abs(x - o) < 0.011 for o in observed) for x in mentioned)
        if not msg or not grounded:
            msg = self._template(turn_ops)
        last_quote = next((op for op in reversed(turn_ops) if op["name"] == "create_quote"), None)
        if last_quote and last_quote["ok"] and "confirm" not in norm(msg):
            msg += " ¿Confirmas el pedido?"
        return msg

    def _template(self, turn_ops: list) -> str:
        a = self.a
        for op in reversed(turn_ops):
            if op["name"] == "create_quote" and op["ok"]:
                return a.fmt_quote(op["result"]["data"])
            if op["name"] == "search_products" and op["ok"]:
                res = op["result"]["data"]["results"][:3]
                if res:
                    return "Encontré: " + "; ".join(f"{k + 1}) {a.fmt_product(r, True)}" for k, r in enumerate(res)) + ". ¿Cuál prefieres?"
            if not op["ok"]:
                return f"No pude completar la operación: {op['result'].get('error')}."
        return "¿En qué más puedo ayudarte?"
