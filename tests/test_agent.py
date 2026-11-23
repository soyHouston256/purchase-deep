"""Escenarios del agente con la política por reglas (sin LLM) y con MockLLM."""

import json

from shopassist.llm import MockLLM
from tests.fixtures import make_toy_env


def names(turn):
    return [o["name"] for o in turn.ops]


def test_purchase_within_constraints_and_no_duplicate(env):
    agent, store = env["agent"], env["store"]
    t = agent.handle("Quiero dos audífonos bluetooth negros con un presupuesto de 100 euros")
    assert "create_quote" in names(t) and "¿Confirmas" in t.reply
    assert agent.state.budget == 100 and agent.state.pending_confirmation
    q = agent.state.last_quote
    assert q["valid"] and q["total"] <= 100 and q["items"][0]["qty"] == 2
    pid = q["items"][0]["product_id"]
    s0 = store.get_stock(pid)
    t = agent.handle("Sí, confirmo")
    assert "confirm_order" in names(t) and t.ops[-1]["ok"] and "registrado" in t.reply
    assert store.get_stock(pid) == s0 - 2 and len(store.list_orders()) == 1
    t2 = agent.handle("sí, confirmo")  # repetición de la confirmación
    assert t2.ops[-1]["result"]["data"]["duplicate"] is True and "no se ha duplicado" in t2.reply
    assert len(store.list_orders()) == 1 and store.get_stock(pid) == s0 - 2
    json.dumps(t2.state)
    assert set(t2.timings) >= {"llm_s", "tools_s", "retrieval_s"}


def test_no_confirm_without_quote_or_explicit_yes(env):
    agent, store = env["agent"], env["store"]
    t = agent.handle("sí, confirmo")
    assert t.needs_clarification and not store.list_orders()
    agent.handle("quiero un mouse inalámbrico")
    t = agent.handle("¿cuánto sería?")
    assert not store.list_orders() and "confirm_order" not in names(t)


def test_ambiguous_requests_ask_clarification(env):
    agent = env["agent"]
    for txt in ("quiero eso", "dame el más barato", "el segundo"):
        agent.reset()
        t = agent.handle(txt)
        assert t.needs_clarification and "add_to_cart" not in names(t), txt
    agent.reset()
    t = agent.handle("quiero unos audífonos")  # sin restricciones: puede avanzar
    assert "search_products" in names(t)


def test_reference_resolved_with_context(env):
    agent = env["agent"]
    t = agent.handle("busco un cargador usb-c")
    assert t.needs_clarification and agent.state.last_result_list
    t = agent.handle("dame el más barato")
    assert "add_to_cart" in names(t)
    added = agent.state.last_quote["items"][0]["product_id"]
    prices = [env["store"].get_price(p) for p in agent.state.last_result_list
              if env["store"].get_stock(p) > 0]
    assert env["store"].get_price(added) == min(prices)


def test_quantity_change_keeps_budget(env):
    agent = env["agent"]
    agent.handle("Quiero dos audífonos bluetooth con un presupuesto de 100 euros")
    q1 = agent.state.last_quote
    t = agent.handle("mejor 3")
    assert "update_cart_quantity" in names(t)
    q2 = agent.state.last_quote
    assert q2["quote_id"] != q1["quote_id"] and q2["items"][0]["qty"] == 3
    assert agent.state.budget == 100 and q2["budget"] == 100


def test_quantity_change_over_budget_rejected(env):
    agent = env["agent"]
    agent.handle("Quiero un audífono bluetooth con un presupuesto de 40 euros")
    q = agent.state.last_quote
    unit = q["items"][0]["unit_price"]
    too_many = int(40 // unit) + 1
    t = agent.handle(f"mejor {too_many}")
    op = next(o for o in t.ops if o["name"] == "update_cart_quantity")
    assert not op["ok"] and op["result"]["data"]["reason"] == "budget"
    assert env["store"].get_cart("c1")[0]["qty"] == 1


def test_preference_change_keeps_qty_and_budget(env):
    agent = env["agent"]
    agent.handle("Quiero dos audífonos bluetooth negros con un presupuesto de 100 euros")
    t = agent.handle("mejor en color blanco")
    cart = env["store"].get_cart("c1")
    assert len(cart) == 1 and cart[0]["qty"] == 2
    assert "blanco" in env["store"].get_product(cart[0]["product_id"])["color"]
    assert agent.state.budget == 100 and agent.state.pending_confirmation


def test_stock_change_between_quote_and_confirm(env):
    agent, store = env["agent"], env["store"]
    agent.handle("Quiero dos audífonos bluetooth con un presupuesto de 150 euros")
    pid = agent.state.last_quote["items"][0]["product_id"]
    store.set_stock(pid, 1)
    stock_before = {p: store.get_stock(p) for p in store.product_ids()}
    t = agent.handle("sí, confirmo")
    assert names(t) == ["confirm_order", "find_alternative"]
    assert not t.ops[0]["ok"] and agent.state.pending_alternative
    assert {p: store.get_stock(p) for p in store.product_ids()} == stock_before  # sin cambios parciales
    assert store.list_orders() == []
    t = agent.handle("sí")  # acepta la alternativa -> nueva cotización
    assert "create_quote" in names(t) and agent.state.pending_confirmation
    alt = agent.state.last_quote["items"][0]["product_id"]
    assert alt != pid and agent.state.last_quote["total"] <= 150
    t = agent.handle("confirmo")
    assert t.ops[-1]["ok"] and len(store.list_orders()) == 1
    t = agent.handle("confirmo")
    assert t.ops[-1]["result"]["data"]["duplicate"] and len(store.list_orders()) == 1


def test_out_of_stock_selected_proposes_alternative(env):
    agent, store = env["agent"], env["store"]
    agent.handle("busco un teclado mecánico")
    first = agent.state.last_result_list[0]
    store.set_stock(first, 0)
    t = agent.handle("quiero la opción 1")
    assert "find_alternative" in names(t) and t.needs_clarification
    assert agent.state.pending_alternative and agent.state.pending_alternative["original"] == first


def test_out_of_stock_no_solution(env):
    agent, store = env["agent"], env["store"]
    for p in store.product_ids():
        if p.startswith("TEC"):
            store.set_stock(p, 0)
    t = agent.handle("quiero un teclado mecánico")
    assert t.needs_clarification and "no hay una solución" in t.reply
    assert store.get_cart("c1") == []


def test_voice_transcript_correction_before_purchase(env):
    agent, store = env["agent"], env["store"]
    t = agent.handle("quiero trece cargadores usb-c", source="voice")
    assert t.ops == [] and "Entendí" in t.reply and "13" in t.reply and agent.state.transcript_pending
    t = agent.handle("no, dije tres", source="voice")
    assert t.ops == [] and "3" in t.reply and agent.state.transcript_pending  # se re-verifica
    t = agent.handle("sí, es correcto", source="voice")
    assert "add_to_cart" in names(t)
    assert store.get_cart("c1")[0]["qty"] == 3
    t = agent.handle("sí, confirmo", source="voice")
    assert t.ops[-1]["name"] == "confirm_order" and t.ops[-1]["ok"]


def test_voice_confirmed_transcript_executes(env):
    agent, store = env["agent"], env["store"]
    agent.handle("quiero dos fundas para iphone", source="voice")
    t = agent.handle("sí", source="voice")
    assert store.get_cart("c1") and store.get_cart("c1")[0]["qty"] == 2 and "create_quote" in names(t)


def test_attribute_unknown_and_compare(env):
    agent = env["agent"]
    agent.handle("busco audífonos bluetooth")
    t = agent.handle("¿qué garantía tiene el primero?")
    assert "desconocido" in t.reply
    t = agent.handle("compara las opciones")
    assert "compare_products" in names(t) and "desconocido" in t.reply


def test_remove_and_multi_item(env):
    agent, store = env["agent"], env["store"]
    agent.handle("quiero dos cables usb-c y un mouse inalámbrico")
    assert len(store.get_cart("c1")) == 2
    t = agent.handle("mejor 3")
    assert t.needs_clarification  # ¿de cuál producto?
    t = agent.handle("el cable")
    assert any(i["qty"] == 3 and i["product_id"].startswith("CAB") for i in store.get_cart("c1"))
    t = agent.handle("quita el mouse")
    assert [i["product_id"][:3] for i in store.get_cart("c1")] == ["CAB"]


def test_grounding_rejects_invented_ids(env):
    agent = env["agent"]
    r = agent.call("add_to_cart", {"product_id": "AUD001", "qty": 1})
    assert not r["ok"] and "no apareció" in r["error"]
    r = agent.call("add_to_cart", {"product_id": "INVENTADO", "qty": 1})
    assert not r["ok"]


def test_confirm_order_tool_requires_explicit_confirmation(env):
    agent = env["agent"]
    agent.handle("quiero un mouse inalámbrico")
    r = agent.call("confirm_order", {})
    assert not r["ok"] and env["store"].list_orders() == []


def test_llm_policy_with_mock(env):
    store = env["store"]
    pid_holder = {}

    def script(messages):
        last = messages[-1]["content"]
        if last.startswith("OBSERVACIÓN (search_products)"):
            obs = json.loads(last.split(": ", 1)[1])
            pid_holder["pid"] = obs["data"]["results"][0]["product_id"]
            return json.dumps({"thought": "añado", "action": "add_to_cart", "args": {"product_id": pid_holder["pid"], "qty": 2}})
        if last.startswith("OBSERVACIÓN (add_to_cart)"):
            return '{"thought": "cotizo", "action": "create_quote", "args": {}}'
        if last.startswith("OBSERVACIÓN (create_quote)"):
            return '{"action": "respond", "message": "Total 9999 euros. ¿Confirmas el pedido?"}'  # importe inventado
        if last.startswith("OBSERVACIÓN"):
            return '{"action": "respond", "message": "ok"}'
        if "mouse" in last:
            return 'Claro: {"thought": "busco", "action": "search_products", "args": {"query": "mouse inalámbrico"}}'
        return '{"action": "confirm_order", "args": {}}'

    cfg, cat, store, retriever, agent = make_toy_env(llm=MockLLM(script))
    t = agent.handle("quiero dos mouse inalámbricos")
    assert t.policy == "llm"
    assert names(t) == ["search_products", "add_to_cart", "create_quote"]
    assert "9999" not in t.reply and "Cotización" in t.reply  # verificación de anclaje
    # El LLM no puede confirmar: la confirmación pasa por la guarda determinista
    t = agent.handle("¿y ahora qué?")
    assert store.list_orders() == []
    t = agent.handle("sí, confirmo")
    assert t.ops[-1]["name"] == "confirm_order" and t.ops[-1]["ok"] and len(store.list_orders()) == 1


def test_llm_garbage_falls_back_to_rules():
    cfg, cat, store, retriever, agent = make_toy_env(llm=MockLLM(lambda m: "no sé qué decir"))
    t = agent.handle("quiero un teclado mecánico")
    assert "search_products" in names(t)


def test_reset_keeps_store(env):
    agent, store = env["agent"], env["store"]
    agent.handle("quiero un mouse inalámbrico")
    agent.handle("sí")
    n = len(store.list_orders())
    agent.reset()
    assert agent.state.last_quote is None and store.get_cart("c1") == [] and len(store.list_orders()) == n
