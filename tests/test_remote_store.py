"""Pruebas del backend EXTRA de tienda real (soyHouston256/store) con HTTP falso.

Nunca se usa la red: ``FakeTransport`` imita las rutas públicas (GET /api/products,
GET /api/products/:id, POST /api/orders con la validación zod del servidor) y los
productos de prueba tienen exactamente la forma de ``ProductDTO``.
"""
import copy
import io
import json

import pytest

from shopassist.config import Config
from shopassist.remote_store import (FakeTransport, HttpTransport, RemoteStoreAdapter, build_remote_demo,
                                     hex_to_color_name, load_fixture_products, resolve_base_url)
from shopassist.scenarios import load_scenarios, run_all
from shopassist.store import Store

CUSTOMER = {"dni": "12345678", "name": "Cliente Demo", "phone": "999888777"}


def _dto(pid, name, price, type_="polo", colors=("#FFF", "#000"), sold_out=False, cuts=None):
    return {"id": pid, "name": name, "price": price, "type": type_, "colors": list(colors),
            "sizes": ["S", "M", "L"] if type_ == "polo" else [],
            "logoPositions": ["pocket", "chest", "back", "front-back"] if type_ == "polo" else [],
            "likes": 0, "published": True, "slug": name.lower().replace(" ", "-"),
            "cuts": (cuts or ["hombre"]) if type_ == "polo" else [], "soldOut": sold_out,
            "createdAt": "1970-01-01T00:00:00.000Z"}


@pytest.fixture
def products():
    return load_fixture_products()


def _demo(products, **kw):
    t = FakeTransport(copy.deepcopy(products), fail_post=kw.pop("fail_post", False))
    d = build_remote_demo(Config(), transport=t, **kw)
    d["transport"] = t
    d["store"].restore(42)
    return d


def _pid(store, name):
    return next(r.product_id for r in store.catalog_df.itertuples() if r.title == name)


def _talk(agent, *turns):
    return [agent.handle(u) for u in turns]


# ------------------------------------------------------------------ catálogo y búsqueda
def test_catalog_build_same_schema_and_prices(products):
    d = _demo(products)
    st, cat = d["store"], d["store"].catalog_df
    assert len(cat) == len(products) == 23
    for col in ("product_id", "title", "description", "bullet_point", "brand", "color"):
        assert col in cat.columns
    redis = cat[cat.title == "Polo Redis"].iloc[0]
    assert redis.color == "blanco, negro"
    assert "Tallas: S, M, L, XL, XXL" in redis.bullet_point and "Redis" in redis.description
    assert "Taza React" in set(cat.title) and "Mousepad Github" in set(cat.title)
    p = st.get_product(redis.product_id)
    assert p["price"] == 60.0 and p["currency"] == "PEN" and d["cfg"].currency == "PEN"
    assert p["options"]["size"] == ["S", "M", "L", "XL", "XXL"] and p["options"]["cut"] == ["hombre"]
    assert st.get_product("demo-taza-react")["options"] == {"cut": [], "size": [], "color": ["blanco", "negro", "verde"]}
    # solo un GET del catálogo al construir (restore usa la copia cacheada)
    assert [c["method"] for c in d["transport"].calls] == ["GET"]


def test_search_finds_polo_redis(products):
    d = _demo(products)
    res = d["retriever"].search("polo redis", k=3)
    assert res[0]["title"] == "Polo Redis" and res[0]["price"] == 60.0
    assert d["retriever"].search("polo docker", k=1)[0]["title"] == "Polo Docker"
    assert d["retriever"].search("taza docker", k=1)[0]["title"] == "Taza Docker"


def test_hex_colors_and_variant_extraction():
    assert hex_to_color_name("#000") == "negro" and hex_to_color_name("#FFFFFF") == "blanco"
    assert hex_to_color_name("#ACCBDD") == "azul" and hex_to_color_name("#6b6b6b") == "gris"
    found, rest, only = RemoteStoreAdapter.extract_variants("quiero 2 polos de redis talla m negros")
    assert found == {"size": "M", "color": "negro"} and not only and "talla" not in rest
    assert RemoteStoreAdapter.extract_variants("Talla L, color blanco")[2] is True
    assert RemoteStoreAdapter.extract_variants("corte mujer")[0] == {"cut": "mujer"}


def test_base_url_resolution(monkeypatch):
    monkeypatch.setenv("STORE_API_URL", "https://api.example.pe/api/")
    assert resolve_base_url() == "https://api.example.pe"
    assert resolve_base_url(cfg=Config(remote_store_url="http://localhost:3000")) == "http://localhost:3000"
    monkeypatch.delenv("STORE_API_URL")
    assert resolve_base_url() is None
    with pytest.raises(Exception):
        RemoteStoreAdapter(None)


def test_http_transport_with_mocked_urlopen(monkeypatch, products):
    import urllib.request

    seen = []

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen.append((req.get_method(), req.full_url))
        return _Resp(json.dumps(products).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    st = RemoteStoreAdapter("https://api.example.pe", transport=HttpTransport())
    assert seen == [("GET", "https://api.example.pe/api/products")]
    assert len(st.product_ids()) == 23


# ------------------------------------------------------------------ agente
def test_buy_two_polos_with_size_and_color_dry_run_does_not_post(products):
    d = _demo(products)
    st, t = d["store"], d["transport"]
    pid = _pid(st, "Polo Redis")
    st.set_stock(pid, 20)
    a = d["agent_factory"]()
    r1, r2, r3 = _talk(a, "Quiero dos polos de Redis talla M negros", "Sí, confirmo el pedido", "Confirmo el pedido")
    assert "talla M" in r1.reply and "120,00 soles" in r1.reply
    assert "dry-run" in r2.reply and "no se ha duplicado" in r3.reply
    assert t.posts == []                                  # por defecto NUNCA se envía
    orders = st.list_orders()
    assert len(orders) == 1 and st.get_stock(pid) == 18
    rem = orders[0]["remote"]
    assert rem["mode"] == "dry_run" and rem["endpoint"].endswith("/api/orders")
    assert rem["payload"] == {"user": {"dni": "<dni>", "name": "<name>", "phone": "<phone>"},
                              "items": [{"productId": pid, "quantity": 2, "cut": "hombre", "size": "M", "color": "#000"}]}
    # la revalidación contra la API se hizo con GET /api/products/:id
    assert any(c["method"] == "GET" and c["url"].endswith(f"/api/products/{pid}") for c in t.calls)


def test_ambiguous_polo_asks_then_asks_size_and_color(products):
    d = _demo(products)
    a = d["agent_factory"]()
    r1 = a.handle("Quiero un polo")
    assert r1.needs_clarification and not a.cart()
    r2 = a.handle("El de Redis")
    assert r2.needs_clarification and "talla" in r2.reply and "color" in r2.reply
    r3 = a.handle("Talla L, color blanco")
    assert "talla L" in r3.reply and "¿Confirmas" in r3.reply


def test_sold_out_product_gets_alternative(products):
    prods = copy.deepcopy(products)
    for p in prods:
        if p["name"] == "Polo Docker":
            p["soldOut"] = True
    d = _demo(prods)
    st = d["store"]
    docker = _pid(st, "Polo Docker")
    assert st.get_stock(docker) == 0 and st.get_product(docker)["sold_out"]
    a = d["agent_factory"]()
    r = a.handle("Quiero un polo de Docker talla M negro")
    assert "agotad" in r.reply.lower() and "alternativa" in r.reply.lower()
    cart = a.cart()
    assert cart and all(i["product_id"] != docker for i in cart)
    assert cart[0]["title"].startswith("Polo")             # mismo tipo: no una taza


def test_price_change_at_confirm_is_detected(products):
    d = _demo(products)
    st, t = d["store"], d["transport"]
    pid = _pid(st, "Polo Redis")
    st.set_stock(pid, 20)
    a = d["agent_factory"]()
    a.handle("Quiero dos polos de Redis talla M negros")
    t.set_product(pid, price=70.0)                         # cambia en la tienda real
    r = a.handle("Sí, confirmo el pedido")
    assert st.list_orders() == [] and "precio cambió" in r.reply
    assert "140,00 soles" in r.reply and st.get_price(pid) == 70.0
    r2 = a.handle("Sí, confirmo el pedido")
    assert len(st.list_orders()) == 1 and st.list_orders()[0]["total"] == 140.0 and "¡Listo!" in r2.reply


def test_sold_out_at_confirm_is_detected(products):
    d = _demo(products)
    st, t = d["store"], d["transport"]
    pid = _pid(st, "Polo Redis")
    st.set_stock(pid, 20)
    a = d["agent_factory"]()
    a.handle("Quiero dos polos de Redis talla M negros")
    t.set_product(pid, soldOut=True)
    r = a.handle("Sí, confirmo el pedido")
    assert st.list_orders() == [] and st.get_stock(pid) == 0
    assert "alternativa" in r.reply.lower() and t.posts == []


def test_real_mode_posts_exactly_once_on_double_confirm(products):
    d = _demo(products, allow_real_orders=True, customer=CUSTOMER)
    st, t = d["store"], d["transport"]
    pid = _pid(st, "Polo Redis")
    st.set_stock(pid, 20)
    a = d["agent_factory"]()
    _, r2, r3 = _talk(a, "Quiero dos polos de Redis talla M negros", "Sí, confirmo el pedido", "Sí, confirmo el pedido")
    assert len(t.posts) == 1
    body = t.posts[0]["payload"]
    assert body == {"user": CUSTOMER, "items": [{"productId": pid, "quantity": 2, "cut": "hombre", "size": "M",
                                                  "color": "#000"}]}
    remote_id = t.orders[0]["id"]
    assert remote_id in r2.reply and "no se ha duplicado" in r3.reply
    assert len(st.list_orders()) == 1 and st.list_orders()[0]["remote"]["status"] == "enviado"
    # incluso llamando directamente a place_order otra vez no hay segundo POST
    q = a.state.last_quote["quote_id"]
    assert st.place_order(q)["duplicate"] and len(t.posts) == 1
    # el registro de pedidos remotos sobrevive a restore (es un efecto real)
    st.restore(42)
    assert [r["remote_order_id"] for r in st.remote_orders()] == [remote_id]


def test_real_mode_requires_customer_data(products):
    d = _demo(products, allow_real_orders=True)            # sin datos del cliente
    st, t = d["store"], d["transport"]
    st.set_stock(_pid(st, "Polo Redis"), 20)
    a = d["agent_factory"]()
    _, r2 = _talk(a, "Quiero dos polos de Redis talla M negros", "Sí, confirmo el pedido")
    assert t.posts == [] and "dry-run" in r2.reply
    assert "faltan datos del cliente" in st.list_orders()[0]["remote"]["detail"]


def test_real_mode_rejection_rolls_back_local_order(products):
    d = _demo(products, allow_real_orders=True, customer=CUSTOMER, fail_post=True)
    st, t = d["store"], d["transport"]
    pid = _pid(st, "Polo Redis")
    st.set_stock(pid, 20)
    a = d["agent_factory"]()
    _, r2 = _talk(a, "Quiero dos polos de Redis talla M negros", "Sí, confirmo el pedido")
    assert len(t.posts) == 1 and "rechazó" in r2.reply
    assert st.list_orders() == [] and st.get_stock(pid) == 20      # compensado
    assert [i["qty"] for i in a.cart()] == [2]                    # el carrito sigue ahí
    q = a.state.last_quote["quote_id"]
    assert not st.place_order(q)["ok"] and len(t.posts) == 1     # misma cotización: nunca se reenvía
    assert st.list_orders() == [] and st.get_stock(pid) == 20
    r3 = a.handle("Sí, confirmo el pedido")                       # el agente re-cotiza (nueva clave)
    assert "¿Confirmas" in r3.reply and len(t.posts) == 1


def test_variant_required_before_quote(products):
    d = _demo(products)
    st = d["store"]
    pid = _pid(st, "Polo Redis")
    st.add_to_cart("c9", pid, 1)
    q = st.create_quote("c9")
    assert not q["valid"] and {p["type"] for p in q["problems"]} == {"variant"}
    assert set(q["problems"][0]["missing"]) == {"size", "color"}
    st.set_line_variant("c9", pid, size="xl", color="blanco")
    q2 = st.create_quote("c9")
    assert q2["valid"] and q2["items"][0]["variant"] == {"cut": "hombre", "size": "XL", "color": "#FFF"}
    with pytest.raises(Exception):
        st.set_line_variant("c9", pid, size="XS")          # XS no existe en corte hombre


def test_mug_and_mousepad_payload_has_no_cut_or_size():
    prods = [_dto("p1", "Redis", 60.0), _dto("t1", "Taza React", 35.0, "taza", ("#FFFFFF", "#111827")),
             _dto("m1", "Mousepad Python", 45.0, "mousepad", ("#111827",))]
    d = _demo(prods)
    st = d["store"]
    st.add_to_cart("c2", "t1", 1)
    st.add_to_cart("c2", "m1", 1)
    st.set_line_variant("c2", "t1", color="negro")
    q = st.create_quote("c2")
    assert q["valid"]
    assert st.build_order_payload(q, CUSTOMER)["items"] == [
        {"productId": "t1", "quantity": 1, "color": "#111827"},
        {"productId": "m1", "quantity": 1, "color": "#111827"}]   # una sola opción: autocompletado


def test_esci_store_has_no_variant_hooks():
    # la Store oficial no se ve afectada por la demo extra
    assert not hasattr(Store, "apply_preferences") and not hasattr(Store, "extract_variants")


# ------------------------------------------------------------------ escenarios
def test_tienda_real_scenarios_pass_in_dry_run(products):
    d = _demo(products)
    scs = load_scenarios("scenarios/scenarios_tienda_real.yaml")
    assert len(scs) == 4
    df, res = run_all(scs, d["agent_factory"], d["store"], out_dir=None)
    assert df["ok"].all(), df[["id", "fallidos"]].to_dict("records")
    assert d["transport"].posts == []
    assert all(r["violations"]["duplicate_orders"] == 0 for r in res)
