import json

import pytest

from shopassist.store import Store, StoreError, synth_price
from tests.fixtures import make_toy_catalog


@pytest.fixture
def store():
    return Store(":memory:", make_toy_catalog(), seed=42)


def _in_stock(store, n=2):
    return [p for p in store.product_ids() if store.get_stock(p) >= 5][:n]


def test_restore_is_deterministic(store):
    s2 = Store(":memory:", make_toy_catalog(), seed=42)
    ids = store.product_ids()
    assert [store.get_price(p) for p in ids] == [s2.get_price(p) for p in ids]
    assert [store.get_stock(p) for p in ids] == [s2.get_stock(p) for p in ids]
    pid = ids[0]
    store.set_stock(pid, 0)
    store.add_to_cart("c1", ids[1], 1)
    store.restore(42)
    assert store.get_stock(pid) == s2.get_stock(pid) and store.get_cart("c1") == [] and store.list_orders() == []


def test_prices_plausible(store):
    prices = [store.get_price(p) for p in store.product_ids()]
    assert all(2 <= p <= 2500 for p in prices)
    assert all(abs(round(p % 1, 2) - 0.99) < 1e-6 for p in prices)
    assert synth_price("X", "Cable USB", 42) != synth_price("X", "Cable USB", 7)


def test_some_zero_stock():
    from shopassist.store import synth_stock
    zeros = sum(synth_stock(f"P{i}", 42) == 0 for i in range(2000))
    assert 120 < zeros < 300  # ~10 %


def test_quote_and_order_decrements_stock(store):
    a, b = _in_stock(store)
    sa, sb = store.get_stock(a), store.get_stock(b)
    store.add_to_cart("c1", a, 2)
    store.add_to_cart("c1", b, 1)
    q = store.create_quote("c1")
    assert q["valid"] and q["total"] == pytest.approx(2 * store.get_price(a) + store.get_price(b))
    r = store.place_order(q["quote_id"])
    assert r["ok"] and not r["duplicate"]
    assert store.get_stock(a) == sa - 2 and store.get_stock(b) == sb - 1
    assert store.get_cart("c1") == []
    json.dumps(store.snapshot())  # serializable


def test_idempotent_order(store):
    a, = _in_stock(store, 1)
    s0 = store.get_stock(a)
    store.add_to_cart("c1", a, 1)
    q = store.create_quote("c1")
    r1 = store.place_order(q["quote_id"])
    r2 = store.place_order(q["quote_id"])
    r3 = store.place_order(q["quote_id"], idempotency_key=q["quote_id"])
    assert r1["ok"] and r2["ok"] and r2["duplicate"] and r3["duplicate"]
    assert r1["order"]["order_id"] == r2["order"]["order_id"]
    assert len(store.list_orders()) == 1 and store.get_stock(a) == s0 - 1


def test_atomicity_on_stock_change(store):
    a, b = _in_stock(store)
    store.add_to_cart("c1", a, 1)
    store.add_to_cart("c1", b, 2)
    q = store.create_quote("c1")
    store.set_stock(b, 1)  # otro cliente compró entre cotización y confirmación
    sa = store.get_stock(a)
    r = store.place_order(q["quote_id"])
    assert not r["ok"] and any(p["type"] == "stock" and p["product_id"] == b for p in r["problems"])
    assert store.get_stock(a) == sa and store.get_stock(b) == 1  # sin cambios parciales
    assert store.list_orders() == [] and len(store.get_cart("c1")) == 2


def test_atomicity_on_midway_failure(store):
    a, b = _in_stock(store)
    store.add_to_cart("c1", a, 1)
    store.add_to_cart("c1", b, 1)
    q = store.create_quote("c1")
    sa, sb = store.get_stock(a), store.get_stock(b)
    r = store.place_order(q["quote_id"], _fail_after_first_line=True)
    assert not r["ok"]
    assert store.get_stock(a) == sa and store.get_stock(b) == sb and store.list_orders() == []
    # tras el fallo, la misma cotización puede confirmarse correctamente
    assert store.place_order(q["quote_id"])["ok"]


def test_budget_violation_rejected(store):
    a, = _in_stock(store, 1)
    store.add_to_cart("c1", a, 2)
    q = store.create_quote("c1", budget=store.get_price(a))
    assert not q["valid"] and any(p["type"] == "budget" for p in q["problems"])
    r = store.place_order(q["quote_id"])
    assert not r["ok"] and store.list_orders() == []


def test_price_change_detected(store):
    a, = _in_stock(store, 1)
    store.add_to_cart("c1", a, 1)
    q = store.create_quote("c1")
    store.set_price(a, store.get_price(a) + 5)
    r = store.place_order(q["quote_id"])
    assert not r["ok"] and r["problems"][0]["type"] == "price"


def test_invalid_qty_and_product(store):
    with pytest.raises(StoreError):
        store.add_to_cart("c1", "NOPE", 1)
    with pytest.raises(StoreError):
        store.add_to_cart("c1", store.product_ids()[0], 0)
    with pytest.raises(StoreError):
        store.add_to_cart("c1", store.product_ids()[0], 1.5)
    with pytest.raises(StoreError):
        store.set_stock(store.product_ids()[0], -1)
