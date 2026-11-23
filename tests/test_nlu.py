import pytest

from shopassist.nlu import apply_correction, normalize_numbers, parse


@pytest.mark.parametrize("text,expected", [
    ("dos", "2"), ("veinte", "20"), ("veintidós", "22"), ("treinta y cinco", "35"), ("cien", "100"),
    ("ciento veinticinco", "125"), ("doscientos cincuenta", "250"), ("mil doscientos", "1200"),
    ("dos mil", "2000"), ("una docena", "12"), ("media docena", "6"), ("un par", "2"), ("dos docenas", "24"),
    ("49,99", "49.99"), ("1.000", "1000"),
])
def test_number_words(text, expected):
    assert normalize_numbers(text) == expected


def test_article_un_is_not_number_out_of_context():
    assert normalize_numbers("busco un mouse") == "busco un mouse"
    assert normalize_numbers("mejor solo uno") == "mejor solo 1"


@pytest.mark.parametrize("text,qty", [
    ("Quiero dos audífonos bluetooth", 2), ("quiero 3 cargadores", 3), ("necesito una docena de pilas", 12),
    ("ponme cinco cables usb-c", 5), ("quiero un iphone 13", 1), ("quiero 2 memorias de 128 gb", 2),
])
def test_quantity(text, qty):
    assert parse(text).quantity == qty


def test_spec_numbers_not_quantity_and_kept_in_query():
    r = parse("quiero 2 memorias de 128 gb")
    assert r.items[0].query == "memorias 128 gb"
    r = parse("quiero un iphone 13")
    assert "13" in r.items[0].query and r.quantity == 1


@pytest.mark.parametrize("text,budget", [
    ("con un presupuesto de 100 euros", 100.0), ("presupuesto máximo de cincuenta euros", 50.0),
    ("máximo 50", 50.0), ("no más de 49,99 €", 49.99), ("tengo 200 euros", 200.0),
    ("no quiero gastar más de ochenta euros", 80.0), ("mil doscientos euros de presupuesto", 1200.0),
    ("hasta 60 euros en total", 60.0),
])
def test_budget(text, budget):
    assert parse(text).budget == pytest.approx(budget)


def test_unit_price_vs_budget():
    r = parse("audífonos de menos de 30 euros")
    assert r.max_unit_price == 30 and r.budget is None
    r = parse("máximo 20 euros cada uno")
    assert r.max_unit_price == 20 and r.budget is None


@pytest.mark.parametrize("text,conf", [
    ("Sí, confirmo", True), ("sí", True), ("confirmo el pedido", True), ("vale, adelante", True),
    ("de acuerdo", True), ("no", False), ("no, cancela", False), ("sí, pero mejor tres", False),
    ("si hay stock dímelo", False), ("¡Sí, confirmo!", True), ("¿sí?", False),
])
def test_explicit_confirmation(text, conf):
    assert parse(text).explicit_confirmation is conf


def test_deny_and_no_mas_de():
    assert "deny" in parse("no, cancela").intents
    assert "deny" not in parse("no más de 40 euros").intents


@pytest.mark.parametrize("text,qty", [
    ("cambia la cantidad a 3", 3), ("mejor 2", 2), ("que sean cuatro", 4), ("mejor solo uno", 1),
    ("quiero 4 en vez de 2", 4), ("en vez de 2, 5", 5),
])
def test_change_qty(text, qty):
    r = parse(text)
    assert "change_qty" in r.intents and r.quantity == qty


def test_correction():
    r = parse("no dije trece, dije tres")
    assert r.correction == ("13", "3")
    assert apply_correction("quiero trece cargadores usb-c", r.correction) == "quiero 3 cargadores usb-c"
    assert apply_correction("quiero 13 cargadores", (None, "tres")) == "quiero 3 cargadores"
    assert "50 euros" in apply_correction("quiero 2 cargadores con presupuesto de 80 euros", (None, "50 euros"))


def test_references_and_ambiguity_markers():
    assert parse("el segundo").ordinal == 1
    assert parse("la opción 3").ordinal == 2
    assert parse("dame el más barato").superlative == "cheapest"
    r = parse("quiero eso")
    assert r.deictic and not r.items


def test_multi_items():
    r = parse("quiero tres cargadores usb-c y un mouse inalámbrico, máximo cien euros")
    assert [(i.qty, i.query) for i in r.items] == [(3, "cargadores usb-c"), (None, "mouse inalambrico")]
    assert r.budget == 100


def test_color_and_pref_change():
    r = parse("mejor en color blanco")
    assert r.color == "blanco" and "change_pref" in r.intents and not r.items
    assert parse("audífonos negros").color == "negro"
    assert parse("de la marca Sony").brand == "sony"
