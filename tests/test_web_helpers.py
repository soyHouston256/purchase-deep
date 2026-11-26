"""Pruebas de las utilidades de la demo web: categorías y troceo del texto para TTS por frases."""
import numpy as np
import pandas as pd

from shopassist.categories import CATEGORY_ICONS, classify_catalog, rule_category
from shopassist.voice import speech_text, split_tts_chunks


def test_rule_category_head_noun_wins():
    assert rule_category("BAIAA Botellas Agua para Deportes, 1 Litro, Tritan")[0] == "Hogar y cocina"
    assert rule_category("Auriculares Bluetooth 5.0 In Ear Inalámbricos")[0] == "Audio"
    assert rule_category("Amazon Basics - Pilas AA recargables, paquete de 4")[0] == "Pilas y baterías"
    # "X para Y": manda el sustantivo que aparece primero
    assert rule_category("Mochila para portátil de 15 pulgadas")[0] == "Bolsos y equipaje"
    assert rule_category("Funda para iPhone 12")[0] == "Móviles y tablets"
    assert rule_category("xyz 123") is None


def test_classify_catalog_isbn_rules_and_knn():
    df = pd.DataFrame({"product_id": ["8445800779", "B000000001", "B000000002", "B000000003"],
                       "title": ["Notas sobre enfermería", "Auriculares inalámbricos", "Cascos con micro", "Zzz qqq"]})
    emb = np.eye(4, dtype=np.float32)
    emb[3] = emb[1]                                   # el producto sin regla es idéntico a unos auriculares
    out = classify_catalog(df, emb)
    assert out["8445800779"]["category"] == "Libros" and out["8445800779"]["source"] == "isbn"
    assert out["B000000001"]["source"] == "regla"
    assert out["B000000003"] == {**out["B000000003"], "category": "Audio", "source": "knn"}
    assert all(v["icon"] for v in out.values())
    assert set(v["category"] for v in out.values()) <= set(CATEGORY_ICONS)


def test_split_tts_chunks_keeps_all_text_and_first_is_short():
    text = ("Encontré varias opciones de botellas de agua. ¿Te gustaría ver más opciones o proceder con la "
            "primera opción? Puedes decirme el número de la que prefieras y la añado al carrito, "
            "o puedo buscar otra cosa si lo necesitas. ¿Algo más?")
    chunks = split_tts_chunks(text)
    assert len(chunks) >= 2
    assert chunks[0] == "Encontré varias opciones de botellas de agua."
    assert " ".join(chunks).split() == text.split()        # no se pierde ni se recorta texto
    assert all(len(c) <= 240 for c in chunks)


def test_speech_text_reads_prices_and_orders():
    assert speech_text("Pedido O00001 por 9,98 EUR.") == "Pedido número 1 por 9 euros con 98."
    assert speech_text("Total: 12.00 euros") == "Total: 12 euros"
    assert speech_text("comprar 2 × «botella»") == "comprar 2 por botella"
