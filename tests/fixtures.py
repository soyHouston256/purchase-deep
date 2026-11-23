"""Catálogo sintético en español (~60 productos) para pruebas y escenarios reproducibles.

``make_toy_catalog()`` devuelve un DataFrame con las mismas columnas que el catálogo real
(product_id, title, description, bullet_point, brand, color). ``make_toy_env()`` construye
tienda + recuperador BM25 + agente con la política por reglas (sin GPU).
"""

from __future__ import annotations

import pandas as pd

_SPECS = [
    # (prefijo, plantilla_título, marcas, colores, viñetas)
    ("AUD", "Audífonos inalámbricos Bluetooth 5.{v} {brand} con micrófono y cancelación de ruido, {h} horas de batería",
     ["Sony", "JBL", "Xiaomi", "Soundcore", "Philips", "Sennheiser"], ["negro", "blanco", "azul"],
     "Conexión Bluetooth estable. Estuche de carga USB-C. Almohadillas de silicona."),
    ("AUC", "Auriculares con cable jack 3,5 mm {brand} estéreo, cable de 1,{v} m",
     ["Sony", "Philips", "Panasonic"], ["negro", "blanco"],
     "Sonido estéreo. Compatible con móviles y portátiles. Peso 20 g."),
    ("CAR", "Cargador USB-C {w}W {brand} carga rápida para móvil y tablet",
     ["Anker", "Ugreen", "Samsung", "Belkin", "Xiaomi"], ["blanco", "negro"],
     "Carga rápida Power Delivery. Protección contra sobrecalentamiento. Garantía de 2 años."),
    ("MOU", "Ratón inalámbrico (mouse) {brand} ergonómico, 2.4 GHz y Bluetooth, {dpi} DPI",
     ["Logitech", "Microsoft", "HP", "Trust"], ["negro", "gris", "rosa"],
     "Receptor USB nano. Batería AA incluida. Silencioso."),
    ("TEC", "Teclado mecánico {brand} retroiluminado RGB, distribución española, USB",
     ["Logitech", "Corsair", "Razer", "Trust"], ["negro", "blanco"],
     "Interruptores mecánicos. Cable USB de 1,8 m. Estructura de aluminio."),
    ("FUN", "Funda de silicona para iPhone 13 {brand} antigolpes",
     ["Spigen", "ESR", "Apple"], ["negro", "rojo", "azul", "transparente"],
     "Material: silicona suave. Protección de cámara. Compatible con carga inalámbrica."),
    ("CAB", "Cable USB-C a USB-C {brand} de {m} m, nailon trenzado, 60W",
     ["Anker", "Ugreen", "Amazon Basics"], ["negro", "gris"],
     "Transferencia de datos 480 Mbps. Nailon trenzado resistente."),
    ("PIL", "Pilas alcalinas AA {brand} paquete de {n} unidades",
     ["Duracell", "Energizer", "Varta"], ["multicolor"],
     "Larga duración. Ideal para mandos y juguetes."),
    ("ALT", "Altavoz Bluetooth portátil {brand} resistente al agua IPX7, {w}W",
     ["JBL", "Sony", "Anker"], ["negro", "azul", "rojo"],
     "Sonido 360. Batería de 12 horas. Micrófono integrado."),
    ("MEM", "Memoria USB {brand} de {gb} GB USB 3.0",
     ["SanDisk", "Kingston", "Samsung"], ["negro", "plateado"],
     "Velocidad de lectura hasta 100 MB/s. Llavero incluido."),
    ("MOC", "Mochila para portátil de 15,6 pulgadas {brand} impermeable con puerto USB",
     ["Samsonite", "Targus", "Amazon Basics"], ["negro", "gris"],
     "Material: poliéster resistente. Compartimento acolchado. Capacidad 25 litros."),
    ("CUA", "Cuaderno A5 {brand} de tapa dura, {n} hojas cuadriculadas",
     ["Oxford", "Moleskine", "Miquelrius"], ["azul", "negro"],
     "Papel de 90 g. Encuadernación espiral."),
    ("BOT", "Botella de agua de acero inoxidable {brand} de 750 ml, sin BPA",
     ["Contigo", "Quechua", "Chilly's"], ["azul", "rosa", "plateado"],
     "Mantiene el frío 24 horas. Tapón antigoteo. Apta para lavavajillas."),
]


def make_toy_catalog() -> pd.DataFrame:
    rows = []
    for prefix, tmpl, brands, colors, bullets in _SPECS:
        n = 0
        for bi, brand in enumerate(brands):
            for ci, color in enumerate(colors[: 2 if len(brands) > 3 else len(colors)]):
                n += 1
                title = tmpl.format(brand=brand, v=(bi + ci) % 3, h=20 + 5 * bi, w=[20, 30, 45, 65][bi % 4],
                                    dpi=[1600, 2400, 4000][ci % 3], m=[1, 2][ci % 2], n=[4, 8, 12][bi % 3] if prefix == "PIL" else [80, 120, 160][bi % 3],
                                    gb=[32, 64, 128][bi % 3])
                rows.append({
                    "product_id": f"{prefix}{n:03d}",
                    "title": f"{title} - {color}",
                    "description": f"{title}. Color {color}. Marca {brand}.",
                    "bullet_point": bullets,
                    "brand": brand,
                    "color": color,
                })
    return pd.DataFrame(rows)


def make_toy_env(seed: int = 42, cart_id: str = "c1", llm=None):
    """Devuelve ``(cfg, catalog_df, store, retriever, agent)`` listos para usar."""
    from shopassist.agent import ShoppingAgent
    from shopassist.config import Config
    from shopassist.retrieval import BM25Retriever
    from shopassist.store import Store

    cfg = Config(SEED=seed, data_dir="data")
    cat = make_toy_catalog()
    store = Store(":memory:", cat, seed=seed)
    retriever = BM25Retriever(cat).attach_store(store)
    agent = ShoppingAgent(store, retriever, llm=llm, cart_id=cart_id, cfg=cfg)
    return cfg, cat, store, retriever, agent
