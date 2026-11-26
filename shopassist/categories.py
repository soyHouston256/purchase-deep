"""Categorías de producto (en español) para la demo web.

El catálogo ESCI (Amazon Shopping Queries) NO trae categoría, migas de pan ni imágenes: solo
``title``, ``description``, ``bullet_point``, ``brand`` y ``color``. Por eso la categoría se infiere:

1. **Libros**: los ASIN que empiezan por dígito son ISBN-10 (en Amazon, libros).
2. **Reglas por palabra clave sobre el título**, con la regla de "la palabra de producto que aparece
   PRIMERO manda" (en los títulos de Amazon el sustantivo principal va al principio: «Botella de agua
   para deporte…», «Funda para iPhone…», «Mochila para portátil…»). Hay palabras *débiles*
   (funda, soporte, kit, set, bolsa, repuesto…) que solo deciden si no aparece ninguna fuerte.
3. **kNN sobre las incrustaciones e5** que ya usa el recuperador (título + viñetas + descripción,
   ``data/emb_*.npy``): los productos sin regla toman el voto ponderado de sus 15 vecinos más
   parecidos entre los que sí tienen regla. Si el vecino más cercano está lejos → «Otros».

Todo es determinista y se calcula una vez al arrancar (≈1 s para 10k productos); el resultado se
cachea en ``data/categories_<hash>.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional

import numpy as np

from .textutil import norm

# categoría -> icono por defecto
CATEGORY_ICONS = {
    "Audio": "🎧",
    "Móviles y tablets": "📱",
    "Informática": "💻",
    "Cables y cargadores": "🔌",
    "Pilas y baterías": "🔋",
    "TV, foto y vídeo": "📷",
    "Electrodomésticos": "🧺",
    "Hogar y cocina": "🏠",
    "Muebles": "🪑",
    "Iluminación": "💡",
    "Ropa y calzado": "👕",
    "Bolsos y equipaje": "🎒",
    "Joyería y relojes": "⌚",
    "Belleza y cuidado personal": "💄",
    "Salud y bienestar": "💊",
    "Bebé": "🍼",
    "Juguetes y juegos": "🧸",
    "Videojuegos": "🎮",
    "Deporte y aire libre": "⚽",
    "Herramientas y bricolaje": "🔧",
    "Jardín": "🌱",
    "Mascotas": "🐾",
    "Papelería y oficina": "📓",
    "Manualidades y fiestas": "🎨",
    "Libros": "📚",
    "Coche y moto": "🚗",
    "Alimentación y bebidas": "🍫",
    "Instrumentos musicales": "🎸",
    "Películas y música": "🎬",
    "Otros": "🛍️",
}

# (categoría, patrón sobre el título normalizado, icono específico o None, fuerte?)
# Los patrones se anclan al inicio de palabra (\b) y aceptan plurales/sufijos.
_R = [
    # ---- audio
    ("Audio", r"auricular|controlador(?:a)? de dj|walkie|audifon|cascos?\b|headphone|headset|earbud|earphone|in-ear|altavo|parlante|speaker|barra de sonido|soundbar|microfono|tocadiscos|amplificador(?! de guitarra)|radio\b|mp3", "🎧", True),
    # ---- móviles
    ("Móviles y tablets", r"smartphone|movil(?:es)?\b|telefono movil|iphone|samsung galaxy|xiaomi redmi|tablet|ipad|protector(?:es)? de pantalla|cristal templado|vidrio templado|selfie|palo selfie", "📱", True),
    # ---- informática
    ("Informática", r"ventilador (?:gpu|cpu|para portatil)|ordenador|portatil(?:es)?\b|laptop|chromebook|notebook\b|raton(?:es)?\b|mouse|teclado(?! electronico| musical)|monitor(?:es)?\b|pendrive|memoria usb|tarjeta (?:de )?memoria|tarjeta micro ?sd|micro ?sd|disco duro|ssd\b|placa base|tarjeta grafica|procesador|router|repetidor wifi|impresora|cartucho(?:s)? de tinta|toner|webcam|alfombrilla|hub usb|switch de red|memoria ram|ram ddr", "💻", True),
    # ---- cables y cargadores
    ("Cables y cargadores", r"cargador|cable(?:s)?\b|adaptador|enchufe|regleta|power ?bank|bateria externa|transformador|ladron", "🔌", True),
    # ---- pilas
    ("Pilas y baterías", r"portapilas|pila(?:s)?\b|bateria(?:s)?\b(?! electronica)|baterias recargables", "🔋", True),
    # ---- tv/foto
    ("TV, foto y vídeo", r"televisor|smart ?tv|television|proyector|camara(?! de aire)|objetivo para|tripode|dron|videocamara|gopro|mando a distancia|soporte (?:de )?tv", "📷", True),
    # ---- electrodomésticos
    ("Electrodomésticos", r"aspirador|robot aspirador|lavadora|secadora|frigorific|nevera|microondas|horno\b|freidora|cafetera|tostador|batidora|licuadora|exprimidor|hervidor|plancha(?! (?:de |para (?:el )?)?pelo| alisadora| de cabello)|ventilador|calefactor|aire acondicionado|deshumidificador|humidificador|purificador|robot de cocina|vaporeta|centro de planchado", None, True),
    # ---- hogar y cocina
    ("Hogar y cocina", r"botella(?:s)?\b|termo(?:s)?\b|cantimplora|taza(?:s)?\b|vaso(?:s)?\b|jarra|tetera", "🥤", True),
    ("Hogar y cocina", r"sarten|olla|cacerola|cazo\b|bandeja de horno|molde|cuchill|tabla de cortar|plato(?:s)?\b|vajilla|cubierto|fiambrera|tupper(?!ware)|recipiente|bascula de cocina|bascula cocina|utensilio|espatula|rallador|colador|abrelatas|sacacorchos|especiero|portarrollos", "🍳", True),
    ("Hogar y cocina", r"sabana|funda nordica|nordico|edredon|almohada|cojin|colcha|manta\b|mantas\b|toalla|alfombra|cortina|estor\b|mantel|felpudo|colchon|protector de colchon|cubre ?colchon|edredon", "🛏️", True),
    ("Hogar y cocina", r"vela(?:s)?\b|ambientador|difusor|jarron|marco de fotos|album(?:es)? de fotos|album fotos|marco(?:s)? para fotos|portafotos|reloj de pared|espejo|cuadro(?:s)?\b|lamina(?:s)?\b|poster|vinilo(?:s)? decorativ|decoracion|figura decorativa|perchero|percha(?:s)?\b|cesta|caja de almacenamiento|organizador|cubo de basura|papelera|tendedero|escoba|fregona|mopa|recogedor|bayeta|estropajo|detergente|suavizante|lavavajillas|limpiador|alcachofa (?:de )?ducha|banera|mampara|cortina de ducha|grifo|dispensador de jabon|alcachofa|biombo|burlete|paragua", "🏠", True),
    # ---- muebles
    ("Muebles", r"silla(?:s)?\b|sillon|sofa\b|mesa(?:s)?\b|escritorio|estanteria|estante|taburete|armario|comoda\b|mesita|zapatero|cabecero|puf\b|banco\b|silla gaming|mueble", "🪑", True),
    # ---- iluminación
    ("Iluminación", r"lampara|tira(?:s)? (?:de )?luz|luz (?:nocturna|de armario|armario)|bombilla|tira(?:s)? led|luces led|luz led|foco(?:s)?\b|aplique|plafon|linterna|guirnalda|flexo", "💡", True),
    # ---- ropa y calzado
    ("Ropa y calzado", r"zapatilla|zapato|bota(?:s)?\b|botin|sandalia|chancla|mocasin|deportivas\b|calzado", "👟", True),
    ("Ropa y calzado", r"camiseta|camisa|pantalon|vaquero|jeans|chaqueta|abrigo|cazadora|sudadera|jersey|sueter|chaleco|vestido|falda|blusa|top\b|leggings|mallas|pijama|camison|bata\b|chandal|polo\b|pullover|traje|bermuda|short(?:s)?\b|banador|bikini|calcetin|media(?:s)?\b|panty|leotardo|ropa interior|calzoncillo|boxer|braga|braguita|tanga|sujetador|jockstrap|bodies?\b|gorra|gorro|sombrero|boina|bufanda|guante(?:s)?\b|cinturon|corbata|tirantes|disfraz|ropa\b|uniforme", "👕", True),
    # ---- bolsos
    ("Bolsos y equipaje", r"bolsa deportiva|mochila(?! portabebe)|bolso|maleta|neceser|cartera|monedero|rinonera|bandolera|bolsa de viaje|bolsa de deporte|tarjetero|trolley", "🎒", True),
    # ---- joyería
    ("Joyería y relojes", r"reloj(?:es)?\b(?! de pared)|smartwatch|pulsera(?! de actividad)|collar(?! para perro| de perro| antiparasit)|colgante|pendiente(?:s)?\b|anillo|sortija|tobillera|broche|gemelos|joya", "⌚", True),
    ("Joyería y relojes", r"pulsera de actividad|smartband|fitness tracker", "⌚", True),
    # ---- belleza
    ("Belleza y cuidado personal", r"champu|limpiador (?:facial|de poros)|puntos negros|acondicionador|mascarilla capilar|tinte|crema|serum|serum|locion|perfume|colonia|eau de|desodorante|maquillaje|pintalabios|labial|rimel|mascara de pestanas|sombra de ojos|base de maquillaje|esmalte|unas\b|manicura|pedicura|pestanas|cepillo de dientes|cepillo electrico|pasta de dientes|dentifrico|hilo dental|irrigador|afeitadora|maquinilla|cuchilla(?:s)? de afeitar|depiladora|recortadora|cortapelos|plancha (?:de |para (?:el )?)?pelo|plancha alisadora|plancha de cabello|coloracion|lampara (?:de )?(?:uv )?(?:para )?unas|lampara uv|secador|rizador|moldeador|cepillo (?:para el )?pelo|peine|gel de ducha|gel\b|jabon|exfoliante|aceite (?:corporal|esencial|de masaje|capilar)|protector solar|cosmetic|tratamiento capilar|extensiones de pelo|peluca|pelo\b", "💄", True),
    # ---- salud
    ("Salud y bienestar", r"vitamina|suplemento|complemento alimenticio|capsulas|comprimidos|probiotic|colageno|proteina(?:s)? en polvo|whey|termometro|tensiometro|pulsioximetro|mascarilla(?:s)? (?:ffp|quirurgica|desechable)|mascarillas\b|tirita|venda|rodillera|tobillera ortopedica|faja|corrector postural|plantilla|masajeador|pistola de masaje|vibrador|succionador|satisfyer|preservativo|lubricante|test de embarazo|bascula(?! de cocina| cocina)|tapones para los oidos|e-liquid|cigarrillo electronico", "💊", True),
    # ---- bebé
    ("Bebé", r"bebe(?:s)?\b|minicuna|colecho|biberon|chupete|panal(?:es)?\b|cambiador|cuna\b|carrito de bebe|cochecito|silla de paseo|sillas de paseo|portabebe|mochila portabebe|trona|vigilabebe|babero|recien nacido", "🍼", True),
    # ---- juguetes
    ("Juguetes y juegos", r"juguete|toys?\b|lego\b|playmobil|muneca|muneco|peluche|puzzle|rompecabezas|juego de mesa|juegos de mesa|cartas (?:de juego|pokemon)|baraja|figura(?:s)? de accion|funko|coche teledirigido|radiocontrol|rc\b|plastilina|pinypon|nenuco|barbie\b|hot wheels|nerf|varita|cubo de rubik|cubo magico|tren de juguete|construccion para ninos|bloques de construccion|cometa|pistola de agua|dados\b", "🧸", True),
    # ---- videojuegos
    ("Videojuegos", r"videojuego|trilogy|remastered|standard edition|edicion estandar|game of the year|playstation|ps4|ps5|xbox|nintendo|switch\b|mando (?:para|de) (?:ps|xbox|switch|consola)|gamepad|consola", "🎮", True),
    # ---- deporte
    ("Deporte y aire libre", r"banco de pesas|banco de musculacion|radios? de bici|bicicleta|bici\b|patinete|patines|monopatin|skate|casco de (?:bici|moto)|esterilla|yoga|pilates|mancuerna|pesas\b|kettlebell|banda(?:s)? elastica|banda(?:s)? de resistencia|comba|cuerda de saltar|balon|pelota|raqueta|pala de padel|padel|tenis\b|futbol|baloncesto|golf|natacion|gafas de natacion|gafas de bucear|bucear|snorkel|tienda de campana|saco de dormir|camping|senderismo|trekking|bastones|pesca|cana de pescar|caña|carrete|esqui|snowboard|gimnasio|fitness|crossfit|guantes de boxeo|boxeo|saco de boxeo|dardos|diana\b|running|ciclismo|abdominales|rodillo(?:s)? de espuma|step\b|cinta de correr|bicicleta estatica|espinilleras", "⚽", True),
    # ---- herramientas
    ("Herramientas y bricolaje", r"taladro|atornillador|destornillador|llave(?:s)? (?:inglesa|allen|de vaso|fija|dinamometrica|torx)|alicate|martillo|sierra|lijadora|amoladora|radial\b|caladora|nivel laser|metro\b|cinta metrica|pistola de (?:silicona|calor|pegamento)|soldador|estano|multimetro|tornillo|tuerca|clavo(?:s)?\b|taco(?:s)?\b|broca|caja de herramientas|herramienta|cola de (?:carpintero|madera)|madera de cola|silicona|cinta americana|cinta adhesiva|pintura(?! acrilica| de dedos)|brocha|rodillo de pintar|masilla|bisagra|cerradura|candado|pomo|tirador|escalera|carretilla|compresor|generador", "🔧", True),
    # ---- jardín
    ("Jardín", r"jardin|maceta|semilla|planta(?:s)?\b|manguera|cortacesped|desbrozadora|tijera de podar|podadora|abono|fertilizante|riego|aspersor|cesped|invernadero|barbacoa(?! salsa)|tumbona|sombrilla|hamaca|piscina|toldo|comedero para pajaros|insecticida|antimosquitos|repelente", "🌱", True),
    # ---- mascotas
    ("Mascotas", r"perro(?:s)?\b|gato(?:s)?\b|mascota|cachorro|arena para gatos|rascador|comedero|bebedero|correa (?:para|de) perro|collar (?:para|de) perro|collar antiparasit|pienso|acuario|pecera|jaula|conejo|hamster|pajaro|transportin|arnes (?:para )?perro", "🐾", True),
    # ---- papelería
    ("Papelería y oficina", r"cuaderno|libreta|boligrafo|lapiz|lapices|rotulador|marcador(?:es)?\b|subrayador|agenda|calendario|archivador|carpeta|folio(?:s)?\b|papel (?:a4|de impresora|termico|kraft)|rollo papel termico|grapadora|grapas|clip(?:s)?\b|post-?it|notas adhesivas|pegamento|celo\b|tijeras\b|calculadora|estuche|sobre(?:s)? (?:de papel|blancos)|sello(?:s)?\b|etiqueta(?:s)?\b|pizarra|corrector|sacapuntas|goma de borrar|regla\b|plumier|papel\b|tinta\b|pluma estilografica|archivo\b|bolsas? de papel", "📓", True),
    # ---- manualidades / fiesta
    ("Manualidades y fiestas", r"abalorio|cuentas para|manualidad|kit de bordado|bordado|punto de cruz|ganchillo|lana(?:s)?\b|ovillo|hilo(?:s)? (?:de|para)|aguja(?:s)? de tejer|maquina de coser|tela(?:s)?\b|fieltro|purpurina|pintura acrilica|acuarela|lienzo|pincel|globo(?:s)?\b|guirnalda de fiesta|decoracion de fiesta|cumpleanos|confeti|pinata|mosaico|tesela|chapa(?:s)?\b|pin(?:es)?\b|resina epoxi|arcilla|scrapbooking|pegatina|sticker|cricut|diy\b|bricolaje\b", "🎨", True),
    # ---- coche y moto
    ("Coche y moto", r"coche(?! teledirigido| de juguete| de bebe)|moto(?:s)?\b|motocicleta|automovil|vehiculo|neumatico|rueda de repuesto|llanta|retrovisor|parabrisas|limpiaparabrisas|escobilla|alfombrilla(?:s)? (?:de|para) coche|funda (?:de|para) (?:asiento|moto|coche)|cubierta de la moto|gato hidraulico|wallbox|cargador (?:para|de) coche electrico|portabicicletas|baca\b|cofre de techo|aceite de motor|casco(?:s)? (?:de )?moto|dashcam|manos libres para coche|soporte (?:movil )?(?:para )?coche", "🚗", True),
    # ---- alimentación
    ("Alimentación y bebidas", r"chocolate|bombon|(?<!color )caramelo|gominola|golosina|regaliz|chicle|galleta|cereales|muesli|avena|cafe\b|cafe en|capsulas de cafe|te (?:verde|negro|rojo|matcha|chai)|bolsitas de te|infusion|azucar|sal\b|aceite de oliva|vinagre|salsa|especia|pimienta|harina|arroz|pasta\b|legumbre|conserva|atun|miel\b|mermelada|cacahuete|almendra|nueces|frutos secos|snack|patatas fritas|palomitas|maiz|batido|zumo|refresco|agua mineral|cerveza|vino\b|licor|ginebra|whisky|ron\b|jamon|queso|embutido|malvavisco|nubes\b|turron|polvoron|dulce(?:s)?\b|proteina vegana|sirope|edulcorante|alimento", "🍫", True),
    # ---- instrumentos
    ("Instrumentos musicales", r"guitarra|ukelele|piano|teclado electronico|teclado musical|bateria electronica|tambor|violin|flauta|armonica|saxofon|trompeta|cuerdas para|pua(?:s)?\b|metronomo|afinador|atril (?:de |para )?(?:partitura|musica|orquesta)|pedal de efectos|amplificador de guitarra|cajon flamenco|kalimba|microfono de condensador", "🎸", True),
    # ---- libros (los que no tienen ISBN como ASIN: ediciones Kindle, etc.)
    ("Libros", r"(?:spanish|english|italian|french|german) edition|edicion kindle|tapa blanda|tapa dura|novela|libro(?:s)?\b(?! de (?:colorear|actividades))", "📚", True),
    # ---- películas y música
    ("Películas y música", r"vinilo\]|vinyl\]|vinyl\)|dvd|blu-?ray|\bcd\b(?! player)|disco de vinilo|vinilo lp|banda sonora|temporada completa", "🎬", True),
    # ---- palabras débiles (solo deciden si no hay ninguna fuerte)
    ("Móviles y tablets", r"funda(?:s)?\b|carcasa(?:s)?\b|protector\b", "📱", False),
    ("Hogar y cocina", r"soporte|organizador|caja(?:s)?\b|bolsa(?:s)?\b|cesta", None, False),
    ("Herramientas y bricolaje", r"kit\b|set\b|repuesto|recambio|pieza(?:s)?\b|accesorio", None, False),
    ("Manualidades y fiestas", r"regalo|personalizad", "🎁", False),
]

_COMPILED = [(cat, re.compile(r"\b(?:" + pat + r")"), icon, strong) for cat, pat, icon, strong in _R]


def rule_category(title: str) -> Optional[tuple]:
    """Categoría por reglas sobre el título: la palabra fuerte que aparece primero manda.
    Devuelve ``(categoría, icono, palabra)`` o ``None`` si ninguna regla aplica."""
    t = norm(title)
    best = None  # (fuerte?, -posición, longitud)
    for cat, rx, icon, strong in _COMPILED:
        m = rx.search(t)
        if not m:
            continue
        key = (strong, -m.start(), len(m.group(0)))
        if best is None or key > best[0]:
            best = (key, cat, icon or CATEGORY_ICONS[cat], m.group(0))
    if best is None:
        return None
    return best[1], best[2], best[3]


def _catalog_hash(ids: list, extra: str = "") -> str:
    h = hashlib.md5(("\n".join(ids) + extra + _RULES_VERSION).encode()).hexdigest()[:12]
    return h


_RULES_VERSION = hashlib.md5(repr(_R).encode()).hexdigest()[:8]


def classify_catalog(catalog_df, emb: Optional[np.ndarray] = None, k: int = 15,
                     min_sim: float = 0.80, cache_dir: Optional[str] = None) -> dict:
    """Asigna ``{product_id: {"category", "icon", "source"}}`` a todo el catálogo.

    ``emb``: incrustaciones normalizadas alineadas con ``catalog_df`` (las del recuperador denso).
    ``source``: ``isbn`` | ``regla`` | ``knn`` | ``ninguna``.
    """
    ids = [str(x) for x in catalog_df["product_id"].tolist()]
    titles = catalog_df["title"].fillna("").astype(str).tolist()
    path = None
    if cache_dir:
        path = os.path.join(cache_dir, f"categories_{_catalog_hash(ids, str(emb is not None))}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    out: dict = {}
    labeled_idx, labeled_cat = [], []
    for i, (pid, title) in enumerate(zip(ids, titles)):
        if pid[:1].isdigit():
            out[pid] = {"category": "Libros", "icon": CATEGORY_ICONS["Libros"], "source": "isbn"}
            labeled_idx.append(i); labeled_cat.append("Libros")
            continue
        r = rule_category(title)
        if r is not None:
            out[pid] = {"category": r[0], "icon": r[1], "source": "regla", "kw": r[2]}
            labeled_idx.append(i); labeled_cat.append(r[0])
    rest = [i for i, pid in enumerate(ids) if pid not in out]
    if rest and emb is not None and labeled_idx:
        L = emb[np.asarray(labeled_idx)]
        cats = np.asarray(labeled_cat)
        for s in range(0, len(rest), 512):
            chunk = rest[s: s + 512]
            sims = emb[np.asarray(chunk)] @ L.T
            top = np.argsort(-sims, axis=1)[:, :k]
            for row, i in enumerate(chunk):
                nn = top[row]
                ss = sims[row, nn]
                if ss[0] < min_sim:
                    continue
                votes: dict = {}
                for j, sv in zip(nn, ss):
                    votes[cats[j]] = votes.get(cats[j], 0.0) + float(np.exp((sv - ss[0]) * 20))
                cat = max(votes, key=votes.get)
                out[ids[i]] = {"category": cat, "icon": CATEGORY_ICONS[cat], "source": "knn",
                               "sim": round(float(ss[0]), 3)}
    for pid in ids:
        out.setdefault(pid, {"category": "Otros", "icon": CATEGORY_ICONS["Otros"], "source": "ninguna"})
    if path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
        except Exception:
            pass
    return out
