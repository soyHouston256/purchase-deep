"""Genera los notebooks de Colab con nbformat.

- Laboratorio4_Asistente_Compras_Voz.ipynb: la entrega, en 3 partes
  (A · Preparar, B · Verlo funcionar, C · Medirlo).
- Demo_Tienda_Real.ipynb: demo opcional con la tienda real (fuera de la evaluación).

Uso: python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "Laboratorio4_Asistente_Compras_Voz.ipynb"
STORE_DEMO = ROOT / "Demo_Tienda_Real.ipynb"

METADATA = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


class Builder:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    def step(self, title, what, expect):
        """Encabezado de sección: qué hace y qué deberías ver."""
        self.md(f"## {title}\n\n**Qué hace:** {what}\n\n**Qué deberías ver:** {expect}")

    def write(self, path):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"] = METADATA
        nbf.validate(nb)
        nbf.write(nb, path)
        print(f"{path.name}: {len(self.cells)} celdas")


# Celdas comunes a los dos notebooks: traer el código e instalar dependencias.
GET_CODE = r"""
# Trae el código a Colab (elige UNA opción)
import os, sys
REPO_URL = ""          # opción 1: URL del repo, p.ej. "https://github.com/<usuario>/purchase.git"
ZIP_IN_DRIVE = "/content/drive/MyDrive/purchase.zip"   # opción 2: zip en tu Drive
WORKDIR = "/content/purchase"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.path.exists(WORKDIR):
    if REPO_URL:
        !git clone -q {REPO_URL} {WORKDIR}
    else:
        from google.colab import drive
        drive.mount("/content/drive")
        !unzip -q -o {ZIP_IN_DRIVE} -d /content/
if os.path.exists(WORKDIR):
    os.chdir(WORKDIR)
sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())
# Recupera grabaciones guardadas antes en Drive (opcional)
if IN_COLAB and os.path.exists("/content/drive/MyDrive/purchase/voice"):
    !cp -rn /content/drive/MyDrive/purchase/voice/* voice/
"""

INSTALL = r"""
# Instala las dependencias (versiones fijadas en requirements.txt)
!pip install -q -r requirements.txt
!apt-get -qq install -y ffmpeg > /dev/null
"""


# ============================================================================
# Notebook principal
# ============================================================================
nb = Builder()

nb.md(r"""
# Laboratorio 4 – Asistente de compras por voz
**Deep Learning – UTEC**

Un asistente al que le hablas en español: **entiende tu voz (Whisper) → busca productos en un catálogo real (RAG) →
compra en una tienda simulada con herramientas → te responde con una voz clonada (XTTS-v2)**.
Todo corre dentro de Colab, sin APIs externas.

### Cómo leer este notebook

| Parte | Para qué sirve | Secciones |
|---|---|---|
| **A · Preparar** | Dejar todo listo: código, datos, tienda, tu voz y los modelos | A0 – A4 |
| **B · Verlo funcionar** | Ver el asistente en acción: buscar, conversar por texto y por voz | B1 – B3 |
| **C · Medirlo** | Las métricas que pide el laboratorio y el reporte final | C1 – C7 |

La lógica vive en el paquete `shopassist/`. Las celdas solo lo llaman y muestran el resultado.

### Antes de empezar
1. `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`.
2. Sube el código: `purchase.zip` en `MyDrive/` o la URL del repo en la celda A0.
3. Ejecuta las celdas en orden. En A3 se te pedirá grabar tu voz.
""")

# ------------------------------------------------------------------ A
nb.md("# Parte A · Preparar")

nb.step("A0. Configuración",
        "comprueba la GPU, trae el código, instala dependencias y fija las semillas.",
        "una GPU *Tesla T4* y, al final, la configuración `cfg`.")
nb.code(r"""
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || echo "Sin GPU"
""")
nb.code(GET_CODE)
nb.code(INSTALL)
nb.code(r"""
import time, json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display

from shopassist.config import Config
from shopassist.notebook_utils import set_seeds, memory_report, store_hash, show_turn, write_report

cfg = Config()
set_seeds(cfg.SEED)
os.makedirs("results", exist_ok=True)
cfg
""")

nb.step("A1. Datos: Amazon ESCI en español (versión *small*)",
        "descarga el dataset y arma el catálogo (≥ 5 000 productos) y las consultas de `train` y `test`, "
        "separadas por `query_id`. Las etiquetas ESCI solo se usan para evaluar: el agente nunca las ve.",
        "todas las comprobaciones en `True` y un resumen del catálogo.")
nb.code(r"""
from shopassist.data import prepare_data, load_data, check_data
prepare_data(cfg)
catalog_df, queries_df, qrels_df = load_data(cfg)
checks = check_data(catalog_df, queries_df, qrels_df, cfg)
display(pd.Series(checks))
""")
nb.code(r"""
# ¿Cómo son los datos?
stats = {
    "productos": len(catalog_df),
    "consultas_train": int((queries_df.split == "train").sum()),
    "consultas_test": int((queries_df.split == "test").sum()),
    "pares_anotados": len(qrels_df),
    "candidatos_por_consulta (media)": round(qrels_df.groupby("query_id").size().mean(), 1),
}
display(pd.Series(stats))

lab = qrels_df.merge(queries_df[["query_id", "split"]], on="query_id").pivot_table(
    index="esci_label", columns="split", values="product_id", aggfunc="count")
display(lab)
lab.plot(kind="bar", figsize=(6, 3), title="Distribución de etiquetas ESCI"); plt.tight_layout(); plt.show()

cov = catalog_df[["description", "bullet_point", "brand", "color"]].apply(lambda c: (c.fillna("").str.len() > 0).mean())
display(cov.rename("cobertura de atributos"))
""")

nb.step("A2. Tienda simulada",
        "crea la tienda en SQLite. ESCI no trae precios ni stock, así que se generan de forma determinista "
        "a partir de la semilla y el id de cada producto.",
        "5 productos con precio y stock, y `reproducible: True` (tras `restore(SEED)` la tienda vuelve exactamente al inicio).")
nb.code(r"""
from shopassist.store import Store
store = Store(cfg.db_path, catalog_df, seed=cfg.SEED, currency=cfg.currency)
sample_ids = catalog_df.product_id.sample(5, random_state=cfg.SEED).tolist()
display(pd.DataFrame([store.get_product(p) for p in sample_ids])[["product_id", "title", "brand", "price", "stock"]])
""")
nb.code(r"""
# ¿restore(SEED) deja la tienda igual que al principio?
h0 = store_hash(store)
store.set_stock(sample_ids[0], 0); store.add_to_cart("demo", sample_ids[1], 2)   # ensuciamos la tienda
h1 = store_hash(store)
store.restore(cfg.SEED)
h2 = store_hash(store)
reproducible = h0 == h2
print("inicial:", h0, "| tras cambios:", h1, "| tras restore:", h2, "| reproducible:", reproducible)
""")

nb.step("A3. Grabar tu voz",
        "graba la voz de referencia para la clonación (15–25 s leyendo `voice/referencia.txt`, tu voz o una voz autorizada) "
        "y los audios de los escenarios de voz (guion en `voice/scripts_lectura.md`). Solo graba lo que falte.",
        "el texto a leer y el audio de referencia para escucharlo.\n\n"
        "> Consejo: copia la carpeta `voice/` a `MyDrive/purchase/voice` para no volver a grabar en otra sesión.")
nb.code(r"""
from shopassist.voice import record_audio_colab, upload_audio_colab
print(open("voice/referencia.txt", encoding="utf-8").read())
if IN_COLAB and not os.path.exists("voice/referencia.wav"):
    record_audio_colab(22, "voice/referencia.wav")    # o: upload_audio_colab("voice/referencia.wav")
display(Audio("voice/referencia.wav"))
""")
nb.code(r"""
# Audios de los escenarios de voz (solo los que falten)
from shopassist.scenarios import load_scenarios
SCENARIOS = load_scenarios("scenarios/scenarios.yaml")
voice_turns = [(sc["id"], t) for sc in SCENARIOS if sc.get("voice") for t in sc["turns"]]
faltan = [(i, t) for i, t in voice_turns if not os.path.exists(t["audio"])]
print(f"{len(voice_turns)} audios de escenarios, faltan {len(faltan)}")
if IN_COLAB:
    for sid, t in faltan:
        input(f"[{sid}] Pulsa Enter y lee: «{t['user']}»")
        record_audio_colab(5, t["audio"])
""")

nb.step("A4. Cargar los modelos",
        "construye el buscador y carga el LLM, el reconocimiento de voz (ASR) y la síntesis de voz (TTS).",
        "la tabla de tiempos de carga y la memoria usada, que debe quedar por debajo de los 15 GB de la T4.")
nb.md(r"""
| Componente | Modelo | VRAM aprox. | Por qué |
|---|---|---|---|
| LLM | `Qwen/Qwen2.5-3B-Instruct` fp16 (o 7B en 4 bits) | ~6.5 GB | Buen español y JSON; cabe junto con la voz en una T4 |
| ASR | `faster-whisper small` float16 | ~1 GB | Buen WER en español y rápido; `medium`/`large-v3-turbo` como opción |
| TTS | Coqui XTTS-v2 | ~2 GB | Clonación *zero-shot* en español con 6–30 s de referencia |
| Embeddings | `multilingual-e5-small` | ~0.5 GB | Multilingüe y rápido |
| Re-ranker | `mmarco-mMiniLMv2-L12` | ~0.5 GB | Cross-encoder multilingüe |
| Similitud de voz | ECAPA (SpeechBrain) | ~0.1 GB | Solo para evaluar (C4) |

Total ≈ 10.5–11 GB < 15 GB.

> Licencia XTTS-v2: Coqui Public Model License (uso no comercial / académico).
""")
nb.code(r"""
# Buscador híbrido: BM25 + denso (e5) + fusión RRF + re-ranker
from shopassist.retrieval import build_retriever
retriever = build_retriever(cfg, catalog_df, "hybrid_rerank", store=store)
""")
nb.code(r"""
from shopassist.notebook_utils import load_voice_and_llm
USE_LLM = True       # False -> el agente decide con reglas, sin LLM (sirve sin GPU)
ASR_SIZE = "small"   # o "medium" / "large-v3-turbo"

llm, asr, tts, cold_df = load_voice_and_llm(cfg, use_llm=USE_LLM, asr_size=ASR_SIZE)
display(cold_df)
memory_report("tras cargar LLM + ASR + TTS")
""")

# ------------------------------------------------------------------ B
nb.md("# Parte B · Verlo funcionar")

nb.step("B1. Buscar productos",
        "prueba el buscador con dos consultas.",
        "5 productos por consulta, ordenados por relevancia, con precio y stock.")
nb.code(r"""
for q in ["audífonos bluetooth", "funda para móvil"]:
    print("\n>>", q)
    display(pd.DataFrame(retriever.search(q, k=5))[["product_id", "title", "price", "stock", "score"]])
""")

nb.step("B2. Conversación por texto",
        "una compra completa en 8 turnos. El agente recuerda el presupuesto y las cantidades y elige qué herramienta usar. "
        "Las validaciones de cantidad, presupuesto y stock se hacen en código, no en el LLM.",
        "la búsqueda, la comparación, el cambio de cantidad, la cotización y **un solo pedido**: "
        "el segundo «Confirmo» devuelve el mismo pedido en vez de crear otro.")
nb.code(r"""
from shopassist.agent import ShoppingAgent
store.restore(cfg.SEED)
agent = ShoppingAgent(store, retriever, llm=llm, cart_id="demo", cfg=cfg)

for u in ["Mi presupuesto es de 60 euros", "Busco audífonos bluetooth", "Compara el primero y el segundo",
          "Añade dos unidades del primero al carrito", "Mejor que sea una sola unidad",
          "Prepara la cotización", "Confirmo el pedido", "Confirmo el pedido"]:
    show_turn(agent, u); print()
print("Pedidos:", [(o["order_id"], o["total"]) for o in store.list_orders()])
""")

nb.step("B3. Un turno completo por voz",
        "grabas una pregunta, Whisper la transcribe, el agente responde y XTTS lo lee con tu voz clonada. "
        "El cronómetro empieza cuando termina la grabación.",
        "la transcripción, la respuesta, el tiempo de cada etapa y el audio de respuesta.")
nb.code(r"""
from shopassist.latency import VoicePipeline
assert tts is not None, "Falta la voz: graba la referencia en A3 y vuelve a ejecutar A4"
store.restore(cfg.SEED)
voice_agent = ShoppingAgent(store, retriever, llm=llm, cart_id="voz", cfg=cfg)
pipe = VoicePipeline(asr, voice_agent, tts, out_dir="results/voice_demo")

DEMO = "results/voice_demo/pregunta.wav"
if IN_COLAB:
    record_audio_colab(6, DEMO)            # di p.ej.: "Busco una botella de agua de menos de veinte euros"
else:
    DEMO = "voice/scenarios/v1_t1.wav"
rec = pipe.run_turn(DEMO, turn_id="demo1")
print("Transcripción:", rec["transcript"]); print("Respuesta:", rec["reply"])
display(pd.Series(rec["timings"]).round(2))
display(Audio(rec["audio_out"]))
""")

# ------------------------------------------------------------------ C
nb.md("# Parte C · Medirlo")

nb.step("C1. Calidad de la búsqueda (nDCG@10)",
        "compara el sistema completo con BM25 (línea base) y con ablaciones: sin re-ranker, sin denso y solo con el título. "
        "Ganancias E=1, S=0.1, C=0.01, I=0. Los parámetros de la fusión se ajustan **solo con `train`**; `test` se reporta una vez.",
        "una tabla con nDCG@10 en train y test (con IC95) por sistema, y un gráfico de barras.")
nb.code(r"""
from shopassist.evaluation import run_retrieval_experiments, success_analysis
exp = run_retrieval_experiments(cfg, catalog_df, queries_df, qrels_df)
print("Parámetros elegidos en train:", exp["params"])
ret_table = exp["table"]
display(ret_table)
ret_table.to_csv("results/retrieval_table.csv", index=False)
if exp.get("tuning") is not None:
    display(exp["tuning"]["table"].head(10))
""")
nb.code(r"""
fig, ax = plt.subplots(figsize=(9, 4))
t = ret_table.set_index("sistema")[[c for c in ["ndcg@10_train", "ndcg@10_test"] if c in ret_table.columns]]
t.plot(kind="barh", ax=ax)
base = ret_table.loc[ret_table.kind == "bm25", "ndcg@10_test"].iloc[0]
ax.axvline(base, ls="--", c="gray", label="BM25 test")
ax.set_xlabel("nDCG@10"); ax.set_title("Búsqueda: línea base vs. sistema y ablaciones"); ax.legend()
plt.tight_layout(); plt.savefig("results/retrieval_bar.png", dpi=120); plt.show()
""")

nb.step("C2. Los 28 escenarios",
        "corre 22 escenarios por texto y 6 por voz (`scenarios/scenarios.yaml`), cada uno desde `store.restore(SEED)`. "
        "Los de voz se corren dos veces: con el audio real y con la transcripción correcta, para separar los errores del ASR. "
        "Todo se guarda en `results/scenarios/<modo>/<id>/`.",
        "una fila por escenario con `ok` y los chequeos que fallaron, y la comparación voz vs. transcripción correcta.")
nb.code(r"""
from shopassist.scenarios import run_all, scenario_metrics, asr_errors_from_results, compare_voice_vs_gold
def agent_factory():
    return ShoppingAgent(store, retriever, llm=llm, cart_id="c1", cfg=cfg)

df_text, res_text = run_all(SCENARIOS, agent_factory, store, input_mode="text")
display(df_text[["id", "titulo", "voz", "ok", "checks_ok", "checks", "fallidos"]])
print(scenario_metrics(df_text))
""")
nb.code(r"""
df_voice, res_voice = run_all(SCENARIOS, agent_factory, store, input_mode="voice", asr=asr, tts=tts)
df_gold, res_gold = run_all(SCENARIOS, agent_factory, store, input_mode="gold_transcript")
display(df_voice[["id", "ok", "fallidos", "error"]])
cmp = compare_voice_vs_gold(df_voice, df_gold); display(cmp)
for r in res_voice:
    print(f"\n=== {r['id']} ===")
    for t in r["turns"]:
        print(f"  [{t['turn']+1}] correcto: {t['user_gold']}\n      asr     : {t['user_text']}\n      bot     : {t['reply'][:160]}")
""")

nb.step("C3. Métricas de escenarios y errores del ASR",
        "calcula el % de escenarios correctos, los incumplimientos (presupuesto superado, stock negativo, "
        "pedido sin confirmación, duplicados) y los errores del ASR en cantidades, importes y productos.",
        "una tabla por modo (texto, voz, transcripción correcta) y el detalle de errores del ASR.")
nb.code(r"""
from shopassist.voice_eval import summarize_asr_errors
metrics_df = pd.DataFrame([{"modo": name, **scenario_metrics(d)} for name, d in
                           [("texto (22+6)", df_text), ("voz (audio)", df_voice), ("voz (transcripción correcta)", df_gold)]])
display(metrics_df)
asr_err = asr_errors_from_results(res_voice)
asr_summary = summarize_asr_errors(asr_err)
display(asr_err[["id", "gold", "hyp", "wer", "qty_errors", "amount_errors", "product_missing"]])
display(pd.Series(asr_summary))
""")

nb.step("C4. Calidad de la voz clonada",
        "sintetiza 6 respuestas típicas (distintas del texto de referencia) y mide si se entienden "
        "(WER/CER de Whisper) y cuánto se parecen a tu voz (coseno ECAPA), comparado con una voz de fábrica de XTTS.",
        "WER/CER y similitud por frase, y un audio de ejemplo.")
nb.code(r"""
from shopassist.voice_eval import evaluate_voice_quality, SpeakerEncoder
texts_eval = [
    "He encontrado tres botellas de agua por menos de veinte euros. La primera cuesta doce con cincuenta.",
    "Tu carrito tiene dos unidades y el total es de veinticinco euros con noventa céntimos.",
    "Ese producto está agotado. Te propongo una alternativa similar de la misma marca.",
    "¿Quieres que confirme el pedido? El total está dentro de tu presupuesto.",
    "No puedo verificar si es resistente al agua; ese dato no aparece en la ficha del producto.",
    "Pedido confirmado. El número de pedido es O cero cero cero uno.",
]
vq_detail, vq_summary = evaluate_voice_quality(tts, asr, texts_eval, out_dir="results/voice_quality", encoder=SpeakerEncoder())
display(vq_summary); display(vq_detail[["id", "voz", "wer", "cer", "sim_ref"]])
display(Audio(vq_detail.iloc[0]["wav"]))
""")

nb.step("C5. Tiempos: arranque y latencia por turno",
        "mide cuánto tarda en construirse cada buscador y el tiempo por etapa (ASR, búsqueda, LLM, herramientas, TTS) "
        "desde el fin de la grabación hasta el audio, separando el arranque en frío de los turnos siguientes.",
        "una tabla de construcción de índices, otra de latencia (media / p90) y un gráfico por etapa.")
nb.code(r"""
# Tiempo de construcción de cada buscador
load_times = []
for kind in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
    t0 = time.perf_counter()
    r = build_retriever(cfg, catalog_df, kind, store=store)
    r.search("calentar", k=1)  # fuerza la carga de los modelos
    load_times.append({"recuperador": kind, "construccion_s": round(time.perf_counter() - t0, 2)})
display(pd.DataFrame(load_times))
""")
nb.code(r"""
# Latencia con turnos "calientes" usando los audios de los escenarios
from shopassist.latency import latency_summary, plot_latency
store.restore(cfg.SEED)
pipe = VoicePipeline(asr, ShoppingAgent(store, retriever, llm=llm, cart_id="lat", cfg=cfg), tts, out_dir="results/latency")
for sc in [s for s in SCENARIOS if s.get("voice")][:3]:
    for t in sc["turns"]:
        if os.path.exists(t["audio"]):
            pipe.run_turn(t["audio"], turn_id=f"{sc['id']}_{t['audio'][-6:-4]}")
lat = latency_summary(pipe.history, load_timings=cold_df.to_dict("records"))
display(lat.round(3))
lat.to_csv("results/latency_summary.csv", index=False)
plot_latency(lat); plt.savefig("results/latency_bar.png", dpi=120); plt.show()
""")

nb.step("C6. 5 aciertos y 5 fallos",
        "elige automáticamente las 5 mejores y 5 peores consultas de `test` (nDCG@10 del sistema completo) "
        "y los escenarios correctos y fallidos.",
        "las tablas de aciertos y fallos y, para cada fallo, qué esperaba el dataset y qué devolvió el sistema. "
        "**Completa las causas** en `REPORTE.md` (sinónimos, marcas, falta de atributos, errores de ASR en números…).")
nb.code(r"""
full = [k for k in exp["results"] if "re-ranker" in k and "BM25" not in k][0]
sa = success_analysis(exp["results"][full]["test"]["per_query"], n=5)
print("Aciertos (búsqueda):"); display(sa["best"])
print("Fallos (búsqueda):"); display(sa["worst"])

cat_idx = catalog_df.set_index("product_id")
def explain(qid):
    g = qrels_df[qrels_df.query_id == qid]
    exact = g[g.esci_label == "E"].product_id.head(2).tolist()
    top = exp["results"][full]["test"]["per_query"].set_index("query_id").loc[qid, "top3"]
    return {"exactos": [cat_idx.loc[p, "title"][:80] for p in exact], "top3_sistema": [cat_idx.loc[p, "title"][:80] for p in top]}
for qid in sa["worst"].query_id[:5]:
    print(qid, json.dumps(explain(qid), ensure_ascii=False, indent=1))
""")
nb.code(r"""
all_sc = pd.concat([df_text.assign(modo="texto"), df_voice.assign(modo="voz")])
failed_sc = all_sc[~all_sc.ok][["modo", "id", "titulo", "fallidos"]]
print("Escenarios correctos:"); display(all_sc[all_sc.ok].head(5)[["modo", "id", "titulo"]])
print("Escenarios fallidos:"); display(failed_sc.head(10))
""")
nb.md(r"""
### Limitaciones (completar con lo observado)
* La NLU basada en reglas cubre un vocabulario acotado; el LLM de 3B puede elegir mal herramientas en peticiones largas.
* Atributos que no están en la ficha (resistencia al agua, compatibilidad) se reportan como *desconocidos*.
* El ASR confunde números parecidos (catorce/cuarenta) y marcas en inglés; por eso el agente repite lo entendido antes de comprar.
* Precios y stock son sintéticos; el catálogo es un subconjunto de ESCI.
* La clonación *zero-shot* depende mucho de la calidad de la grabación de referencia.
""")

nb.step("C7. Reporte automático",
        "junta todas las tablas de las partes A y C en `results/REPORTE_auto.md`.",
        "el mensaje `results/REPORTE_auto.md escrito`. Copia los números a la plantilla del README (§8).")
nb.code(r"""
REPORT = {
    "datos": checks, "stats": stats, "etiquetas": lab, "restore_reproducible": reproducible,
    "carga_indices": pd.DataFrame(load_times), "carga_modelos": cold_df,
    "recuperacion": ret_table,
    "escenarios": metrics_df, "asr_errores": pd.Series(asr_summary).to_frame("valor"), "voz_vs_gold": cmp,
    "calidad_voz": vq_summary, "latencia": lat.round(3),
    "aciertos_recuperacion": sa["best"], "fallos_recuperacion": sa["worst"], "escenarios_fallidos": failed_sc,
}
write_report(REPORT, "results/REPORTE_auto.md")
""")

nb.write(MAIN)


# ============================================================================
# Demo opcional: tienda real
# ============================================================================
demo = Builder()

demo.md(r"""
# Demo extra: el mismo agente en una tienda real (soyHouston256/store)

**Opcional. No forma parte de la evaluación oficial**, que está en `Laboratorio4_Asistente_Compras_Voz.ipynb`
(ESCI + tienda simulada).

Aquí el *mismo* agente, sin cambiar su lógica, compra en una tienda real de merch (polos, tazas y mousepads)
a través de `shopassist.remote_store.RemoteStoreAdapter`, que tiene la misma interfaz que `Store`:

* catálogo y precios (PEN) desde `GET {STORE_API_URL}/api/products`, indexados con BM25;
* la API no tiene stock numérico: el stock se **simula** en una copia SQLite reproducible (`soldOut` → 0);
* antes de confirmar se vuelve a leer cada producto (`GET /api/products/:id`) para detectar cambios de precio o agotados;
* el agente pide **talla / color / corte** cuando faltan.

Sin `STORE_API_URL` se usa una copia offline del catálogo (`scenarios/tienda_real_fixture.json`).

> ⚠️ **Pedidos reales.** Por defecto todo es **DRY-RUN**: el pedido se registra solo en local y se muestra el *payload*
> que se enviaría a `POST /api/orders`. Un pedido **real** solo se crea con `allow_real_orders=True` **y**
> `customer={"dni", "name", "phone"}`. Nunca se hace POST dos veces para la misma cotización.
""")
demo.code(GET_CODE)
demo.code(INSTALL)
demo.code(r"""
import os, json
import pandas as pd
from IPython.display import display
from shopassist.config import Config
from shopassist.remote_store import build_remote_demo

cfg = Config()
os.environ.setdefault("STORE_API_URL", "https://api.devhaus.pe")   # vacío = copia offline
demo = build_remote_demo(cfg)                # dry-run (allow_real_orders=False por defecto)
tienda, agent_factory_tienda = demo["store"], demo["agent_factory"]
print("Modo:", "OFFLINE (fixture)" if demo["offline"] else f"API {tienda.base_url}", "| productos:", len(tienda.product_ids()),
      "| moneda:", demo["cfg"].currency)
display(tienda.catalog_df[["product_id", "title", "color", "bullet_point"]].head(8))
""")
demo.code(r"""
tienda.restore(cfg.SEED)
agente_tienda = agent_factory_tienda("demo_tienda")
for u in ["Quiero un polo", "El de Redis", "Talla M, color negro",
          "Quiero también una taza de React blanca", "Sí, confirmo el pedido", "Sí, confirmo el pedido"]:
    t = agente_tienda.handle(u)
    print(f"👤 {u}\n🤖 {t.reply}\n")
for o in tienda.list_orders():
    print(o["order_id"], o["total"], o["remote"]["mode"], "→", o["remote"]["endpoint"])
    print(json.dumps(o["remote"]["payload"], ensure_ascii=False, indent=1))
""")
demo.code(r"""
# Escenarios de la tienda real (dry-run): ambigüedad, talla/color, agotado, taza
from shopassist.scenarios import load_scenarios, run_all
df_tienda, res_tienda = run_all(load_scenarios("scenarios/scenarios_tienda_real.yaml"), agent_factory_tienda, tienda,
                                out_dir="results/tienda_real")
display(df_tienda[["id", "titulo", "ok", "checks_ok", "checks", "fallidos"]])
""")
demo.code(r"""
# (NO ejecutar salvo que se quiera crear un pedido REAL en la tienda)
# demo_real = build_remote_demo(cfg, allow_real_orders=True,
#                               customer={"dni": "...", "name": "...", "phone": "..."})
""")

demo.write(STORE_DEMO)
