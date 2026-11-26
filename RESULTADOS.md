# Resultados – Laboratorio 4 · Asistente de compras por voz

Todos los números salen de la ejecución en Colab del notebook `Laboratorio4_Asistente_Compras_Voz_ejecutado.ipynb`
(tablas completas en `results/REPORTE_auto.md`).

**Configuración**: GPU T4 (15 GB) · semilla 42 · LLM `Qwen/Qwen2.5-3B-Instruct` fp16 · ASR faster-whisper `small`
float16 · TTS Coqui XTTS-v2 · catálogo 10 199 productos · consultas train/test 100/250 · 9 398 pares anotados.

## 1. Recuperación (nDCG@10, ganancias E=1 · S=0,1 · C=0,01 · I=0)

| Sistema | Papel | train | test | IC95 test |
|---|---|---|---|---|
| Aleatorio | cota inferior | 0,536 | 0,548 | [0,520; 0,577] |
| **BM25** | **modelo base léxico** | 0,687 | 0,708 | [0,679; 0,736] |
| BM25 solo título | ablación: texto indexado | 0,672 | 0,700 | [0,669; 0,729] |
| Denso e5 | componente | 0,737 | 0,744 | [0,717; 0,768] |
| Híbrido (fusión) | ablación: sin re-ranker | 0,725 | 0,743 | [0,717; 0,768] |
| BM25 + re-ranker | ablación: sin denso | 0,756 | 0,776 | [0,749; 0,800] |
| **Híbrido + re-ranker** | **sistema completo** | 0,756 | **0,776** | [0,749; 0,800] |

- Fusión elegida **solo con train**: lineal, α = 0,7 (rejilla RRF k ∈ {10, 30, 60, 100}; lineal α ∈ {0,3; 0,5; 0,7}).
- Frente a BM25, el sistema completo gana **+0,068** de nDCG@10 (IC95 de la diferencia pareada [0,046; 0,094]):
  mejora en 143 consultas, empeora en 81 y empata en 26 de 250.
- **Ablación**: quitar el re-ranker cuesta 0,033; quitar el denso no cuesta nada. Cada consulta tiene como máximo
  59 candidatos anotados y el re-ranker reordena los 50 primeros, así que casi siempre ve todos los candidatos
  y la primera etapa deja de importar. El denso sí aporta cuando no hay re-ranker (0,744 frente a 0,708).
  Indexar solo el título baja 0,008 frente a BM25 con todos los campos.

## 2. Escenarios (28: 22 de texto + 6 de voz, cada uno desde `store.restore(SEED)`)

| Modo | Escenarios | Correctos | % | Presupuesto superado | Stock negativo | Pedido sin confirmar | Duplicados |
|---|---|---|---|---|---|---|---|
| Texto, con LLM | 28 | 19 | 67,9 | 0 | 0 | 0 | 0 |
| Voz (audio) | 6 | 2 | 33,3 | 0 | 0 | 0 | 0 |
| Voz (transcripción correcta) | 6 | 2 | 33,3 | 0 | 0 | 0 | 0 |
| Texto, política por reglas (ablación sin LLM, ejecución local) | 28 | 27 | 96,4 | 0 | 0 | 0 | 0 |

**Cero incumplimientos de restricciones en todos los modos.** Los fallos del LLM son de *elección de herramienta*,
nunca de seguridad: las guardas en código y la revalidación atómica de `place_order` bloquean cualquier
operación que pase el presupuesto, deje stock negativo, cree un pedido sin confirmación o lo duplique.

Los 6 casos de voz fallan exactamente igual con el audio que con la transcripción correcta
(`results/REPORTE_auto.md`, tabla *voz_vs_gold*): **ningún fallo es atribuible al ASR**.

## 3. Voz

**ASR (29 enunciados grabados)**: WER 10,4 % · CER 5,6 % · errores en cantidades 1/6 · importes 0/3 ·
palabras de producto 5/20.

| Voz sintetizada | WER | CER | Similitud ECAPA con la referencia |
|---|---|---|---|
| Clonada (XTTS-v2) | 0,120 | 0,055 | **0,661** |
| Hablante de fábrica XTTS (línea base) | 0,068 | 0,023 | −0,046 |
| Mitad vs. mitad de la referencia (cota) | – | – | 0,814 |

La voz clonada es inteligible (WER 12 %) y se parece a la referencia (0,66 frente a una cota de 0,81); la voz de
fábrica es algo más inteligible pero no se parece en nada al hablante.

## 4. Latencia (desde que termina la grabación hasta que la respuesta puede reproducirse)

| Fase | n | ASR | Recuperación | LLM | Herramientas | TTS | Total |
|---|---|---|---|---|---|---|---|
| Primer turno | 1 | 0,69 | 0,27 | 9,66 | 0,00 | 3,86 | 14,48 |
| Turnos posteriores, media | 11 | 0,63 | 0,10 | 5,19 | 0,00 | 5,23 | 11,15 |
| Turnos posteriores, p90 | 11 | 0,73 | 0,27 | 8,55 | 0,00 | 8,60 | 14,98 |

**Carga inicial** (una sola vez): LLM 31,6 s · ASR 2,7 s · TTS 33,7 s · latentes de la referencia 3,3 s.
VRAM de PyTorch tras cargar LLM + TTS: 7,5 GB de 15 GB (Whisper corre en CTranslate2, fuera de esa cuenta).

## 5. Cinco aciertos

| # | Caso | Resultado | Causa |
|---|---|---|---|
| 1 | Recuperación: «atornillador sin escobillas», «bisagras para puertas de armarios», «alisado sin formol» | nDCG@10 = 1,0 | Consultas descriptivas cuyo vocabulario aparece literal en títulos y viñetas; muchos candidatos Exact (27–34 de 40) y el re-ranker los sube todos al top 3. |
| 2 | t03 · presupuesto insuficiente | Correcto | La guarda de presupuesto en código rechaza el `add_to_cart` y el agente lo explica, sin depender del LLM. |
| 3 | t14 · el stock cambia antes de confirmar | Correcto | `place_order` revalida stock, precio y presupuesto dentro de `BEGIN IMMEDIATE`; detecta el cambio y no crea un pedido parcial. |
| 4 | t15 · confirmación repetida | Correcto, un solo pedido | La confirmación es idempotente por `quote_id`: el segundo «confirmo» devuelve el mismo pedido marcado como duplicado. Se ve también en la demo B2 (`Pedidos: [('O00001', 51.99)]`). |
| 5 | v1 · compra completa por voz (5 turnos) y v2 · aclaración por voz | Correctos con audio y con transcripción | Whisper transcribe bien cantidades e importes (0/3 errores en importes) y la NLU determinista convierte números en palabras antes de llegar al LLM. |

## 6. Cinco fallos

| # | Caso | Resultado | Causa |
|---|---|---|---|
| 1 | Recuperación: «pen f olympus» (nDCG 0,07), «elevadores de coches de 2 columnas» (0,19), «hilo soldar sin gas 1mm» (0,22) | Top 3 con sustitutos o irrelevantes | Pocos Exact (3–5 de 40) y la relevancia depende de un modelo o una medida exacta (*Pen F*, *2 columnas*, *1 mm*) que ni BM25 ni e5-small distinguen de productos parecidos. |
| 2 | Recuperación: «pompom mini» (0,13) | Top 3 irrelevante | Consulta corta y ambigua (¿manualidad, accesorio, mascota?) con solo 16 candidatos y 3 Exact. |
| 3 | t10 · quitar del carrito | Incorrecto | Con solo «busco…» el LLM ya añade al carrito y cotiza en el primer turno. Después intenta `add_to_cart` dos veces (las bloquea la guarda de presupuesto) y nunca llama `remove_from_cart`. La política por reglas sí lo resuelve. |
| 4 | t12 / v4 · producto agotado | Incorrecto | El LLM añadió el producto en el primer turno, antes de que se agotara. Cuando el usuario pide más, la guarda de stock rechaza el `add_to_cart` («Stock insuficiente: disponible 0») y el agente pregunta si quiere otra opción, pero no busca la alternativa ni quita el artículo agotado. No se crea ningún pedido. |
| 5 | t21 · cantidad cero | Incorrecto | Otra vez el LLM añade y cotiza en el primer turno; ante «añade cero unidades» actualiza la cantidad y **responde que el pedido «ha sido confirmado» sin que exista ningún pedido** (`orders: []`). La guarda impidió el pedido, pero el texto de la respuesta no está anclado a las operaciones: el anclaje actual solo revisa importes. |

Otros fallos del LLM: t06 (no usa la palabra «desconocido» para un atributo no verificable), t22, v3, v5 (cantidad
final distinta de la pedida tras una corrección) y v6 (no menciona el cambio de stock con las palabras esperadas,
aunque no duplica el pedido).

## 7. Limitaciones

- **El LLM de 3B elige mal la herramienta** en peticiones de varios pasos: 19/28 frente a 27/28 con reglas. Las
  guardas evitan que esos errores se vuelvan pedidos incorrectos, pero no los corrigen.
- **El LLM se adelanta**: añade al carrito y cotiza ante un simple «busco…», lo que arrastra errores a los turnos
  siguientes (t10, t12, t21). Una guarda que exija una intención explícita de compra antes de `add_to_cart` lo evitaría.
- **El texto de la respuesta puede afirmar algo que no ocurrió** (t21: «pedido confirmado» sin pedido). El anclaje
  solo comprueba importes; habría que extenderlo a estados como «confirmado» o «agotado».
- **Latencia de 11 s por turno**: el LLM y XTTS suman unos 10 s. Se puede bajar con streaming del TTS o un LLM en 4 bits.
- **NLU por reglas** con vocabulario acotado; marcas en inglés y modelos con números son difíciles para el ASR y la búsqueda.
- **Precios y stock son sintéticos** (semilla fija) y el catálogo es un subconjunto de ESCI.
- **Siguientes pasos**: Qwen2.5-7B en NF4, Whisper `large-v3-turbo`, validar stock en el carrito y F5-TTS como alternativa de voz.
