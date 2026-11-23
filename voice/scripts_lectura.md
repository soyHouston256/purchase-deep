# Guion de lectura para las grabaciones de voz

> Archivo generado por `scripts/gen_scripts_lectura.py` desde `scenarios/scenarios.yaml`.
> No lo edites a mano: cambia el YAML y vuelve a generarlo.

## 1. Grabación de referencia para la clonación (`voice/referencia.wav`)

* Duración: **15–25 s** (XTTS-v2 admite 6–30 s). Mono, sin música ni eco, a ~20 cm del micrófono.
* Debe ser **tu propia voz** (o una voz con autorización explícita y por escrito).
* Lee con tono natural el siguiente texto (está en `voice/referencia.txt`). Es un texto neutro,
  **distinto de cualquier respuesta del asistente**; el código (`TTS._check_text`) impide
  sintetizar frases que coincidan con él.

> Esta mañana salí temprano a caminar por el parque. El cielo estaba despejado y el aire se sentía fresco. Me detuve junto al lago para mirar a los patos, y luego seguí hasta la biblioteca, donde pasé un buen rato leyendo una novela sobre viajes por la costa.

## 2. Audios de los 6 escenarios de voz

Convención de nombres: `voice/scenarios/v{n}_t{k}.wav` (escenario *n*, turno *k* empezando en 1).
Lee **exactamente** la frase (es la transcripción correcta, *gold*, usada para aislar errores del ASR).
Graba cada turno por separado, 2–6 s, con una pausa breve al inicio y al final.

### v1_compra_confirmada — Voz: compra que cumple las restricciones y termina en pedido confirmado

| Archivo | Frase a leer (exacta) |
|---|---|
| `voice/scenarios/v1_t1.wav` | Busco una botella de agua, mi presupuesto es de cuarenta euros |
| `voice/scenarios/v1_t2.wav` | Añade dos unidades del primero al carrito |
| `voice/scenarios/v1_t3.wav` | Sí, es correcto |
| `voice/scenarios/v1_t4.wav` | Prepara la cotización |
| `voice/scenarios/v1_t5.wav` | Sí, confirmo el pedido |

### v2_aclaracion — Voz: solicitud ambigua que requiere aclaración

| Archivo | Frase a leer (exacta) |
|---|---|
| `voice/scenarios/v2_t1.wav` | Añade eso al carrito |
| `voice/scenarios/v2_t2.wav` | Busco audífonos bluetooth |

### v3_cambio_cantidad — Voz: cambio de cantidad conservando el presupuesto

| Archivo | Frase a leer (exacta) |
|---|---|
| `voice/scenarios/v3_t1.wav` | Mi presupuesto es de treinta euros, busco un cuaderno |
| `voice/scenarios/v3_t2.wav` | Añade tres unidades del primero al carrito |
| `voice/scenarios/v3_t3.wav` | Sí, es correcto |
| `voice/scenarios/v3_t4.wav` | Mejor que sean dos unidades |
| `voice/scenarios/v3_t5.wav` | Sí, es correcto |

### v4_agotado — Voz: producto agotado, se busca alternativa o se informa que no hay solución

| Archivo | Frase a leer (exacta) |
|---|---|
| `voice/scenarios/v4_t1.wav` | Busco un ratón inalámbrico |
| `voice/scenarios/v4_t2.wav` | Añade una unidad del primero al carrito |
| `voice/scenarios/v4_t3.wav` | Sí, es correcto |

### v5_correccion_transcripcion — Voz: corrección de la transcripción antes de ejecutar la compra

| Archivo | Frase a leer (exacta) |
|---|---|
| `voice/scenarios/v5_t1.wav` | Busco pilas AA |
| `voice/scenarios/v5_t2.wav` | Añade tres unidades del primero al carrito |
| `voice/scenarios/v5_t3.wav` | No, quería cuatro unidades |
| `voice/scenarios/v5_t4.wav` | Sí, es correcto |

### v6_stock_cambia_y_reconfirmacion — Voz: cambio de stock entre cotización y confirmación, reconfirmación sin duplicar

| Archivo | Frase a leer (exacta) |
|---|---|
| `voice/scenarios/v6_t1.wav` | Busco una mochila, mi presupuesto es de doscientos euros |
| `voice/scenarios/v6_t2.wav` | Añade dos unidades del primero al carrito |
| `voice/scenarios/v6_t3.wav` | Sí, es correcto |
| `voice/scenarios/v6_t4.wav` | Prepara la cotización |
| `voice/scenarios/v6_t5.wav` | Confirmo el pedido |
| `voice/scenarios/v6_t6.wav` | Entonces que sea una unidad |
| `voice/scenarios/v6_t7.wav` | Sí, es correcto |
| `voice/scenarios/v6_t8.wav` | Prepara la cotización |
| `voice/scenarios/v6_t9.wav` | Confirmo el pedido |
| `voice/scenarios/v6_t10.wav` | Confirmo el pedido |


## 3. Consejos

* Graba en el mismo ambiente que la demo del video. Evita leer números como dígitos: di "cuarenta", no "cuatro cero".
* Puedes grabar desde Colab (`record_audio_colab`) o con el móvil y subir los archivos (`upload_audio_colab`).
* Total: 29 audios + 1 referencia.
