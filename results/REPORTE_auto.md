# Reporte automático – Laboratorio 4

Generado: 2026-11-25 23:19

## datos

|                          |   valor |
|:-------------------------|--------:|
| catalog_size             |   10199 |
| catalog_ok               |    True |
| n_test_queries           |     250 |
| n_train_queries          |     100 |
| eval_queries_ok          |    True |
| all_annotated_in_catalog |    True |
| catalog_has_no_labels    |    True |

## stats

|                                 |   valor |
|:--------------------------------|--------:|
| productos                       | 10199   |
| consultas_train                 |   100   |
| consultas_test                  |   250   |
| pares_anotados                  |  9398   |
| candidatos_por_consulta (media) |    26.9 |

## etiquetas

| esci_label   |   test |   train |
|:-------------|-------:|--------:|
| C            |    512 |     195 |
| E            |   2835 |    1209 |
| I            |   1164 |     357 |
| S            |   2104 |    1022 |

## restore_reproducible

True

## carga_indices

| recuperador   |   construccion_s |
|:--------------|-----------------:|
| bm25          |             0    |
| dense         |             0.02 |
| hybrid        |             0.02 |
| hybrid_rerank |             0.36 |

## carga_modelos

| model            |   load_s |   vram_gb_after |
|:-----------------|---------:|----------------:|
| llm              | 31.5679  |         5.74801 |
| asr              |  2.70491 |         5.74801 |
| tts              | 33.676   |         7.49806 |
| tts_latentes_ref |  3.271   |       nan       |

## recuperacion

| sistema                      | kind          | descripcion                                                                 |   ndcg@10_train |   n_train |   ndcg@10_test |   n_test | ic95_test      |   s_por_consulta |
|:-----------------------------|:--------------|:----------------------------------------------------------------------------|----------------:|----------:|---------------:|---------:|:---------------|-----------------:|
| Aleatorio                    | random        | Cota inferior: orden aleatorio                                              |          0.5359 |       100 |         0.5481 |      250 | [0.520, 0.577] |           0.0001 |
| BM25 (baseline)              | bm25          | Línea base léxica: BM25 propio sobre título+marca+color+viñetas+descripción |          0.687  |       100 |         0.7078 |      250 | [0.679, 0.736] |           0.0001 |
| BM25 solo título             | bm25_title    | Ablación del texto indexado (solo título)                                   |          0.6716 |       100 |         0.6995 |      250 | [0.669, 0.729] |           0.0001 |
| Denso e5                     | dense         | multilingual-e5-small, coseno                                               |          0.7373 |       100 |         0.7444 |      250 | [0.717, 0.768] |           0.0118 |
| Híbrido RRF                  | hybrid        | BM25 + denso fusionados (parámetros ajustados en train)                     |          0.7254 |       100 |         0.7429 |      250 | [0.717, 0.768] |           0.0134 |
| Híbrido + re-ranker          | hybrid_rerank | Sistema completo: híbrido + cross-encoder mMARCO top-50                     |          0.7557 |       100 |         0.7762 |      250 | [0.749, 0.800] |           0.1431 |
| BM25 + re-ranker (sin denso) | bm25_rerank   | Ablación: sin componente denso                                              |          0.7557 |       100 |         0.7762 |      250 | [0.749, 0.800] |           0.1282 |

## escenarios

| modo                         |   escenarios |   correctos |   pct_correctos |   budget_exceeded |   stock_negative |   order_without_confirmation |   duplicate_orders |   stock_inconsistent |   violaciones_restricciones |
|:-----------------------------|-------------:|------------:|----------------:|------------------:|-----------------:|-----------------------------:|-------------------:|---------------------:|----------------------------:|
| texto (22+6)                 |           28 |          19 |            67.9 |                 0 |                0 |                            0 |                  0 |                    0 |                           0 |
| voz (audio)                  |            6 |           2 |            33.3 |                 0 |                0 |                            0 |                  0 |                    0 |                           0 |
| voz (transcripción correcta) |            6 |           2 |            33.3 |                 0 |                0 |                            0 |                  0 |                    0 |                           0 |

## asr_errores

|      valor |
|-----------:|
| 29         |
|  0.103704  |
|  0.0563187 |
|  1         |
|  6         |
|  0         |
|  3         |
|  5         |
| 20         |

## voz_vs_gold

| id                               | ok_voz   | fallidos_voz                                                                             | ok_gold   | fallidos_gold                                     | fallo_atribuible_asr   |
|:---------------------------------|:---------|:-----------------------------------------------------------------------------------------|:----------|:--------------------------------------------------|:-----------------------|
| v1_compra_confirmada             | True     |                                                                                          | True      |                                                   | False                  |
| v2_aclaracion                    | True     |                                                                                          | True      |                                                   | False                  |
| v3_cambio_cantidad               | False    | qty_of_item(cantidad=3)                                                                  | False     | qty_of_item(cantidad=1)                           | False                  |
| v4_agotado                       | False    | any_of(sin alternativa | no contiene); cart_qty_le_stock(excesos=[('B08P36BRPM', 1, 0)]) | False     | cart_qty_le_stock(excesos=[('B08QVJV76V', 1, 0)]) | False                  |
| v5_correccion_transcripcion      | False    | qty_of_item(cantidad=None)                                                               | False     | qty_of_item(cantidad=10)                          | False                  |
| v6_stock_cambia_y_reconfirmacion | False    | reply_contains_any(no contiene)                                                          | False     | reply_contains_any(no contiene)                   | False                  |

## calidad_voz

| voz                       |   wer_mean |   wer_std |   cer_mean |   cer_std |   sim_ref_mean |   sim_ref_std |
|:--------------------------|-----------:|----------:|-----------:|----------:|---------------:|--------------:|
| base_Claribel Dervla      |      0.068 |     0.133 |      0.023 |     0.045 |         -0.046 |         0.035 |
| clonada                   |      0.12  |     0.127 |      0.055 |     0.06  |          0.661 |         0.034 |
| referencia_mitad_vs_mitad |    nan     |   nan     |    nan     |   nan     |          0.814 |       nan     |

## latencia

| fase                   | estadistico   |   n |   asr_s |   retrieval_s |   llm_s |   tools_s |   tts_s |   agent_other_s |   total_s |   vram_gb_after |
|:-----------------------|:--------------|----:|--------:|--------------:|--------:|----------:|--------:|----------------:|----------:|----------------:|
| caliente               | media         |  11 |   0.625 |         0.095 |   5.192 |     0     |   5.231 |           0.001 |    11.145 |         nan     |
| caliente               | mediana       |  11 |   0.587 |         0     |   4.236 |     0     |   5.4   |           0.001 |     9.622 |         nan     |
| caliente               | p90           |  11 |   0.73  |         0.269 |   8.55  |     0     |   8.597 |           0.002 |    14.982 |         nan     |
| primer_turno           | media         |   1 |   0.691 |         0.268 |   9.662 |     0.001 |   3.856 |           0.002 |    14.48  |         nan     |
| primer_turno           | mediana       |   1 |   0.691 |         0.268 |   9.662 |     0.001 |   3.856 |           0.002 |    14.48  |         nan     |
| primer_turno           | p90           |   1 |   0.691 |         0.268 |   9.662 |     0.001 |   3.856 |           0.002 |    14.48  |         nan     |
| todos                  | media         |  12 |   0.63  |         0.11  |   5.565 |     0     |   5.116 |           0.001 |    11.423 |         nan     |
| todos                  | mediana       |  12 |   0.588 |         0     |   4.976 |     0     |   5.149 |           0.001 |     9.73  |         nan     |
| todos                  | p90           |  12 |   0.727 |         0.269 |   9.551 |     0.001 |   8.454 |           0.002 |    14.932 |         nan     |
| carga_llm              | valor         |   1 | nan     |       nan     | nan     |   nan     | nan     |         nan     |    31.568 |           5.748 |
| carga_asr              | valor         |   1 | nan     |       nan     | nan     |   nan     | nan     |         nan     |     2.705 |           5.748 |
| carga_tts              | valor         |   1 | nan     |       nan     | nan     |   nan     | nan     |         nan     |    33.676 |           7.498 |
| carga_tts_latentes_ref | valor         |   1 | nan     |       nan     | nan     |   nan     | nan     |         nan     |     3.271 |         nan     |

## aciertos_recuperacion

|   query_id | query                             |   ndcg |   n_candidates |   n_exact | top3_gains      |
|-----------:|:----------------------------------|-------:|---------------:|----------:|:----------------|
|       5378 | 8gb 512gb                         |      1 |             39 |        27 | [1.0, 1.0, 1.0] |
|       8035 | alisado sin formol                |      1 |             40 |        27 | [1.0, 1.0, 1.0] |
|      11004 | atornillador sin escobillas       |      1 |             40 |        31 | [1.0, 1.0, 1.0] |
|      15979 | biombo blanco barato              |      1 |             40 |        25 | [1.0, 1.0, 1.0] |
|      16154 | bisagras para puertas de armarios |      1 |             40 |        34 | [1.0, 1.0, 1.0] |

## fallos_recuperacion

|   query_id | query                              |     ndcg |   n_candidates |   n_exact | top3_gains        |
|-----------:|:-----------------------------------|---------:|---------------:|----------:|:------------------|
|      78694 | pen f olympus                      | 0.068514 |             40 |         3 | [0.1, 0.01, 0.01] |
|      82056 | pompom mini                        | 0.12857  |             16 |         3 | [0.0, 0.0, 0.0]   |
|      37290 | elevadores de coches de 2 columnas | 0.186096 |             40 |         3 | [0.0, 0.0, 0.0]   |
|       7789 | album guardería                    | 0.210688 |             39 |        15 | [0.1, 0.1, 0.0]   |
|      50658 | hilo soldar sin gas 1mm            | 0.220669 |             40 |         5 | [0.1, 0.0, 0.1]   |

## escenarios_fallidos

| modo   | id                               | titulo                                                                            | fallidos                                                                                 |
|:-------|:---------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|
| texto  | t06_atributo_desconocido         | Atributo no verificable debe tratarse como desconocido                            | reply_contains_any(no contiene)                                                          |
| texto  | t10_quitar_del_carrito           | Quitar un producto del carrito                                                    | op_called(remove_from_cart no llamada); cart_empty(items=1)                              |
| texto  | t12_agotado_alternativa          | Producto agotado al añadir, se propone alternativa                                | alternative_proposed(sin alternativa); cart_qty_le_stock(excesos=[('B08QVJV76V', 1, 0)]) |
| texto  | t21_cantidad_invalida            | Cantidad cero no se añade al carrito                                              | cart_empty(items=1)                                                                      |
| texto  | t22_ver_carrito                  | Ver el carrito con dos artículos                                                  | qty_of_item(cantidad=1)                                                                  |
| texto  | v3_cambio_cantidad               | Voz: cambio de cantidad conservando el presupuesto                                | qty_of_item(cantidad=1)                                                                  |
| texto  | v4_agotado                       | Voz: producto agotado, se busca alternativa o se informa que no hay solución      | cart_qty_le_stock(excesos=[('B08QVJV76V', 1, 0)])                                        |
| texto  | v5_correccion_transcripcion      | Voz: corrección de la transcripción antes de ejecutar la compra                   | qty_of_item(cantidad=10)                                                                 |
| texto  | v6_stock_cambia_y_reconfirmacion | Voz: cambio de stock entre cotización y confirmación, reconfirmación sin duplicar | reply_contains_any(no contiene)                                                          |
| voz    | v3_cambio_cantidad               | Voz: cambio de cantidad conservando el presupuesto                                | qty_of_item(cantidad=3)                                                                  |
| voz    | v4_agotado                       | Voz: producto agotado, se busca alternativa o se informa que no hay solución      | any_of(sin alternativa | no contiene); cart_qty_le_stock(excesos=[('B08P36BRPM', 1, 0)]) |
| voz    | v5_correccion_transcripcion      | Voz: corrección de la transcripción antes de ejecutar la compra                   | qty_of_item(cantidad=None)                                                               |
| voz    | v6_stock_cambia_y_reconfirmacion | Voz: cambio de stock entre cotización y confirmación, reconfirmación sin duplicar | reply_contains_any(no contiene)                                                          |
