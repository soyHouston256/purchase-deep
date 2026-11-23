"""shopassist: asistente de compras por voz (Laboratorio 4, Deep Learning UTEC).

Módulos principales:
- config: configuración global y justificación de modelos.
- data: preparación del catálogo, consultas y qrels a partir de Amazon ESCI.
- store: tienda simulada (SQLite) con precios, stock, carritos y pedidos.
- retrieval: BM25 propio, recuperación densa (e5), fusión RRF y re-ranking.
- evaluation: nDCG@10 sobre candidatos anotados, baseline y ablaciones.
- attributes: extracción de atributos verificables ("desconocido" si no consta).
- state / nlu / tools / llm / agent: agente conversacional con herramientas.
"""

__version__ = "0.1.0"
