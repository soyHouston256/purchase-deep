"""Configuración global del asistente.

Justificación de modelos (Colab T4, ~15 GB VRAM, ~12 GB RAM):

- LLM ``Qwen/Qwen2.5-3B-Instruct``: buen seguimiento de instrucciones y salida JSON
  en español con un tamaño pequeño. En fp16 ocupa ~6.5 GB de VRAM; en 4 bits
  (bitsandbytes NF4) ~2.5 GB. La alternativa ``Qwen/Qwen2.5-7B-Instruct`` en 4 bits
  (~5.5 GB) mejora el razonamiento a costa de latencia.
- Embeddings ``intfloat/multilingual-e5-small`` (118M parámetros, ~0.5 GB): multilingüe,
  entrenado para búsqueda asimétrica con prefijos "query: " / "passage: ". Indexar
  ~5k productos tarda < 1 min en T4.
- Re-ranker ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` (~0.5 GB): cross-encoder
  multilingüe entrenado en mMARCO (incluye español); se aplica solo al top-50.
- Voz (otro módulo): Whisper (ASR) + XTTS-v2 (TTS con clonación) ≈ 4 GB.

Presupuesto total aproximado: 6.5 (LLM fp16) + 1 (recuperación) + 4 (voz) ≈ 11.5 GB < 15 GB.
Con ``llm_4bit=True`` el total baja a ~8 GB, dejando margen para el caché KV.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    """Parámetros reproducibles del sistema (semilla, rutas, modelos, tienda)."""

    SEED: int = 42
    data_dir: str = "data"
    # --- Datos ---
    catalog_size: int = 5000          # tamaño mínimo del catálogo
    # Consultas: el split "test" de ESCI es la evaluación (>=200) y "train" el desarrollo.
    # Cada consulta trae ~27 productos anotados, así que 250+100 consultas ya dan ~9.5k
    # productos; se añaden ``fill_extra`` productos aleatorios como distractores.
    n_queries: Optional[int] = None   # si se da, total de consultas (70% test / 30% train)
    n_test_queries: int = 250
    n_train_queries: int = 100
    min_queries: int = 200
    fill_extra: int = 1000
    locale: str = "es"
    esci_base_url: str = (
        "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/"
    )
    # --- Modelos ---
    llm_name: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_alt_name: str = "Qwen/Qwen2.5-7B-Instruct"   # usar con llm_4bit=True
    llm_4bit: bool = False
    embed_name: str = "intfloat/multilingual-e5-small"
    reranker_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    embed_batch_size: int = 64
    rerank_top: int = 50
    # --- Recuperación ---
    k_rrf: int = 60
    text_max_chars: int = 1200        # truncado del texto del producto para indexar
    # --- Tienda ---
    currency: str = "EUR"
    db_path: str = ":memory:"
    zero_stock_frac: float = 0.10
    max_qty: int = 50
    # --- Demo extra: tienda real (soyHouston256/store); NO forma parte de la evaluación ESCI ---
    remote_store_url: Optional[str] = None   # si es None se usa la variable de entorno STORE_API_URL
    remote_store_currency: str = "PEN"       # la API devuelve precios en soles
    # --- Agente ---
    max_steps: int = 6
    max_new_tokens: int = 384
    search_k: int = 5

    def to_dict(self) -> dict:
        return asdict(self)
