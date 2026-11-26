"""Genera (offline) imágenes de producto para la demo web con SD-Turbo en Apple Silicon (MPS).

El catálogo ESCI no trae imágenes (solo título/descripción/viñetas/marca/color), así que se
generan localmente a partir del texto del producto. NUNCA se ejecuta durante un turno de voz:
es un paso previo; el servidor solo sirve lo que haya en la caché (``webapp/product_images/``,
ignorada por git) y para el resto muestra un icono coherente con la categoría.

Pasos por producto:
1. Descripción visual corta EN INGLÉS (el codificador de texto CLIP de SD entiende mal el español
   y los títulos de Amazon son larguísimos) con el mismo ``qwen2.5:3b`` de Ollama; se cachea en
   ``prompts.json``. Con ``--no-llm`` se usa el título recortado.
2. ``stabilityai/sd-turbo`` (fp16, ~2.4 GB, 1 paso, 512 px, guidance 0) en ``mps``: ~1.5-3 s/imagen
   en un M1 Pro. Semilla fija por producto (reproducible).
3. Se guarda ``<product_id>.webp`` a 256 px (~10-15 KB).

Qué productos (por defecto, unos cientos): los resultados del recuperador para las consultas de la
demo (las transcripciones de ``video/demo_web_local_log.json`` si existe, más una lista fija:
botellas de agua, auriculares Bluetooth, pilas AA…) y una muestra estratificada por categoría
(``--per-category``), que es lo que la grilla muestra primero.

Uso (desde la raíz del repo, con el servidor web PARADO para no competir por CPU/GPU)::

    pip install diffusers==0.35.1                  # una vez (torch ya está)
    .venv/bin/python scripts/generate_product_images.py               # selección por defecto
    .venv/bin/python scripts/generate_product_images.py --ids B08KH65GW5,B09DFTCQQM --force
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_QUERIES = [
    "botella de agua", "auriculares bluetooth", "audífonos bluetooth", "pilas AA", "cuaderno", "mochila",
    "ratón inalámbrico", "funda para móvil", "cargador USB",
]
STYLE = ("professional e-commerce product photo of {desc}, single isolated product centered on a plain "
         "white background, soft studio lighting, sharp focus, no people, no text")


def ollama_describe(title: str, color: str, model: str, host: str, book: bool = False) -> str:
    if book:  # libros: SD no sabe escribir títulos; se pide solo el tema para ilustrar la portada
        prompt = ("In at most 6 English words, what is the topic of this book? Answer with the topic only.\n"
                  f"Book title: {title}")
        return "a closed hardcover book with an illustrated cover about " + _ask(prompt, model, host, title)[:60]
    prompt = ("Describe in English, in at most 14 words, the physical object sold in this product listing, "
              "as it would appear in a product photo (object type, material, shape, color). No brand names. "
              "Answer with the description only.\n"
              f"Listing title: {title}\nColor: {color or 'unknown'}")
    return _ask(prompt, model, host, title)


def _ask(prompt: str, model: str, host: str, fallback: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0, "num_predict": 40, "seed": 0}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = json.loads(r.read())["response"].strip().strip('"').strip()
    return txt.splitlines()[0][:160].rstrip('.') if txt else fallback[:80]


def select_products(args, cfg, catalog_df, store) -> list:
    from shopassist.categories import classify_catalog
    from shopassist.retrieval import build_retriever

    ids: list = []

    def add(pid):
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)

    seen: set = set()
    if args.ids:
        for p in args.ids.split(","):
            add(p.strip())
        return ids
    queries = list(DEFAULT_QUERIES)
    log = Path(args.from_log)
    if log.exists():
        d = json.loads(log.read_text())
        for t in reversed(d.get("turns", [])):
            if t.get("transcript") and any(o.get("name") == "search_products" for o in t.get("ops", [])):
                queries.insert(0, t["transcript"])       # solo los turnos que buscaron
            for s in t.get("stock", []) + [l for o in t.get("orders", []) for l in o.get("lines", [])]:
                if s.get("product_id"):
                    add(s["product_id"])
    if args.per_query > 0:
        r = build_retriever(cfg, catalog_df, "hybrid_rerank", store=store)
        for q in queries:
            for it in r.search(q, k=args.per_query):
                add(it["product_id"])
    if args.per_category > 0:
        emb = None
        npy = sorted(Path(cfg.data_dir).glob(f"emb_{cfg.embed_name.replace('/', '_')}_full_*.npy"))
        if npy:
            import numpy as np
            e = np.load(npy[-1])
            emb = e if len(e) == len(catalog_df) else None
        cats = classify_catalog(catalog_df, emb, cache_dir=cfg.data_dir)
        by: dict = {}
        for pid, v in cats.items():
            by.setdefault(v["category"], []).append(pid)
        rng = random.Random(cfg.SEED)
        for cat in sorted(by):
            pool = sorted(by[cat])
            rng.shuffle(pool)
            for pid in pool[: args.per_category]:
                add(pid)
    return ids[: args.limit] if args.limit else ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "webapp/product_images"))
    ap.add_argument("--from-log", default=str(ROOT / "video/demo_web_local_log.json"))
    ap.add_argument("--per-query", type=int, default=10, help="top-k del recuperador por consulta de la demo")
    ap.add_argument("--per-category", type=int, default=2, help="muestra estratificada por categoría")
    ap.add_argument("--ids", default="", help="lista de product_id separada por comas (ignora la selección)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--size", type=int, default=256, help="lado de la imagen guardada (se genera a 512)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-llm", action="store_true", help="usar el título como prompt (sin Ollama)")
    ap.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    ap.add_argument("--force", action="store_true", help="regenerar aunque exista la imagen")
    args = ap.parse_args()

    from shopassist.config import Config
    from shopassist.data import load_data
    from shopassist.store import Store

    cfg = Config()
    cfg.data_dir = str(ROOT / cfg.data_dir)
    catalog_df, _, _ = load_data(cfg)
    store = Store(":memory:", catalog_df, seed=cfg.SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = select_products(args, cfg, catalog_df, store)
    todo = [p for p in ids if args.force or not (out / f"{p}.webp").exists()]
    print(f"{len(ids)} productos seleccionados, {len(todo)} por generar -> {out}", flush=True)
    if not todo:
        return

    prompts_path = out / "prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
    t0 = time.time()
    for i, pid in enumerate(todo):
        if pid in prompts and not args.force:
            continue
        p = store.catalog_text(pid)
        title, color = p.get("title", ""), p.get("color", "")
        if args.no_llm:
            desc = title[:90]
        else:
            try:
                desc = ollama_describe(title, color, args.ollama_model, args.ollama_host, book=pid[:1].isdigit())
            except Exception as e:  # Ollama caído: se usa el título
                print("  (aviso) Ollama:", e, flush=True)
                desc = title[:90]
        prompts[pid] = {"title": title, "desc": desc}
        if (i + 1) % 25 == 0:
            prompts_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=1))
            print(f"  descripciones {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    prompts_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=1))

    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                     variant="fp16").to(args.device)
    pipe.set_progress_bar_config(disable=True)
    t0 = time.time()
    for i, pid in enumerate(todo):
        seed = int.from_bytes(pid.encode()[:8].ljust(8, b"0"), "big") % (2**31)
        img = pipe(prompt=STYLE.format(desc=prompts[pid]["desc"]), num_inference_steps=args.steps,
                   guidance_scale=0.0, width=512, height=512,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        img.resize((args.size, args.size)).save(out / f"{pid}.webp", "WEBP", quality=80, method=6)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"  imágenes {i + 1}/{len(todo)} · {el / (i + 1):.2f} s/img · {el:.0f} s", flush=True)
    total = sum(f.stat().st_size for f in out.glob("*.webp"))
    print(f"listo: {len(list(out.glob('*.webp')))} imágenes, {total / 1e6:.1f} MB en {out}")


if __name__ == "__main__":
    main()
