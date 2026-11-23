"""Escenarios operativos reproducibles y su evaluación.

Cada escenario parte de ``store.restore(SEED)`` y aplica *hooks* opcionales. Se
ejecuta en uno de tres modos:

* ``text``: los turnos se envían como texto (``source="text"``; los escenarios de
  voz usan ``source="voice"`` con su texto correcto).
* ``voice``: el audio de cada turno se transcribe con el ASR (``source="voice"``).
* ``gold_transcript``: escenarios de voz con la transcripción correcta como entrada
  (``source="voice"``), para aislar los errores del ASR.

Además de las expectativas propias de cada escenario, siempre se miden las
violaciones de restricciones: presupuesto superado, stock negativo, pedido sin
confirmación del usuario y pedidos duplicados.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

DEFAULT_SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios" / "scenarios.yaml"

CONFIRM_WORDS = ("confirm", "si ", "si,", "sí", "adelante", "de acuerdo", "procede", "dale", "ok", "vale")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_scenarios(path: Optional[str] = None) -> List[dict]:
    import yaml

    with open(path or DEFAULT_SCENARIOS, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    scs = data["scenarios"] if isinstance(data, dict) else data
    ids = [s["id"] for s in scs]
    assert len(ids) == len(set(ids)), "IDs de escenario duplicados"
    return scs


# ---------------------------------------------------------------------------
# Adaptadores robustos sobre las salidas de Store / herramientas
# ---------------------------------------------------------------------------
_PID_KEYS = ("product_id", "pid", "id", "asin")
_QTY_KEYS = ("qty", "quantity", "cantidad")


def _to_records(x: Any) -> List[dict]:
    if x is None:
        return []
    if hasattr(x, "to_dict") and hasattr(x, "columns"):  # DataFrame
        return x.to_dict("records")
    if isinstance(x, dict):
        for k in ("items", "lines", "results", "products", "orders", "data"):
            if k in x and isinstance(x[k], (list, tuple)) or (k in x and hasattr(x[k], "columns")):
                return _to_records(x[k])
        return [x]
    if isinstance(x, (list, tuple)):
        return [r if isinstance(r, dict) else getattr(r, "__dict__", {"value": r}) for r in x]
    return [getattr(x, "__dict__", {})]


def _pid(rec: dict) -> Optional[str]:
    for k in _PID_KEYS:
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return None


def _qty(rec: dict) -> int:
    for k in _QTY_KEYS:
        if k in rec and rec[k] is not None:
            return int(rec[k])
    return 0


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def cart_items(store, cart_id: str) -> List[dict]:
    try:
        cart = store.get_cart(cart_id)
    except Exception:
        return []
    return [r for r in _to_records(cart) if _pid(r) is not None]


def cart_total(store, cart_id: str) -> float:
    try:
        cart = store.get_cart(cart_id)
    except Exception:
        return 0.0
    if isinstance(cart, dict):
        for k in ("total", "subtotal", "total_amount"):
            if _num(cart.get(k)) is not None:
                return float(cart[k])
    tot = 0.0
    for r in cart_items(store, cart_id):
        line = _num(r.get("line_total")) or _num(r.get("subtotal"))
        if line is None:
            price = _num(r.get("unit_price")) or _num(r.get("price")) or 0.0
            line = price * _qty(r)
        tot += line
    return tot


def list_orders(store) -> List[dict]:
    try:
        return _to_records(store.list_orders())
    except Exception:
        return []


def _order_id(o: dict) -> str:
    return str(o.get("order_id", o.get("id", json.dumps(o, sort_keys=True, default=str))))


def _order_items(o: dict) -> List[dict]:
    for k in ("items", "lines", "products"):
        if k in o:
            v = o[k]
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    v = []
            return _to_records(v)
    return []


def _order_total(o: dict) -> Optional[float]:
    for k in ("total", "total_amount", "amount"):
        if _num(o.get(k)) is not None:
            return float(o[k])
    return None


def get_stock(store, pid: str) -> Optional[int]:
    try:
        v = store.get_stock(pid)
        return None if v is None else int(v)
    except Exception:
        return None


def all_stock(store) -> Dict[str, int]:
    """Inventario completo {pid: stock}. Usa ``store.inventory``; si no está
    disponible para todos los ids, usa ``snapshot``."""
    pids = None
    if hasattr(store, "product_ids"):
        try:
            pids = [str(p) for p in store.product_ids()]
        except Exception:
            pids = None
    cat = None if pids is not None else getattr(store, "catalog_df", None)
    if cat is None:
        cat = getattr(store, "catalog", None)
    if pids is None and cat is not None and hasattr(cat, "columns"):
        col = next((c for c in _PID_KEYS if c in cat.columns), None)
        if col:
            pids = [str(x) for x in cat[col].tolist()]
    if pids is not None:
        try:
            inv = store.inventory(pids)
            if isinstance(inv, dict):
                return {str(k): int(v if not isinstance(v, dict) else v.get("stock", 0)) for k, v in inv.items()}
            return {(_pid(r) or ""): int(r.get("stock", 0)) for r in _to_records(inv)}
        except Exception:
            pass
    try:
        snap = store.snapshot()
        inv = snap.get("inventory", snap.get("stock", snap.get("products")))
        if isinstance(inv, dict):
            return {str(k): int(v if not isinstance(v, dict) else v.get("stock", 0)) for k, v in inv.items()}
        return {(_pid(r) or ""): int(r.get("stock", 0)) for r in _to_records(inv)}
    except Exception:
        return {}


def _result_pids(result: Any) -> List[str]:
    """IDs de producto en el resultado de una herramienta (search_products, ...)."""
    data = result.get("data", result) if isinstance(result, dict) else result
    return [p for p in (_pid(r) for r in _to_records(data)) if p]


def _ops(turns: Iterable[dict]) -> List[dict]:
    return [op for t in turns for op in (t.get("ops") or [])]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o))))


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def _retriever_top(retriever, query: str, k: int) -> List[str]:
    for call in (
        lambda: retriever.search(query, k=k),
        lambda: retriever.search(query, k),
        lambda: retriever.retrieve(query, k=k),
        lambda: retriever(query, k),
    ):
        try:
            res = call()
        except (AttributeError, TypeError):
            continue
        if res is None:
            continue
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], (tuple, list)):
            return [str(r[0]) for r in res][:k]
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], str):
            return list(res)[:k]
        return _result_pids(res)[:k]
    return []


def apply_hook(hook: dict, store, agent, turn_log: List[dict]) -> dict:
    """Aplica un hook y devuelve un registro {hook, pid(s), stock}."""
    typ, stock = hook["type"], int(hook.get("stock", 0))
    pids: List[str] = []
    if typ == "set_stock_cart_item":
        items = cart_items(store, getattr(agent, "cart_id", "c1"))
        idx = int(hook.get("index", 0))
        if idx < len(items):
            pids = [_pid(items[idx])]
    elif typ == "set_stock_last_result":
        for op in reversed(_ops(turn_log)):
            if op.get("name") in ("search_products", "find_alternative"):
                res = _result_pids(op.get("result"))
                if res:
                    r = int(hook.get("rank", 0))
                    pids = [res[r]] if r < len(res) else []
                    break
    elif typ == "set_stock_query":
        retriever = getattr(agent, "retriever", None)
        if retriever is not None:
            pids = _retriever_top(retriever, hook["query"], int(hook.get("top_k", 10)))
    else:
        raise ValueError(f"Hook desconocido: {typ}")
    for p in pids:
        store.set_stock(p, stock)
    return {"hook": hook, "pids": pids, "stock": stock, "applied": bool(pids)}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _sel_turns(ctx: dict, chk: dict) -> List[dict]:
    t = chk.get("turn")
    turns = ctx["turns"]
    if t is None:
        return turns
    return [turns[t]] if -len(turns) <= t < len(turns) else []


def _check(chk: dict, ctx: dict) -> tuple:
    """Devuelve (ok: bool, detalle: str)."""
    typ = chk["type"]
    turns = _sel_turns(ctx, chk)
    ops = _ops(turns)
    new_orders = ctx["new_orders"]
    store, cart_id = ctx["store"], ctx["cart_id"]
    if typ == "order_count":
        return len(new_orders) == chk["n"], f"pedidos nuevos={len(new_orders)}"
    if typ == "no_order":
        return len(new_orders) == 0, f"pedidos nuevos={len(new_orders)}"
    if typ == "no_duplicate_orders":
        v = ctx["violations"]["duplicate_orders"]
        return v == 0, f"duplicados={v}"
    if typ == "cart_total_le_budget":
        b = ctx["budget"]
        if b is None:
            return True, "sin presupuesto"
        totals = [cart_total(store, cart_id)] + [t for t in (_order_total(o) for o in new_orders) if t is not None]
        return max(totals) <= b + 1e-6, f"max_total={max(totals):.2f} presupuesto={b}"
    if typ == "budget_preserved":
        b = ctx["budget"]
        return b is not None and abs(float(b) - float(chk["value"])) < 1e-6, f"presupuesto={b}"
    if typ == "clarification_asked":
        ok = any(t.get("needs_clarification") for t in turns) or any(
            o.get("name") == "ask_clarification" for o in ops
        )
        return ok, "aclaración" if ok else "sin aclaración"
    if typ == "qty_of_item":
        items = cart_items(store, cart_id)
        if not items and new_orders:
            items = _order_items(new_orders[-1])
        idx = int(chk.get("index", 0))
        q = _qty(items[idx]) if idx < len(items) else None
        return q == chk["n"], f"cantidad={q}"
    if typ == "cart_empty":
        n = len(cart_items(store, cart_id))
        return n == 0, f"items={n}"
    if typ == "cart_qty_le_stock":
        bad = []
        for it in cart_items(store, cart_id):
            s = get_stock(store, _pid(it))
            if s is not None and _qty(it) > s:
                bad.append((_pid(it), _qty(it), s))
        return not bad, f"excesos={bad}"
    if typ == "alternative_proposed":
        ok = any(o.get("name") == "find_alternative" for o in ops) or any(
            "alternativ" in _norm(t.get("reply", "")) for t in turns
        )
        return ok, "alternativa" if ok else "sin alternativa"
    if typ == "stock_consistent":
        v = ctx["violations"]
        ok = v["stock_inconsistent"] == 0 and v["stock_negative"] == 0
        return ok, f"inconsistencias={v['stock_inconsistent']} negativos={v['stock_negative']}"
    if typ == "reply_contains_any":
        words = [_norm(w) for w in chk["words"]]
        ok = any(w in _norm(t.get("reply", "")) for t in turns for w in words)
        return ok, "contiene" if ok else "no contiene"
    if typ == "op_called":
        ok = any(o.get("name") == chk["name"] for o in ops)
        return ok, f"{chk['name']} {'llamada' if ok else 'no llamada'}"
    if typ == "op_not_called":
        ok = not any(o.get("name") == chk["name"] and o.get("ok", True) for o in ops)
        return ok, f"{chk['name']} {'no llamada' if ok else 'llamada'}"
    if typ == "any_of":
        res = [_check(c, ctx) for c in chk["checks"]]
        return any(r[0] for r in res), " | ".join(r[1] for r in res)
    raise ValueError(f"Check desconocido: {typ}")


def _violations(turn_log: List[dict], store, budget_hist: List[Optional[float]]) -> dict:
    v = {"budget_exceeded": 0, "stock_negative": 0, "order_without_confirmation": 0,
         "duplicate_orders": 0, "stock_inconsistent": 0}
    seen_ids: Counter = Counter()
    for i, t in enumerate(turn_log):
        new = t.get("_new_orders", [])
        budget = budget_hist[i]
        for o in new:
            seen_ids[_order_id(o)] += 1
            tot = _order_total(o)
            if budget is not None and tot is not None and tot > budget + 1e-6:
                v["budget_exceeded"] += 1
            if not any(w in _norm(t.get("user_text", "")) + " " for w in CONFIRM_WORDS):
                v["order_without_confirmation"] += 1
        # >1 pedido en el mismo turno o pedido creado por una reconfirmación del mismo carrito
        if len(new) > 1:
            v["duplicate_orders"] += len(new) - 1
        # inventario: delta == unidades pedidas en este turno
        before, after = t.get("_stock_before", {}), t.get("_stock_after", {})
        ordered: Counter = Counter()
        for o in new:
            for it in _order_items(o):
                ordered[_pid(it)] += _qty(it)
        for pid in set(before) | set(after):
            b, a = before.get(pid), after.get(pid)
            if a is not None and a < 0:
                v["stock_negative"] += 1
            if a is not None and b is not None and (b - a) != ordered.get(pid, 0):
                v["stock_inconsistent"] += 1
    v["duplicate_orders"] += sum(c - 1 for c in seen_ids.values() if c > 1)
    # dos pedidos con exactamente los mismos artículos y cantidades => duplicado
    sigs = Counter()
    for t in turn_log:
        for o in t.get("_new_orders", []):
            sig = tuple(sorted((_pid(it), _qty(it)) for it in _order_items(o)))
            if sig:
                sigs[sig] += 1
    v["duplicate_orders"] += sum(c - 1 for c in sigs.values() if c > 1)
    return v


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def _state_dict(agent) -> dict:
    st = getattr(agent, "state", None)
    if st is None:
        return {}
    if hasattr(st, "to_dict"):
        try:
            return st.to_dict()
        except Exception:
            pass
    return dict(getattr(st, "__dict__", {}))


def _budget(agent) -> Optional[float]:
    st = getattr(agent, "state", None)
    return _num(getattr(st, "budget", None)) if st is not None else None


def run_scenario(
    sc: dict,
    agent_factory: Callable[[], Any],
    store,
    input_mode: str = "text",
    asr=None,
    tts=None,
    seed: Optional[int] = None,
    out_dir: Optional[str] = "results/scenarios",
    audio_root: str = ".",
) -> dict:
    """Ejecuta un escenario desde un estado restaurado y evalúa sus checks.

    ``agent_factory()`` debe devolver un agente nuevo ligado a ``store``.
    ``input_mode``: "text" | "voice" | "gold_transcript".
    Devuelve dict(id, passed, checks, violations, turns, final_store_snapshot, ...).
    """
    if seed is None:
        try:
            from .config import Config

            seed = Config().SEED
        except Exception:
            seed = 42
    if input_mode == "voice" and asr is None:
        raise ValueError("input_mode='voice' requiere asr")
    store.restore(seed)
    agent = agent_factory()
    if hasattr(agent, "reset"):
        agent.reset()
    cart_id = getattr(agent, "cart_id", "c1")
    source = "voice" if (sc.get("voice") or input_mode in ("voice", "gold_transcript")) else "text"
    run_dir = Path(out_dir) / input_mode / sc["id"] if out_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    turn_log: List[dict] = []
    hook_log = [apply_hook(h, store, agent, turn_log) for h in sc.get("setup", []) or []]
    orders_before = {_order_id(o) for o in list_orders(store)}
    initial_orders = set(orders_before)
    budget_hist: List[Optional[float]] = []

    for k, turn in enumerate(sc["turns"]):
        for h in turn.get("hooks_before", []) or []:
            hook_log.append({**apply_hook(h, store, agent, turn_log), "turn": k})
        gold = turn["user"]
        timings: Dict[str, float] = {}
        audio = turn.get("audio")
        asr_info = {}
        if input_mode == "voice":
            apath = str(Path(audio_root) / audio) if audio else None
            if not apath or not Path(apath).exists():
                raise FileNotFoundError(f"Falta el audio {apath} del escenario {sc['id']}")
            t0 = time.perf_counter()
            text, asr_info = asr.transcribe(apath)
            timings["asr_s"] = time.perf_counter() - t0
            if run_dir:
                shutil.copy(apath, run_dir / f"t{k + 1}_input.wav")
        else:
            text = gold
            timings["asr_s"] = 0.0
        stock_before = all_stock(store)
        t0 = time.perf_counter()
        res = agent.handle(text, source=source)
        agent_s = time.perf_counter() - t0
        timings.update({k2: float(v) for k2, v in (getattr(res, "timings", {}) or {}).items()})
        timings["agent_s"] = agent_s
        audio_out = None
        if tts is not None and getattr(res, "reply", ""):
            t0 = time.perf_counter()
            audio_out = tts.synthesize(res.reply, str(run_dir / f"t{k + 1}_reply.wav")) if run_dir else None
            timings["tts_s"] = time.perf_counter() - t0
        stock_after = all_stock(store)
        now = list_orders(store)
        new = [o for o in now if _order_id(o) not in orders_before]
        orders_before |= {_order_id(o) for o in now}
        budget_hist.append(_budget(agent))
        turn_log.append({
            "turn": k,
            "user_gold": gold,
            "user_text": text,
            "transcript": text if input_mode == "voice" else None,
            "asr_info": asr_info,
            "audio_in": audio,
            "audio_out": audio_out,
            "reply": getattr(res, "reply", ""),
            "ops": _jsonable(getattr(res, "ops", [])),
            "needs_clarification": bool(getattr(res, "needs_clarification", False)),
            "state": _jsonable(getattr(res, "state", None) or _state_dict(agent)),
            "timings": timings,
            "_new_orders": _jsonable(new),
            "_stock_before": stock_before,
            "_stock_after": stock_after,
        })

    final_orders = list_orders(store)
    new_orders = [o for o in final_orders if _order_id(o) not in initial_orders]
    violations = _violations(turn_log, store, budget_hist)
    ctx = {"turns": turn_log, "store": store, "cart_id": cart_id, "new_orders": new_orders,
           "budget": _budget(agent), "violations": violations}
    checks = []
    for chk in sc.get("expectations", []):
        try:
            ok, detail = _check(chk, ctx)
        except Exception as e:  # un check mal formado no detiene la batería
            ok, detail = False, f"error: {e!r}"
        checks.append({"check": chk, "ok": bool(ok), "detail": detail})
    hard_violations = violations["budget_exceeded"] + violations["stock_negative"] + \
        violations["order_without_confirmation"] + violations["duplicate_orders"]
    passed = all(c["ok"] for c in checks) and hard_violations == 0

    try:
        snapshot = _jsonable(store.snapshot())
    except Exception:
        snapshot = {}
    # quitar el inventario completo de los turnos (pesado) y dejar sólo los cambios
    for t in turn_log:
        b, a = t.pop("_stock_before"), t.pop("_stock_after")
        t["stock_changes"] = {p: [b.get(p), a.get(p)] for p in set(b) | set(a) if b.get(p) != a.get(p)}
        t["new_orders"] = t.pop("_new_orders")
    result = {
        "id": sc["id"], "title": sc.get("title", ""), "category": sc.get("category", ""),
        "voice": bool(sc.get("voice")), "input_mode": input_mode, "passed": passed,
        "checks": checks, "violations": violations, "hooks": hook_log, "turns": turn_log,
        "new_orders": _jsonable(new_orders), "final_store_snapshot": snapshot,
    }
    if run_dir:
        (run_dir / "result.json").write_text(json.dumps(_jsonable({k: v for k, v in result.items() if k != "final_store_snapshot"}), ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "ops.json").write_text(json.dumps([{"turn": t["turn"], "ops": t["ops"]} for t in turn_log], ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "final_store.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (run_dir / "transcripts.txt").write_text(
            "\n".join(f"[t{t['turn'] + 1}] gold: {t['user_gold']}\n     entrada: {t['user_text']}\n     agente: {t['reply']}" for t in turn_log),
            encoding="utf-8",
        )
    return result


def run_all(
    scenarios: List[dict],
    agent_factory: Callable[[], Any],
    store,
    input_mode: str = "text",
    asr=None,
    tts=None,
    only_voice: bool = False,
    out_dir: Optional[str] = "results/scenarios",
    audio_root: str = ".",
    seed: Optional[int] = None,
):
    """Ejecuta una batería y devuelve ``(df, results)``. En modos "voice" y
    "gold_transcript" sólo se ejecutan los escenarios de voz."""
    import pandas as pd

    results, rows = [], []
    for sc in scenarios:
        if (only_voice or input_mode in ("voice", "gold_transcript")) and not sc.get("voice"):
            continue
        try:
            r = run_scenario(sc, agent_factory, store, input_mode, asr, tts, seed, out_dir, audio_root)
        except FileNotFoundError as e:
            r = {"id": sc["id"], "title": sc.get("title", ""), "category": sc.get("category", ""),
                 "voice": bool(sc.get("voice")), "input_mode": input_mode, "passed": False,
                 "checks": [], "violations": {}, "turns": [], "error": str(e)}
        results.append(r)
        v = r.get("violations", {})
        rows.append({
            "id": r["id"], "titulo": r["title"], "categoria": r["category"], "voz": r["voice"],
            "modo": input_mode, "ok": r["passed"],
            "checks_ok": sum(c["ok"] for c in r["checks"]), "checks": len(r["checks"]),
            "fallidos": "; ".join(f"{c['check']['type']}({c['detail']})" for c in r["checks"] if not c["ok"]),
            "turnos": len(r["turns"]), **{f"v_{k}": n for k, n in v.items()},
            "error": r.get("error", ""),
        })
    df = pd.DataFrame(rows)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir) / f"summary_{input_mode}.csv", index=False)
    return df, results


def scenario_metrics(df) -> dict:
    """% de escenarios correctos y total de violaciones de restricciones."""
    if df is None or len(df) == 0:
        return {}
    vcols = [c for c in df.columns if c.startswith("v_")]
    out = {"escenarios": int(len(df)), "correctos": int(df["ok"].sum()),
           "pct_correctos": round(100.0 * df["ok"].mean(), 1)}
    for c in vcols:
        out[c[2:]] = int(df[c].fillna(0).sum())
    out["violaciones_restricciones"] = int(sum(out.get(k, 0) for k in
                                               ("budget_exceeded", "stock_negative", "order_without_confirmation", "duplicate_orders")))
    return out


def asr_errors_from_results(voice_results: List[dict]):
    """Tabla de errores del ASR (cantidades, importes, productos) a partir de
    los resultados en modo voz (gold vs transcripción)."""
    from .voice_eval import asr_error_table

    pairs = [(f"{r['id']}_t{t['turn'] + 1}", t["user_gold"], t["user_text"])
             for r in voice_results for t in r.get("turns", [])]
    return asr_error_table(pairs)


def compare_voice_vs_gold(df_voice, df_gold):
    """Une los resultados voz vs. transcripción correcta para atribuir fallos al ASR."""
    a = df_voice[["id", "ok", "fallidos"]].rename(columns={"ok": "ok_voz", "fallidos": "fallidos_voz"})
    b = df_gold[["id", "ok", "fallidos"]].rename(columns={"ok": "ok_gold", "fallidos": "fallidos_gold"})
    m = a.merge(b, on="id", how="outer")
    m["fallo_atribuible_asr"] = (~m["ok_voz"].fillna(False).astype(bool)) & m["ok_gold"].fillna(False).astype(bool)
    return m
