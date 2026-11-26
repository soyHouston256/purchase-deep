"""Demo interactiva por voz con Gradio (micrófono -> ASR -> agente -> TTS).

Reutiliza los objetos ya cargados en el notebook (``asr``, ``tts``, la tienda y una
fábrica de agentes) y el pipeline cronometrado :class:`shopassist.latency.VoicePipeline`:
los tiempos que muestra la interfaz son los que mide ``run_turn`` (desde que el audio
grabado está en disco hasta que el wav de respuesta está escrito), sin retardos añadidos.

Uso (en el notebook, con los modelos de A4 ya cargados)::

    from shopassist.demo_ui import build_demo
    demo = build_demo(lambda cart_id: ShoppingAgent(store, retriever, llm=llm, cart_id=cart_id, cfg=cfg),
                      asr, tts, store, seed=cfg.SEED)
    demo.queue(default_concurrency_limit=1).launch(share=True, debug=False)

Cada pestaña del navegador tiene su propia sesión (``gr.State``): su agente, su carrito
(``cart_id`` único) y su historial. La tienda es la misma para todas.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional

from .latency import VoicePipeline
from .session_view import session_view, short

LAT_COLS = ["turno", "entrada", "asr_s", "retrieval_s", "llm_s", "tools_s", "tts_s", "agent_other_s", "total_s"]
CART_COLS = ["producto", "cantidad", "precio_unit", "subtotal", "stock"]
ORDER_COLS = ["pedido", "estado", "total", "líneas"]
STOCK_COLS = ["product_id", "producto", "precio", "stock"]


def _df(rows, cols):
    import pandas as pd

    return pd.DataFrame(rows, columns=cols)


_short = short


class DemoSession:
    """Estado de una sesión del navegador: agente + pipeline cronometrado + historial."""

    def __init__(self, agent_factory: Callable[[str], object], asr, tts, out_dir: Path):
        self.sid = uuid.uuid4().hex[:6]
        self.cart_id = f"gradio_{self.sid}"
        self.agent = agent_factory(self.cart_id)
        self.dir = Path(out_dir) / self.sid
        self.pipe = VoicePipeline(asr, self.agent, tts, out_dir=str(self.dir))
        self.chat: list = []          # mensajes {"role", "content"}
        self.lat_rows: list = []      # una fila por turno
        self.n = 0

    def reset(self) -> None:
        """Nueva conversación: vacía el carrito de esta sesión y la memoria del agente."""
        self.agent.reset()
        self.pipe.history.clear()
        self.chat, self.lat_rows, self.n = [], [], 0


def _panels(sess: Optional[DemoSession], store):
    """Carrito, pedidos de la sesión y stock de los productos relevantes, leídos de la tienda."""
    if sess is None:
        return _df([], CART_COLS), _df([], ORDER_COLS), _df([], STOCK_COLS)
    v = session_view(store, sess.cart_id, sess.agent)
    cart = v["cart"]["lines"]
    cart_rows = [[_short(l["title"]), l["qty"], l["unit_price"], l["line_total"], l["stock"]] for l in cart]
    if cart:
        cart_rows.append(["TOTAL", v["cart"]["n_items"], "", v["cart"]["total"], ""])
    order_rows = [[o["order_id"], o["status"], o["total"],
                   "; ".join(f"{l['qty']}× {_short(l['title'], 40)}" for l in o["lines"])]
                  for o in v["orders"]]
    stock_rows = [[r["product_id"], _short(r["title"]), r["price"], r["stock"]] for r in v["stock"]]
    return _df(cart_rows, CART_COLS), _df(order_rows, ORDER_COLS), _df(stock_rows, STOCK_COLS)


def _latency_md(rec: dict, tid: str, ops: list) -> str:
    t = rec["timings"]
    tools = ", ".join(f"{o['name']}{'' if o.get('ok') else ' (falló)'}" for o in ops) or "ninguna"
    return (f"**Turno {tid} · espera real {t['total_s']:.2f} s** (fin de grabación → audio de respuesta listo)\n\n"
            f"ASR {t['asr_s']:.2f} s · recuperación {t['retrieval_s']:.2f} s · LLM {t['llm_s']:.2f} s · "
            f"herramientas {t['tools_s']:.2f} s · TTS {t['tts_s']:.2f} s · otros {t['agent_other_s']:.2f} s\n\n"
            f"Herramientas usadas: {tools}")


def _chatbot(gr, **kw):
    """``gr.Chatbot`` en formato de mensajes (Gradio 4.44/5 necesita ``type``; Gradio 6 ya no lo acepta)."""
    try:
        return gr.Chatbot(type="messages", **kw)
    except TypeError:
        return gr.Chatbot(**kw)


def build_demo(agent_factory: Callable[[str], object], asr, tts, store, seed: Optional[int] = None,
               out_dir: str = "results/gradio_demo", title: str = "ShopAssist · compra por voz"):
    """Construye la interfaz Gradio (no la lanza).

    - ``agent_factory(cart_id) -> ShoppingAgent``: se llama una vez por sesión del navegador.
    - ``asr`` / ``tts``: los objetos ya cargados (``tts`` puede ser ``None``: responde solo con texto).
    - ``store``: la tienda compartida; ``seed`` habilita la opción de restaurarla con ``store.restore(seed)``.
    """
    import gradio as gr

    out = Path(out_dir)

    def run(audio_path, text, sess, progress=gr.Progress()):
        text = (text or "").strip()
        if not audio_path and not text:
            gr.Warning("Graba un audio (o escribe un mensaje) antes de enviar.")
            cart, orders, stock = _panels(sess, store)
            return (gr.update(), (sess.chat if sess else []), gr.update(), gr.update(), cart, orders, stock,
                    sess, gr.update(), gr.update())
        if sess is None:
            sess = DemoSession(agent_factory, asr, tts, out)
        sess.n += 1
        tid = f"t{sess.n:02d}"
        progress(0, desc="ASR → agente (recuperación + LLM + herramientas) → TTS …")
        try:
            if audio_path:
                src = Path(audio_path)
                sess.dir.mkdir(parents=True, exist_ok=True)
                saved = sess.dir / f"{tid}_in{src.suffix or '.wav'}"   # se conserva la grabación
                shutil.copyfile(src, saved)
                rec = sess.pipe.run_turn(str(saved), turn_id=tid)
                kind = "voz"
            else:
                rec = sess.pipe.run_turn("(texto)", turn_id=tid, transcript=text)
                kind = "texto"
        except Exception as e:  # que la interfaz no se rompa; el error se ve en pantalla
            sess.n -= 1
            raise gr.Error(f"Falló el turno: {type(e).__name__}: {e}")
        t = rec["timings"]
        sess.chat.append({"role": "user", "content": f"[{kind}] {rec['transcript']}"})
        sess.chat.append({"role": "assistant", "content": rec["reply"]})
        sess.lat_rows.append([tid, kind] + [round(float(t.get(k, 0.0)), 2) for k in LAT_COLS[2:]])
        cart, orders, stock = _panels(sess, store)
        return (rec["audio_out"], sess.chat, _latency_md(rec, tid, rec.get("ops", [])),
                _df(sess.lat_rows, LAT_COLS), cart, orders, stock, sess, None, "")

    def reset(sess, restore_store):
        if restore_store and seed is not None:
            store.restore(seed)
        if sess is not None:
            sess.reset()
        msg = "Sesión reiniciada" + (f" y tienda restaurada con la semilla {seed}." if restore_store and seed is not None
                                     else " (carrito vacío, memoria del agente borrada).")
        cart, orders, stock = _panels(sess, store)
        return (None, [], msg, _df([], LAT_COLS), cart, orders, stock, sess, None, "")

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n"
            "Habla en español: **Whisper** transcribe, el **agente** busca en el catálogo y opera la tienda, "
            "y **XTTS-v2** responde con la voz clonada. El turno se envía al **detener la grabación** "
            "(o con *Enviar*). El contador de Gradio sobre las salidas muestra la espera real; "
            "debajo queda el tiempo medido por etapa."
        )
        sess_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Tu voz (micrófono)")
                text_in = gr.Textbox(label="o escribe (respaldo, se omite el ASR)",
                                     placeholder="Busco una botella de agua, mi presupuesto es de cuarenta euros")
                with gr.Row():
                    send_btn = gr.Button("Enviar", variant="primary")
                    reset_btn = gr.Button("Reiniciar sesión")
                restore_chk = gr.Checkbox(value=False, label=f"Al reiniciar, restaurar la tienda (store.restore({seed}))",
                                          visible=seed is not None)
                audio_out = gr.Audio(label="Respuesta (voz clonada)", type="filepath", autoplay=True,
                                     interactive=False)
            with gr.Column(scale=2):
                chat = _chatbot(gr, label="Conversación (transcripción → respuesta)", height=380)
                lat_md = gr.Markdown("Aún no hay turnos.")
        lat_df = gr.Dataframe(value=_df([], LAT_COLS), label="Latencia por turno (s), medida con VoicePipeline",
                              interactive=False)
        with gr.Row():
            cart_df = gr.Dataframe(value=_df([], CART_COLS), label="Carrito actual", interactive=False)
            orders_df = gr.Dataframe(value=_df([], ORDER_COLS), label="Pedidos de esta sesión", interactive=False)
        stock_df = gr.Dataframe(value=_df([], STOCK_COLS), label="Stock de productos relevantes (tienda)",
                                interactive=False)

        outputs = [audio_out, chat, lat_md, lat_df, cart_df, orders_df, stock_df, sess_state, audio_in, text_in]
        audio_in.stop_recording(run, [audio_in, text_in, sess_state], outputs)
        send_btn.click(run, [audio_in, text_in, sess_state], outputs)
        text_in.submit(run, [audio_in, text_in, sess_state], outputs)
        reset_btn.click(reset, [sess_state, restore_chk], outputs, show_progress="hidden")
    return demo
