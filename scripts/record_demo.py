"""Graba la demo en vídeo (MP4 con audio) de la app web local, conducida con Playwright.

La entrada de voz son las GRABACIONES REALES del usuario (``results/scenarios/voice/*/tN_input.wav``):
se inyectan como micrófono sobrescribiendo ``navigator.mediaDevices.getUserMedia`` para que devuelva un
``MediaStream`` de un ``AudioContext`` que reproduce el clip elegido. El script pulsa y mantiene el botón
de micrófono REAL de la página durante el clip, así que el camino es el de verdad:
MediaRecorder (webm/opus) -> ``POST /api/voice`` -> ffmpeg -> Whisper -> agente -> XTTS (1.ª frase) -> ``<audio>``;
las frases siguientes se sintetizan mientras suena la primera y la página las pide con ``/api/tts_chunk``.
Las esperas NO se recortan ni se aceleran: el vídeo muestra el contador de espera real de la página.

El vídeo de Playwright no lleva audio, así que se registran marcas de tiempo (``Date.now()`` del
navegador) de cada clip de entrada y de CADA trozo de respuesta que empieza a sonar (evento ``playing``
del ``<audio>``, con su URL), se descargan todos los wav de los trozos y luego ffmpeg mezcla todo
(adelay + amix) y lo multiplexa con el vídeo (h264 + aac); los silencios entre frases quedan donde
ocurrieron de verdad.
La sincronización vídeo/página se calibra con un destello magenta de 0.5 s al inicio, que se recorta.

Uso (desde la raíz del repo, con el servidor ya arrancado: ``.venv/bin/python -m webapp``)::

    pip install playwright && python -m playwright install chromium     # una vez
    .venv/bin/python scripts/record_demo.py                              # graba + mezcla
    .venv/bin/python scripts/record_demo.py --post-only video/demo_web_local_log.json   # solo re-mezclar

Salidas: ``video/demo_web_local.mp4``, ``video/demo_web_local_log.json`` (marcas, turnos, tiempos)
y ``video/demo_work/`` (webm crudo, wav de respuesta, mezcla de audio).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICE = ROOT / "results/scenarios/voice"

# Guion: (segmento, título del rótulo, [clips]). Restaurar la tienda solo al principio.
SCRIPT = [
    ("1. Compra completa por voz",
     "Búsqueda con presupuesto → añadir 2 unidades → confirmar cantidad → cotización → confirmar pedido",
     ["v1_compra_confirmada/t1_input.wav", "v1_compra_confirmada/t2_input.wav",
      "v1_compra_confirmada/t3_input.wav", "v1_compra_confirmada/t4_input.wav",
      "v1_compra_confirmada/t5_input.wav"]),
    ("2. Caso que requiere aclaración",
     "Sesión reiniciada (sin carrito ni contexto): «Añade eso al carrito» → el agente pregunta → el usuario aclara",
     ["v2_aclaracion/t1_input.wav", "v2_aclaracion/t2_input.wav"]),
]

INIT_JS = r"""
(() => {
  const log = (ev, extra) => { try { window.__demoLog({ ev, t: Date.now(), ...(extra || {}) }); } catch (_) {} };
  let ctx = null, clip = null, clipUrl = null;
  const getCtx = () => { if (!ctx) ctx = new AudioContext({ sampleRate: 48000 }); return ctx; };
  window.__setMicClip = async (url) => {
    const c = getCtx(); await c.resume();
    const buf = await (await fetch(url)).arrayBuffer();
    clip = await c.decodeAudioData(buf); clipUrl = url;
    return clip.duration;
  };
  let dest = null;
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    const c = getCtx(); await c.resume();
    dest = c.createMediaStreamDestination();
    dest.channelCount = 1;
    return dest.stream;
  };
  const origStart = MediaRecorder.prototype.start;
  MediaRecorder.prototype.start = function (...a) {
    const r = origStart.apply(this, a);
    if (clip && dest) {
      const c = getCtx();
      const src = c.createBufferSource();
      src.buffer = clip; src.connect(dest);
      const lead = 0.25;
      src.onended = () => log("clip_end", { url: clipUrl });
      src.start(c.currentTime + lead);
      log("clip_start", { url: clipUrl, t: Date.now() + lead * 1000, dur: clip.duration });
    }
    log("rec_start");
    return r;
  };
  const origStop = MediaRecorder.prototype.stop;
  MediaRecorder.prototype.stop = function (...a) { log("rec_stop"); return origStop.apply(this, a); };
  document.addEventListener("playing", (e) => { if (e.target.id === "player") log("reply_play", { src: e.target.currentSrc }); }, true);
  document.addEventListener("ended", (e) => { if (e.target.id === "player") log("reply_end", { src: e.target.currentSrc }); }, true);
  document.addEventListener("tts-done", (e) => log("reply_done", { turn: e.detail && e.detail.turn }));
  document.addEventListener("pause", (e) => { if (e.target.id === "player" && !e.target.ended) log("reply_pause", { src: e.target.currentSrc, at: e.target.currentTime }); }, true);
})();
"""

CAPTION_JS = r"""
([title, sub]) => {
  let d = document.getElementById("demo-caption");
  if (!d) {
    d = document.createElement("div"); d.id = "demo-caption";
    d.style.cssText = "position:fixed;left:24px;bottom:18px;z-index:50;max-width:900px;background:rgba(29,35,48,.92);color:#fff;" +
      "padding:10px 16px;border-radius:12px;font:600 17px/1.35 -apple-system,sans-serif;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.2)";
    document.body.appendChild(d);
  }
  d.innerHTML = title ? `${title}<div style="font-weight:400;font-size:13.5px;opacity:.85;margin-top:2px">${sub || ""}</div>` : "";
  d.style.display = title ? "block" : "none";
}
"""


def wait_ready(base: str, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(urllib.request.urlopen(base + "/api/health", timeout=5).read())
            if h.get("ready"):
                return h
            print("  servidor:", h.get("stage"), flush=True)
        except Exception as e:  # noqa: BLE001
            print("  servidor no responde:", e, flush=True)
        time.sleep(3)
    raise SystemExit("El servidor no quedó listo a tiempo.")


def record(args) -> Path:
    from playwright.sync_api import sync_playwright

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    (work / "replies").mkdir(parents=True)
    health = wait_ready(args.base)
    events: list = []
    turns: list = []
    W, H = args.width, args.height

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed, args=[
            "--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream"])
        context = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1,
                                      record_video_dir=str(work), record_video_size={"width": W, "height": H},
                                      permissions=["microphone"], locale="es-ES")

        def serve_clip(route):
            name = route.request.url.split("/__democlips/", 1)[1].split("?")[0]
            path = (VOICE / name).resolve()
            if VOICE.resolve() not in path.parents or not path.exists():
                return route.fulfill(status=404)
            route.fulfill(path=str(path), content_type="audio/wav")

        context.route("**/__democlips/**", serve_clip)
        context.expose_function("__demoLog", lambda e: events.append(e))
        context.add_init_script(INIT_JS)
        page = context.new_page()
        page.on("console", lambda m: print("  [consola]", m.type, m.text, flush=True) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: print("  [pageerror]", e, flush=True))
        page.goto(args.base + "/", wait_until="domcontentloaded")
        page.wait_for_selector("#overlay.hidden", state="attached", timeout=120000)
        page.wait_for_selector("#mic:not([disabled])", timeout=60000)
        page.wait_for_timeout(1500)

        # destello de sincronización (se recorta del vídeo final)
        t_flash = page.evaluate("""() => new Promise((res) => {
            const d = document.createElement('div'); d.id = 'sync-flash';
            d.style.cssText = 'position:fixed;inset:0;background:#ff00ff;z-index:999';
            document.body.appendChild(d);
            requestAnimationFrame(() => requestAnimationFrame(() => {
              const t = Date.now(); setTimeout(() => { d.remove(); res(t); }, 500); }));
        })""")
        events.append({"ev": "flash", "t": t_flash})
        page.wait_for_timeout(600)
        events.append({"ev": "content_start", "t": page.evaluate("Date.now()")})

        def caption(title, sub=""):
            page.evaluate(CAPTION_JS, [title, sub])

        def side_scroll(where):
            js = {"top": "0", "bottom": "s.scrollHeight"}[where] if where in ("top", "bottom") else where
            page.evaluate(f"() => {{ const s = document.querySelector('.side'); s.scrollTo({{top: {js}, behavior: 'smooth'}}); }}")

        def reset(restore: bool):
            page.set_checked("#restore", restore)
            page.wait_for_timeout(700)
            with page.expect_response("**/api/reset"):
                page.click("#reset")
            page.wait_for_timeout(1200)

        # ---------------- apertura: estado inicial, catálogo, reinicio con tienda restaurada
        caption("ShopAssist · demo web local (todo en este Mac: Whisper → Qwen2.5-3B → XTTS-v2)",
                "Entrada: grabaciones reales de la voz del usuario, enviadas por el micrófono de la página. Esperas sin recortar.")
        page.wait_for_timeout(3000)
        caption("Reiniciar + «restaurar tienda (semilla)»", "Se parte de la tienda original: stock y pedidos de la semilla.")
        reset(True)
        page.evaluate("() => window.scrollTo({top: 500, behavior: 'smooth'})")
        page.wait_for_timeout(2500)
        page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
        page.wait_for_timeout(1500)

        for si, (title, sub, clips) in enumerate(SCRIPT):
            if si > 0:
                caption(title, "Reiniciar (sin restaurar): se borra el carrito y la memoria de la conversación.")
                side_scroll("top")
                reset(False)
                page.wait_for_timeout(1500)
            for ci, clip in enumerate(clips):
                side_scroll("top")
                caption(title, sub + f"  ·  turno {ci + 1}/{len(clips)}")
                url = f"{args.base}/__democlips/{clip}"
                dur = page.evaluate("(u) => window.__setMicClip(u)", url)
                page.wait_for_timeout(1200)
                box = page.locator("#mic").bounding_box()
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=8)
                page.wait_for_timeout(300)
                n_end = sum(1 for e in events if e["ev"] == "clip_end")
                page.mouse.down()
                t_wait = time.time()
                while sum(1 for e in events if e["ev"] == "clip_end") == n_end:
                    if time.time() - t_wait > dur + 10:
                        raise RuntimeError("el clip no terminó")
                    page.wait_for_timeout(50)
                page.wait_for_timeout(450)
                with page.expect_response("**/api/voice", timeout=300000) as resp_info:
                    page.mouse.up()
                    t_up = page.evaluate("Date.now()")
                resp = resp_info.value
                t_resp = page.evaluate("Date.now()")
                data = resp.json()
                if resp.status != 200:
                    print("  ERROR turno:", data, flush=True)
                    turns.append({"segment": si, "clip": clip, "error": data})
                    continue
                page.mouse.move(box["x"] - 200, box["y"] + 40, steps=5)
                n_done = sum(1 for e in events if e["ev"] == "reply_done")
                rec = {"segment": si, "clip": clip, "turn_id": data["turn_id"], "transcript": data["transcript"],
                       "reply": data["reply"], "ops": data["ops"], "timings": data["timings"],
                       "clarification": data["clarification"], "audio_url": data["audio_url"],
                       "t_release": t_up, "t_response": t_resp, "browser_wait_s": round((t_resp - t_up) / 1000, 2),
                       "cart": data["state"]["cart"], "orders": data["state"]["orders"], "stock": data["state"]["stock"],
                       "results": [r["product_id"] for r in data["state"].get("results", [])]}
                print(f"  [{data['turn_id']}] «{data['transcript']}» -> «{data['reply'][:120]}» "
                      f"ops={[o['name'] for o in data['ops']]} 1.ª voz (servidor)={data['timings'].get('ttfa_s')} s "
                      f"respuesta en {rec['browser_wait_s']} s", flush=True)
                if data.get("audio_url"):
                    # mostrar carrito/pedidos/stock mientras suenan las frases
                    page.wait_for_timeout(2500)
                    side_scroll("bottom")
                    t_w = time.time()
                    while sum(1 for e in events if e["ev"] == "reply_done") == n_done:
                        if time.time() - t_w > 300:
                            print("  (aviso) la respuesta no terminó de sonar", flush=True)
                            break
                        page.wait_for_timeout(100)
                    sid = page.evaluate("SID")
                    fin = context.request.get(f"{args.base}/api/turn_timings?session={sid}&turn={data['turn_id']}").json()
                    rec["final_timings"] = fin.get("timings")
                    rec["chunks"] = []
                    for c in fin.get("chunks", []):
                        if not c.get("url"):
                            continue
                        local = work / "replies" / f"s{si}_{data['turn_id']}_{c['i']:02d}.wav"
                        local.write_bytes(context.request.get(args.base + c["url"]).body())
                        rec["chunks"].append({**c, "wav": str(local)})
                    plays = [e for e in events if e["ev"] == "reply_play" and e["t"] >= t_up]
                    if plays:
                        rec["ttfa_browser_s"] = round((plays[0]["t"] - t_up) / 1000, 2)
                    ends = [e for e in events if e["ev"] == "reply_end" and e["t"] >= t_up]
                    if ends:
                        rec["reply_audio_end_s"] = round((ends[-1]["t"] - t_up) / 1000, 2)
                    print(f"     frases={len(rec['chunks'])} 1.ª voz (navegador)={rec.get('ttfa_browser_s')} s · "
                          f"total servidor={(rec['final_timings'] or {}).get('total_s')} s · "
                          f"fin del audio a los {rec.get('reply_audio_end_s')} s", flush=True)
                    rec["lat_row"] = page.evaluate("() => [...document.querySelectorAll('#lat-hist tbody tr')].pop()?.innerText")
                    page.wait_for_timeout(1800)
                else:
                    page.wait_for_timeout(4000)
                turns.append(rec)
                if si == 0 and ci == len(clips) - 1:
                    shot = Path(args.shots) / "03_pedido_confirmado.png"
                    page.screenshot(path=str(shot))
                if si == 0 and ci == 0:
                    page.screenshot(path=str(Path(args.shots) / "01_busqueda.png"))
                if si == 1 and ci == 0:
                    side_scroll("top"); page.wait_for_timeout(800)
                    page.screenshot(path=str(Path(args.shots) / "04_aclaracion.png"))
            if si == 0:
                page.wait_for_timeout(1000)

        # ---------------- cierre: carrito, pedido y stock
        caption("Resumen final", "Pedido registrado en la tienda (persiste tras Reiniciar), carrito de la 2.ª sesión y stock actualizado.")
        side_scroll("bottom")
        page.wait_for_timeout(5000)
        page.screenshot(path=str(Path(args.shots) / "05_final.png"))
        side_scroll("top")
        page.wait_for_timeout(3000)
        events.append({"ev": "content_end", "t": page.evaluate("Date.now()")})
        page.wait_for_timeout(500)
        video_path = Path(page.video.path())
        context.close()
        browser.close()

    log = {"base": args.base, "health_models": health.get("models"), "settings": health.get("settings"),
           "video_raw": str(video_path), "events": events, "turns": turns}
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print("log ->", log_path)
    return log_path


def find_flash(video: str, fps: int = 50) -> float:
    """Segundo del vídeo crudo en el que aparece el destello magenta (pixel central)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-t", "60", "-vf",
                          f"fps={fps},crop=8:8:(iw-8)/2:(ih-8)/2,scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    for i in range(0, len(out) - 2, 3):
        r, g, b = out[i], out[i + 1], out[i + 2]
        if r > 200 and g < 80 and b > 200:
            return (i // 3) / fps
    raise RuntimeError("no se encontró el destello de sincronización")


def post(log_path: Path, out: Path) -> None:
    log = json.loads(Path(log_path).read_text())
    ev = log["events"]
    video = log["video_raw"]
    t_flash = next(e["t"] for e in ev if e["ev"] == "flash")
    flash_v = find_flash(video)
    to_v = lambda t: (t - t_flash) / 1000 + flash_v  # noqa: E731
    start_v = to_v(next(e["t"] for e in ev if e["ev"] == "content_start"))
    end_v = to_v(next(e["t"] for e in ev if e["ev"] == "content_end"))
    print(f"destello en {flash_v:.3f} s del vídeo crudo; contenido {start_v:.2f}–{end_v:.2f} s")

    tracks = []  # (archivo, segundo en vídeo crudo, ganancia)
    for e in ev:
        if e["ev"] == "clip_start":
            name = e["url"].split("/__democlips/", 1)[1]
            tracks.append((str(VOICE / name), to_v(e["t"]), 0.8))   # margen para no saturar en AAC
    # clave = URL completa con ?t=...: tras Reiniciar el servidor reinicia la numeración (t01_reply.wav se repite)
    replies = {t["audio_url"]: t["reply_wav"] for t in log["turns"] if t.get("reply_wav")}   # logs antiguos
    for t in log["turns"]:
        for c in t.get("chunks", []):
            replies[c["url"]] = c["wav"]
    seen = set()
    for e in ev:
        if e["ev"] == "reply_play":
            key = "/" + e["src"].split("/", 3)[3]
            if key in replies and key not in seen:          # solo la primera reproducción de cada trozo
                seen.add(key)
                tracks.append((replies[key], to_v(e["t"]), 1.0))
    work = Path(video).parent
    mix = work / "mix.wav"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for f, _, _ in tracks:
        cmd += ["-i", f]
    flt = []
    for i, (_, tv, g) in enumerate(tracks):
        ms = max(0, int(round((tv - start_v) * 1000)))
        flt.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,volume={g},adelay={ms}:all=1[a{i}]")
    flt.append("".join(f"[a{i}]" for i in range(len(tracks))) + f"amix=inputs={len(tracks)}:normalize=0:duration=longest,"
               f"apad=whole_dur={end_v - start_v + 1:.2f}[m]")
    cmd += ["-filter_complex", ";".join(flt), "-map", "[m]", "-ac", "1", "-ar", "48000", str(mix)]
    subprocess.run(cmd, check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start_v:.3f}", "-to", f"{end_v:.3f}", "-i", video,
                    "-i", str(mix), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-b:a", "160k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
    log["sync"] = {"flash_video_s": flash_v, "trim_start_s": start_v, "trim_end_s": end_v,
                   "tracks": [{"file": f, "at_s": round(tv - start_v, 3)} for f, tv, _ in tracks]}
    Path(log_path).write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print("vídeo ->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default=str(ROOT / "video/demo_web_local.mp4"))
    ap.add_argument("--log", default=str(ROOT / "video/demo_web_local_log.json"))
    ap.add_argument("--work", default=str(ROOT / "video/demo_work"))
    ap.add_argument("--shots", default=str(ROOT / "video/demo_work_shots"), help="carpeta para capturas PNG")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--post-only", metavar="LOG", help="no grabar: re-mezclar a partir de un log existente")
    args = ap.parse_args()
    if args.post_only:
        post(Path(args.post_only), Path(args.out))
        return
    Path(args.shots).mkdir(parents=True, exist_ok=True)
    log_path = record(args)
    post(log_path, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
