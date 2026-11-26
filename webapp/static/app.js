// ShopAssist · demo web local (JS plano, sin build).
"use strict";

const $ = (id) => document.getElementById(id);
const PAGE = 24;
const STAGES = [
  ["asr_s", "ASR (Whisper)", "--c-asr"],
  ["retrieval_s", "Recuperación", "--c-ret"],
  ["llm_s", "LLM (Qwen2.5-3B)", "--c-llm"],
  ["tools_s", "Herramientas", "--c-tools"],
  ["tts_s", "TTS (todas las frases)", "--c-tts"],
  ["agent_other_s", "Otros (NLU, control)", "--c-other"],
  ["decode_s", "Decodificar audio", "--c-dec"],
];
const f2 = (v) => (v === null || v === undefined ? "—" : Number(v).toFixed(2));

// Una sesión por pestaña (sessionStorage): su propio carrito y memoria del agente.
function sessionId() {
  let sid = null;
  try { sid = sessionStorage.getItem("shopassist_sid"); } catch (_) { /* sin storage */ }
  if (!sid) {
    sid = "s" + Math.random().toString(36).slice(2, 10);
    try { sessionStorage.setItem("shopassist_sid", sid); } catch (_) { /* ignorar */ }
  }
  return sid;
}
const SID = sessionId();

const ui = {
  offset: 0, total: 0, q: "", currency: "EUR", busy: false, turns: [],
  state: { cart: { lines: [], total: 0, n_items: 0 }, orders: [], stock: [], results: [] },
};

// ------------------------------------------------------------------ utilidades
function money(v) {
  if (v === null || v === undefined || v === "") return "";
  return Number(v).toLocaleString("es-ES", { style: "currency", currency: ui.currency || "EUR" });
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function short(s, n = 70) { s = String(s || "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function stockBadge(n) {
  if (n <= 0) return `<span class="stock out">agotado</span>`;
  if (n <= 3) return `<span class="stock low">quedan ${n}</span>`;
  return `<span class="stock ok">stock ${n}</span>`;
}
function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 6000);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) { /* sin cuerpo */ }
  if (!r.ok) throw new Error((body && body.detail) || `HTTP ${r.status}`);
  return body;
}

// ------------------------------------------------------------------ salud / carga
async function pollHealth() {
  let h;
  try { h = await api("/api/health"); } catch (e) { $("overlay-stage").textContent = "servidor no disponible…"; setTimeout(pollHealth, 1500); return; }
  const models = h.models || {};
  $("overlay-stage").textContent = h.stage;
  $("overlay-models").innerHTML = Object.entries(models).map(([k, v]) =>
    `<li>${v.ok ? "✅" : "⚠️"} <b>${esc(k)}</b> ${esc(v.modelo || v.tipo || "")} ${v.device ? "· " + esc(v.device) : ""} · ${v.load_s ?? "?"} s ${v.error ? "<br><small>" + esc(v.error) + "</small>" : ""}</li>`).join("");
  if (!h.ready) { setTimeout(pollHealth, 1000); return; }
  $("overlay").classList.add("hidden");
  const bad = Object.entries(models).filter(([, v]) => !v.ok && !v.disabled).map(([k]) => k);
  const llm = models.llm && models.llm.ok ? models.llm.modelo : "reglas";
  const tts = models.tts && models.tts.ok ? (models.tts.engine === "fast" ? "Piper (rápido, voz no clonada)" : `XTTS-v2 voz clonada (${models.tts.device})`) : "sin voz";
  $("health").textContent = `${llm} · ${tts}${bad.length ? " · fallo: " + bad.join(", ") : ""}`;
  $("health").className = "badge " + (bad.length ? "badge-bad" : "badge-ok");
  $("health").title = JSON.stringify(models, null, 1);
  if (!h.ffmpeg) toast("Falta ffmpeg: instala con `brew install ffmpeg` (necesario para la voz).");
  for (const e of h.errors || []) toast(e);
  const mic = $("mic");
  mic.disabled = !(models.asr && models.asr.ok);
  $("mic-label").textContent = mic.disabled ? "ASR no disponible" : "Mantén pulsado para hablar";
  await Promise.all([loadProducts(), refreshState()]);
}

// ------------------------------------------------------------------ catálogo
function card(p, opts = {}) {
  const inCart = ui.state.cart.lines.some((l) => l.product_id === p.product_id);
  const hit = opts.hit || (ui.state.results || []).some((r) => r.product_id === p.product_id);
  return `<div class="prod ${hit ? "hit" : ""} ${opts.flash ? "flash" : ""}" data-pid="${esc(p.product_id)}" title="${esc(p.title)}">
    ${opts.rank ? `<span class="rank">${opts.rank}º</span>` : ""}
    ${inCart ? `<span class="incart">en carrito</span>` : ""}
    <div class="thumb">${p.image
      ? `<img src="${esc(p.image)}" alt="" loading="lazy" decoding="async" onerror="this.replaceWith(Object.assign(document.createElement('span'),{textContent:'${p.emoji || "🛍️"}'}))">`
      : `<span>${p.emoji || "🛍️"}</span>`}</div>
    <div class="title">${esc(p.title)}</div>
    <div class="meta">${esc(p.category)} · ${esc(p.brand && p.brand !== "desconocido" ? p.brand : "marca desconocida")}</div>
    <div class="row"><span class="price">${money(p.price)}</span>${stockBadge(p.stock)}</div>
  </div>`;
}

async function loadProducts() {
  const qs = new URLSearchParams({ q: ui.q, offset: ui.offset, limit: PAGE, in_stock: $("instock").checked });
  try {
    const d = await api("/api/products?" + qs);
    ui.total = d.total;
    $("grid").innerHTML = d.items.map((p) => card(p)).join("") || `<p class="muted">Sin productos para “${esc(ui.q)}”.</p>`;
    const pages = Math.max(1, Math.ceil(d.total / PAGE));
    $("page").textContent = `página ${Math.floor(ui.offset / PAGE) + 1} de ${pages}`;
    $("cat-count").textContent = `(${d.total.toLocaleString("es-ES")} productos)`;
    $("prev").disabled = ui.offset <= 0;
    $("next").disabled = ui.offset + PAGE >= d.total;
  } catch (e) { toast("Catálogo: " + e.message); }
}

// ------------------------------------------------------------------ estado de la sesión
async function refreshState() {
  try { renderState(await api("/api/state?session=" + SID), false); } catch (e) { /* antes del primer turno */ }
}

function renderState(st, flash) {
  if (!st) return;
  const prevResults = (ui.state.results || []).map((r) => r.product_id).join(",");
  ui.state = st;
  ui.currency = st.currency || ui.currency;
  // resultados del agente (resaltados, con su orden)
  const res = st.results || [];
  $("results-wrap").classList.toggle("hidden", res.length === 0);
  const changed = res.map((r) => r.product_id).join(",") !== prevResults;
  $("results").innerHTML = res.map((p, i) => card(p, { hit: true, rank: i + 1, flash: flash && changed })).join("");
  // carrito
  const c = st.cart;
  $("cart-count").textContent = c.n_items ? `(${c.n_items} uds.)` : "";
  $("cart").innerHTML = c.lines.length ? `<table class="tbl"><thead><tr><th>producto</th><th class="n">cant.</th><th class="n">precio</th><th class="n">subtotal</th><th class="n">stock</th></tr></thead><tbody>
    ${c.lines.map((l) => `<tr><td title="${esc(l.title)}">${esc(short(l.title, 46))}</td><td class="n">${l.qty}</td><td class="n">${money(l.unit_price)}</td><td class="n">${money(l.line_total)}</td><td class="n">${l.stock}</td></tr>`).join("")}
    <tr class="total"><td>TOTAL</td><td class="n">${c.n_items}</td><td></td><td class="n">${money(c.total)}</td><td></td></tr></tbody></table>`
    : `<p class="muted">Vacío.</p>`;
  // pedidos
  $("orders").innerHTML = st.orders.length ? st.orders.map((o) => `<div class="order">
      <div class="oh"><span>${esc(o.order_id)}</span><span class="pill">${esc(o.status)}</span><span>${money(o.total)}</span></div>
      ${o.lines.map((l) => `<div>${l.qty}× ${esc(short(l.title, 52))} · ${money(l.unit_price)}</div>`).join("")}
    </div>`).join("") : `<p class="muted">Ninguno todavía.</p>`;
  // stock relevante
  $("stock").innerHTML = st.stock.length ? `<table class="tbl"><thead><tr><th>producto</th><th class="n">precio</th><th class="n">stock</th></tr></thead><tbody>
    ${st.stock.map((s) => `<tr><td title="${esc(s.product_id)}">${esc(short(s.title, 50))}</td><td class="n">${money(s.price)}</td><td class="n">${s.stock}</td></tr>`).join("")}</tbody></table>`
    : `<p class="muted">—</p>`;
}

// ------------------------------------------------------------------ chat y latencia
function addMsg(role, html, cls = "") {
  const chat = $("chat");
  if (chat.querySelector(".muted")) chat.innerHTML = "";
  const d = document.createElement("div");
  d.className = `msg ${role} ${cls}`;
  d.innerHTML = html;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

function totalOf(t) { return (t.total_s || 0) + 0; }

function renderLatency(r, clientS, final) {
  const t = Object.assign({}, r.timings, final ? final.timings : {});
  const parts = STAGES.filter(([k]) => (t[k] || 0) > 0.0005);
  const sum = parts.reduce((a, [k]) => a + t[k], 0) || 1;
  const tools = r.ops.map((o) => `${o.name}${o.ok ? "" : " ✗"}`).join(" → ") || "ninguna";
  const n = r.tts ? r.tts.n : 0;
  const ttfaB = r._ttfaBrowser;
  $("latency").innerHTML = `
    <div class="lat-total">Turno <b>${esc(r.turn_id)}</b> (${esc(r.kind)})
      <div class="lat-kpis">
        <div class="kpi"><span>Tiempo hasta primera voz</span><b>${f2(r.timings.ttfa_s)} s</b><small>servidor: fin de grabación → 1.er audio listo${ttfaB ? ` · navegador: soltar → suena ${ttfaB.toFixed(2)} s` : ""}</small></div>
        <div class="kpi"><span>Total (respuesta entera sintetizada)</span><b>${final ? f2(t.total_s) : "…"} s</b><small>${n} frase(s)${final && t.audio_s ? ` · ${t.audio_s.toFixed(1)} s de audio` : ""}${t.cola_s > 0.05 ? ` · en cola ${t.cola_s.toFixed(1)} s` : ""}</small></div>
      </div>
      <small>Medido en el navegador (envío → respuesta con el 1.er audio): ${clientS.toFixed(2)} s${r.asr_info && r.asr_info.duration_s ? ` · audio de entrada ${r.asr_info.duration_s.toFixed(1)} s` : ""}</small></div>
    <div class="bar">${parts.map(([k, , c]) => `<span style="width:${(100 * t[k] / sum).toFixed(2)}%;background:var(${c})" title="${k} ${t[k].toFixed(2)} s"></span>`).join("")}</div>
    <div class="legend">${STAGES.map(([k, label, c]) => `<div><span><i style="background:var(${c})"></i>${label}${k === "tts_s" && !final && n > 1 ? " (en curso)" : ""}</span><b>${(t[k] || 0).toFixed(2)} s</b></div>`).join("")}</div>
    ${final && final.chunks ? `<div class="lat-tools">Frases: ${final.chunks.map((c) => `${c.synth_s ? c.synth_s.toFixed(1) : "?"} s → ${c.duration_s ? c.duration_s.toFixed(1) : "?"} s audio`).join(" · ")}</div>` : ""}
    <div class="lat-tools">Herramientas: ${esc(tools)}</div>`;
  const row = `<td>${esc(r.turn_id)}</td><td>${esc(r.kind)}</td><td class="n">${f2(t.asr_s)}</td><td class="n">${f2(t.retrieval_s)}</td><td class="n">${f2(t.llm_s)}</td><td class="n">${f2(t.tts_s)}</td><td class="n"><b>${f2(r.timings.ttfa_s)}</b></td><td class="n">${final ? f2(t.total_s) : "…"}</td><td class="n">${clientS.toFixed(2)}</td>`;
  const tb = $("lat-hist").querySelector("tbody");
  let tr = tb.querySelector(`tr[data-key="${r._key}"]`);
  if (!tr) { tr = document.createElement("tr"); tr.dataset.key = r._key; tb.appendChild(tr); ui.turns.push(r); }
  tr.innerHTML = row;
}

// ------------------------------------------------------------------ reproducción por frases
let playGen = 0;                       // se incrementa para cancelar la cola (nuevo turno, reinicio)
function stopPlayback() { playGen++; const p = $("player"); p.pause(); }
function playUrl(url, gen) {
  return new Promise((resolve) => {
    const p = $("player");
    if (gen !== playGen) return resolve(false);
    const done = (ok) => { p.removeEventListener("ended", onEnd); p.removeEventListener("error", onErr); resolve(ok); };
    const onEnd = () => done(true);
    const onErr = () => done(false);
    p.addEventListener("ended", onEnd);
    p.addEventListener("error", onErr);
    p.src = url;
    p.play().catch(() => { toast("El navegador bloqueó la reproducción automática: pulsa ▶ en el reproductor."); done(false); });
  });
}
async function playReply(r, gen) {
  const n = r.tts ? r.tts.n : 0;
  const waits = [];
  const waitChunk = (i) => (waits[i] = waits[i] || api(`/api/tts_chunk?session=${SID}&turn=${r.turn_id}&i=${i}`));
  for (let i = 0; i < n; i++) {
    if (gen !== playGen) return;
    let url = i === 0 ? r.audio_url : null;
    if (i + 1 < n) waitChunk(i + 1).catch(() => null);         // pide ya la siguiente frase
    if (!url) {
      $("status").textContent = `Sintetizando la frase ${i + 1} de ${n}…`;
      try { url = (await waitChunk(i)).url; } catch (e) { toast("TTS: " + e.message); break; }
    }
    if (gen !== playGen) return;
    $("status").textContent = n > 1 ? `Reproduciendo frase ${i + 1} de ${n}…` : "Reproduciendo la respuesta…";
    const ok = await playUrl(url, gen);
    if (!ok) break;
  }
  if (gen === playGen) document.dispatchEvent(new CustomEvent("tts-done", { detail: { turn: r.turn_id } }));
}

// ------------------------------------------------------------------ temporizador de espera (tiempo real)
let timerId = null, timerT0 = 0;
function startTimer() {
  timerT0 = performance.now();
  $("timer").classList.remove("hidden");
  const tick = () => { $("timer-val").textContent = ((performance.now() - timerT0) / 1000).toFixed(1); };
  tick();
  timerId = setInterval(tick, 100);
}
function stopTimer() {
  clearInterval(timerId); timerId = null;
  $("timer").classList.add("hidden");
  return (performance.now() - timerT0) / 1000;
}

function setBusy(b, label) {
  ui.busy = b;
  const mic = $("mic");
  mic.classList.toggle("busy", b);
  mic.disabled = b;
  $("mic-label").textContent = b ? (label || "Procesando…") : "Mantén pulsado para hablar";
  document.querySelector("#text-form button").disabled = b;
}

// ------------------------------------------------------------------ envío de un turno
async function sendTurn(kind, payload, pendingText) {
  setBusy(true, kind === "voz" ? "Transcribiendo y pensando…" : "Pensando…");
  stopPlayback();
  const pending = addMsg("user", `<span class="tag">${kind}</span><i>${esc(pendingText)}</i>`);
  $("status").textContent = "Procesando: ASR → recuperación → LLM → herramientas → TTS (1.ª frase). El contador es la espera real.";
  startTimer();
  try {
    const r = kind === "voz"
      ? await api("/api/voice", { method: "POST", body: payload })
      : await api("/api/text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const clientS = stopTimer();
    const tSent = timerT0;
    r._key = `${r.turn_id}-${Date.now()}`;
    pending.innerHTML = `<span class="tag">${r.kind}</span>${esc(r.transcript)}`;
    const ops = r.ops.map((o) => o.name + (o.ok ? "" : " ✗")).join(" → ");
    const ttfa = r.timings.ttfa_s;
    addMsg("bot", `${esc(r.reply)}<span class="ops">${r.clarification ? "❓ pide aclaración · " : ""}${esc(ops || "sin herramientas")}${ttfa ? ` · 1.ª voz ${ttfa.toFixed(1)} s` : ""}</span>`, r.clarification ? "clar" : "");
    renderLatency(r, clientS, null);
    renderState(r.state, true);
    loadProducts();
    setBusy(false);
    if (r.audio_url) {
      const gen = ++playGen;
      const p = $("player");
      const onPlay = () => { r._ttfaBrowser = (performance.now() - tSent) / 1000; renderLatency(r, clientS, r._final || null); };
      p.addEventListener("playing", onPlay, { once: true });
      playReply(r, gen).then(() => {
        if (gen === playGen) $("status").textContent = r.clarification ? "El agente pidió una aclaración: responde por voz o texto." : "Listo. Pulsa el micrófono para el siguiente turno.";
      });
      api(`/api/turn_timings?session=${SID}&turn=${r.turn_id}`).then((fin) => { r._final = fin; renderLatency(r, clientS, fin); }).catch(() => null);
    } else {
      $("status").textContent = r.clarification ? "El agente pidió una aclaración: responde por voz o texto." : "Listo. Pulsa el micrófono para el siguiente turno.";
    }
  } catch (e) {
    stopTimer();
    pending.remove();
    addMsg("bot", `⚠️ ${esc(e.message)}`, "clar");
    $("status").textContent = "Error en el turno (ver mensaje).";
    setBusy(false);
  }
}

// ------------------------------------------------------------------ micrófono (MediaRecorder)
let rec = null, chunks = [], stream = null, pressT = 0, recMime = "";
function pickMime() {
  const c = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg", "audio/mp4"];
  for (const m of c) if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return "";
}
async function startRec() {
  if (ui.busy || rec) return;
  if (!navigator.mediaDevices || !window.MediaRecorder) { toast("Este navegador no soporta MediaRecorder. Usa Chrome/Firefox/Safari recientes en http://localhost."); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) { toast("No hay acceso al micrófono: " + e.message); return; }
  recMime = pickMime();
  rec = new MediaRecorder(stream, recMime ? { mimeType: recMime } : undefined);
  chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = onRecStop;
  rec.start();
  stopPlayback();
  $("mic").classList.add("recording");
  $("mic-label").textContent = "Grabando… suelta (o pulsa) para enviar";
  $("status").textContent = "🔴 Grabando. Habla con naturalidad.";
}
function stopRec() {
  if (!rec || rec.state === "inactive") return;
  rec.stop();
  $("mic").classList.remove("recording");
}
function onRecStop() {
  const type = (rec && rec.mimeType) || recMime || "audio/webm";
  const blob = new Blob(chunks, { type });
  stream && stream.getTracks().forEach((t) => t.stop());
  rec = null; stream = null;
  if (blob.size < 1500) { $("status").textContent = "Grabación demasiado corta; inténtalo de nuevo."; setBusy(false); return; }
  const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
  const fd = new FormData();
  fd.append("session", SID);
  fd.append("audio", blob, `grabacion.${ext}`);
  sendTurn("voz", fd, "transcribiendo…");
}

// Mantener pulsado (> 400 ms) = push-to-talk; clic corto = alternar.
const mic = $("mic");
mic.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  if (mic.disabled) return;
  if (rec) { stopRec(); return; }
  pressT = performance.now();
  startRec();
});
mic.addEventListener("pointerup", () => { if (rec && performance.now() - pressT > 400) stopRec(); });
mic.addEventListener("pointerleave", () => { if (rec && performance.now() - pressT > 400) stopRec(); });
let spaceDown = false;
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || spaceDown || ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName) || mic.disabled) return;
  e.preventDefault(); spaceDown = true; pressT = performance.now(); startRec();
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space" || !spaceDown) return;
  e.preventDefault(); spaceDown = false; stopRec();
});

// ------------------------------------------------------------------ texto, reinicio, catálogo
$("text-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text || ui.busy) return;
  $("text").value = "";
  sendTurn("texto", { session: SID, text }, text);
});
$("reset").addEventListener("click", async () => {
  if (ui.busy) return;
  try {
    const r = await api("/api/reset", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: SID, restore_store: $("restore").checked }) });
    $("chat").innerHTML = `<p class="muted">Sesión reiniciada${r.restored ? ` y tienda restaurada (semilla ${r.seed})` : ""}.</p>`;
    $("latency").innerHTML = `<p class="muted">—</p>`;
    $("lat-hist").querySelector("tbody").innerHTML = "";
    stopPlayback();
    $("player").removeAttribute("src");
    ui.turns = [];
    renderState(r.state, false);
    loadProducts();
  } catch (e) { toast("Reinicio: " + e.message); }
});
let qTimer = null;
$("q").addEventListener("input", () => { clearTimeout(qTimer); qTimer = setTimeout(() => { ui.q = $("q").value.trim(); ui.offset = 0; loadProducts(); }, 250); });
$("instock").addEventListener("change", () => { ui.offset = 0; loadProducts(); });
$("prev").addEventListener("click", () => { ui.offset = Math.max(0, ui.offset - PAGE); loadProducts(); });
$("next").addEventListener("click", () => { ui.offset += PAGE; loadProducts(); });

pollHealth();
