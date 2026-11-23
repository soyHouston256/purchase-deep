"""Genera voice/scripts_lectura.md a partir de scenarios/scenarios.yaml.

Uso: python scripts/gen_scripts_lectura.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

HEADER = """# Guion de lectura para las grabaciones de voz

> Archivo generado por `scripts/gen_scripts_lectura.py` desde `scenarios/scenarios.yaml`.
> No lo edites a mano: cambia el YAML y vuelve a generarlo.

## 1. Grabación de referencia para la clonación (`voice/referencia.wav`)

* Duración: **15–25 s** (XTTS-v2 admite 6–30 s). Mono, sin música ni eco, a ~20 cm del micrófono.
* Debe ser **tu propia voz** (o una voz con autorización explícita y por escrito).
* Lee con tono natural el siguiente texto (está en `voice/referencia.txt`). Es un texto neutro,
  **distinto de cualquier respuesta del asistente**; el código (`TTS._check_text`) impide
  sintetizar frases que coincidan con él.

> {ref}

## 2. Audios de los 6 escenarios de voz

Convención de nombres: `voice/scenarios/v{{n}}_t{{k}}.wav` (escenario *n*, turno *k* empezando en 1).
Lee **exactamente** la frase (es la transcripción correcta, *gold*, usada para aislar errores del ASR).
Graba cada turno por separado, 2–6 s, con una pausa breve al inicio y al final.

{tables}

## 3. Consejos

* Graba en el mismo ambiente que la demo del video. Evita leer números como dígitos: di "cuarenta", no "cuatro cero".
* Puedes grabar desde Colab (`record_audio_colab`) o con el móvil y subir los archivos (`upload_audio_colab`).
* Total: {n_audios} audios + 1 referencia.
"""


def main() -> None:
    scs = yaml.safe_load((ROOT / "scenarios" / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]
    ref = (ROOT / "voice" / "referencia.txt").read_text(encoding="utf-8").strip()
    out, n = [], 0
    for s in scs:
        if not s.get("voice"):
            continue
        out.append(f"### {s['id']} — {s['title']}\n")
        out.append("| Archivo | Frase a leer (exacta) |\n|---|---|")
        for k, t in enumerate(s["turns"], 1):
            out.append(f"| `{t.get('audio', '')}` | {t['user']} |")
            n += 1
        out.append("")
    (ROOT / "voice" / "scripts_lectura.md").write_text(
        HEADER.format(ref=ref, tables="\n".join(out), n_audios=n), encoding="utf-8"
    )
    print(f"voice/scripts_lectura.md: {n} audios")


if __name__ == "__main__":
    main()
