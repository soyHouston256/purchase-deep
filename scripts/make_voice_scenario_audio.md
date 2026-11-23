# Cómo preparar los audios de los escenarios de voz

1. Abre `voice/scripts_lectura.md`: contiene el texto de referencia y la frase exacta de cada turno
   de los 6 escenarios de voz (v1–v6), con su nombre de archivo `voice/scenarios/v{n}_t{k}.wav`.
2. **Referencia de clonación**: graba `voice/referencia.wav` (15–25 s) leyendo `voice/referencia.txt`.
   En Colab (sección 7 del notebook):
   ```python
   from shopassist.voice import record_audio_colab
   record_audio_colab(22, "voice/referencia.wav")
   ```
3. **Turnos de los escenarios**: en la celda "Grabación de escenarios" del notebook se recorre la lista
   de turnos de voz y se graba cada uno (pulsa *Run* y habla cuando aparezca "● Grabando"):
   ```python
   from shopassist.scenarios import load_scenarios
   for sc in [s for s in load_scenarios() if s["voice"]]:
       for t in sc["turns"]:
           print(t["audio"], "→", t["user"])
           record_audio_colab(5, t["audio"])
   ```
   Alternativa: graba con el móvil (m4a/mp3), súbelos con `upload_audio_colab(ruta_destino)`; se
   convierten a wav mono 16 kHz con ffmpeg.
4. Verifica cada audio con `IPython.display.Audio(ruta)` y regraba si hay ruido o cortes.
5. Copia la carpeta `voice/` a Drive (`/content/drive/MyDrive/purchase/voice`) para no perderla al
   reiniciar el entorno; el notebook la restaura en la sección 0.
6. Si cambias frases en `scenarios/scenarios.yaml`, regenera el guion:
   `python scripts/gen_scripts_lectura.py`.

Las grabaciones deben ser de tu voz o de una persona que haya dado su autorización explícita.
