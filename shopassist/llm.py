"""Modelos de lenguaje locales (Hugging Face transformers) y un LLM simulado para pruebas.

``torch``/``transformers`` se importan de forma perezosa para que el paquete funcione en
máquinas sin GPU ni esas bibliotecas (pruebas con ``MockLLM`` o la política por reglas).
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional, Union


class LocalLLM:
    """LLM local con ``transformers`` (fp16, ``device_map='auto'``; opcional 4 bits NF4).

    Memoria (T4 15 GB): Qwen2.5-3B fp16 ≈ 6.5 GB, 4 bits ≈ 2.5 GB; Qwen2.5-7B 4 bits ≈ 5.5 GB.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", load_in_4bit: Optional[bool] = None,
                 device_map: str = "auto", max_context_tokens: int = 6144):
        """``model_name`` puede ser un nombre de HF o un ``Config`` (usa ``llm_name``/``llm_4bit``)."""
        if not isinstance(model_name, str):  # se pasó un Config
            cfg = model_name
            model_name = cfg.llm_name
            load_in_4bit = cfg.llm_4bit if load_in_4bit is None else load_in_4bit
        load_in_4bit = bool(load_in_4bit)
        import torch  # noqa: F401  (import perezoso)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"device_map": device_map, "torch_dtype": torch.float16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.load_time_s = time.perf_counter() - t0
        self.last_stats: dict = {}

    def generate(self, messages: list[dict], max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if ids["input_ids"].shape[1] > self.max_context_tokens:  # recorta historial antiguo
            ids = {k: v[:, -self.max_context_tokens:] for k, v in ids.items()}
        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**ids, **gen_kwargs)
        new = out[0, ids["input_ids"].shape[1]:]
        self.last_stats = {"prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)),
                           "gen_s": time.perf_counter() - t0}
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def unload(self):
        """Libera VRAM (útil para gestionar memoria en Colab)."""
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MockLLM:
    """LLM simulado: devuelve respuestas guionizadas (lista de str/dict) o de una función.

    ``MockLLM(["{...}", {...}])`` o ``MockLLM(lambda messages: "...")``.
    """

    def __init__(self, script: Union[list, Callable, None] = None):
        self.script = script if script is not None else []
        self.calls: list = []
        self._i = 0

    def generate(self, messages, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        self.calls.append(messages)
        if callable(self.script):
            out = self.script(messages)
        elif self._i < len(self.script):
            out = self.script[self._i]
            self._i += 1
        else:
            out = {"action": "respond", "message": "¿En qué más puedo ayudarte?"}
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


# ------------------------------------------------------------------ parseo robusto
def _balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
        start = text.find("{", start + 1)
    return None


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)                    # comas finales
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)  # claves sin comillas
    return s


def parse_llm_json(text: str) -> Optional[dict]:
    """Extrae el primer objeto JSON de la salida del LLM (tolerante a ```json, texto extra,
    comas finales, comillas simples o literales de Python). Devuelve None si no hay JSON."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    i0 = text.find("{")
    cand = _balanced_json(text)
    if cand is not None and i0 >= 0 and not text.startswith(cand, i0):
        # el objeto exterior quedó truncado: se prefiere cerrarlo antes que tomar uno interior
        tail = text[i0:]
        closed = tail + "}" * max(tail.count("{") - tail.count("}"), 0)
        try:
            obj = json.loads(closed)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    if cand is None:
        # JSON truncado: intenta cerrar llaves
        i = text.find("{")
        if i < 0:
            return None
        cand = text[i:] + "}" * max(text[i:].count("{") - text[i:].count("}"), 0)
    for attempt in (cand, _repair(cand)):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
