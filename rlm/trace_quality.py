"""Lightweight, conservative diagnostic for teacher traces in the wrong language.

Not a full language classifier: it flags only clear, lengthy English reasoning
while allowing short mathematical or mixed-language completions to pass.
"""
from __future__ import annotations

import re

ENGLISH = frozenset(
    "the and is are for with this that given first then therefore need should "
    "will from which because their has have in as amount income calculate "
    "monthly case user let me check would they each answer what we can if of "
    "since by considering based".split()
)
SPANISH = frozenset(
    "el la las los del un una por para con según ingreso ingresos importe "
    "mensual cálculo calcula resultado renta garantizada familia año años "
    "menores derecho cumple corresponde es son tiene total se debe cada "
    "que del sus y de al si sumamos aplicamos".split()
)


def is_predominantly_english_reasoning(trace: str) -> bool:
    """Flag obvious English text; never infer legal or numeric correctness."""
    if "<think>" not in trace or "</think>" not in trace:
        return False
    thought = trace.split("<think>", 1)[1].split("</think>", 1)[0]
    words = re.findall(r"[a-záéíóúüñ]+", thought.lower())
    english = sum(word in ENGLISH for word in words)
    spanish = sum(word in SPANISH for word in words)
    return english >= 12 and english > 2 * spanish
