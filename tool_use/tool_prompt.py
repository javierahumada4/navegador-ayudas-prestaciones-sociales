"""One fixed, non-evaluation demonstration for the Qwen3 IMV tool-call protocol.

The reasoning GRPO checkpoint was trained for numerical answers, not tool
invocations. A real Qwen3 tool-call exchange can help bootstrap the *format*
without putting any train/test oracle answer in the current problem prompt.
Use exactly the same demonstration for both tool-equipped evaluation arms.
"""

from __future__ import annotations

import json
from copy import deepcopy

# Standalone invented illustrative case, not a Phase-1 benchmark question.
_DEMO_CASE = {
    "application_date": "2026-09-26",
    "applicant_id": "p1",
    "persons": [
        {
            "id": "p1", "date_of_birth": "1985-05-10",
            "legal_residence_in_spain": True, "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": "2010-01-01",
        },
        {
            "id": "p2", "date_of_birth": "1984-01-01",
            "legal_residence_in_spain": True, "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": "2010-01-01",
        },
        {
            "id": "p3", "date_of_birth": "2010-06-01",
            "legal_residence_in_spain": True, "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": "2010-01-01",
        },
    ],
    "domicile": {
        "co_resident_ids": ["p1", "p2", "p3"],
        "same_domicile_since": "2010-06-01",
    },
    "relationships": [
        {"person_a": "p1", "person_b": "p2", "type": "spouse"},
        {"person_a": "p1", "person_b": "p3", "type": "parent_child", "custody": "shared"},
        {"person_a": "p2", "person_b": "p3", "type": "parent_child", "custody": "shared"},
    ],
    "economic": {
        "countable_income_annual_eur": 9600,
        "net_worth_eur": 5000,
        "non_corporate_assets_eur": 5000,
    },
}
_DEMO_ANSWER = "431.26"


def tool_demonstration() -> list[dict]:
    """Use normal Qwen3 conversation messages with a *real* structured tool call.

    The return value is fresh so generation code cannot mutate shared prompts.
    Not all user queries are numerical: only attach it for the IMV benchmark.
    """
    case = deepcopy(_DEMO_CASE)
    return [
        {
            "role": "user",
            "content": (
                "EJEMPLO INDEPENDIENTE DE FORMATO. Calcula este caso sintético "
                "normalizado usando calcular_imv: "
                + json.dumps(case, ensure_ascii=False)
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "calcular_imv",
                    "arguments": {"case": case},
                },
            }],
            "content": "",
        },
        {
            "role": "tool", "name": "calcular_imv",
            "content": json.dumps({"total_monthly_eur": 431.26}),
        },
        {"role": "assistant", "content": _DEMO_ANSWER},
    ]


def demo_is_correct() -> bool:
    """Validate the example against the production deterministic calculator."""
    from tool_use.tools import default_registry

    record = default_registry().call("calcular_imv", {"case": deepcopy(_DEMO_CASE)})
    return bool(
        record.ok and
        record.result.get("total_monthly_eur") == float(_DEMO_ANSWER)
    )
