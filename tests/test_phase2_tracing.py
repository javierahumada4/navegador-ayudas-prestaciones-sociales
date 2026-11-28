"""Full-fidelity phase-2 tool traces are durable and isolated per evaluation case."""
import json
import os
import stat
from pathlib import Path

import pytest

from tool_use.evaluate import load_bench
from tool_use.executor import ToolLoop
from tool_use.stage_eval import (
    evaluate_stage_cases, registry_for_stage, seed_report_observations,
)
from tool_use.tools import default_registry
from tool_use.tracing import TraceRecorder, case_trace_path, create_artifact_run


def _tool_call(name, args):
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": args}, ensure_ascii=False
    ) + "</tool_call>"


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _example():
    return load_bench("tool_use/bench/imv.jsonl")[0]["expected_arguments"]["case"]


def _scripted(*completions, registry=None):
    loop = ToolLoop(registry=registry or default_registry())
    responses = iter(completions)
    loop.generate = lambda messages: next(responses)
    return loop


def test_each_turn_records_input_raw_output_parse_tool_observation_and_answer(tmp_path):
    case = _example()
    query = "Calcula el IMV/CAPI de este caso y explica el resultado."
    raw_call = _tool_call("calcular_imv", {"case": case})
    final = "Simulación orientativa IMV y CAPI: 733,60 €, 0,00 € y total 733,60 €."
    trace = tmp_path / "full" / "case.jsonl"
    loop = _scripted(raw_call, final)
    # A stub can provide the same decoded generation metadata as real Qwen.
    real_generate_stub = loop.generate

    def generate_with_metadata(messages):
        result = real_generate_stub(messages)
        loop._last_generation_metadata = {
            "rendered_chat_template": "SAMPLE CHAT TEMPLATE " + str(len(messages)),
            "decoded_with_special_tokens": "<begin>" + result + "<end>",
            "output_token_ids": [1, 2, 3],
        }
        return result

    loop.generate = generate_with_metadata
    reply = loop.run(query, trace_path=trace)
    events = _records(trace)

    assert reply.tool_calls[0].ok
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["event"] == "run_start"
    assert events[0]["query"] == query
    assert events[0]["available_tools"] == list(default_registry().tools)
    inputs = [e for e in events if e["event"] == "model_input"]
    outputs = [e for e in events if e["event"] == "model_output"]
    parses = [e for e in events if e["event"] == "parsed_output"]
    executed = [e for e in events if e["event"] == "tool_execution"]

    assert len(inputs) == len(outputs) == len(parses) == 2
    assert inputs[0]["messages"][1]["content"] == query
    assert any(m["role"] == "tool" for m in inputs[1]["messages"])
    assert outputs[0]["raw_text"] == raw_call
    assert outputs[0]["decoded_with_special_tokens"] == "<begin>" + raw_call + "<end>"
    assert outputs[0]["rendered_chat_template"] == "SAMPLE CHAT TEMPLATE 2"
    assert outputs[0]["output_token_ids"] == [1, 2, 3]
    assert outputs[1]["raw_text"] == final
    assert parses[0]["tool_calls"][0]["arguments"] == {"case": case}
    assert executed[0]["record"]["result"]["imv_monthly_eur"] == 733.6
    assert executed[0]["observation_message"]["role"] == "tool"
    assert events[-1]["event"] == "final_response"
    assert events[-1]["response"]["answer"] == reply.answer
    assert stat.S_IMODE(os.stat(trace).st_mode) == 0o600


def test_malformed_and_repaired_calls_are_both_preserved(tmp_path):
    case = _example()
    malformed = '<tool_call>{"name":"calcular_imv","arguments":{} garbage}</tool_call>'
    valid = _tool_call("calcular_imv", {"case": case})
    trace = tmp_path / "parse.jsonl"
    result = _scripted(
        malformed, valid, "Simulación IMV/CAPI orientativa, terminada."
    ).run("Calcula IMV", trace_path=trace)
    assert not result.tool_calls[0].ok and result.tool_calls[1].ok
    calls = [e for e in _records(trace) if e["event"] == "tool_execution"]
    assert len(calls) == 2
    assert "Malformed tool call" in calls[0]["record"]["error"]
    assert calls[1]["record"]["ok"]
    assert [e["event"] for e in _records(trace)].count("model_output") == 3


def test_persist_model_authored_report_and_raw_write_arguments(tmp_path, monkeypatch):
    report_root = tmp_path / "reports"
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(report_root))
    case = _example()
    calc = default_registry().call("calcular_imv", {"case": case})
    assert calc.ok
    values = calc.result
    text = (
        "Informe de simulación orientativa del IMV y CAPI. "
        f"IMV: {values['imv_monthly_eur']:.2f} euros. "
        f"CAPI: {values['capi_monthly_eur']:.2f} euros. "
        f"Total: {values['total_monthly_eur']:.2f} euros. "
        "Esto no es una resolución administrativa."
    )
    trace = tmp_path / "report-run.jsonl"
    result = _scripted(
        _tool_call("calcular_imv", {"case": case}),
        _tool_call("guardar_informe", {"contenido": text}),
    ).run("Calcula IMV y guarda un informe TXT.", trace_path=trace)

    assert [call.name for call in result.tool_calls] == ["calcular_imv", "guardar_informe"]
    assert all(call.ok for call in result.tool_calls)
    saved = next(report_root.glob("*.txt"))
    assert saved.read_text(encoding="utf-8") == text
    assert [e for e in _records(trace) if e["event"] == "model_output"][1]["raw_text"] == (
        _tool_call("guardar_informe", {"contenido": text})
    )
    assert "guardado correctamente" in _records(trace)[-1]["response"]["answer"]


def test_traces_capture_real_prior_role_observations(tmp_path):
    case = next(c for c in load_bench("tool_use/bench/report.jsonl")
                if c["api_fixture"] == "with_results")
    prior = seed_report_observations(case)
    captured = []
    loop = ToolLoop(registry=registry_for_stage("report"))

    def generate(messages):
        captured.extend(messages)
        return "No puedo redactar todavía este informe."

    loop.generate = generate
    trace = tmp_path / "history.jsonl"
    result = loop.run(case["query"], max_turns=1,
                      prior_observations=prior, trace_path=trace)
    events = _records(trace)
    assert not any(call.ok and call.name == "guardar_informe" for call in result.tool_calls)
    assert events[0]["prior_observations"][0]["name"] == "calcular_imv"
    assert events[0]["prior_observations"][1]["result"]["fixture"] is True
    model_input = next(e for e in events if e["event"] == "model_input")
    assert [message["role"] for message in model_input["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "tool",
    ]
    assert model_input["messages"] == captured


def test_generation_error_is_written_before_propagating(tmp_path):
    loop = ToolLoop()
    def broken(_messages):
        raise RuntimeError("simulated Qwen GPU failure")
    loop.generate = broken
    trace = tmp_path / "broken.jsonl"
    with pytest.raises(RuntimeError, match="GPU failure"):
        loop.run("Dime qué es el IMV", trace_path=trace)
    events = _records(trace)
    assert [e["event"] for e in events] == [
        "run_start", "model_input", "run_exception",
    ]
    assert events[-1]["error"] == "simulated Qwen GPU failure"


def test_run_artifacts_are_unique_and_never_overwrite_traces(tmp_path):
    first = create_artifact_run(tmp_path / "phase2_base.json")
    second = create_artifact_run(tmp_path / "phase2_base.json")
    assert first != second
    assert (first / "traces").is_dir()
    assert (first / "reports").is_dir()
    assert stat.S_IMODE(os.stat(first).st_mode) == 0o700
    trace = case_trace_path(first, "imv", 1, "P2 /001")
    with TraceRecorder(trace) as writer:
        writer.event("test", full_text="cifra 1.234,56 €")
    with pytest.raises(FileExistsError):
        with TraceRecorder(trace):
            pass
    assert _records(trace)[0]["full_text"] == "cifra 1.234,56 €"


def test_stage_evaluator_links_case_to_durable_trace(tmp_path):
    artifact_run = create_artifact_run(tmp_path / "stage.json")
    case = next(c for c in load_bench("tool_use/bench/imv.jsonl")
                if c["category"] == "missing_facts")
    loop = _scripted(
        "Necesito tu residencia y patrimonio para poder calcular."
    , registry=registry_for_stage("imv"))
    result = evaluate_stage_cases(
        [case], loop, "imv",
        report_root=artifact_run / "reports", artifact_run=artifact_run,
    )
    trace = Path(result["rows"][0]["trace_path"])
    assert trace.is_file()
    assert any(row["event"] == "model_output" for row in _records(trace))
    assert result["summary"]["n_cases"] == 1


def test_api_trace_directory_is_opt_in(tmp_path, monkeypatch):
    trace_root = tmp_path / "api_logs"
    monkeypatch.setenv("ARCA_TOOLS_TRACE_DIR", str(trace_root))
    _scripted("IMV significa Ingreso Mínimo Vital.").run("Define IMV")
    files = list(trace_root.glob("tools-*.jsonl"))
    assert len(files) == 1
    assert _records(files[0])[-1]["event"] == "final_response"
    assert stat.S_IMODE(os.stat(files[0]).st_mode) == 0o600
