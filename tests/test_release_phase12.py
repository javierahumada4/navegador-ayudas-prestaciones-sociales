"""Consolidated phase-1/2 tool registry contract and optional BOE integration."""

from tool_use.registry import ToolRegistry
from tool_use.tools import (
    calcular_imv, consultar_ayudas, consultar_boe, default_registry, guardar_informe,
)


def test_default_tools_keep_phase2_benchmark_contract(monkeypatch):
    monkeypatch.delenv("ARCA_TOOLS_ENABLE_BOE", raising=False)
    assert set(default_registry().tools) == {
        "calcular_imv", "consultar_ayudas", "guardar_informe",
    }


def test_boe_is_available_for_deployment_only_when_enabled(monkeypatch):
    monkeypatch.setenv("ARCA_TOOLS_ENABLE_BOE", "1")
    names = set(default_registry().tools)
    assert names == {
        "calcular_imv", "consultar_ayudas", "guardar_informe", "consultar_boe",
    }
    assert consultar_boe.definition["function"]["name"] == "consultar_boe"
    assert ToolRegistry().register(consultar_boe).tools["consultar_boe"] is consultar_boe


def test_truthy_and_falsey_boe_values(monkeypatch):
    for value in ("true", "yes", "on", "1"):
        monkeypatch.setenv("ARCA_TOOLS_ENABLE_BOE", value)
        assert "consultar_boe" in default_registry().tools
    for value in ("0", "false", "", "no"):
        monkeypatch.setenv("ARCA_TOOLS_ENABLE_BOE", value)
        assert "consultar_boe" not in default_registry().tools
