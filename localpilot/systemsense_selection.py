"""Semantic raw sensor selection and its adapter to the generic answer contract."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from localpilot.answer_contract import AnswerContract, AnswerField, normalized


MAX_REQUESTED_METRICS = 32


SENSOR_ALIASES = {
    "CPU Tctl/Tdie": ("CPU Tctl/Tdie", "Tctl/Tdie", "Core (Tctl/Tdie)"),
    "CPU CCD1": ("CPU CCD1", "CCD1", "CCD1 (Tdie)"),
    "GPU core": ("GPU core",),
    "GPU memory": ("GPU memory",),
    "GPU hot spot": ("GPU hot spot", "GPU hotspot", "hot spot", "hotspot"),
}


def sensor_matches(row: dict[str, Any], requested: str) -> bool:
    label = normalized(requested)
    name = normalized(row.get("Name", ""))
    hardware = normalized(row.get("HardwareType", ""))
    aliases = next((aliases for key, aliases in SENSOR_ALIASES.items()
                    if label in {normalized(key), *(normalized(a) for a in aliases)}), (requested,))
    component = "cpu" if label.startswith("cpu") or "tctl" in label or "ccd" in label else (
        "gpu" if "gpu" in label or "hot" in label else "")
    if component and not (hardware.startswith(component) or component in name):
        return False
    return any(normalized(alias) == name or normalized(alias) in name for alias in aliases)


def select_sensors(rows: list[dict[str, Any]], *, requested_sensors: list[str],
                   sensor_type: str, components: list[str], limit: int) -> tuple[list[dict], dict]:
    if len(requested_sensors) > MAX_REQUESTED_METRICS:
        raise ValueError("at most 32 requested sensor names are supported per bounded read")
    selected = [row for row in rows if
                (not sensor_type or normalized(row.get("SensorType", "")) == normalized(sensor_type))
                and (not components or any(normalized(row.get("HardwareType", "")).startswith(normalized(c))
                                          for c in components))
                and (not requested_sensors or any(sensor_matches(row, name) for name in requested_sensors))]
    returned = selected[:max(1, min(int(limit), 500))]
    return returned, {
        "requested": requested_sensors, "matched_count": len(selected),
        "returned_count": len(returned), "truncated": len(returned) < len(selected),
        "missing": [name for name in requested_sensors if not any(
            sensor_matches(row, name) and isinstance(row.get("Value"), (int, float)) for row in returned)],
    }


def sensor_payload(payload: dict) -> tuple[list[dict], dict, dict, dict]:
    truth = payload.get("hardware_truth") or {}
    raw = payload.get("raw_sensors") or {}
    rows = truth.get("sensors", raw.get("sensors", payload.get("items", [])))
    return (rows, truth.get("sensor_provider", payload.get("sensor_provider", raw)),
            truth.get("windows_performance", payload.get("windows_performance", payload.get("performance", {}))),
            payload.get("selection") or {})


@dataclass(frozen=True)
class SensorRequest:
    names: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    provider: bool = False
    windows_thermal: bool = False

    @classmethod
    def from_prompt(cls, prompt: str) -> SensorRequest:
        text = normalized(prompt)
        if not re.search(r"\btemperatures?\b|\btctl/tdie\b|\bccd1\b", text):
            return cls()
        components = tuple(c for c in ("cpu", "gpu") if re.search(rf"\b{c}\b", text))
        if not components:
            return cls()
        names = []
        for key, aliases in SENSOR_ALIASES.items():
            if any(normalized(alias) in text for alias in aliases):
                names.append(key)
        # A grouped GPU list commonly spells the component only once.
        gpu_clause = text[text.find("gpu"):] if "gpu" in components else ""
        for word, key in (("core", "GPU core"), ("memory", "GPU memory"), ("hot spot", "GPU hot spot")):
            if word in gpu_clause and key not in names:
                names.append(key)
        return cls(tuple(key for key in SENSOR_ALIASES if key in names), components,
                   "provider" in text,
                   bool(re.search(r"windows thermal|thermal zone|\bwmi\b", text)))

    @property
    def active(self) -> bool:
        return bool(self.components)

    def tool_arguments(self, name: str, args: dict) -> dict:
        if not self.active or name != "inspect_raw_system_sense":
            return args
        # The owner's requested field set takes precedence over a model's
        # prefix slice. Selection still precedes the unchanged 500-row ceiling.
        # Broad component questions use the existing bounded field capacity.
        try:
            limit = int(args.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        required_capacity = len(self.names) or MAX_REQUESTED_METRICS
        return {"category": "sensors", "limit": min(500, max(required_capacity, limit)),
                "requested_sensors": list(self.names), "sensor_type": "Temperature",
                "components": list(self.components)}

    def evaluate(self, result: str) -> tuple[list[str], AnswerContract]:
        try:
            payload = json.loads(str(result).split("\n\n", 1)[-1] if str(result).startswith("[Observation ID:") else str(result))
        except (ValueError, TypeError):
            return (["current requested temperature metrics"] if self.active else []), AnswerContract()
        if not self.active:
            return [], AnswerContract()
        if not isinstance(payload, dict):
            return ["current requested temperature metrics"], AnswerContract()
        rows, provider, windows, selection = sensor_payload(payload)
        # Compatibility with registered integrations that expose flat CPU/GPU metrics.
        if not rows:
            rows = [{"Name": f"{c.upper()} temperature", "HardwareType": c,
                     "SensorType": "Temperature", "Value": payload[f"{c}_temperature_c"]}
                    for c in self.components if isinstance(payload.get(f"{c}_temperature_c"), (int, float))]
        selected, detail = select_sensors(rows, requested_sensors=list(self.names),
                                         sensor_type="Temperature", components=list(self.components), limit=500)
        missing = list(detail["missing"])
        for component in self.components:
            if not any(normalized(row.get("HardwareType", "")).startswith(component)
                       and isinstance(row.get("Value"), (int, float)) for row in selected):
                missing.append(f"{component.upper()} temperature")
        if selection.get("truncated"):
            missing.append("complete requested sensor selection")
        fields = []
        if self.names:
            for name in self.names:
                row = next((r for r in selected if sensor_matches(r, name) and isinstance(r.get("Value"), (int, float))), None)
                if row:
                    fields.append(AnswerField(name, SENSOR_ALIASES[name], row["Value"]))
        else:
            for row in selected[:MAX_REQUESTED_METRICS]:
                if isinstance(row.get("Value"), (int, float)):
                    name = str(row.get("Name"))
                    aliases = next((aliases for key, aliases in SENSOR_ALIASES.items()
                                    if sensor_matches(row, key)), (name,))
                    if name in {"CPU temperature", "GPU temperature"}:
                        aliases = (name, name.split()[0])
                    fields.append(AnswerField(name, aliases, row["Value"]))
            if len(selected) > MAX_REQUESTED_METRICS:
                missing.append("bounded named selection needed for more than 32 requested metrics")
        if self.provider:
            if provider.get("source"):
                source = str(provider["source"])
                fields.append(AnswerField("hardware provider", ("provider", source), source))
            else:
                missing.append("hardware provider identity")
        if provider.get("available") is False:
            missing.append("available hardware provider")
        if self.windows_thermal:
            if not windows:
                missing.append("separate Windows thermal-zone status")
            else:
                errors = [str(error) for error in windows.get("errors", []) if "thermal" in str(error).casefold()]
                value = errors[0] if errors else "Windows thermal-zone status"
                alternatives = ("unavailable", "error", "failed") if errors else ("thermal", "wmi")
                fields.append(AnswerField("Windows thermal-zone status", ("Windows thermal", "thermal-zone", "thermal zone", "WMI"), value, alternatives))
        return list(dict.fromkeys(missing)), AnswerContract(tuple(fields))
