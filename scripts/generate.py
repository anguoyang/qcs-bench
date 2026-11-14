#!/usr/bin/env python3
"""Generate the leaderboard tables in README.md from data/leaderboard.json.

The JSON file is the single source of truth; numbers are never hand-edited
in the README. Run: python3 scripts/generate.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "leaderboard.json").read_text())

DEVICES = DATA["devices"]


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def table_for(model):
    lines = [
        f"### {model['model_full']}",
        "",
        f"Input {model['input']} · COCO mAP {model['coco_map'] if model['coco_map'] is not None else '—'}",
        "",
        "| Device | Runtime | Precision | Latency (estimate / median) | NPU layers |",
        "|---|---|---|---|---|",
    ]
    for c in model["cells"]:
        dev = DEVICES[c["device"]]["name"]
        lat = f"{fmt(c['latency_ms_estimate'])} ms" + (
            f" / {fmt(c['latency_ms_median'])} ms"
            if c["latency_ms_median"] is not None
            else ""
        )
        npu = (
            f"{c['layers_npu']}/{c['layers_total']}"
            if c["layers_total"] is not None
            else "—"
        )
        note = f" — {c['notes']}" if c["notes"] else ""
        lines.append(
            f"| {dev} | {c['runtime']} | {c['precision']} | {lat} | {npu} |{note}"
        )
    lines.append("")
    return "\n".join(lines)


def genai_table_for(model):
    def g(v):
        return "UNKNOWN" if v is None else f"{v:.2f}"

    lines = [
        f"### {model['model_full']}",
        "",
        f"Checkpoint {model['checkpoint']}",
        "",
        "| Device | Runtime | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) |",
        "|---|---|---|---|---|",
    ]
    for c in model["cells"]:
        dev = DEVICES[c["device"]]["name"]
        job = f" ([job]({c['job']}))" if c.get("job") else ""
        note = f" — {c['notes']}" if c.get("notes") else ""
        lines.append(
            f"| {dev} | {c['runtime']} | {g(c['prefill_tok_s'])} | "
            f"{g(c['decode_tok_s'])} | {g(c['ttft_ms'])} |{job}{note}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    md = [
        "# qcs-bench",
        "",
        "Public models × Qualcomm silicon, measured the same way every time. "
        "Answers two questions: which model to pick, and how fast it runs on "
        "your QCS / Snapdragon chip.",
        "",
        "Sister repo of [rk-bench](https://github.com/anguoyang/rk-bench), "
        "same rules: one measurement protocol, one JSON source of truth, "
        "failures recorded as UNKNOWN, never filled in.",
        "",
        "## Measurement protocol (v1.0)",
        "",
        f"- Devices: {', '.join(d['name'] + ' (' + d['chipset'] + ')' for d in DEVICES.values())}, via Qualcomm AI Hub Workbench cloud devices.",
        f"- Runtimes: {', '.join(DATA['protocol']['runtimes'])}; precisions: {', '.join(DATA['protocol']['precisions'])}.",
        f"- {DATA['protocol']['timing']}",
        f"- {DATA['protocol']['calibration']}",
        f"- {DATA['protocol']['failures']}",
        f"- Layer split: {DATA['protocol']['layer_split']}.",
        "",
        "## Leaderboard",
        "",
    ]
    for m in DATA["entries"]:
        md.append(table_for(m))
    md.append("## Why the same model can be 27x faster")
    md.append("")
    md.append(
        "YOLOv11-N on the RB3 (QCS6490): 134.7 ms with the default TFLite "
        "float path, 5.03 ms with w8a8 + precompiled QNN — same model, same "
        "chip. Most published numbers are the slow path. This repo exists so "
        "you pick the fast one on purpose."
    )
    md.append("")
    (ROOT / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"README.md regenerated from data/leaderboard.json ({len(DATA['entries'])} models)")


if __name__ == "__main__":
    main()
