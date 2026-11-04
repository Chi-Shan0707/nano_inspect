"""Reproducible research workflows (PLAN.md Section 6).

``run_experiment_command`` loads the local model once and runs the studies
enabled in the config: Study A (retrieval across distance/distractors),
Study B (clean-to-corrupted activation patching) and Study C (contrastive
steering). Large per-example data go to ``details/``; ``experiment.json``
holds summaries, controls and provenance; the report is regenerated at the end.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
import torch
from ..runtime import load_model, read_config, write_json


def run_experiment_command(config, output):
    cfg = read_config(config)
    output = Path(output)
    (output / "details").mkdir(parents=True, exist_ok=True)
    runtime = load_model(cfg)
    torch.manual_seed(int(cfg.get("seed", 0)))
    result = {"schema_version": 1, "kind": "experiment", "config_path": str(config), "config": cfg,
              "provenance": runtime.provenance, "model_revision": runtime.provenance.get("download_revision"),
              "dtype": str(runtime.dtype), "device": str(runtime.device), "studies": {}, "timing_seconds": {}}
    details = {"_dir": output / "details"}
    studies = cfg.get("studies", ["retrieval", "patching", "steering"])
    try:
        if "retrieval" in studies:
            from .retrieval import study_a
            started = time.perf_counter()
            result["studies"]["retrieval"] = study_a(runtime, cfg, details)
            result["timing_seconds"]["retrieval"] = time.perf_counter() - started
        if "patching" in studies:
            from .retrieval import study_b
            started = time.perf_counter()
            result["studies"]["patching"] = study_b(runtime, cfg, details)
            result["timing_seconds"]["patching"] = time.perf_counter() - started
        if "steering" in studies:
            from .steering import study_c
            started = time.perf_counter()
            result["studies"]["steering"] = study_c(runtime, cfg, details)
            result["timing_seconds"]["steering"] = time.perf_counter() - started
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    for name, value in details.items():
        if name != "_dir":
            write_json(output / "details" / name, value)
    controls = {name: study.get("controls_passed") for name, study in result["studies"].items() if "controls_passed" in study}
    result["summary"] = {"studies": list(result["studies"]), "controls_passed": controls,
                         "all_controls_passed": all(controls.values()) if controls else None}
    write_json(output / "experiment.json", result)
    from ..reporting import generate_report
    generate_report(output)
    return result
