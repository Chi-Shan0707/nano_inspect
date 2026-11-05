# Implementation acceptance record

This checklist tracks the original `PLAN.md` scope. A checkbox is only checked
after inspecting the corresponding executable result and retained artifact.

- [x] M0: dedicated conda environment, reproducible dependency locks, hardware and dtype policy.
  - Evidence: [`environment.yml`](../environment.yml), [`conda-linux-64.lock`](../conda-linux-64.lock), [`requirements.lock`](../requirements.lock), [`scripts/bootstrap.sh`](../scripts/bootstrap.sh), [`runs/doctor.json`](../runs/doctor.json), [`runs/repro/bootstrap.log`](../runs/repro/bootstrap.log), [`runs/repro/baseline/comparison.json`](../runs/repro/baseline/comparison.json).
- [x] M0: complete ModelScope snapshot, revision identity and artifact hashes; local-only loading with loading-key review.
  - Evidence: [`models/Qwen3.5-0.8B/nanoinspect_provenance.json`](../models/Qwen3.5-0.8B/nanoinspect_provenance.json), [`src/nanoinspect/runtime.py`](../src/nanoinspect/runtime.py), [`runs/download-verified.log`](../runs/download-verified.log).
- [x] M0: deterministic baseline and second offline execution, tokenizer/template/settings, selected logits and IDs, timing and memory.
  - Evidence: [`runs/baseline/baseline.json`](../runs/baseline/baseline.json), [`runs/baseline/selected_logits.pt`](../runs/baseline/selected_logits.pt), [`runs/baseline-offline/baseline.json`](../runs/baseline-offline/baseline.json), [`runs/baseline-comparison.json`](../runs/baseline-comparison.json).
- [x] M1: complete checkpoint/module/parameter atlas, logical versus unique storage, vision/text/auxiliary reconciliation.
  - Evidence: [`src/nanoinspect/inventory.py`](../src/nanoinspect/inventory.py), [`runs/inventory/inventory.json`](../runs/inventory/inventory.json), [`runs/inventory/report.html`](../runs/inventory/report.html), [`runs/inventory/parameter-allocation.png`](../runs/inventory/parameter-allocation.png).
- [x] M1: layer schedule and expanded mixer diagrams with measured widths and source links.
  - Evidence: [`docs/architecture.md`](architecture.md), [`runs/inventory/block-0.svg`](../runs/inventory/block-0.svg), [`runs/inventory/block-3.svg`](../runs/inventory/block-3.svg), [`runs/inventory/operations-0.svg`](../runs/inventory/operations-0.svg), [`runs/inventory/operations-3.svg`](../runs/inventory/operations-3.svg).
- [x] M1: executed text-path accounting and cache measurements at multiple lengths, separated from formulas.
  - Evidence: [`runs/inventory/inventory.json`](../runs/inventory/inventory.json) (`accounting` and `cache_measurements` at lengths 8, 32, 64), [`docs/findings.md`](findings.md).
- [x] M2: selectable prompt/layer/token/head/site with bounded values, statistics, axis semantics and source locations.
  - Evidence: [`src/nanoinspect/trace.py`](../src/nanoinspect/trace.py), [`configs/trace_short.yaml`](../configs/trace_short.yaml), [`runs/trace/trace.json`](../runs/trace/trace.json), [`runs/trace/report.html`](../runs/trace/report.html).
- [x] M2: full-attention Q/K/V, grouping, RoPE, masks, scores, probabilities, aggregation, gating and projection.
  - Evidence: [`src/nanoinspect/adapters/qwen35.py`](../src/nanoinspect/adapters/qwen35.py), [`runs/trace/trace.json`](../runs/trace/trace.json) (layer 15 operations & worked example), [`runs/trace/trace-layer-15.png`](../runs/trace/trace-layer-15.png).
- [x] M2: gated-delta convolution, normalization, decay/update, state updates, readout and gating.
  - Evidence: [`src/nanoinspect/adapters/qwen35.py`](../src/nanoinspect/adapters/qwen35.py), [`runs/trace/trace.json`](../runs/trace/trace.json) (layer 14 operations & worked example), [`runs/trace/trace-layer-14.png`](../runs/trace/trace-layer-14.png) (recurrent state norms and controls, never depicted as a softmax attention heatmap).
- [x] M2: worked numeric examples, residual and selected next-token effects, measured error tolerances.
  - Evidence: [`scripts/walkthrough.py`](../scripts/walkthrough.py), [`runs/trace/trace.json`](../runs/trace/trace.json) (`worked_examples`, `residual_effects`, `tolerances`, `calibration`).
- [x] M2: untouched/instrumented layer and logit parity for prefill and cached decode, padding and causality checks.
  - Evidence: [`runs/trace/trace.json`](../runs/trace/trace.json) (`parity`: all 25 BF16 checks and 9 FP32 calibration checks pass), [`tests/test_qwen35_adapter.py`](../tests/test_qwen35_adapter.py).
- [x] M3: scoped semantic capture, metadata, budget estimate/enforcement, cloned mutable observations and exception cleanup.
  - Evidence: [`src/nanoinspect/sites.py`](../src/nanoinspect/sites.py), [`src/nanoinspect/capture.py`](../src/nanoinspect/capture.py), [`tests/test_capture_interventions.py`](../tests/test_capture_interventions.py).
- [x] M3: clean-to-corrupt patching, exact position/site isolation, zero intervention and read-only parity, repeat/reset checks.
  - Evidence: [`src/nanoinspect/interventions.py`](../src/nanoinspect/interventions.py), [`src/nanoinspect/experiments/retrieval.py`](../src/nanoinspect/experiments/retrieval.py) (`study_b`), [`runs/experiment/experiment.json`](../runs/experiment/experiment.json) (`studies.patching.controls`, all 10 controls passed).
- [x] M3: training-only aligned contrastive direction, signed strength sweep, matched random controls, prefill/decode policies.
  - Evidence: [`src/nanoinspect/experiments/steering.py`](../src/nanoinspect/experiments/steering.py), [`runs/experiment/experiment.json`](../runs/experiment/experiment.json) (`studies.steering`), [`runs/experiment/details/steering_directions.pt`](../runs/experiment/details/steering_directions.pt).
- [x] Studies: approximately 20 pilot examples; fixed held-out set of at least 50 where feasible; baseline-solvable task selection.
  - Evidence: [`configs/retrieval_patch.yaml`](../configs/retrieval_patch.yaml), [`runs/experiment/experiment.json`](../runs/experiment/experiment.json) (Study A: 20 pilot + 50 held-out per condition across 18 conditions; Study B: 60 held-out aligned solvable pairs; Study C: 24 train pairs, 20 pilot prompts, 60 held-out prompts, 50 held-out capital QA prompts).
- [x] Studies: per-example target and quality results, uncertainty, distance/distractor variation and failure examples.
  - Evidence: [`runs/experiment/details/retrieval_examples.json`](../runs/experiment/details/retrieval_examples.json), [`runs/experiment/details/patching_examples.json`](../runs/experiment/details/patching_examples.json), [`runs/experiment/details/steering_examples.json`](../runs/experiment/details/steering_examples.json), [`runs/experiment/details/steering_generations.json`](../runs/experiment/details/steering_generations.json).
- [x] Reports: readable HTML/Markdown, plots, expandable tables, run comparison and artifact/source links.
  - Evidence: [`src/nanoinspect/reporting.py`](../src/nanoinspect/reporting.py), [`runs/inventory/report.html`](../runs/inventory/report.html), [`runs/trace/report.html`](../runs/trace/report.html), [`runs/experiment/report.html`](../runs/experiment/report.html), [`runs/baseline-offline/report.html`](../runs/baseline-offline/report.html).
- [x] Workbench: `nanoinspect serve` exposes architecture, logit lens, mixer internals, patching and steering on localhost.
  - Evidence: [`src/nanoinspect/server.py`](../src/nanoinspect/server.py), [`src/nanoinspect/workbench.py`](../src/nanoinspect/workbench.py), [`tests/test_workbench.py`](../tests/test_workbench.py).
- [x] Findings: question, prediction, controlled change, observation, alternatives and next discriminating experiment.
  - Evidence: [`docs/findings.md`](findings.md).
- [x] Final: documented CLI commands execute, tests pass, temporary artifacts/processes cleaned up.
  - Evidence: [`README.md`](../README.md), [`tests/`](../tests/), [`runs/tests.log`](../runs/tests.log) (38 passed, none skipped), [`runs/final-check.json`](../runs/final-check.json) (link check and cleanup record).
- [x] Rebuild: a second environment built by `scripts/bootstrap.sh` reproduced the offline baseline bit-identically.
  - Evidence: [`runs/repro/bootstrap.log`](../runs/repro/bootstrap.log), [`runs/repro/doctor.json`](../runs/repro/doctor.json), [`runs/repro/baseline/comparison.json`](../runs/repro/baseline/comparison.json).
  - Deviation: that rebuild used a temporary `nano-interpretability-repro` environment (since removed), not the dedicated environment. Package sets were not compared exhaustively; the log shows it received `setuptools 81.0.0` (the dedicated environment has `78.1.0`) because `requirements.lock` does not pin setuptools.

## Documented limitations

- Dtype-specific parity tolerances in [`src/nanoinspect/trace.py`](../src/nanoinspect/trace.py) are calibrated on short prompts (single 64-token chunk); longer sequences crossing multiple chunk boundaries can accumulate larger BF16 rounding differences between chunked and recurrent gated-delta kernels.
- Single-site activation patching in Study B establishes sufficiency for restoring the clean logit margin on the synthetic retrieval template, not uniqueness when multiple layers act redundantly.
- Contrastive additive steering in Study C demonstrates directional controllability on review continuations but causes steep collateral degradation on one-word capital QA at positive strengths (`alpha >= +0.125`), showing that the mean-difference vector entangles sentiment with formatting/punctuation behavior.
