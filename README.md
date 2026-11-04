# NanoInterpretability

A local Qwen3.5-0.8B research workbench. The implementation follows [PLAN.md](PLAN.md): architecture accounting, concrete computation traces, and controlled activation interventions. Model artifacts come from ModelScope; inference loads only a local snapshot. There are no notebooks.

## Environment

All Python, installation, acquisition, test and report commands run inside the dedicated `nano-interpretability` conda environment. `scripts/bootstrap.sh` creates it from the pinned locks (`conda-linux-64.lock`, conda-forge only; `requirements.lock`, pip with the CUDA 12.8 PyTorch index) and installs this package in editable mode. A from-scratch rebuild was verified: it produced identical package sets and a bit-identical offline baseline (`runs/repro/`). The default configuration targets CUDA BF16 with eager attention; consult the recorded hardware and dtype metadata for a particular run.

```bash
bash scripts/bootstrap.sh                      # ENV_NAME=... to build a separate verification environment
```

## Commands

```bash
conda run -n nano-interpretability nanoinspect doctor
conda run -n nano-interpretability nanoinspect download --config configs/model.yaml
conda run -n nano-interpretability nanoinspect baseline --config configs/model.yaml --output runs/baseline
conda run -n nano-interpretability nanoinspect baseline --config configs/model.yaml --output runs/baseline-offline --offline-check
conda run -n nano-interpretability nanoinspect inventory --config configs/model.yaml
conda run -n nano-interpretability nanoinspect trace --config configs/trace_short.yaml
conda run -n nano-interpretability nanoinspect experiment --config configs/retrieval_patch.yaml
conda run -n nano-interpretability nanoinspect report --run runs/inventory
conda run -n nano-interpretability nanoinspect report --run runs/trace
conda run -n nano-interpretability nanoinspect report --run runs/experiment
conda run -n nano-interpretability nanoinspect report --run runs/baseline-offline --compare runs/baseline
conda run -n nano-interpretability python scripts/walkthrough.py
conda run -n nano-interpretability python -m pytest
```

| Command | Output | What it shows |
| --- | --- | --- |
| `inventory` | `runs/inventory/` | Checkpoint/loaded/text-path parameter accounting, layer schedule, block diagrams, measured cache sizes |
| `trace` | `runs/trace/` | One prompt through a gated-delta layer and a full-attention layer: every operation with shapes, axes, statistics, value slices and pinned source lines; worked numerical examples; residual and logit effects; parity checks against measured dtype-specific tolerances (with an FP32 calibration pass). Fails if any parity check fails |
| `experiment` | `runs/experiment/` | Study A (retrieval vs distance/distractors), Study B (clean-to-corrupted patching with controls), Study C (contrastive steering vs random directions, prefill-only vs decode-time, quality cost); writes `experiment.json`, `details/`, figures and the report |
| `scripts/walkthrough.py` | `runs/trace/` + stdout | The trace printed as a step-by-step text walkthrough of both mixer types |

`trace` accepts `--prompt`, `--layer`, `--token`, `--head` (an integer or `auto`) and `--site` to change the selection. Configurations merge over `configs/model.yaml`.

Open the generated `report.html` directly in a browser. Reports link to the JSON and tensor artifacts in their run directory; keep those files together when sharing a report. Comparisons display shared scalar fields; verify matching model, prompts, token selections and generation settings before interpreting them.

## Evidence and scope

The full multimodal model is inventoried, while execution and interventions use the text path. Loaded unique storage, logical parameter references, checkpoint tensors and executed text-path parameters are separate quantities. Full attention and gated-delta recurrence have different state semantics; recurrent states are reported as matrices and trajectories, never as softmax attention maps.

Capture selects explicit (optionally per-example) positions, records run/layer/token/position/phase/site/shape/dtype/revision metadata, and enforces a byte budget estimated before the forward pass. Interventions use scoped hooks that are removed on exit, including after exceptions. A steering effect demonstrates controllability under the tested intervention; it does not establish a unique semantic representation.

Findings, with their predictions, controls, alternatives and next experiments, are in [docs/findings.md](docs/findings.md). See [the acceptance record](docs/acceptance.md) for verified milestones and remaining limitations. `models/` and `runs/` are project-owned, ignored persistent deliverables. Preserve the downloaded snapshot and retained experiment evidence; they are not disposable caches.
