# NanoInterpretability

A local Qwen3.5-0.8B research workbench. The implementation follows [PLAN.md](PLAN.md): architecture accounting, concrete computation traces, and controlled activation interventions. Model artifacts come from ModelScope; inference loads only a local snapshot. There are no notebooks.

## Environment and commands

Run all Python, installation, acquisition, test and report commands inside the dedicated `nano-interpretability` conda environment. Dependency locks and reproduction instructions are finalized after the compatibility and numerical gates pass. The default configuration targets CUDA BF16 with eager attention; consult recorded hardware and dtype metadata for a particular run.

```bash
conda run -n nano-interpretability nanoinspect doctor
conda run -n nano-interpretability nanoinspect download --config configs/model.yaml
conda run -n nano-interpretability nanoinspect baseline --config configs/model.yaml --output runs/baseline
conda run -n nano-interpretability nanoinspect baseline --config configs/model.yaml --output runs/baseline-offline --offline-check
conda run -n nano-interpretability nanoinspect inventory --config configs/model.yaml
conda run -n nano-interpretability nanoinspect trace --config configs/trace_short.yaml
conda run -n nano-interpretability nanoinspect experiment --config configs/retrieval_patch.yaml
conda run -n nano-interpretability nanoinspect report --run runs/inventory
conda run -n nano-interpretability nanoinspect report --run runs/baseline-offline --compare runs/baseline
conda run -n nano-interpretability python -m pytest
```

Open the generated `report.html` directly in a browser. Reports link to the JSON and tensor artifacts in their run directory. Keep those files together when sharing a report. Comparisons display shared scalar fields; verify matching model, prompts, token selections and generation settings before interpreting them.

## Evidence and scope

The full multimodal model is inventoried, while initial execution and interventions use the text path. Loaded unique storage, logical parameter references, checkpoint tensors and executed text-path parameters are separate quantities. Full attention and gated-delta recurrence have different state semantics; recurrent states are not softmax attention maps.

Capture selects explicit positions and enforces a byte budget. Interventions use scoped hooks and restore them on exit, including exceptions. A steering effect demonstrates controllability under the tested intervention; it does not establish a unique semantic representation.

See [the acceptance record](docs/acceptance.md) for verified milestones and remaining work. `models/` and `runs/` are project-owned, ignored persistent deliverables. Preserve the downloaded snapshot and retained experiment evidence; they are not disposable caches.
