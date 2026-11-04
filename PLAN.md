# NanoInterpretability: Qwen3.5-0.8B Research Workbench

Status: implementation plan only. No model weights have been downloaded and no runtime has been installed in this planning pass.

The goal is to build a small local research workbench that makes a real model inspectable: first its structure, then its computation, and finally its causal response to interventions. Deliver concrete observations backed by tensors and reproducible experiments, rather than a generic explanation of Q/K/V.

## 1. Requirements and priorities

1. **Architecture first:** inspect depth, widths, layer types, parameter allocation, and the relationship between modules.
2. **Computation second:** follow actual token inputs through selected layers, with tensor shapes, values, intermediate operations, and numerical checks.
3. **Interpretability third:** capture activations, perform controlled patching and steering, and evaluate effects on held-out examples.

All installation, downloading, execution, testing, and report generation must use one dedicated conda environment, provisionally named `nano-interpretability`. Invoke commands through `conda run -n nano-interpretability` when activation is ambiguous. Do not install into the system Python or conda base. Obtain model artifacts through **ModelScope**, then load exclusively from the resolved local directory. Do not silently fall back to Hugging Face Hub downloads.

**No Jupyter notebooks.** Use ordinary Python modules, CLI commands, configuration files, and generated HTML/Markdown reports. A local interactive viewer may follow after the core inspection workflow works; it must remain independent of Jupyter.

This phase produces the plan. Downloading the model and building the runtime are the first implementation milestone.

## 2. Model-specific starting point

Target the post-trained `Qwen/Qwen3.5-0.8B`, available on [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B). Do not substitute the Base model or another Qwen generation.

The currently published [official configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json) provides these provisional architectural facts; verify them against the downloaded ModelScope snapshot before treating them as run metadata:

| Property | Published value |
| --- | --- |
| Text depth / residual width | 24 / 1024 |
| MLP intermediate width | 3584 |
| Mixer pattern | Three linear-attention layers, then one full-attention layer |
| Full-attention layers, zero-based | 3, 7, 11, 15, 19, 23 |
| Full-attention query / KV heads | 8 / 2 |
| Full-attention head dimension | 256 |
| Linear-attention key / value heads | 16 / 16 |
| Linear-attention key / value dimensions | 128 / 128 |
| Convolution kernel size | 4 |
| Vocabulary / tied embeddings | 248320 / enabled |
| Vision tower depth / width | 12 / 768 |

The checkpoint is multimodal. Inventory the entire checkpoint, while initially tracing text-only execution. Report loaded parameters separately from parameters executed by the text path. Identify any auxiliary/MTP tensors and loading discrepancies rather than assuming every checkpoint tensor participates in ordinary generation.

A useful first structural observation: residual width is not necessarily attention projection width. Here, query heads times head dimension is 2048, despite a residual width of 1024. The workbench must derive dimensions from actual weights instead of assuming hidden width divided by head count.

## 3. Reuse strategy and boundaries

Use ModelScope for acquisition, Transformers for the authoritative model execution, PyTorch for tensor access and hooks, and established plotting/report libraries for presentation. Pin working versions after a compatibility smoke test. The checkpoint's historical `transformers_version` field is not a sufficient installation specification.

Build only the research-specific layer: an architecture inventory, stable inspection sites, bounded tensor capture, experiment definitions, intervention controls, and evidence reports. Do not copy an entire interpretability repository or rewrite the full model.

Start with native PyTorch hooks for module boundaries. Internal functional operations may require small, version-specific instrumented adapters; module hooks alone do not expose every intermediate. Keep those adapters isolated, attributed, and tested against the untouched implementation. Preserve upstream licenses when adapting code.

TransformerLens, NNsight, and similar packages are optional candidates, not assumed dependencies. Before adopting one, demonstrate support for this exact hybrid checkpoint, faithful logits, and the needed intervention sites. If a small native adapter already meets the requirement, avoid another framework layer.

The [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) is a reference for the computation: it includes recurrent/chunked gated-delta operations and separate attention machinery. Record an immutable source revision during implementation; moving `main` links in this plan are discovery references.

## 4. User workflow and deliverables

Proposed CLI commands, to be implemented rather than treated as currently available:

```text
nanoinspect doctor
nanoinspect download --config configs/model.yaml
nanoinspect inventory --config configs/model.yaml
nanoinspect trace --config configs/trace_short.yaml
nanoinspect experiment --config configs/retrieval_patch.yaml
nanoinspect report --run runs/<run_id>
```

The user should be able to select a prompt, layer, token, head, and inspection site; inspect dimensions and selected values; compare runs; and follow a result back to the responsible source operation. Start with CLI selection and a generated HTML report with plots and expandable tables. Defer a dedicated web application until a real usability need appears.

Suggested repository structure:

```text
configs/                  # Model, trace, and experiment definitions
src/nanoinspect/
  runtime.py              # Local loading, device/dtype policy, provenance
  inventory.py            # Config, modules, weights, parameter accounting
  sites.py                # Semantic names mapped to version-specific modules
  capture.py              # Scoped hooks, tensor selection, storage budget
  adapters/qwen35.py      # Minimal internal-computation instrumentation
  interventions.py        # Patching, ablation, additive steering
  experiments/            # Reproducible research workflows
  reporting.py            # HTML/Markdown and figure export
  cli.py
 tests/                   # Numerical and behavioral correctness checks
 docs/                    # Architecture guide and evidence-based findings
```

Use project-owned, ignored directories for persistent model artifacts and experiment outputs. The model snapshot and reusable conda environment are intended implementation deliverables. Remove temporary downloads, scratch environments, conversion files, and background processes; preserve original inputs and explicitly retained runs.

## 5. Milestones and acceptance gates

### M0 — Reproducible local baseline

Inspect conda availability, CPU/GPU, RAM/VRAM, and disk capacity. Choose a supported Python/PyTorch/Transformers combination based on the actual machine. Use a short context, batch size one, and no quantization initially; choose BF16 where supported and a documented fallback otherwise. Do not promise performance before measuring it.

Download the complete checkpoint through ModelScope. Its [download CLI](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md) supports model, revision, and local-directory arguments. Resolve an available revision, record its immutable identifier where exposed, and hash downloaded artifacts. Record tokenizer, chat template, generation settings, package versions, hardware, and source revision.

Load using the local path with offline/local-only behavior. Review missing/unexpected weight keys. Run deterministic text inference and save token IDs, logits for selected positions, and generated output. Measure peak memory and elapsed time. A second run must work without contacting a model hub.

**Gate:** local inference succeeds inside the dedicated conda environment, the environment can be reproduced, and artifact provenance is complete.

### M1 — Architecture atlas, highest priority

Produce a machine-readable inventory and a readable architecture report from the loaded model. Include all layer types, module paths, weight shapes, trainable/frozen status, dtype, device, and parameter counts. Count tied storage once in total parameters and show logical references separately. Reconcile checkpoint, loaded-model, and active text-path totals.

Show the complete stack plus expanded diagrams for one full-attention block and one gated-delta block. Annotate residual paths, normalization, mixer, and MLP boundaries. Include widths before and after each projection, and distinguish configuration-derived facts from runtime measurements.

Initial questions: Where are parameters concentrated? How much belongs to embeddings? Which dimensions expand inside a block? How do the two mixer types differ in cached state and sequence-length dependence? Report measured cache sizes at several short lengths, with formulas and measured bytes clearly distinguished.

**Gate:** every displayed shape is backed by a weight or observed tensor; the user can identify the model's depth, widths, layer schedule, and major parameter costs without reading the implementation.

### M2 — Computation microscope

Use short, fixed prompts with visible token boundaries. Capture a selected full-attention layer and a selected gated-delta layer, plus their normalization, residual additions, and MLP boundaries.

For full attention, expose the stages needed to explain query/key/value formation, head grouping, positional transformation, masking, scores, probabilities, value aggregation, output gating, and projection. For the recurrent mixer, expose convolution effects, normalized inputs, decay/update controls, selected state updates, readout, and output gating. These are inspection requirements; derive precise ordering and equations from the pinned implementation.

For each selected operation, show input/output shape, axis meaning, dtype, a small value slice, summary statistics, and source location. Add one worked numerical example for a token/head. Explain how its output changes the residual stream and selected next-token scores.

Use an eager or reference path when optimized kernels hide internals. Compare instrumented and untouched layer outputs and final logits. Establish dtype-specific tolerances from baseline measurements, report absolute/relative errors, and avoid requiring exact equality across different kernels. Validate prefill and cached one-token decoding separately, including padding and causal behavior.

**Gate:** an ordinary Python script reproduces a concrete computation walkthrough for both mixer types, with passing parity checks. Do not represent a recurrent state as an ordinary softmax attention heatmap.

### M3 — Activations and controlled interventions

Define semantic sites such as residual input, mixer output before residual addition, post-mixer residual, MLP output, and residual output. Map names explicitly to concrete module paths and tensor semantics. Capture metadata must include run, layer, token IDs/positions, phase, site, shape, dtype, and model revision.

Default to selected sites/tokens and summary statistics. Require explicit selection for large tensors, estimate storage before capture, and enforce a configurable memory/disk budget. Hooks must be removed even after exceptions; mutable cache tensors must be copied when necessary to preserve observations.

Implement clean-to-corrupted activation patching first. Then add additive steering at one well-defined residual site: `h' = h + alpha * v`. Estimate directions on a separate training split using aligned token positions and contrastive examples; record normalization and scale conventions. Sweep positive, zero, and negative strengths and include norm-matched random directions. Compare prefill-only and decode-time intervention policies explicitly.

**Gate:** read-only capture and zero-strength intervention preserve baseline outputs; a patch affects only the declared site/positions; repeated runs reset caches and hooks; held-out results include target effects and quality degradation. A steering effect is evidence of controllability, not by itself proof of a unique semantic representation.

## 6. Experiments designed to produce insight

| Question | Experiment | Evidence and limitation |
| --- | --- | --- |
| How is model capacity allocated? | Parameter/storage inventory and observed width expansion | Exact accounting; parameter count alone does not establish functional importance |
| How is earlier information retained? | Synthetic key/value retrieval with increasing distance and distractors | Baseline accuracy, cache/state measurements, selected traces; exclude tasks the baseline cannot solve |
| Which layers mediate a specific answer? | Corrupt one key/value fact and restore selected clean activations | Recovery of target-vs-distractor logit difference across layers; controls for token alignment and patch magnitude |
| Does a candidate direction generalize? | Contrastive steering with held-out prompts and strength sweep | Target likelihood/behavior, unrelated-task quality, random-direction controls, and examples of failure |

Begin with approximately 20 pilot prompts to establish that the behavior exists, then use a fixed held-out set of at least 50 examples for the first intervention study where feasible. Record per-example results and uncertainty, not only a mean or a striking generation. These experiments are proposals, not claims about what the model will reveal.

Each finding should state the question, prediction, controlled change, observation, alternative explanation, and next discriminating experiment. Publish negative results when they invalidate an attractive hypothesis. Do not infer causality from attention weights or activation magnitude alone.

## 7. Agent organization

A single **Astra lead** owns research direction, task decomposition, interface contracts, integration decisions, and evidence review. **Sol and Terra implement the code**. This is the intended implementation team; this planning pass does not launch that team.

| Role | Responsibility |
| --- | --- |
| Astra lead | Maintain priorities, define acceptance criteria, assign bounded tasks, review numerical fidelity, and synthesize findings |
| Sol implementer | Environment/bootstrap, ModelScope acquisition, runtime, inventory, and CLI integration |
| Terra implementer | Capture/adapters, numerical validation, interventions, and experiment machinery |

Astra should sequence M0 and the shared runtime/site interfaces before parallelizing independent modules. Assign clear file ownership or separate worktrees. Each task must specify inputs, outputs, files, numerical checks, and completion criteria. Implementation agents return changes, commands run, results, and unresolved limitations. Astra integrates each milestone before expanding scope; cross-review critical adapters and intervention semantics.

Do not spend early effort on steering demos or interface polish while architecture accounting is incomplete. A useful first delivery is the local baseline plus architecture atlas; a verified two-layer computation walkthrough is the next delivery.

## 8. Completion definition

The initial workbench is complete when a fresh conda environment can reproduce the local ModelScope-backed baseline; the atlas explains the actual architecture; a script exposes and checks both mixer computations; and a controlled intervention report includes baseline, controls, held-out outcomes, and limitations. Reports must link every substantive observation to run artifacts and the relevant source implementation.

Remain explicit about what was measured, inferred, or left untested. Preserve useful final outputs and verify that task-created temporary artifacts and processes have been cleaned up.
