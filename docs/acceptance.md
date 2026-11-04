# Implementation acceptance record

This checklist tracks the original `PLAN.md` scope. A checkbox is only checked
after inspecting the corresponding executable result and retained artifact.

- [ ] M0: dedicated conda environment, reproducible dependency locks, hardware and dtype policy.
- [ ] M0: complete ModelScope snapshot, revision identity and artifact hashes; local-only loading with loading-key review.
- [ ] M0: deterministic baseline and second offline execution, tokenizer/template/settings, selected logits and IDs, timing and memory.
- [ ] M1: complete checkpoint/module/parameter atlas, logical versus unique storage, vision/text/auxiliary reconciliation.
- [ ] M1: layer schedule and expanded mixer diagrams with measured widths and source links.
- [ ] M1: executed text-path accounting and cache measurements at multiple lengths, separated from formulas.
- [ ] M2: selectable prompt/layer/token/head/site with bounded values, statistics, axis semantics and source locations.
- [ ] M2: full-attention Q/K/V, grouping, RoPE, masks, scores, probabilities, aggregation, gating and projection.
- [ ] M2: gated-delta convolution, normalization, decay/update, state updates, readout and gating.
- [ ] M2: worked numeric examples, residual and selected next-token effects, measured error tolerances.
- [ ] M2: untouched/instrumented layer and logit parity for prefill and cached decode, padding and causality checks.
- [ ] M3: scoped semantic capture, metadata, budget estimate/enforcement, cloned mutable observations and exception cleanup.
- [ ] M3: clean-to-corrupt patching, exact position/site isolation, zero intervention and read-only parity, repeat/reset checks.
- [ ] M3: training-only aligned contrastive direction, signed strength sweep, matched random controls, prefill/decode policies.
- [ ] Studies: approximately 20 pilot examples; fixed held-out set of at least 50 where feasible; baseline-solvable task selection.
- [ ] Studies: per-example target and quality results, uncertainty, distance/distractor variation and failure examples.
- [ ] Reports: readable HTML/Markdown, plots, expandable tables, run comparison and artifact/source links.
- [ ] Findings: question, prediction, controlled change, observation, alternatives and next discriminating experiment.
- [ ] Final: documented CLI commands execute, tests pass, fresh-environment reproduction verified, temporary artifacts/processes cleaned up.

Evidence links and any remaining limitations will be recorded here as milestones
are verified. An unchecked item is not a completion claim.
