# Evidence-based findings

## Parameter allocation and width expansion

**Question.** Where is capacity allocated, and does residual width determine attention width?

**Prediction.** The hybrid model has distinct mixer projections; tied embedding/head references must not be counted twice.

**Controlled change.** None: this is a structural inventory of the complete ModelScope snapshot and loaded multimodal model, followed by a text-only forward. It does not test functional importance.

**Observation.** The [inventory](../runs/inventory/inventory.json) records 873,438,784 checkpoint tensor parameters and 852,985,920 unique loaded parameters. The 20,452,864-parameter difference consists of 15 MTP tensors. The loader suppresses these unexpected keys by design ([pinned source, line 912](https://github.com/huggingface/transformers/blob/446a570e62b04a89ea001604ada7366baac2ea67/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L912)); an empty loading warning list does not mean every checkpoint tensor participates in inference.

The text path accounts for 752,393,024 unique parameters; the inactive vision tower accounts for 100,592,896. Tied embedding/head storage contains 254,279,680 parameters, counted once. Text MLPs contain 264,241,152 and token mixers 233,822,016. Full-attention query/gate projection weights have 4,096 output rows from 1,024 residual inputs, split into 2,048 query and 2,048 gate features. The key and value projections each produce 512 features. See the [atlas](../runs/inventory/report.html) for complete weight evidence and expanded block diagrams.

**Alternative explanation.** Parameter allocation describes storage and dimensions; it does not measure how much each component contributes to a task. The active text-path count relies on observed module execution plus source-confirmed functional access to convolution weights.

**Next discriminating experiment.** Compare answer recovery after matched interventions across mixer/MLP boundaries on baseline-solvable retrieval examples.

## Cached state growth

**Question.** How does retained state depend on short sequence length?

**Prediction.** Full-attention KV state grows with token count, while gated-delta recurrent and convolution states retain fixed dimensions.

**Controlled change.** Repeat a single token with batch size one at lengths 8, 32 and 64; keep checkpoint, device, dtype and cache policy fixed.

**Observation.** Unique cache storage is respectively 19,857,408, 20,152,320 and 20,545,536 bytes. The increase is 12,288 bytes per token across the six full-attention layers. Per-tensor shapes/dtypes and calculated versus measured bytes appear in [cache measurements](../runs/inventory/inventory.json). These figures describe retained cache storage, not total peak GPU memory or inference latency.

**Alternative explanation.** Fixed state size does not imply perfect retention over arbitrary distances; repeated-token inputs also do not establish retrieval quality.

**Next discriminating experiment.** Vary retrieval distance and distractors, measuring baseline success before interpreting activation patching.

## Deterministic local baseline

The [baseline](../runs/baseline/baseline.json) answers the fixed capital-of-France prompt with “Paris”. The first measured inference/generation took 1.8238 seconds and allocated 1,775,333,376 peak GPU bytes. The [network-blocked repeat](../runs/baseline-offline/baseline.json) produced exactly identical inputs, generated IDs and saved selected-position logits, with zero network attempts; see [comparison evidence](../runs/baseline-comparison.json). These measurements apply to this short prompt and hardware, not a general performance benchmark.

## Chunked versus recurrent gated-delta numerics

**Question.** Why did an earlier instrumented trace match untouched logits exactly for unpadded prefill but differ by 0.34 under left padding, and what tolerance should parity checks use?

**Prediction.** Upstream prefill uses the chunked rule (a 64×64 unit-triangular solve in FP32, [line 381](https://github.com/huggingface/transformers/blob/446a570e62b04a89ea001604ada7366baac2ea67/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L381)) while one-token cached decoding uses the token recurrence ([line 612–613](https://github.com/huggingface/transformers/blob/446a570e62b04a89ea001604ada7366baac2ea67/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L612)). If the earlier adapter always ran the recurrence, the two kernels agree mathematically and differ only by rounding. The final BF16 cast would hide most of that rounding, but it would surface as occasional 1-ULP flips that later layers amplify. The same magnitude should then appear in the *untouched* model whenever an input change perturbs rounding, for example under padding.

**Controlled change.** (1) The [adapter](../src/nanoinspect/adapters/qwen35.py) now dispatches exactly like upstream and runs the explicit recurrence only as an observation; a `recurrent` mode substitutes the recurrence for one layer. (2) Same prompt, layer and inputs, run in BF16 and again in FP32 (`calibration_dtypes`). (3) Untouched-model perturbations: 1, 3 and 8 left-padding tokens; cached versus uncached final-token decoding; truncated prefixes.

**Observation.** All values come from [trace.json](../runs/trace/trace.json) (`parity` and `calibration`).
- Upstream-dispatch instrumentation is **bit-exact** against untouched execution in 18 same-kernel checks. These cover prefill logits, mixer and layer outputs, cached prefill, one-token cached decode, and 3-token left padding, for both traced layers. The read-only capture check is also exact.
- Substituting the recurrence at gated-delta layer 14 changes its BF16 mixer output by at most 2.4e-4 and its layer output by 0.0078 (0.0019 of the layer's max |value|, under 2 BF16 eps). Final logits move by at most 0.25 (mean 0.014), and the greedy token is unchanged.
- In FP32 the same substitution moves logits by only 2.3e-5. An earlier probe on layer 0 showed the FP32 cores differ at about 1e-8 in 83% of elements, while the BF16-cast cores differ in only 0.02% of elements, each by 1 ULP.
- The **untouched** BF16 model moves by 0.25–0.28 (mean ≈ 0.024) under left padding and by 0.16 for cached versus uncached decoding. FP32 shows ≤ 3.0e-5 in both cases.
- The 0.34 gap is therefore a property of BF16 kernels, not an adapter error.
- The explicit recurrence reproduces the chunked kernel's core readout to 9.5e-7.

The [dtype-specific tolerances](../src/nanoinspect/trace.py) require exact equality for same-kernel comparisons. Cross-kernel layer outputs must stay within 2 eps × max|ref| in BF16 (64 eps in FP32). Cross-kernel logits must stay within max 0.5 / mean 0.05 with an unchanged greedy token in BF16 (1e-4 / 1e-5 in FP32), roughly 1.6–3× the measured maxima. All 25 BF16 checks and 9 FP32 calibration checks pass.

**Alternative explanation.** The tolerances come from one 37-token prompt. Longer prompts span several 64-token chunks, accumulate more rounding, and could exceed them without any bug. A tolerance calibrated on a single input is a measurement, not a guarantee.

**Next discriminating experiment.** Repeat the calibration across prompt lengths that cross chunk boundaries (63, 64, 65, 200 tokens) and across padding counts. Fit how cross-kernel error grows with length before reusing the BF16 tolerance on long contexts.

## Two-mixer walkthrough of one retrieval answer

**Question.** For "The blue box contains the" → " coin" (distractor " apple"), what do a gated-delta layer and a full-attention layer actually compute at the prediction position, and how do they change the answer's logit margin?

**Prediction.** A full-attention head should retrieve the value token by content (high softmax probability on " coin"). A gated-delta layer has no token-to-token weights; any contribution must pass through its matrix state's decay/write/readout.

**Controlled change.** Trace layers 14 (gated delta) and 15 (full attention) at the final position, selecting per layer the head with the largest direct coin-minus-apple effect ([configs/trace_short.yaml](../configs/trace_short.yaml)). Layers were chosen because zero-ablating layer 14's mixer output at that position removed most of the 2.5-logit margin in a 24-layer scan. Components are zero-ablated one at a time.

**Observation.** See [the trace report](../runs/trace/report.html) and [trace.json](../runs/trace/trace.json) (`worked_examples`, `residual_effects`).
- **Layer 15, query head 5** attends from the final token with probability 0.707 to the earlier " coin" token (position 17). Its scaled score there is 13.09 versus 10.48 for the next key. It shares KV head 1 with heads 4–7 ([repeat_kv, line 721](https://github.com/huggingface/transformers/blob/446a570e62b04a89ea001604ada7366baac2ea67/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L721)).
- Its 1024-wide contribution (norm 1.06) adds +0.87 to the coin-minus-apple direct logit. The whole layer-15 mixer adds +1.25 directly. Zero-ablating that mixer output at the final token lowers the margin by 0.50; zero-ablating the layer-15 MLP lowers it by 1.13.
- **Layer 14, value head 1** keeps almost all of its state at the final token (exp(g) = 0.998, beta = 0.16). The state norm grows from 0.27 after token 0 to 3.32. The write is a rank-one correction of size beta · (v − kᵀS), and the readout qᵀS is gated by SiLU(z) after RMSNorm. The [recurrent-state figure](../runs/trace/trace-layer-14.png) shows state norms and controls, not attention.
- Layer 14's mixer has a small *direct* effect (−0.08) but a large *total* effect: zero-ablating it at the final token lowers the margin by 2.38. Its influence is therefore mediated by later layers.
- Zero-ablating the layer-14 MLP *raises* the margin by 2.63.
- Residual identities hold exactly in BF16 (`post_mixer_residual = residual_input + mixer_output` and `residual_output = post_mixer_residual + mlp_output`). Frozen-scale direct attribution of the full final residual gives 20.14 / 17.60 versus model logits 20.125 / 17.625.

**Alternative explanation.** Zero ablation pushes activations off-distribution, so its large total effects may overstate normal function, and opposite-sign mixer and MLP effects may partly be compensation. One prompt and one head per layer are examples, not population claims. Attention probability is not causal evidence by itself; the patching study below tests mediation directly.

**Next discriminating experiment.** Use clean-to-corrupted patching on held-out pairs across all layers and sites, which avoids zero-ablation's distribution shift. Then patch individual heads (layer 15 head 5; layer 14 state) to separate the direct path from the path through later layers.

## Retrieval across distance, distractors, and format (Study A)

**Question.** Over what distances and distractor counts does Qwen3.5-0.8B reliably retrieve a stored key/value fact, and how does cached state scale across those conditions?

**Prediction.** Full-attention KV cache bytes scale linearly with prompt token length while gated-delta recurrent and convolution state bytes remain constant. Retrieval accuracy should be verified on a 20-prompt pilot before selecting a solvable condition for causal patching.

**Controlled change.** Vary prompt format (`sentence`: `"The {color} {container} contains the {object}."` vs terse `list`: `"{color} {container}: {object}"`), number of stored facts (`2`, `8`, `16` — i.e., `1`, `7`, `15` distractors), and neutral filler sentences between the facts and the query (`0`, `40`, `160`, spanning `10` to `1,370` tokens). Each of the 18 conditions runs on a 20-example pilot and a disjoint 50-example held-out set ([configs/retrieval_patch.yaml](../configs/retrieval_patch.yaml)).

**Observation.** Complete tables, 95% bootstrap confidence intervals, cache measurements, and failure cases are in [runs/experiment/report.html](../runs/experiment/report.html), [runs/experiment/experiment.json](../runs/experiment/experiment.json), and [runs/experiment/details/retrieval_examples.json](../runs/experiment/details/retrieval_examples.json):
- In `sentence` format, held-out accuracy is `0.980 [0.940, 1.000]` at 2 facts / 0 filler (mean distance `23.2` tokens, logit margin `2.316 [2.074, 2.579]`) and `1.000` across all 8 other `sentence` conditions up to `16` facts and `160` filler sentences (`1,370` tokens, mean distance `1,305` tokens, margin `4.380 [4.053, 4.681]`).
- In terse `list` format with `0` filler sentences, pilot accuracy drops to `0.60–0.70` and held-out accuracy to `0.700–0.860` (`2`, `8`, `16` facts), because without a sentence frame or filler context the model often predicts generic nouns (`" cow"`, `" chair"`, `" book"`) over the listed value tokens. Adding `40` or `160` filler sentences restores `1.000` held-out accuracy. Following `PLAN.md` Section 6, the unsolvable `0`-filler `list` conditions are marked `solvable: false` and excluded from causal patching.
- Measured cache storage across the 18 conditions ([retrieval-cache.png](../runs/experiment/retrieval-cache.png)) confirms that the 18 gated-delta layers hold a constant `18,874,368` recurrent-state bytes plus `884,736` convolution-state bytes regardless of length, while the 6 full-attention layers grow at exactly `12,288` bytes per token (from `122,880` bytes at 10 tokens to `16,834,560` bytes at 1,370 tokens).

**Alternative explanation.** Perfect accuracy up to 1,370 tokens in this synthetic template does not imply that fixed-size recurrent state alone retains arbitrary information: the 6 full-attention layers have direct KV access to all earlier positions. Higher logit margins with filler sentences reflect stronger context framing rather than improved memory capacity.

**Next discriminating experiment.** Ablate or freeze full-attention KV access to the fact span while keeping gated-delta state updates intact (and vice versa) across increasing filler distances to separate recurrent-state retention from full-attention KV lookup.

## Layer- and site-resolved activation patching (Study B)

**Question.** Which layers, mixer types, and token positions mediate the retrieved answer when one key/value fact is corrupted?

**Prediction.** At the source `value` token position, restoring the clean residual in early layers should restore the answer because all downstream layers see the restored value. At the `final` prediction position, information must be routed from the value position to the final position by token mixers; full-attention mixers should show sharp step increases in `post_mixer_residual` recovery at the final position.

**Controlled change.** On a fixed held-out set of `n = 60` baseline-solvable examples (`sentence` format, `4` facts, `10` filler sentences; clean top-1 = clean value, corrupted top-1 = corrupted value; clean logit difference `10.206 [9.806, 10.641]`, corrupted `-10.011 [-10.536, -9.508]`), corrupt only the single queried value token (`token_alignment` verified) and patch a single clean activation into the corrupted run across all `24` layers, `4` semantic sites (`mixer_output`, `post_mixer_residual`, `mlp_output`, `residual_output`), and `4` aligned positions (`value`, `query_key`, `final`, and an unrelated distractor `control` position), plus norm-matched random-displacement controls ([src/nanoinspect/experiments/patching.py](../src/nanoinspect/experiments/patching.py)).

**Observation.** See [patching-heatmap.png](../runs/experiment/patching-heatmap.png), [patching-lines.png](../runs/experiment/patching-lines.png), and [runs/experiment/details/patching_examples.json](../runs/experiment/details/patching_examples.json). All 10 validation controls (`read_only_capture`, `zero_strength_steering`, `identity_patch`, `position_site_isolation`, `pre_corruption_position`, `cached_decode_reset`, `repeat_after_sweep`, `hook_cleanup`, `position_wise_mlp_equivalence`, `token_alignment`) pass:
- **At the `value` position**, patching `post_mixer_residual` or `residual_output` achieves `~1.00` recovery in layers `0–4` (e.g., layer 2 `1.000 [0.999, 1.001]`, layer 3 `1.002 [1.000, 1.004]`), stays high through layer 14 (`~0.95`), and drops sharply after layer 15 once the critical cross-position read has already occurred. Patching `mixer_output` alone at the `value` position is near zero everywhere except layer 0 (`0.966`), where the token mixer first transforms the raw token embedding.
- **At the `final` prediction position**, residual recovery (`post_mixer_residual` / `residual_output`) is `0.00` through layer 14, jumps at full-attention **layer 15** (`mixer_output` alone recovers `0.567` of the 20.2-logit gap), and climbs through layers 16–23 to `1.065 [1.047, 1.082]` at layer 22 and `1.000` at layer 23. Across all layers at the `final` position, `full_attention` `mixer_output` mean recovery is `0.1085` (max `0.567` at layer 15) versus `0.0101` for `linear_attention` `mixer_output` (max `0.050`).
- **Position-wise MLP equivalence holds exactly**: because `Qwen3_5MLP` operates independently at each token position, patching `post_mixer_residual` at a single position automatically causes that layer's MLP to compute the exact clean MLP output at that position, making single-position `post_mixer_residual` and `residual_output` recovery identical.
- **Specificity controls**: patching at `query_key` or `control` positions yields `< 0.002` recovery across all layers and sites, and norm-matched random displacements at the patched site fail to recover the clean answer.

**Alternative explanation.** Single-site sufficiency (restoring the clean residual at one layer recovers the logit margin) does not prove that a single layer is the unique mediator when multiple layers act redundantly, and low single-layer `mixer_output` recovery in `linear_attention` layers at the `final` position reflects distributed multi-layer state propagation rather than complete absence of recurrent contribution.

**Next discriminating experiment.** Perform path patching and multi-layer subset patching (comparing blocking all 6 full-attention mixers vs blocking the 18 gated-delta mixers between the `value` and `final` positions) to quantify how much retrieval signal travels through recurrent state versus direct KV attention.

## Contrastive additive steering, controls, and quality trade-off (Study C)

**Question.** Does a contrastive residual direction estimated only on training pairs generalize to held-out prompts under positive and negative strength sweeps, how do `prefill_only` and `decode_time` policies differ, and what is the collateral cost on unrelated factual tasks?

**Prediction.** Adding `alpha * v` along a positive-minus-negative review direction should monotonically shift held-out target token log-odds and continuation sentiment, outperforming norm-matched random directions, while large `|alpha|` should degrade unrelated factual QA accuracy before or alongside strong generation control.

**Controlled change.** Estimate a unit contrastive direction `v_hat` at `residual_output` from `24` aligned positive/negative review pairs (training split only; raw direction norm `unit_value = 2.398` at layer 12, selected via a separate `20`-prompt pilot over candidate layers `[3, 6, 9, 12, 15, 18]`). Evaluate on `n = 60` disjoint held-out prompts across relative strengths `alpha in {-0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5}`, comparing against `5` norm-matched random directions (`random_0..4`), comparing `prefill_only` vs `decode_time` policies, and measuring unrelated-task quality on `50` baseline-correct world-capital QA prompts and `12` neutral factual passages ([src/nanoinspect/experiments/steering.py](../src/nanoinspect/experiments/steering.py)).

**Observation.** See [steering-sweep.png](../runs/experiment/steering-sweep.png), [steering-policies.png](../runs/experiment/steering-policies.png), and [runs/experiment/details/steering_examples.json](../runs/experiment/details/steering_examples.json). All 6 steering controls pass:
- **Monotonic bidirectional target shift**: on the 60 held-out prompts, the contrastive direction shifts positive-minus-negative first-token log-odds monotonically from `-3.599 [-3.689, -3.500]` at `alpha = -0.5` through `0.000` at `alpha = 0` to `+4.459 [4.380, 4.538]` at `alpha = +0.5`. At `|alpha| = 0.5`, the 5 norm-matched random directions produce shifts bounded within `[-1.455, +0.565]` with inconsistent signs.
- **Prefill-only versus decode-time policy**: `prefill_only` and `decode_time` produce the exact same first-step logits (`policies_share_prefill: true`), but diverge over 12-token continuations. For negative steering (`alpha = -0.125`), `prefill_only` fades during generation (unsteered sentiment judge `+5.040 [3.342, 6.692]`), whereas persistent `decode_time` steering flips continuations negative (`-4.679 [-6.377, -2.788]`, and `-8.079` at `alpha = -0.25`).
- **Collateral quality degradation and asymmetry**: positive steering (`alpha >= +0.125`) degrades unrelated capital-city QA accuracy steeply (`1.00` at `alpha = 0` → `0.70` at `+0.0625` → `0.44` at `+0.125` → `0.10` at `+0.25` → `0.00` at `+0.5`, where prompts collapse into `"..."` punctuation tokens), whereas negative steering preserves `1.00` capital accuracy down to `alpha = -0.25` (`0.94` at `-0.5`).

**Alternative explanation.** As noted in `PLAN.md` M3, a steering effect demonstrates controllability under the tested linear perturbation; it does not prove that the model uses this direction as a unique, isolated semantic feature. The severe degradation of capital QA under positive `alpha` shows that the mean difference vector entangled review sentiment with punctuation/formatting features that suppress direct one-word factual answers.

**Next discriminating experiment.** Project out formatting and punctuation subspaces (or estimate the direction across diverse prompt templates with sparse / orthogonalized probing) and re-test whether target sentiment shift can be achieved at `alpha = +0.25` without collapsing factual QA accuracy.

