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

Computation and intervention findings will be added only after their numerical and held-out validation runs pass.
