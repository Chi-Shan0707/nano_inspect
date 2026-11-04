# Reading the architecture atlas

The downloaded ModelScope configuration identifies `Qwen3_5ForConditionalGeneration`, with 24 text blocks and a 12-block vision tower. These are **configuration facts** until checked against the loaded module and weight inventory. The historical `transformers_version` in the checkpoint is not the installed runtime version; run provenance records both the installed package and its source hash.

The text path is token IDs → tied embedding lookup → 24 residual blocks → final RMSNorm → tied vocabulary projection. Each block computes:

```text
x ──────────────────────────────┐
└─ input RMSNorm → token mixer ──+ → y ───────────────────────┐
                                    └─ RMSNorm → gated MLP ─+ → output
```

The schedule repeats three gated-delta blocks followed by one full-attention block. The atlas displays every block and derives projection input/output widths from weight shapes. For a linear weight stored as `[out, in]`, the data moves from `in` to `out`; the shape order is not the execution order.

In the configured full-attention block, eight query heads each have 256 dimensions. Their combined 2,048 dimensions exceed the 1,024-dimensional residual width. The query projection also produces an output gate, so its raw output width is twice the combined query width. Two key/value heads are repeated across query-head groups. Query/key RMS normalization precedes partial rotary position transformation. Scores, the causal/padding mask, softmax, value aggregation, sigmoid output gating, and output projection are distinct operations.

The gated-delta block projects Q/K/V, applies a depthwise causal convolution and activation, normalizes Q/K, and updates a recurrent matrix state. Its controls are `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`. Each token decays the old state by `exp(g)`, computes the error between its value and the state's key-conditioned readout, and adds a key/error outer product scaled by `beta`. The normalized, scaled query reads the updated state; gated RMS normalization and output projection return to residual width. The implementation's prefill chunk algorithm and one-token recurrent algorithm must agree within measured numerical tolerances.

## Parameter and cache accounting

The inventory separates checkpoint tensors, loaded logical references, unique loaded storage and the executed text path. Tied embedding/head storage is counted once in unique totals. Vision parameters remain loaded but do not execute in the initial text-only trace. Auxiliary/MTP tensors and missing/unexpected loading keys are listed explicitly; a checkpoint total is not automatically the model's active parameter total.

Full-attention key/value state grows with sequence length: per layer, the logical KV bytes are `2 × batch × tokens × KV heads × head dimension × element bytes`. A gated-delta recurrent matrix has fixed logical size `batch × value heads × key dimension × value dimension × state element bytes`; convolution history adds a fixed buffer. These are formulas, not measurements of allocator memory. See the run's cache tensors for actual dtype, stored history length, tensor bytes and unique storage bytes at each measured length.

## Evidence navigation

Generate `runs/inventory/report.html` for the complete stack, expanded block diagrams, measured projection tables and parameter plot. The adjacent `inventory.json` contains every module, parameter and observed boundary. Trace reports expose individual internal operations and link to the pinned source. A large activation, a large parameter allocation or an attention probability does not establish causal importance; use the controlled intervention study for causal claims.
