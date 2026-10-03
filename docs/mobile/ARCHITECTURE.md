# Laya Mobile architecture

Status: implementation baseline for the first iOS prototype.

## Objective

Run an existing Laya checkpoint entirely inside an iOS application while preserving the
observable behavior of the reference Python runtime. The first prototype targets the English
`convaiinnovations/laya` checkpoint, FP16 Core ML inference, an iPhone 17 Pro, batch size one,
and a 512-token context.

The prototype is not a model redesign. PyTorch remains the conversion-time source of truth and
is not shipped in the application. Python and network access are not required at inference time.

## Compatibility baseline

The prototype intentionally pins each moving part:

- Neural architecture: upstream Laya commit
  `573e5b62696ba441230cd6be71d593331b5d23af`.
- English checkpoint revision: `c5d78730f3493e4fe16d61507ef4b78eef7318cf`.
- Mobile behavior: the public behavior of this repository at `v0.3.0`, including its documented
  post-v0.3.5 validation, rendering, truncation, routing, and temperature-clamping fixes.
- Initial precision: FP16 weights and computation where Core ML supports it.

Changing one of these inputs requires regenerating the parity fixtures. Checkpoint bundles carry
their source revision and SHA-256 digests so an application cannot silently combine incompatible
weights, tokenizer data, and configuration.

## Inference pipeline

```text
state + ordered question definitions
        |
        v
validation and JSON serialization
        |
        v
question and option rendering
        |
        v
checkpoint tokenizer
        |
        v
[CLS] <type> question: <instructions> [SEP]
[MASK] <option 0> [MASK] <option 1> ... [SEP]
<state> [SEP]
        |
        v
input_ids, attention_mask, marker_pos, marker_mask, qtype
        |
        v
Core ML: ModernBERT + decision head + scorer + action head
        |
        v
option logits + action logits
        |
        v
temperature calibration, softmax, typed decoding, usage metadata
```

## Runtime boundary

The Core ML model accepts five integer/boolean tensors and returns two floating-point tensors:

| Name | Prototype shape | Meaning |
| --- | --- | --- |
| `input_ids` | `[1, 512]` | Token IDs padded with the checkpoint pad token |
| `attention_mask` | `[1, 512]` | Valid-token mask |
| `marker_pos` | `[1, 48]` | Positions of option marker tokens |
| `marker_mask` | `[1, 48]` | Valid-option mask |
| `qtype` | `[1]` | `choice=0`, `score=1`, `noul=2` |
| `logits` | `[1, 48]` | One score per padded option slot |
| `action_logits` | checkpoint-defined | Action-head scores |

Forty-eight marker slots cover the maximum useful option count under the current 192-token head
budget and four-token minimum allocation. Export validation must reject a checkpoint whose
configuration can exceed this contract.

Questions are independent rows in the reference implementation. The iOS runtime therefore runs
multiple questions sequentially for the first milestone. This preserves decisions while bounding
peak activation memory. Batching can be added later behind the same public API.

## Portable and platform-specific responsibilities

The behavioral specification and parity fixtures are shared. Implementations remain native to
their platform instead of introducing a cross-platform ABI prematurely.

| Layer | Canonical definition | iOS implementation |
| --- | --- | --- |
| Input validation | `laya_mlx.agent.Agent._to_internal` | Swift value types and errors |
| State serialization | `laya_mlx.common.serialize_state` | Swift JSON serialization |
| Question rendering | `laya_mlx.common.render_options` | Swift |
| Sequence construction | `laya_mlx.common.build_sequence` | Swift |
| Tokenization | Checkpoint `tokenizer.json` | Swift tokenizer; Rust bridge fallback |
| Neural inference | Pinned PyTorch checkpoint | Core ML `.mlpackage` / `.mlmodelc` |
| Calibration and decoding | `laya_mlx.agent.Agent.system_one` | Swift |
| Parity truth | Generated fixture bundle | XCTest fixture consumer |

## Observable behavior that must not drift

- Question and choice insertion order controls model and output index order.
- Literal mask tokens in user-controlled text are replaced by spaces.
- Each rendered option is capped at 48 tokenizer tokens.
- The question head uses a 192-token budget and preserves at least eight instruction tokens.
- String and dictionary states keep their beginning when truncated. Conversation-list states keep
  their end.
- `choice` returns the original label and the calibrated probability for every option.
- `score` returns the expected zero-based rubric index, not the argmax index.
- `noul` returns `P(true)`; its semantic option order is always false, then true.
- Choice and score `confidence` use normalized inverse entropy. `answer_confidence` is maximum
  probability. Noul overrides `confidence` with `max(P(false), P(true))`.
- Calibration temperatures are selected by question type and option-count bucket, then clamped to
  `[0.5, 5.0]`.
- Public floating-point values are rounded to four decimal places.
- Truncation and collapsed-option diagnostics remain visible in usage metadata.

## Model artifacts

Generated models and downloaded checkpoints are ignored by Git. A distributable model bundle
contains:

```text
Laya.bundle/
  laya.mlpackage or laya.mlmodelc
  tokenizer/
    tokenizer.json
    tokenizer_config.json
  rl_agent_config.json
  encoder/config.json
  manifest.json
```

`manifest.json` records the source repository and revision, hashes of every bundled input,
conversion versions, tensor shapes, precision, and parity results.

## Acceptance gates

1. Token IDs, marker positions, masks, and question types exactly match every golden fixture.
2. Core ML produces finite outputs for every fixture.
3. Decisions agree on 100% of the parity corpus.
4. Calibrated probability drift stays within an explicitly recorded tolerance; the initial FP16
   ceiling is `0.02`, matching the existing MLX validation ceiling.
5. The Swift public result matches the reference schema and four-decimal rounding rules.
6. A physical iPhone completes inference in airplane mode without Python, a server, or runtime
   downloads.
7. Load time, first and warm latency, peak memory, model size, and sustained-run thermals are
   recorded before calling the prototype complete.

## Deferred work

- Android and ONNX Runtime integration.
- Multi-checkpoint routing and simultaneous checkpoint residency.
- Quantization, sparsity, palettization, or model distillation.
- Batched Core ML requests.
- Custom Metal kernels.
- Training, fine-tuning, and adapter support.

