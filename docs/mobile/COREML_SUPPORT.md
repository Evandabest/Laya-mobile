# Core ML support boundary

The first profile exports an FP16 ML Program with a minimum deployment target of iOS 18. The
runtime requests `MLComputeUnits.all`, leaving placement across CPU, GPU, and Neural Engine to
Core ML. The model has no custom layers and requires no network or Python runtime.

## Conversion findings

Core ML Tools 9 could not lower three constructs in the unmodified PyTorch graph:

- Transformers' dynamic ModernBERT attention-mask callbacks (`new_ones`, dynamic integer shape
  helpers, and bitwise mask operations).
- PyTorch's fused `_transformer_encoder_layer_fwd` decision-head operation.
- The expanded `torch.gather` marker indices emitted by the TorchScript frontend.

The export boundary replaces those constructs with fixed-shape, mathematically equivalent
matmul/softmax attention, ordinary tensor masks, explicit decision-head layers, and an
`index_select` marker gather. The conversion-time wrapper differs from the FP32 reference by at
most `5.24521e-06` on decision logits and `0.00146484` on action logits in the sampled wrapper
check. The resulting FP16 Core ML model passed all 44 parity rows; see
`parity_tests/reports/laya-ios-english-v1-coreml.json`.

## Placement and fallback status

Conversion and native Core ML execution have been verified on Apple silicon macOS. There are no
unsupported Core ML operations or custom CPU implementations in the exported package. Per-op
CPU/GPU/Neural Engine placement has not yet been measured on the target iPhone 17 Pro, so this
document does not claim that every operation runs on the Neural Engine. That allocation and any
CPU fallback will be recorded during the physical-device milestone.

The current development Mac cannot build or launch the iOS simulator because its installed
CoreSimulator framework is one patch older than Xcode expects and the iOS 26.5 platform component
is absent. This does not affect the successful native Core ML execution or Swift package tests,
but an iPhone build remains part of physical-device validation.
