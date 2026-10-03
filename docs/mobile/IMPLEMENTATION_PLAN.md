# Laya Mobile implementation plan

This checklist is the task ledger for the iOS-first prototype. A checked item must be backed by
source, tests, generated validation output, or a physical-device result. Generated checkpoints and
model packages are stored under ignored `models/` and `artifacts/` directories.

## 1. Freeze the contract

- [x] Document the inference pipeline and backend boundary.
- [x] Pin the upstream implementation and English checkpoint revisions.
- [x] Define fixed first-prototype tensor shapes and parity tolerances.
- [x] Add a machine-readable compatibility manifest schema.

## 2. Portable parity fixtures

- [x] Define a versioned fixture schema.
- [x] Add representative request cases covering every typed output and edge case.
- [ ] Generate exact tokenizer and padded tensor inputs from the pinned reference.
- [ ] Record raw logits, calibrated probabilities, decoded results, and file hashes.
- [ ] Add schema and determinism tests.

## 3. Export backends

- [ ] Add a conversion-time PyTorch wrapper with five inputs and two outputs.
- [ ] Export the pinned English checkpoint directly to an FP16 Core ML ML Program.
- [ ] Validate Core ML numerics and decisions against fixtures on macOS.
- [ ] Record unsupported or CPU-fallback operations and the minimum deployment target.
- [ ] Add an ONNX export and validator as the later Android source artifact.

## 4. Native Swift runtime

- [ ] Create the `LayaMobile` Swift package.
- [ ] Add strongly typed choice, score, boolean/noul, result, usage, and error types.
- [ ] Port validation, serialization, rendering, truncation, and tensor construction.
- [ ] Load the tokenizer entirely from bundled local assets.
- [ ] Prove exact tokenizer parity, including Unicode and adversarial whitespace.
- [ ] Load and invoke the compiled Core ML model.
- [ ] Port calibration, decoding, confidence, and output formatting.
- [ ] Support multiple questions sequentially behind one `predict` call.
- [ ] Pass fixture-driven XCTest parity tests.

## 5. Example application

- [ ] Create a minimal iOS application using the package.
- [ ] Support user text and sample choice, score, and boolean questions.
- [ ] Display decision, probability/confidence values, latency, truncation, and backend.
- [ ] Make offline execution evident and testable.

## 6. Physical-device validation

- [ ] Run on an iPhone 17 Pro in airplane mode.
- [ ] Measure model load, first inference, and warm inference latency.
- [ ] Measure peak memory at short and 512-token contexts.
- [ ] Run a sustained inference loop and record thermal behavior.
- [ ] Publish a reproducible device report.

## 7. Prototype completion gate

- [ ] Load a real pinned Laya checkpoint with no Python runtime.
- [ ] Match tokenizer inputs exactly.
- [ ] Match every fixture decision with acceptable confidence drift.
- [ ] Demonstrate all primary typed decisions in the iOS app.
- [ ] Confirm no server or inference-time network dependency.
- [ ] Document remaining limitations and the Android follow-up plan.
