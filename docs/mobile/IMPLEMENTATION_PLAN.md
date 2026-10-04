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
- [x] Generate exact tokenizer and padded tensor inputs from the pinned reference.
- [x] Record raw logits, calibrated probabilities, decoded results, and file hashes.
- [x] Add schema and determinism tests.

## 3. Export backends

- [x] Add a conversion-time PyTorch wrapper with five inputs and two outputs.
- [x] Export the pinned English checkpoint directly to an FP16 Core ML ML Program.
- [x] Validate Core ML numerics and decisions against fixtures on macOS.
- [x] Record unsupported or CPU-fallback operations and the minimum deployment target.
- [ ] Add an ONNX export and validator as the later Android source artifact.

## 4. Native Swift runtime

- [x] Create the `LayaMobile` Swift package.
- [x] Add strongly typed choice, score, boolean/noul, result, usage, and error types.
- [ ] Port validation, serialization, rendering, truncation, and tensor construction.
- [x] Load the tokenizer entirely from bundled local assets.
- [x] Prove exact tokenizer parity, including Unicode and adversarial whitespace.
- [x] Load and invoke the compiled Core ML model.
- [x] Port calibration, decoding, confidence, and output formatting.
- [x] Support multiple questions sequentially behind one `predict` call.
- [x] Pass fixture-driven XCTest parity tests.

## 5. Example application

- [x] Create a minimal iOS application using the package.
- [x] Support user text and sample choice, score, and boolean questions.
- [x] Display decision, probability/confidence values, latency, truncation, and backend.
- [x] Make offline execution evident and testable.

## 6. Physical-device validation

- [ ] Run on an iPhone 17 Pro in airplane mode.
- [ ] Measure model load, first inference, and warm inference latency.
- [ ] Measure peak memory at short and 512-token contexts.
- [ ] Run a sustained inference loop and record thermal behavior.
- [ ] Publish a reproducible device report.

## 7. Prototype completion gate

- [x] Load a real pinned Laya checkpoint with no Python runtime.
- [x] Match tokenizer inputs exactly.
- [x] Match every fixture decision with acceptable confidence drift.
- [ ] Demonstrate all primary typed decisions in the iOS app.
- [x] Confirm no server or inference-time network dependency.
- [x] Document remaining limitations and the Android follow-up plan.
