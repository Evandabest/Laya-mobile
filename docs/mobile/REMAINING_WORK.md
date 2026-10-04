# Remaining work after the first Core ML prototype

## iOS limitations

- Physical-device execution is pending. The iPhone 17 Pro pass must record load, first/warm
  inference, peak memory, sustained thermals, and Core ML compute-unit placement; Simulator
  execution does not answer those hardware questions.
- The public Swift API accepts text, ordered JSON objects, and chronological conversation lists.
  It serializes structured state with upstream-compatible JSON formatting and keeps the newest
  conversation tokens when the context is truncated.
- Choice and rubric criteria currently use string values. The reference accepts structured JSON
  criterion and instruction values, which require the same canonical renderer in Swift.
- The first profile is fixed to batch 1, 512 sequence tokens, 48 marker slots, English, FP16, and
  iOS 18 or newer. Dynamic contexts and the multilingual/typed-decision checkpoints are not yet
  exported.
- The app expects a generated model bundle and does not yet implement optional downloaded model
  bundles, integrity verification, or model-version migration.

## Android follow-up

Android work starts from the same pinned fixture contract rather than translating Swift code:

1. Add and validate an ONNX export against all 44 reference rows.
2. Package ONNX Runtime Mobile with CPU execution first.
3. Load the same serialized tokenizer assets through an exact Kotlin-compatible tokenizer.
4. Port the shared rendering, sequence, calibration, confidence, and typed-result behavior.
5. Run the portable tensor fixtures and decoded-result fixtures on Android instrumentation tests.
6. Only after parity, measure NNAPI/XNNPACK providers and FP16/INT8 variants on flagship and
   midrange devices.

No Android optimization should change model semantics or become a second source of truth for
question formatting.
