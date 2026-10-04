# Laya iOS example

1. Generate the model bundle as described in `Resources/LayaModel/README.md`. The generated model
   is intentionally ignored by Git, so this step is required after every fresh checkout.
2. Run `xcodegen generate` in this directory after changing `project.yml`.
3. Open `LayaExample.xcodeproj`, select an iPhone target, and run.

The screen includes examples for plain text, ordered JSON objects, chronological conversations,
Unicode, empty input, and truncation. It exercises choice, score, boolean/noul, custom boolean
labels, and multiple questions in one request through the local `LayaMobile` Swift package.

Each result shows the selected value, the complete option distribution, confidence,
answer-confidence, action probability, and its individual latency. The summary reports total
request latency, token usage, truncation, and the active Core ML backend. Inference runs away from
the UI actor so progress remains visible during multi-question requests. No inference-time network
request is made.
