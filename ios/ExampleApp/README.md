# Laya iOS example

1. Generate the model bundle as described in `Resources/LayaModel/README.md`. The generated model
   is intentionally ignored by Git, so this step is required after every fresh checkout.
2. Run `xcodegen generate` in this directory after changing `project.yml`.
3. Open `LayaExample.xcodeproj`, select an iPhone target, and run.

The screen accepts user text and exercises choice, score, and boolean decisions through the local
`LayaMobile` Swift package. It displays confidence, inference latency, truncation, and the active
Core ML backend. No inference-time network request is made.
