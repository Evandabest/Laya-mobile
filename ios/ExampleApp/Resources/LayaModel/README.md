# Local model bundle

Generate the ignored `Generated/` directory before running the example app:

```bash
.venv-coreml/bin/python -m conversion.export_coreml \
  --output ios/ExampleApp/Resources/LayaModel/Generated
```

Xcode packages that directory as an app resource. It contains the Core ML model, tokenizer, and
calibration configuration; the application performs no model or tokenizer downloads at runtime.
