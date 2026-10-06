# Local model bundle

Generate the ignored `Generated/` directory before running the example app:

```bash
.venv-coreml/bin/python -m conversion.export_coreml \
  --output ios/ExampleApp/Resources/LayaModel/Generated \
  --compile-model
```

The compile flag replaces the source `.mlpackage` with an `.mlmodelc` produced by Xcode's Core ML
compiler. Xcode packages that directory as an app resource, so launch never recompiles the model.
The directory also contains the tokenizer and calibration configuration; the application performs
no model or tokenizer downloads at runtime.
