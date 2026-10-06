"""Export the pinned English Laya checkpoint directly from PyTorch to Core ML."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    MARKER_SLOTS,
    PROFILE,
    SEQUENCE_LENGTH,
    UPSTREAM_COMMIT,
    load_upstream_agent,
    sha256,
    validate_checkpoint,
)
from .model_wrapper import MobileLayaModel, example_inputs


def compile_model_package(model_path, output):
    """Compile an mlpackage once so applications can bundle the resulting mlmodelc."""
    compiler = shutil.which("xcrun")
    if compiler is None:
        raise RuntimeError("Compiling Core ML models requires Xcode's xcrun command")

    subprocess.run(
        [compiler, "coremlcompiler", "compile", str(model_path), str(output)],
        check=True,
    )
    compiled_path = output / f"{model_path.stem}.mlmodelc"
    if not compiled_path.is_dir():
        raise RuntimeError(f"Core ML compiler did not create {compiled_path}")
    shutil.rmtree(model_path)
    return compiled_path


def export(checkpoint, upstream, output, *, target="iOS18", compile_model=False):
    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError(
            "Core ML export requires the conversion extra: uv sync --extra conversion"
        ) from error

    checkpoint, config, files = validate_checkpoint(checkpoint)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    target_value = getattr(ct.target, target, None)
    if target_value is None:
        raise ValueError(
            f"coremltools {ct.__version__} does not support deployment target {target}"
        )

    agent = load_upstream_agent(upstream, checkpoint, "cpu")
    # Core ML Tools 9 cannot lower the shape-to-integer helpers emitted by the SDPA integration
    # for this ModernBERT graph. Eager attention is mathematically the same QK^T/softmax/V
    # operation with the same masks and rotary embeddings, but lowers to ordinary matmul and
    # softmax operations. The pinned reference and parity fixtures remain SDPA-based.
    agent.model.encoder.config._attn_implementation = "eager"
    wrapped = MobileLayaModel(agent.model).eval()
    inputs = example_inputs(sequence_length=SEQUENCE_LENGTH, marker_slots=MARKER_SLOTS)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, inputs, strict=False, check_trace=False)
        traced = torch.jit.freeze(traced.eval())

    names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    shapes = (
        (1, SEQUENCE_LENGTH),
        (1, SEQUENCE_LENGTH),
        (1, MARKER_SLOTS),
        (1, MARKER_SLOTS),
        (1,),
    )
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=target_value,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        inputs=[
            ct.TensorType(name=name, shape=shape, dtype=np.int32)
            for name, shape in zip(names, shapes)
        ],
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="action_logits")],
    )
    converted.author = "Laya Mobile contributors"
    converted.license = "Apache-2.0; checkpoint licensing and attribution are retained in NOTICE"
    converted.short_description = "Laya typed-decision model for fully offline iOS inference"
    converted.user_defined_metadata.update(
        {
            "com.evandabest.laya.profile": PROFILE,
            "com.evandabest.laya.checkpoint": CHECKPOINT_ID,
            "com.evandabest.laya.checkpoint_revision": CHECKPOINT_REVISION,
            "com.evandabest.laya.upstream_commit": UPSTREAM_COMMIT,
            "com.evandabest.laya.precision": "float16",
        }
    )

    output.mkdir(parents=True)
    try:
        model_path = output / "laya.mlpackage"
        converted.save(str(model_path))
        if compile_model:
            model_path = compile_model_package(model_path, output)
        shutil.copytree(checkpoint / "tokenizer", output / "tokenizer")
        shutil.copy2(checkpoint / "rl_agent_config.json", output / "rl_agent_config.json")
        (output / "encoder").mkdir()
        shutil.copy2(checkpoint / "encoder/config.json", output / "encoder/config.json")
        manifest = {
            "schema_version": 1,
            "profile": PROFILE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "checkpoint": CHECKPOINT_ID,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "upstream_commit": UPSTREAM_COMMIT,
            },
            "conversion": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "coremltools": ct.__version__,
                "deployment_target": target,
                "compute_precision": "float16",
                "packaging": model_path.suffix.removeprefix("."),
            },
            "model_io": {
                "batch_size": 1,
                "sequence_length": SEQUENCE_LENGTH,
                "marker_slots": MARKER_SLOTS,
                "inputs": list(names),
                "outputs": ["logits", "action_logits"],
            },
            "checkpoint_files": {name: sha256(path) for name, path in files.items()},
            "agent_config": config,
            "validation": None,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/laya"))
    parser.add_argument("--upstream", type=Path, default=Path(".upstream"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/coreml/laya-ios-english-v1"))
    parser.add_argument("--target", default="iOS18")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="replace the source mlpackage with a precompiled mlmodelc for app packaging",
    )
    args = parser.parse_args()
    try:
        result = export(
            args.checkpoint,
            args.upstream,
            args.output,
            target=args.target,
            compile_model=args.compile_model,
        )
    except Exception as error:
        print(f"Core ML export failed: {error}", file=sys.stderr)
        raise
    print(f"wrote {result}")


if __name__ == "__main__":
    main()
