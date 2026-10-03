"""Dependency-free structural validation for generated mobile fixture bundles."""

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = 1
PROFILE = "laya-ios-english-v1"
SEQUENCE_LENGTH = 512
MARKER_SLOTS = 48


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_bundle(bundle):
    _require(isinstance(bundle, dict), "fixture root must be an object")
    _require(bundle.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(bundle.get("profile") == PROFILE, "unexpected fixture profile")
    source = bundle.get("source")
    _require(isinstance(source, dict), "source must be an object")
    for key in ("upstream_commit", "checkpoint_revision"):
        value = source.get(key)
        _require(isinstance(value, str) and len(value) == 40, f"invalid source.{key}")
    _require(isinstance(source.get("checkpoint"), str), "invalid source.checkpoint")

    model_io = bundle.get("model_io")
    _require(isinstance(model_io, dict), "model_io must be an object")
    _require(model_io.get("batch_size") == 1, "fixtures must use batch size one")
    _require(model_io.get("sequence_length") == SEQUENCE_LENGTH, "unexpected sequence length")
    _require(model_io.get("marker_slots") == MARKER_SLOTS, "unexpected marker slot count")

    files = bundle.get("files")
    _require(isinstance(files, dict) and files, "files must contain artifact hashes")
    for name, digest in files.items():
        _require(isinstance(name, str) and name, "invalid hashed file name")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"invalid SHA-256 for {name}",
        )

    cases = bundle.get("cases")
    _require(isinstance(cases, list) and cases, "cases must be a nonempty array")
    names = [case.get("name") for case in cases if isinstance(case, dict)]
    _require(
        len(names) == len(cases) and len(set(names)) == len(names), "case names must be unique"
    )
    for case in cases:
        questions = case.get("questions")
        rows = case.get("rows")
        _require(isinstance(questions, dict) and questions, f"{case['name']}: missing questions")
        _require(isinstance(rows, list), f"{case['name']}: rows must be an array")
        _require(len(rows) == len(questions), f"{case['name']}: row count differs from questions")
        _require(isinstance(case.get("result"), dict), f"{case['name']}: missing result")
        for row, question_id in zip(rows, questions):
            prefix = f"{case['name']}/{question_id}"
            _require(row.get("question_id") == question_id, f"{prefix}: row order mismatch")
            for key in ("input_ids", "attention_mask"):
                _require(len(row.get(key, [])) == SEQUENCE_LENGTH, f"{prefix}: invalid {key}")
            for key in ("marker_pos", "marker_mask", "logits"):
                _require(len(row.get(key, [])) == MARKER_SLOTS, f"{prefix}: invalid {key}")
            option_count = row.get("option_count")
            _require(
                isinstance(option_count, int) and 1 <= option_count <= MARKER_SLOTS,
                f"{prefix}: invalid option_count",
            )
            _require(
                sum(bool(value) for value in row["marker_mask"]) == option_count,
                f"{prefix}: marker mask differs from option_count",
            )
            _require(
                len(row.get("probabilities", [])) == option_count,
                f"{prefix}: invalid probabilities",
            )
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    validate_bundle(json.loads(args.fixture.read_text()))
    print(f"validated {args.fixture}")


if __name__ == "__main__":
    main()
