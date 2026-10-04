"""Validate a converted Core ML package against the checked-in parity fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np

from parity_tests.validate import validate_bundle


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def compare_row(row, logits, action_logits, probability_tolerance):
    count = row["option_count"]
    expected_probabilities = np.asarray(row["probabilities"], dtype=np.float64)
    expected_logits = np.asarray(row["logits"], dtype=np.float64)
    expected_action_logits = np.asarray(row["action_logits"], dtype=np.float64)

    # The fixture probabilities already include the checkpoint's calibrated temperature. Recover
    # that scale from the non-constant logits and probabilities so this validator stays driven by
    # the portable fixture rather than importing either Python implementation's decoder.
    actual_logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    actual_action_logits = np.asarray(action_logits, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(actual_logits)) or not np.all(np.isfinite(actual_action_logits)):
        raise AssertionError(f"{row['question_id']}: Core ML returned non-finite outputs")
    pairs = [
        (i, j)
        for i in range(count)
        for j in range(i + 1, count)
        if abs(expected_logits[i] - expected_logits[j]) > 1e-8
        and expected_probabilities[i] > 0
        and expected_probabilities[j] > 0
    ]
    if pairs:
        i, j = pairs[0]
        temperature = (expected_logits[i] - expected_logits[j]) / np.log(
            expected_probabilities[i] / expected_probabilities[j]
        )
    else:
        temperature = 1.0
    probabilities = softmax(actual_logits[:count] / temperature)
    action_probabilities = softmax(actual_action_logits)
    expected_action_probabilities = softmax(expected_action_logits)
    probability_error = float(np.max(np.abs(probabilities - expected_probabilities)))
    action_probability_error = float(
        np.max(np.abs(action_probabilities - expected_action_probabilities))
    )
    decision_match = int(np.argmax(probabilities)) == int(np.argmax(expected_probabilities))
    if (
        not decision_match
        or probability_error > probability_tolerance
        or action_probability_error > probability_tolerance
    ):
        raise AssertionError(
            f"{row['question_id']}: decision_match={decision_match}, "
            f"max_probability_error={probability_error:.6g}, "
            f"action_probability_error={action_probability_error:.6g}"
        )
    return {
        "decision_match": decision_match,
        "max_probability_error": probability_error,
        "max_action_probability_error": action_probability_error,
        "max_logit_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "max_action_logit_error": float(
            np.max(np.abs(actual_action_logits - expected_action_logits))
        ),
    }


def validate(model_path, fixture_path, limit=None):
    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError("Core ML validation requires the conversion extra") from error

    bundle = validate_bundle(json.loads(Path(fixture_path).read_text()))
    tolerance = bundle["model_io"]["probability_tolerance"]
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
    metrics = []
    for case in bundle["cases"]:
        for row in case["rows"]:
            if limit is not None and len(metrics) >= limit:
                break
            inputs = {
                "input_ids": np.asarray([row["input_ids"]], dtype=np.int32),
                "attention_mask": np.asarray([row["attention_mask"]], dtype=np.int32),
                "marker_pos": np.asarray([row["marker_pos"]], dtype=np.int32),
                "marker_mask": np.asarray([row["marker_mask"]], dtype=np.int32),
                "qtype": np.asarray([row["qtype"]], dtype=np.int32),
            }
            outputs = model.predict(inputs)
            metrics.append(compare_row(row, outputs["logits"], outputs["action_logits"], tolerance))
        if limit is not None and len(metrics) >= limit:
            break
    return {
        "rows": len(metrics),
        "decision_agreement": sum(item["decision_match"] for item in metrics) / len(metrics),
        "max_probability_error": max(item["max_probability_error"] for item in metrics),
        "max_action_probability_error": max(
            item["max_action_probability_error"] for item in metrics
        ),
        "max_logit_error": max(item["max_logit_error"] for item in metrics),
        "max_action_logit_error": max(item["max_action_logit_error"] for item in metrics),
        "probability_tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/coreml/laya-ios-english-v1/laya.mlpackage")
    )
    parser.add_argument(
        "--fixture", type=Path, default=Path("parity_tests/fixtures/laya-ios-english-v1.json")
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(validate(args.model, args.fixture, args.limit), indent=2))


if __name__ == "__main__":
    main()
