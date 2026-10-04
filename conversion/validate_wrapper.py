"""Check that the Core ML export wrapper preserves the reference PyTorch graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from parity_tests.validate import validate_bundle

from .common import load_upstream_agent
from .model_wrapper import MobileLayaModel

LOGIT_TOLERANCE = 1e-4
ACTION_LOGIT_TOLERANCE = 5e-3


def validate(upstream, checkpoint, fixture, device, limit=None):
    bundle = validate_bundle(json.loads(Path(fixture).read_text()))
    agent = load_upstream_agent(upstream, checkpoint, device)
    # Compute the reference outputs before constructing the wrapper.  The wrapper intentionally
    # replaces encoder attention modules in-place, so constructing it first would silently turn
    # the reference into the graph under test.
    reference_rows = []
    max_logit_error = 0.0
    max_action_error = 0.0
    seen = 0
    with torch.inference_mode():
        for case in bundle["cases"]:
            for row in case["rows"]:
                if limit is not None and seen >= limit:
                    break
                arrays = {
                    "input_ids": np.asarray([row["input_ids"]], dtype=np.int32),
                    "attention_mask": np.asarray([row["attention_mask"]], dtype=np.int32),
                    "marker_pos": np.asarray([row["marker_pos"]], dtype=np.int32),
                    "marker_mask": np.asarray([row["marker_mask"]], dtype=np.int32),
                    "qtype": np.asarray([row["qtype"]], dtype=np.int32),
                }
                tensors = {
                    key: torch.from_numpy(value).to(agent.device) for key, value in arrays.items()
                }
                expected = agent.model(
                    input_ids=tensors["input_ids"].to(torch.int64),
                    attention_mask=tensors["attention_mask"].to(torch.int64),
                    marker_pos=tensors["marker_pos"].to(torch.int64),
                    marker_mask=tensors["marker_mask"].bool(),
                    qtype=tensors["qtype"].to(torch.int64),
                )
                reference_rows.append((tensors, expected))
                seen += 1
            if limit is not None and seen >= limit:
                break
    wrapper = MobileLayaModel(agent.model).eval().to(agent.device)
    with torch.inference_mode():
        for tensors, expected in reference_rows:
            actual = wrapper(**tensors)
            max_logit_error = max(
                max_logit_error,
                float(np.max(np.abs(actual[0].cpu().numpy() - expected[0].cpu().numpy()))),
            )
            max_action_error = max(
                max_action_error,
                float(np.max(np.abs(actual[1].cpu().numpy() - expected[1].cpu().numpy()))),
            )
    if max_logit_error > LOGIT_TOLERANCE or max_action_error > ACTION_LOGIT_TOLERANCE:
        raise AssertionError(
            f"wrapper drift: max logits {max_logit_error:.6g}, "
            f"max action logits {max_action_error:.6g}"
        )
    return max_logit_error, max_action_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/laya"))
    parser.add_argument(
        "--fixture", type=Path, default=Path("parity_tests/fixtures/laya-ios-english-v1.json")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    errors = validate(args.upstream, args.checkpoint, args.fixture, args.device, args.limit)
    print(f"wrapper parity passed: logits={errors[0]:.6g}, action_logits={errors[1]:.6g}")


if __name__ == "__main__":
    main()
