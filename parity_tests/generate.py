"""Generate portable, fixed-shape mobile fixtures from a pinned real Laya checkpoint."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from laya_mlx.agent import Agent
from laya_mlx.common import (
    QTYPES,
    answer_confidence,
    build_sequence,
    clamp_temperature,
    collapsed_options,
    confidence_from_probs,
    render_options,
    serialize_state,
    temp_bucket,
)
from laya_mlx.tokenizer import Tokenizer

from .cases import parity_cases
from .validate import MARKER_SLOTS, PROFILE, SCHEMA_VERSION, SEQUENCE_LENGTH, validate_bundle

UPSTREAM_COMMIT = "573e5b62696ba441230cd6be71d593331b5d23af"
CHECKPOINT = "convaiinnovations/laya"
CHECKPOINT_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
PROBABILITY_TOLERANCE = 0.02


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.exp(values - values.max())
    return values / values.sum()


def load_reference(upstream, checkpoint, device):
    upstream = Path(upstream).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {commit}")
    sys.path.insert(0, str(upstream))
    import laya

    return laya.load(str(checkpoint), device=device)


def prepare_question(tok, cfg, state, question_id, definition):
    question = Agent._question(question_id, definition)
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False)[
        "input_ids"
    ]
    ids, markers, option_stats, state_stats = build_sequence(
        tok,
        state,
        question,
        cfg.get("max_len", SEQUENCE_LENGTH),
        cfg.get("head_max_len", 192),
        truncate_left=isinstance(state, list),
        state_ids=state_ids,
        return_stats=True,
        return_truncation_stats=True,
    )
    if cfg.get("max_len", SEQUENCE_LENGTH) != SEQUENCE_LENGTH:
        raise ValueError("The iOS English profile requires max_len=512")
    if len(markers) != len(render_options(question)):
        raise ValueError(f"Question {question_id!r} exceeds the token budget")
    if not 1 <= len(markers) <= MARKER_SLOTS:
        raise ValueError(f"Question {question_id!r} has unsupported option count {len(markers)}")

    pad = tok.pad_token_id
    input_ids = np.full((1, SEQUENCE_LENGTH), pad, dtype=np.int64)
    attention_mask = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int64)
    marker_pos = np.zeros((1, MARKER_SLOTS), dtype=np.int64)
    marker_mask = np.zeros((1, MARKER_SLOTS), dtype=bool)
    input_ids[0, : len(ids)] = ids
    attention_mask[0, : len(ids)] = 1
    marker_pos[0, : len(markers)] = markers
    marker_mask[0, : len(markers)] = True
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": np.asarray([QTYPES[question["t"]]], dtype=np.int64),
    }
    item = {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[question["t"]],
        "options": option_stats,
        "state_stats": state_stats,
    }
    return question, item, batch


def decode_answer(question, item, logits, action_logits, temperatures, by_options):
    count, question_type = len(item["markers"]), item["qtype"]
    scale = by_options.get(temp_bucket(question_type, count), temperatures[question_type])
    probabilities = softmax(np.asarray(logits)[:count] / scale)
    action = softmax(action_logits)
    answer = {
        "type": question["t"],
        "confidence": round(confidence_from_probs(probabilities, count), 4),
        "answer_confidence": round(answer_confidence(probabilities, count), 4),
        "action": {"act_probability": round(float(action[0]), 4)},
    }
    if question["t"] == "choice":
        labels = list(question["crit"])
        answer.update(
            choice=labels[int(probabilities.argmax())],
            probabilities={
                label: round(float(value), 4) for label, value in zip(labels, probabilities)
            },
        )
    elif question["t"] == "score":
        answer.update(
            score=round(float((np.arange(count) * probabilities).sum()), 4),
            legend={str(index): value for index, value in enumerate(question["crit"])},
            probabilities={
                str(index): round(float(value), 4) for index, value in enumerate(probabilities)
            },
        )
    else:
        answer.update(
            noul=round(float(probabilities[1]), 4),
            confidence=round(max(float(probabilities[1]), 1.0 - float(probabilities[1])), 4),
        )
    return answer, probabilities


def generate(upstream, checkpoint, device):
    import torch

    checkpoint = Path(checkpoint).resolve()
    reference = load_reference(upstream, checkpoint, device)
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    tok = Tokenizer(checkpoint / "tokenizer")
    temperatures = [clamp_temperature(value) for value in cfg.get("temperature", [1, 1, 1])]
    by_options = {
        key: clamp_temperature(value)
        for key, value in cfg.get("temperature_by_options", {}).items()
    }
    output_cases = []
    for case in parity_cases():
        answers, rows, items = {}, [], []
        for question_id, definition in case["questions"].items():
            question, item, arrays = prepare_question(
                tok, cfg, case["state"], question_id, definition
            )
            tensors = {
                key: torch.from_numpy(value).to(reference.device) for key, value in arrays.items()
            }
            with torch.inference_mode():
                logits, action_logits = reference.model(**tensors)
            logits = logits[0].detach().cpu().float().numpy()
            action_logits = action_logits[0].detach().cpu().float().numpy()
            answer, probabilities = decode_answer(
                question, item, logits, action_logits, temperatures, by_options
            )
            answers[question_id] = answer
            items.append(item)
            rows.append(
                {
                    "question_id": question_id,
                    "input_ids": arrays["input_ids"][0].astype(np.int32).tolist(),
                    "attention_mask": arrays["attention_mask"][0].astype(bool).tolist(),
                    "marker_pos": arrays["marker_pos"][0].astype(np.int32).tolist(),
                    "marker_mask": arrays["marker_mask"][0].tolist(),
                    "qtype": int(arrays["qtype"][0]),
                    "option_count": len(item["markers"]),
                    "logits": logits.tolist(),
                    "action_logits": action_logits.tolist(),
                    "probabilities": probabilities.tolist(),
                }
            )
        stats = [item["state_stats"] for item in items]
        dropped = max((entry["state_tokens_dropped"] for entry in stats), default=0)
        question_ids = list(case["questions"])
        usage = {
            "input_tokens": sum(len(item["ids"]) for item in items),
            "output_tokens": 0,
            "state_tokens": stats[0]["state_tokens"] if stats else 0,
            "state_tokens_dropped": dropped,
            "truncated": dropped > 0,
            "truncated_questions": [
                question_id
                for question_id, state_stats in zip(question_ids, stats)
                if state_stats["truncated"]
            ],
        }
        collapsed = collapsed_options(question_ids, items)
        if collapsed:
            usage["options"] = collapsed
        output_cases.append(
            {
                **case,
                "rows": rows,
                "result": {"model": "laya-rl-agent", "answers": answers, "usage": usage},
            }
        )

    hashed_files = [
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "behavioral_reference": "laya-mlx-v0.3.0",
            "torch_device": str(reference.device),
        },
        "files": {name: sha256(checkpoint / name) for name in hashed_files},
        "model_io": {
            "batch_size": 1,
            "sequence_length": SEQUENCE_LENGTH,
            "marker_slots": MARKER_SLOTS,
            "probability_tolerance": PROBABILITY_TOLERANCE,
        },
        "cases": output_cases,
    }
    return validate_bundle(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/laya"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/parity/laya-ios-english-v1.json")
    )
    args = parser.parse_args()
    bundle = generate(args.upstream, args.checkpoint, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"wrote {args.output} ({len(bundle['cases'])} cases)")


if __name__ == "__main__":
    main()
