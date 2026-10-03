import hashlib

import pytest

from parity_tests.cases import parity_cases
from parity_tests.validate import MARKER_SLOTS, SEQUENCE_LENGTH, validate_bundle


def test_mobile_case_names_and_question_order_are_stable():
    cases = parity_cases()
    names = [case["name"] for case in cases]
    assert len(names) == len(set(names))
    assert {
        "typed_email",
        "empty_state",
        "unicode_mixed",
        "maximum_context_string",
        "maximum_context_conversation",
    } <= set(names)
    assert list(cases[0]["questions"]) == ["department", "urgency", "refund"]


def test_case_definitions_are_fresh_copies():
    first, second = parity_cases(), parity_cases()
    first[0]["questions"]["department"]["criteria"]["new"] = "mutation"
    assert "new" not in second[0]["questions"]["department"]["criteria"]


def test_structural_fixture_validator_accepts_complete_bundle():
    option_count = 2
    case = {
        "name": "synthetic",
        "state": "hello",
        "questions": {"q": {"type": "choice"}},
        "rows": [
            {
                "question_id": "q",
                "input_ids": [0] * SEQUENCE_LENGTH,
                "attention_mask": [False] * SEQUENCE_LENGTH,
                "marker_pos": [0] * MARKER_SLOTS,
                "marker_mask": [True] * option_count + [False] * (MARKER_SLOTS - option_count),
                "qtype": 0,
                "option_count": option_count,
                "logits": [0.0] * MARKER_SLOTS,
                "action_logits": [0.0, 0.0],
                "probabilities": [0.5, 0.5],
            }
        ],
        "result": {},
    }
    bundle = {
        "schema_version": 1,
        "profile": "laya-ios-english-v1",
        "generated_at": "2026-10-03T00:00:00+00:00",
        "source": {
            "upstream_commit": "a" * 40,
            "checkpoint": "test/model",
            "checkpoint_revision": "b" * 40,
        },
        "files": {"model": hashlib.sha256(b"model").hexdigest()},
        "model_io": {
            "batch_size": 1,
            "sequence_length": SEQUENCE_LENGTH,
            "marker_slots": MARKER_SLOTS,
            "probability_tolerance": 0.02,
        },
        "cases": [case],
    }
    assert validate_bundle(bundle) is bundle


def test_structural_fixture_validator_rejects_tensor_length():
    with pytest.raises(ValueError, match="input_ids"):
        bundle = {
            "schema_version": 1,
            "profile": "laya-ios-english-v1",
            "source": {
                "upstream_commit": "a" * 40,
                "checkpoint": "test/model",
                "checkpoint_revision": "b" * 40,
            },
            "files": {"model": "0" * 64},
            "model_io": {
                "batch_size": 1,
                "sequence_length": SEQUENCE_LENGTH,
                "marker_slots": MARKER_SLOTS,
            },
            "cases": [
                {
                    "name": "bad",
                    "questions": {"q": {}},
                    "rows": [
                        {
                            "question_id": "q",
                            "input_ids": [],
                            "attention_mask": [False] * SEQUENCE_LENGTH,
                            "marker_pos": [0] * MARKER_SLOTS,
                            "marker_mask": [True] + [False] * (MARKER_SLOTS - 1),
                            "logits": [0.0] * MARKER_SLOTS,
                            "probabilities": [1.0],
                            "option_count": 1,
                        }
                    ],
                    "result": {},
                }
            ],
        }
        validate_bundle(bundle)
