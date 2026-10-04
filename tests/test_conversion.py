import torch

from conversion.model_wrapper import StaticDecisionLayer, example_inputs
from conversion.validate_coreml import compare_row


def test_static_decision_layer_matches_pytorch_encoder_layer():
    torch.manual_seed(7)
    reference = torch.nn.TransformerEncoderLayer(
        d_model=16,
        nhead=4,
        dim_feedforward=64,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    ).eval()
    converted = StaticDecisionLayer(reference, sequence_length=12).eval()
    hidden = torch.randn(1, 12, 16)
    padding = torch.tensor([[False] * 9 + [True] * 3])
    with torch.inference_mode():
        expected = reference(hidden, src_key_padding_mask=padding)
        actual = converted(hidden, src_key_padding_mask=padding)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_mobile_example_inputs_are_deterministic_and_static():
    first = example_inputs(sequence_length=512, marker_slots=48)
    second = example_inputs(sequence_length=512, marker_slots=48)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert [tuple(value.shape) for value in first] == [
        (1, 512),
        (1, 512),
        (1, 48),
        (1, 48),
        (1,),
    ]


def test_coreml_comparison_uses_calibrated_probabilities():
    row = {
        "question_id": "route",
        "option_count": 3,
        "logits": [4.0, 2.0, 1.0, -10000.0],
        "action_logits": [10.0, -10.0],
        "probabilities": [0.8437947344813395, 0.11419519938459449, 0.04201006613406605],
    }
    result = compare_row(
        row,
        logits=[4.001, 2.0, 1.0, -10000.0],
        action_logits=[10.001, -10.0],
        probability_tolerance=0.02,
    )
    assert result["decision_match"]
    assert result["max_probability_error"] < 0.001
