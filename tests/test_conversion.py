import torch

from conversion.model_wrapper import StaticDecisionLayer, example_inputs


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
