import torch

from conversion.model_wrapper import MobileLayaModel, example_inputs


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        self.seen = tuple(
            value.dtype for value in (input_ids, attention_mask, marker_pos, marker_mask, qtype)
        )
        return marker_pos.float(), torch.stack([qtype.float(), qtype.float()], dim=-1)


def test_mobile_wrapper_exposes_int32_and_casts_for_pytorch():
    model = RecordingModel()
    wrapper = MobileLayaModel(model)
    inputs = example_inputs(sequence_length=32, marker_slots=8)
    logits, action_logits = wrapper(*inputs)
    assert all(value.dtype == torch.int32 for value in inputs)
    assert model.seen == (
        torch.int64,
        torch.int64,
        torch.int64,
        torch.bool,
        torch.int64,
    )
    assert logits.shape == (1, 8)
    assert action_logits.shape == (1, 2)


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
