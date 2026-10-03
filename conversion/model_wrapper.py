"""Static mobile inference boundary around the pinned PyTorch Laya model."""

import torch


class MobileLayaModel(torch.nn.Module):
    """Expose Core ML-friendly int32 inputs and stable output names.

    Core ML does not expose int64 multi-arrays. The wrapper accepts int32 tensors and performs
    the exact casts expected by the PyTorch embedding and gather operations inside the graph.
    Marker and attention masks are also int32 at the public boundary to keep Swift tensor
    construction uniform.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, action_logits = self.model(
            input_ids=input_ids.to(torch.int64),
            attention_mask=attention_mask.to(torch.int64),
            marker_pos=marker_pos.to(torch.int64),
            marker_mask=marker_mask.to(torch.bool),
            qtype=qtype.to(torch.int64),
        )
        return logits, action_logits


def example_inputs(device="cpu", sequence_length=512, marker_slots=48):
    """Deterministic, fully valid inputs used only to capture the fixed-shape graph."""
    input_ids = torch.zeros((1, sequence_length), dtype=torch.int32, device=device)
    attention_mask = torch.ones((1, sequence_length), dtype=torch.int32, device=device)
    marker_pos = torch.arange(marker_slots, dtype=torch.int32, device=device)[None, :]
    marker_mask = torch.ones((1, marker_slots), dtype=torch.int32, device=device)
    qtype = torch.zeros((1,), dtype=torch.int32, device=device)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype
