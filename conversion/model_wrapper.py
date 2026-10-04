"""Static mobile inference boundary around the pinned PyTorch Laya model."""

import torch
import torch.nn.functional as F


class StaticDecisionLayer(torch.nn.Module):
    """Explicit fixed-shape equivalent of nn.TransformerEncoderLayer."""

    def __init__(self, layer, sequence_length):
        super().__init__()
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.in_proj_weight = layer.self_attn.in_proj_weight
        self.in_proj_bias = layer.self_attn.in_proj_bias
        self.out_proj = layer.self_attn.out_proj
        self.num_heads = layer.self_attn.num_heads
        self.head_dim = layer.self_attn.head_dim
        self.sequence_length = sequence_length

    def forward(self, hidden_states, src_key_padding_mask=None):
        x = self.norm1(hidden_states)
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(
            1, self.sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) * (self.head_dim**-0.5)
        if src_key_padding_mask is not None:
            blocked = torch.finfo(scores.dtype).min
            scores = scores + src_key_padding_mask[:, None, None, :].to(scores.dtype) * blocked
        probs = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probs, value).transpose(1, 2).reshape(1, self.sequence_length, -1)
        x = hidden_states + self.out_proj(attended)
        # Laya constructs TransformerEncoderLayer without an activation override, so the
        # decision head uses PyTorch's default ReLU (the scorer and action head use GELU).
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class StaticModernBertAttention(torch.nn.Module):
    """ModernBERT attention with fixed sequence reshapes for Core ML conversion."""

    def __init__(self, attention, cos, sin, sequence_length):
        super().__init__()
        self.Wqkv = attention.Wqkv
        self.Wo = attention.Wo
        self.num_heads = attention.config.num_attention_heads
        self.head_dim = attention.head_dim
        self.sequence_length = sequence_length
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        qkv = self.Wqkv(hidden_states).reshape(
            1, self.sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos = self.cos.unsqueeze(1)
        sin = self.sin.unsqueeze(1)
        half = self.head_dim // 2
        query_rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
        key_rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
        query = query.float() * cos + query_rotated.float() * sin
        key = key.float() * cos + key_rotated.float() * sin
        query = query.to(value.dtype)
        key = key.to(value.dtype)
        scores = torch.matmul(query, key.transpose(2, 3)) * (self.head_dim**-0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(1, self.sequence_length, -1)
        return self.Wo(output), None


class MobileLayaModel(torch.nn.Module):
    """Expose Core ML-friendly int32 inputs and stable output names.

    Core ML does not expose int64 multi-arrays. The wrapper accepts int32 tensors and performs
    the exact casts expected by the PyTorch embedding and gather operations inside the graph.
    Marker and attention masks are also int32 at the public boundary to keep Swift tensor
    construction uniform.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder.eval()
        self.sequence_length = 512
        with torch.no_grad():
            position_ids = torch.arange(self.sequence_length, device="cpu").unsqueeze(0)
            dummy = torch.zeros(
                (1, self.sequence_length, self.encoder.config.hidden_size), dtype=torch.float32
            )
            position_embeddings = {
                kind: self.encoder.rotary_emb(dummy, position_ids, kind)
                for kind in set(self.encoder.config.layer_types)
            }
        for layer in self.encoder.layers:
            cos, sin = position_embeddings[layer.attention_type]
            layer.attn = StaticModernBertAttention(layer.attn, cos, sin, self.sequence_length)
        self.head = model.head
        self.type_emb = model.type_emb
        self.scorer = model.scorer
        self.act_head = model.act_head
        self.local_attention = self.encoder.config.local_attention
        if self.head is not None:
            self.head_layers = torch.nn.ModuleList(
                [StaticDecisionLayer(layer, self.sequence_length) for layer in self.head.layers]
            )
        else:
            self.head_layers = None

    def _attention_masks(self, attention_mask):
        """Build ModernBERT's fixed-shape masks without Transformers' mask callbacks.

        Transformers 5.17 constructs a scalar with ``q_idx.new_ones`` inside its vmap mask
        callback. Core ML Tools 9 does not lower that operator. This is the same mask expressed
        as ordinary tensor operations, matching the MLX implementation and the reference model's
        inclusive local window. The sequence length is static for the first mobile profile.
        """
        valid = attention_mask.to(torch.bool)
        length = attention_mask.shape[1]
        positions = torch.arange(length, device=attention_mask.device)
        radius = self.local_attention // 2
        local = (positions[:, None] - positions[None, :]).abs() <= radius
        full = valid[:, None, None, :]
        sliding = torch.logical_and(
            torch.logical_or(local[None, None, :, :], ~valid[:, None, :, None]), full
        )
        zero = torch.zeros((), dtype=torch.float32, device=attention_mask.device)
        blocked = torch.full((), torch.finfo(torch.float32).min, device=attention_mask.device)
        return {
            "full_attention": torch.where(full, zero, blocked),
            "sliding_attention": torch.where(sliding, zero, blocked),
        }

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        input_ids = input_ids.to(torch.int64)
        attention_mask = attention_mask.to(torch.int64)
        # marker_pos is already int32 at the public boundary. Avoid a redundant traced cast:
        # Core ML Tools 9 mis-types that particular TorchScript cast as fp32 before gather.
        marker_mask = marker_mask.to(torch.bool)
        qtype = qtype.to(torch.int64)
        h = self.encoder(
            input_ids=input_ids,
            attention_mask=self._attention_masks(attention_mask),
        ).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head_layers is not None:
            pad = ~attention_mask.bool()
            for layer in self.head_layers:
                h = layer(h, src_key_padding_mask=pad)
        # The first mobile profile has a fixed batch of one. index_select lowers to Core ML's
        # ordinary gather op, while torch.gather expands the indices across the hidden width and
        # is incorrectly typed as fp32 by the Core ML Tools TorchScript frontend.
        # Preprocessing pads unused marker slots with zero, so all indices are already in range.
        marker_indices = marker_pos[0]
        markers = torch.index_select(h, 1, marker_indices)
        logits = self.scorer(markers).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        features = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], dim=-1)
        pooled = h[:, 0].float()
        action_logits = self.act_head(torch.cat([pooled, features], dim=-1))
        return logits, action_logits


def example_inputs(device="cpu", sequence_length=512, marker_slots=48):
    """Deterministic, fully valid inputs used only to capture the fixed-shape graph."""
    input_ids = torch.zeros((1, sequence_length), dtype=torch.int32, device=device)
    attention_mask = torch.ones((1, sequence_length), dtype=torch.int32, device=device)
    marker_pos = torch.arange(marker_slots, dtype=torch.int32, device=device)[None, :]
    marker_mask = torch.ones((1, marker_slots), dtype=torch.int32, device=device)
    qtype = torch.zeros((1,), dtype=torch.int32, device=device)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype
