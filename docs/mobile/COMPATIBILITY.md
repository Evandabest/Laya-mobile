# Compatibility policy

Laya Mobile treats behavior as a versioned contract. A checkpoint is compatible only when its
weights, encoder configuration, agent configuration, tokenizer assets, preprocessing rules, and
decoding rules were validated together.

## First prototype profile

| Field | Value |
| --- | --- |
| Profile | `laya-ios-english-v1` |
| Upstream source | `NandhaKishorM/laya` |
| Upstream commit | `573e5b62696ba441230cd6be71d593331b5d23af` |
| Checkpoint | `convaiinnovations/laya` |
| Checkpoint revision | `c5d78730f3493e4fe16d61507ef4b78eef7318cf` |
| Behavioral reference | `laya-mlx` `v0.3.0` |
| Core ML precision | FP16 |
| Batch | 1 |
| Context length | 512 |
| Marker slots | 48 |
| Probability tolerance | 0.02 absolute |
| Decision tolerance | zero mismatches |

## Versioning rules

- Fixture schema changes increment `schema_version`.
- Behavioral changes increment `behavior_version` and regenerate every expected result.
- Changing weights, tokenizer files, or either configuration creates a new model profile.
- Conversion-tool upgrades require re-running numerical validation even when artifact hashes are
  otherwise unchanged.
- Mobile implementations may add diagnostics but may not reinterpret existing result fields.

## Source-of-truth order

When implementations disagree, resolve them in this order:

1. Pinned upstream tensors and neural outputs.
2. This repository's documented public behavior and regression tests.
3. Stored portable fixtures generated from those two sources.
4. Platform implementation behavior.

The mobile implementation must change to match the contract unless a deliberate compatibility
revision is documented and fixtures are regenerated.
