"""Shared constants and helpers for reproducible mobile model conversion."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROFILE = "laya-ios-english-v1"
UPSTREAM_COMMIT = "573e5b62696ba441230cd6be71d593331b5d23af"
CHECKPOINT_ID = "convaiinnovations/laya"
CHECKPOINT_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
SEQUENCE_LENGTH = 512
MARKER_SLOTS = 48


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(checkpoint):
    checkpoint = Path(checkpoint)
    return {
        name: checkpoint / name
        for name in (
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        )
    }


def validate_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    files = checkpoint_files(checkpoint)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {', '.join(missing)}")
    config = json.loads(files["rl_agent_config.json"].read_text())
    if config.get("max_len", 512) != SEQUENCE_LENGTH:
        raise ValueError(f"{PROFILE} requires max_len={SEQUENCE_LENGTH}")
    if config.get("head_max_len", 192) > MARKER_SLOTS * 4 + 16:
        raise ValueError("Configured question head may exceed the fixed marker-slot contract")
    return checkpoint, config, files


def load_upstream_agent(upstream, checkpoint, device):
    upstream = Path(upstream).resolve()
    if not (upstream / "laya/agent.py").is_file():
        raise FileNotFoundError(f"Pinned upstream Laya source not found at {upstream}")
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {commit}")
    sys.path.insert(0, str(upstream))
    import laya

    return laya.load(str(checkpoint), device=device)
