# SPDX-License-Identifier: Apache-2.0
"""Lightweight Hub reference parsing and startup path resolution."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_METADATA_PATTERNS = [
    "*.json",
    "*.py",
    "*.model*",
    "*.txt",
    "*.tiktoken",
    "*.jinja",
    "*.bpe",
    "*.vocab",
    "*.spm",
]


def parse_model_hub_reference(path: str | None) -> tuple[str, str] | None:
    """Parse hf://owner/repo or modelscope://owner/repo, leaving other paths alone."""
    if not isinstance(path, str) or not path:
        return None
    source, separator, repo_id = path.partition("://")
    if not separator or source not in ("hf", "modelscope"):
        return None
    try:
        if repo_id.count("/") != 1:
            raise HFValidationError("Expected owner/repo")
        validate_repo_id(repo_id)
    except HFValidationError as exc:
        raise ValueError(
            f"Expected {source}://owner/repo without a file path, query, or "
            "embedded revision; use the existing revision arguments."
        ) from exc
    return source, repo_id


def download_hf_weights(
    snapshot, repo_id, revision, cache_dir, patterns, ignore, download, subdir=""
):
    """Reuse the HF weight downloader while keeping the metadata snapshot pinned."""
    path = Path(snapshot)
    commit = path.name if path.parent.name == "snapshots" else revision
    if subdir:
        patterns = [f"{subdir}/{pattern}" for pattern in patterns]
    folder = download(
        repo_id, cache_dir, patterns, commit, ignore_patterns=ignore, force_remote=True
    )
    if os.path.realpath(folder) != os.path.realpath(snapshot):
        raise ValueError("HF metadata and weight snapshots do not match")
    return str(Path(folder) / subdir) if subdir else folder


def _download_model(reference, revision, cfg, metadata_only):
    from huggingface_hub.constants import HF_HUB_OFFLINE

    source, repo_id = reference
    if source == "hf":
        from huggingface_hub import snapshot_download
    else:
        from modelscope.hub.snapshot_download import snapshot_download

    def check_config(config):
        if "--" in json.dumps(config.get("auto_map", {})):
            raise ValueError(
                "Explicit Hub sources do not support cross-repository auto_map; "
                "download the model and code locally first."
            )
        return config

    try:
        snapshot = Path(
            snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=cfg.download_dir,
                local_files_only=HF_HUB_OFFLINE,
                allow_patterns=_METADATA_PATTERNS if metadata_only else None,
            )
        )
        for path in snapshot.rglob("*config*.json"):
            try:
                json.loads(path.read_text(), object_hook=check_config)
            except json.JSONDecodeError:
                # Required configs are validated by the regular loader.
                continue
        return str(snapshot)
    except Exception:
        logger.error("Failed to resolve %s://%s", source, repo_id)
        raise


def resolve_model_hub_paths(cfg: Any) -> dict[str, str]:
    """Validate explicit sources, then resolve models before tokenizer-only paths."""
    references = {
        field: parse_model_hub_reference(getattr(cfg, field))
        for field in ("model_path", "speculative_draft_model_path", "tokenizer_path")
    }
    model_reference = references["model_path"]
    if model_reference is not None and cfg.tokenizer_path is None:
        references["tokenizer_path"] = model_reference
    formats = {}
    for field, reference in references.items():
        if reference is None:
            continue
        load_format = (
            cfg.speculative_draft_load_format
            if field == "speculative_draft_model_path"
            else None
        ) or cfg.load_format
        formats[field] = "dummy" if field == "tokenizer_path" else load_format.lower()
        if formats[field] not in (
            "auto",
            "safetensors",
            "fastsafetensors",
            "instanttensor",
            "mistral",
            "pt",
            "npcache",
            "dummy",
        ):
            raise ValueError(
                f"Explicit Hub sources do not support load_format={load_format!r}; "
                "download a local snapshot for this loader."
            )
    resolved = {}
    if model_reference is not None and cfg.served_model_name is None:
        resolved["served_model_name"] = model_reference[1]
    snapshots = {}
    for field, reference in references.items():
        if reference is None:
            continue
        # The normal HF loader already selects the weight files to download.
        # Preserve that path when the process default is HF.
        if (
            reference[0] == "hf"
            and not envs.SGLANG_USE_MODELSCOPE.get()
            and not os.path.exists(reference[1])
        ):
            resolved[field] = reference[1]
            continue
        draft = field == "speculative_draft_model_path"
        revision = cfg.speculative_draft_model_revision if draft else cfg.revision
        key = (reference, revision, formats[field] == "dummy")
        if field == "tokenizer_path":
            key = next((k for k in snapshots if k[:2] == key[:2]), key)
        if key not in snapshots:
            snapshots[key] = _download_model(
                reference, revision, cfg, key[2] or reference[0] == "hf"
            )
        resolved[field] = snapshots[key]
    return resolved
