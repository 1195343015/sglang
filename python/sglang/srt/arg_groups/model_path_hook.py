# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the model source paths."""

from __future__ import annotations

import glob
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import (
    _gguf_quantization,
    declare_resolution,
    run_post_process_pass,
)
from sglang.srt.environ import envs
from sglang.srt.hf_transformers import check_gguf_file
from sglang.srt.utils.common import find_local_repo_dir, is_remote_url
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from sglang.utils import parse_model_hub_reference

logger = logging.getLogger(__name__)


_HUB_METADATA_PATTERNS = [
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
_HUB_LOAD_FORMATS = {
    "auto",
    "safetensors",
    "fastsafetensors",
    "instanttensor",
    "mistral",
    "pt",
    "npcache",
    "dummy",
}


def _resolve_or_download(
    repo: str,
    revision: str | None,
    cache_dir: str | None,
    *,
    hub: str | None,
    metadata_only: bool,
) -> str:
    """An unprefixed ModelScope ID can reuse the legacy on-disk cache layout."""
    if hub is None:
        if not repo or os.path.exists(repo):
            return repo
        from modelscope.utils.file_utils import get_model_cache_root

        for root in (get_model_cache_root(), cache_dir):
            if root and os.path.exists(os.path.join(root, repo)):
                return os.path.join(root, repo)
        options = (
            {"ignore_patterns": ["*.bin", "*.safetensors"]} if metadata_only else {}
        )
    else:
        from huggingface_hub.constants import HF_HUB_OFFLINE

        options = dict(
            local_files_only=HF_HUB_OFFLINE,
            allow_patterns=_HUB_METADATA_PATTERNS if metadata_only else None,
        )

    if hub == "hf":
        from huggingface_hub import snapshot_download
    else:
        from modelscope.hub.snapshot_download import snapshot_download
    try:
        folder = snapshot_download(
            repo, revision=revision, cache_dir=cache_dir, **options
        )
    except Exception:
        if hub is not None:
            logger.error("Failed to resolve %s://%s", hub, repo)
        raise
    if hub is not None:
        for path in Path(folder).rglob("*config*.json"):
            try:
                json.loads(path.read_text(), object_hook=_check_hub_auto_map)
            except json.JSONDecodeError:
                continue
    return folder


def handle_model_hub_paths(server_args: Any, *, default_modelscope: bool = False):
    """Resolve prefixed paths before GGUF, or unprefixed IDs at the legacy ModelScope hook."""
    cfg = resolving_view(server_args)
    resolved = {}
    downloads = {}
    # An explicit tokenizer can reuse the full target or draft snapshot. Keep
    # the original resolution order for unprefixed ModelScope checkpoints.
    fields = (
        ("model_path", "tokenizer_path", "speculative_draft_model_path")
        if default_modelscope
        else ("model_path", "speculative_draft_model_path", "tokenizer_path")
    )
    for field in fields:
        path = getattr(cfg, field)
        draft = field == "speculative_draft_model_path"
        tokenizer = field == "tokenizer_path"
        revision = cfg.speculative_draft_model_revision if draft else cfg.revision
        if default_modelscope:
            if path is not None:
                resolved[field] = _resolve_or_download(
                    path,
                    (revision or "main") if draft else revision,
                    cfg.download_dir,
                    hub=None,
                    metadata_only=tokenizer,
                )
            continue
        ref = parse_model_hub_reference(path)
        if ref is None:
            continue
        hub, repo = ref
        load_format = (
            (cfg.speculative_draft_load_format if draft else None) or cfg.load_format
        ).lower()
        if not tokenizer and load_format not in _HUB_LOAD_FORMATS:
            raise ValueError(f"Use a local snapshot for load_format={load_format!r}")
        if field == "model_path" and cfg.served_model_name is None:
            resolved["served_model_name"] = repo
        if (
            hub == "hf"
            and not envs.SGLANG_USE_MODELSCOPE.get()
            and not os.path.exists(repo)
        ):
            resolved[field] = repo
            continue
        downloads[field] = (
            hub,
            repo,
            revision,
            tokenizer or hub == "hf" or load_format == "dummy",
        )

    snapshots = {}
    for field, key in downloads.items():
        hub, repo, revision, metadata_only = key
        full = (hub, repo, revision, False)
        if field == "tokenizer_path" and full in snapshots:
            resolved[field] = snapshots[full]
            continue
        if key not in snapshots:
            snapshots[key] = _resolve_or_download(
                repo, revision, cfg.download_dir, hub=hub, metadata_only=metadata_only
            )
        resolved[field] = snapshots[key]
    if (
        "model_path" in resolved
        and cfg.tokenizer_path is None
        and not default_modelscope
    ):
        resolved["tokenizer_path"] = resolved["model_path"]
    if resolved:
        declare_resolution(
            server_args,
            "_handle_modelscope_paths"
            if default_modelscope
            else "_handle_model_hub_paths",
            **resolved,
        )


def _check_hub_auto_map(config):
    if "--" in json.dumps(config.get("auto_map", {})):
        raise ValueError(
            "Explicit Hub sources do not support cross-repository auto_map"
        )
    return config


def handle_model_source_paths(server_args: Any):
    """Resolve explicit Hub sources before generic remote-connector handling."""
    cfg = resolving_view(server_args)
    handle_model_hub_paths(server_args)
    resolve_hf_gguf_model_path(server_args)

    seen_paths = set()
    for model_path in (
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.speculative_draft_model_path,
    ):
        if (
            model_path is not None
            and model_path not in seen_paths
            and is_runai_obj_uri(model_path)
        ):
            ObjectStorageModel.download_and_get_path(model_path)
            seen_paths.add(model_path)


def resolve_hf_gguf_model_path(server_args: Any):
    """Turn a Hub reference to a .gguf into a local file path."""
    cfg = resolving_view(server_args)
    from sglang.srt.hf_transformers import resolve_hf_gguf_reference

    resolved = resolve_hf_gguf_reference(cfg.model_path, revision=cfg.revision)
    if resolved is not None:
        logger.info("Resolved GGUF %s -> %s", cfg.model_path, resolved)
        if cfg.tokenizer_path == cfg.model_path:
            declare_resolution(
                server_args,
                "_resolve_hf_gguf_model_path",
                tokenizer_path=resolved,
            )
        declare_resolution(
            server_args,
            "_resolve_hf_gguf_model_path",
            model_path=resolved,
        )

    # A speculative draft can be a .gguf too, and it is loaded by path, so it
    # needs the same Hub-reference resolution as the target.
    if cfg.speculative_draft_model_path:
        resolved_draft = resolve_hf_gguf_reference(
            cfg.speculative_draft_model_path,
            revision=cfg.speculative_draft_model_revision,
        )
        if resolved_draft is not None:
            logger.info(
                "Resolved draft GGUF %s -> %s",
                cfg.speculative_draft_model_path,
                resolved_draft,
            )
            declare_resolution(
                server_args,
                "_resolve_hf_gguf_model_path",
                speculative_draft_model_path=resolved_draft,
            )


def handle_load_format(server_args: Any):
    # The quantization side of the gguf coupling moved to the pipeline
    # (arg_groups/overrides.py: _gguf_quantization); load_format itself is
    # genuine config (runtime user updates write it) and stays imperative.
    cfg = resolving_view(server_args)

    run_post_process_pass(server_args, _gguf_quantization)
    if (cfg.load_format == "auto" or cfg.load_format == "gguf") and check_gguf_file(
        cfg.model_path
    ):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="gguf",
        )

    if cfg.load_format == "auto" and is_mistral_native_format(server_args):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="mistral",
        )
        logger.info(
            "Detected Mistral native format checkpoint, setting load_format='mistral'"
        )

    if is_runai_obj_uri(cfg.model_path):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="runai_streamer",
        )
    elif is_remote_url(cfg.model_path):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="remote",
        )

    if (
        cfg.speculative_draft_model_path is not None
        and is_runai_obj_uri(cfg.speculative_draft_model_path)
        and cfg.speculative_draft_load_format is None
    ):
        declare_resolution(
            server_args,
            "_handle_load_format",
            speculative_draft_load_format="runai_streamer",
        )

    if cfg.custom_weight_loader is None:
        declare_resolution(server_args, "_handle_load_format", custom_weight_loader=[])

    if cfg.load_format == "remote_instance":
        if cfg.remote_instance_weight_loader_backend != "modelexpress" and (
            cfg.remote_instance_weight_loader_seed_instance_ip is None
            or cfg.remote_instance_weight_loader_seed_instance_service_port is None
        ):
            logger.warning(
                "Fallback load_format to 'auto' due to incomplete remote instance weight loader settings."
            )
            declare_resolution(
                server_args,
                "_handle_load_format",
                load_format="auto",
            )
        elif (
            cfg.remote_instance_weight_loader_send_weights_group_ports is None
            and cfg.remote_instance_weight_loader_backend == "nccl"
        ):
            logger.warning(
                "Fallback load_format to 'auto' due to incomplete remote instance weight loader NCCL group ports settings."
            )
            declare_resolution(
                server_args,
                "_handle_load_format",
                load_format="auto",
            )
        elif (
            cfg.remote_instance_weight_loader_backend == "transfer_engine"
            and not validate_transfer_engine(server_args)
        ):
            logger.warning(
                "Fallback load_format to 'auto' due to 'transfer_engine' backend is not supported."
            )
            declare_resolution(
                server_args,
                "_handle_load_format",
                load_format="auto",
            )

    # Check whether TransferEngine can be used when users want to start seed service that supports TransferEngine backend.
    if cfg.remote_instance_weight_loader_start_seed_via_transfer_engine:
        declare_resolution(
            server_args,
            "_handle_load_format",
            remote_instance_weight_loader_start_seed_via_transfer_engine=validate_transfer_engine(
                server_args
            ),
        )

    # "ipc_cache" is an internal-only load format: ModelRunner sets it
    # automatically when the weight cache is enabled, and it is not a public
    # --load-format choice. Setting it directly is always wrong (no daemon is
    # launched, and fallback_load_format inherits a nonsensical format), so
    # reject it and point at the knob (defense-in-depth; the CLI already
    # rejects it via LOAD_FORMAT_CHOICES).
    if cfg.load_format == "ipc_cache":
        raise ValueError(
            "load_format='ipc_cache' is an internal-only format and must not "
            "be set directly. Enable the weight cache via --weight-cache-mode "
            "client (connect to an existing daemon) or daemon (launch one); "
            "that selects IPC loading automatically."
        )

    # Speculative decoding loads an extra draft model whose weights the
    # daemon does not export, so refuse the combination up front instead of
    # failing deep inside draft-worker load (draft-model daemon TBD).
    if cfg.weight_cache_mode != "off" and cfg.speculative_algorithm is not None:
        raise ValueError(
            "--weight-cache-mode is not supported together with speculative "
            "decoding (--speculative-algorithm): the weight cache daemon does "
            "not export the draft model's weights. Disable one of them "
            "(--weight-cache-mode off) for this configuration."
        )


def validate_transfer_engine(server_args: Any):
    cfg = resolving_view(server_args)
    try:
        mooncake_available = importlib.util.find_spec("mooncake.engine") is not None
    except (ModuleNotFoundError, ValueError):
        mooncake_available = False
    if not mooncake_available:
        logger.warning(
            "Failed to import mooncake.engine. Does not support using TransferEngine as remote instance weight loader backend."
        )
        return False
    elif cfg.enable_memory_saver:
        logger.warning(
            "Memory saver is enabled, which is not compatible with TransferEngine. Does not support using TransferEngine as remote instance weight loader backend."
        )
        return False
    else:
        return True


def is_mistral_native_format(server_args: Any) -> bool:
    """True iff the checkpoint requires load_format=mistral.

    Looks for consolidated*.safetensors with no competing
    model*.safetensors; when both weight formats ship in the
    same checkpoint (e.g. Mistral-7B-Instruct-v0.3) the HF path is
    preferred to avoid loading Mistral-named weights into an
    HF-named architecture.

    Name override: mistral-large-3 / mistral-small-4 /
    leanstral always treat as Mistral-native when params.json
    is present -- those families need Mistral weight loading
    regardless of which weight files happen to be present.
    """
    cfg = resolving_view(server_args)
    _MISTRAL_NATIVE_PATTERNS = (
        "mistral-large-3",
        "mistral-small-4",
        "leanstral",
    )
    name_matches = any(
        p in str(cfg.model_path).lower() for p in _MISTRAL_NATIVE_PATTERNS
    )

    def _check_format(has_params, has_consolidated, has_hf_weights) -> bool:
        if has_params and name_matches:
            return True
        return has_consolidated and not has_hf_weights

    if os.path.isdir(cfg.model_path):
        return _check_format(
            has_params=os.path.exists(os.path.join(cfg.model_path, "params.json")),
            has_consolidated=bool(
                glob.glob(os.path.join(cfg.model_path, "consolidated*.safetensors"))
            ),
            has_hf_weights=bool(
                glob.glob(os.path.join(cfg.model_path, "model*.safetensors"))
            ),
        )

    try:
        from huggingface_hub import HfApi

        files = {s.rfilename for s in HfApi().model_info(cfg.model_path).siblings}
    except Exception:
        # Hub unreachable (offline, rate-limited): judge from the local snapshot.
        snapshot_dir = find_local_repo_dir(cfg.model_path, cfg.revision)
        if snapshot_dir is None:
            return False
        files = set(os.listdir(snapshot_dir))
    return _check_format(
        has_params="params.json" in files,
        has_consolidated=any(
            f.startswith("consolidated") and f.endswith(".safetensors") for f in files
        ),
        has_hf_weights=any(
            f.startswith("model") and f.endswith(".safetensors") and "/" not in f
            for f in files
        ),
    )
