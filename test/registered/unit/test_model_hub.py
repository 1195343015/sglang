# SPDX-License-Identifier: Apache-2.0

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pytest

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.model_path_hook import (
    explicit_hf_source,
    handle_model_source_paths,
    handle_modelscope_paths,
)
from sglang.srt.model_hub import (
    download_hf_weights,
    parse_model_hub_reference,
    resolve_model_hub_paths,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def hubs(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", "true")
    for source in ("hf", "ms"):
        folder = tmp_path / source
        folder.mkdir()
        (folder / "model.safetensors").touch()
    hf, ms = (Mock(return_value=str(tmp_path / source)) for source in ("hf", "ms"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", hf)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        SimpleNamespace(snapshot_download=ms),
    )
    return {"hf": hf, "modelscope": ms}


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_validation_and_legacy_paths(hubs, source):
    assert parse_model_hub_reference(f"{source}://Org/Model-1.5") == (
        source,
        "Org/Model-1.5",
    )
    for repo in (
        "",
        "owner",
        "owner/repo/file",
        "owner/repo@v1",
        "owner/repo?x=1",
        "owner/repo#v1",
        "../repo",
        "owner/repo.git",
        "owner/.repo",
        "owner/repo-",
    ):
        with pytest.raises(ValueError, match="Expected"):
            resolve_model_hub_paths(
                ServerArgs(
                    model_path="hf://Org/valid", tokenizer_path=f"{source}://{repo}"
                )
            )
    for path in (
        "owner/repo",
        "/models/repo",
        "./model",
        "s3://bucket/model",
        "gs://bucket/model",
        "az://bucket/model",
        "redis://host/model",
        "instance://host/model",
        "https://huggingface.co/Org/Model",
        "owner/repo:Q4_K_M",
    ):
        assert resolve_model_hub_paths(ServerArgs(model_path=path)) == {}
    for download in hubs.values():
        download.assert_not_called()


@pytest.mark.parametrize("global_source", ["true", "false"])
@pytest.mark.parametrize("source", ["hf", "modelscope"])
@pytest.mark.parametrize(
    "field", ["model_path", "tokenizer_path", "speculative_draft_model_path"]
)
def test_fields_override_global_source(hubs, monkeypatch, global_source, source, field):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", global_source)
    args = ServerArgs(**{"model_path": "Org/target", field: f"{source}://Org/repo"})
    resolved = resolve_model_hub_paths(args)
    use_hf_loader = source == "hf" and global_source == "false"
    assert resolved[field] == (
        "Org/repo" if use_hf_loader else hubs[source].return_value
    )
    if field == "model_path":
        assert resolved["tokenizer_path"] == resolved["model_path"]
        assert resolved["served_model_name"] == "Org/repo"
    if use_hf_loader:
        hubs[source].assert_not_called()
    else:
        hubs[source].assert_called_once_with(
            "Org/repo",
            revision=None,
            cache_dir=None,
            local_files_only=True,
            allow_patterns=ANY,
        )
        assert (hubs[source].call_args.kwargs["allow_patterns"] is None) == (
            source == "modelscope" and field != "tokenizer_path"
        )
    hubs["hf" if source == "modelscope" else "modelscope"].assert_not_called()
    assert getattr(args, field) == f"{source}://Org/repo"
    assert os.environ["SGLANG_USE_MODELSCOPE"] == global_source


@pytest.mark.parametrize(
    "source,draft_source", [("hf", "modelscope"), ("modelscope", "hf")]
)
@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize("global_source", ["true", "false"])
def test_mixed_sources_through_server_args(
    hubs, monkeypatch, source, draft_source, cli, global_source
):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", global_source)
    options = dict(
        model_path=f"{source}://Org/target",
        tokenizer_path=f"{draft_source}://Org/tokenizer",
        speculative_draft_model_path=f"{draft_source}://Org/draft",
        revision="v1",
        speculative_draft_model_revision="v2",
        download_dir="/custom",
    )
    if cli:
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            item
            for key, value in options.items()
            for item in ("--" + key.replace("_", "-"), value)
        ]
        args = ServerArgs.from_cli_args(parser.parse_args(argv))
    else:
        args = ServerArgs(**options)
    with patch("sglang.srt.arg_groups.model_path_hook.resolve_hf_gguf_model_path"):
        handle_model_source_paths(args)
    view = resolving_view(args)

    def expected(hub, repo):
        return (
            f"Org/{repo}"
            if hub == "hf" and global_source == "false"
            else hubs[hub].return_value
        )

    assert view.model_path == expected(source, "target")
    assert view.tokenizer_path == expected(draft_source, "tokenizer")
    assert view.speculative_draft_model_path == expected(draft_source, "draft")
    assert view.served_model_name == "Org/target"
    if source == "hf" and global_source == "false":
        hubs[source].assert_not_called()
    else:
        hubs[source].assert_called_once_with(
            "Org/target",
            revision="v1",
            cache_dir="/custom",
            local_files_only=True,
            allow_patterns=ANY,
        )
        assert (hubs[source].call_args.kwargs["allow_patterns"] is None) == (
            source == "modelscope"
        )
    assert [call.kwargs["revision"] for call in hubs[draft_source].call_args_list] == (
        [] if draft_source == "hf" and global_source == "false" else ["v2", "v1"]
    )
    assert args.model_path == options["model_path"]


def test_resolution_pipeline_preserves_implicit_dspark_draft(hubs, monkeypatch):
    from transformers import LlamaConfig

    snapshot = hubs["hf"].return_value
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=2048,
        dspark_block_size=2,
        dspark_markov_rank=1,
        dspark_noise_token_id=1,
        dspark_target_layer_ids=[0],
    ).save_pretrained(snapshot)
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", "true")
    args = ServerArgs(
        model_path="hf://Org/repo",
        device="cuda",
        speculative_algorithm="DSPARK",
        revision="v1",
    )
    args.resolve_once()
    cfg = resolving_view(args)
    assert (
        cfg.model_path
        == cfg.tokenizer_path
        == cfg.speculative_draft_model_path
        == snapshot
    )
    assert cfg.speculative_draft_model_revision == "v1"
    assert cfg.served_model_name == "Org/repo"
    hubs["hf"].assert_called_once()
    hubs["modelscope"].assert_not_called()


def test_legacy_modelscope_main_with_explicit_hf_draft(hubs, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", "true")
    monkeypatch.setitem(sys.modules, "modelscope.utils", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "modelscope.utils.file_utils",
        SimpleNamespace(get_model_cache_root=lambda: str(tmp_path / "ms-cache")),
    )
    args = ServerArgs(
        model_path="Org/target", speculative_draft_model_path="hf://Org/draft"
    )
    with patch("sglang.srt.arg_groups.model_path_hook.resolve_hf_gguf_model_path"):
        handle_model_source_paths(args)
    handle_modelscope_paths(args)
    view = resolving_view(args)
    assert view.model_path == hubs["modelscope"].return_value
    assert view.speculative_draft_model_path == hubs["hf"].return_value
    assert hubs["modelscope"].call_args.args == ("Org/target",)
    assert hubs["hf"].call_args.args == ("Org/draft",)
    assert hubs["hf"].call_args.kwargs["allow_patterns"] is not None


def test_explicit_hf_does_not_use_a_same_named_local_directory(
    hubs, monkeypatch, tmp_path
):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", "false")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Org" / "repo").mkdir(parents=True)
    result = resolve_model_hub_paths(ServerArgs(model_path="hf://Org/repo"))
    assert result["model_path"] == hubs["hf"].return_value
    assert hubs["hf"].call_args.kwargs["allow_patterns"] is not None


def test_hf_weight_downloader_can_force_remote_for_same_named_directory(
    hubs, monkeypatch, tmp_path
):
    from sglang.srt.model_loader import weight_utils

    monkeypatch.chdir(tmp_path)
    (tmp_path / "Org" / "repo").mkdir(parents=True)
    monkeypatch.setenv("SGLANG_IS_IN_CI", "false")
    download = Mock(return_value=str(tmp_path / "remote-snapshot"))
    monkeypatch.setattr(weight_utils, "snapshot_download", download)
    args = ("Org/repo", str(tmp_path / "cache"), ["*.safetensors"])
    assert weight_utils.download_weights_from_hf(*args) == "Org/repo"
    download.assert_not_called()
    assert weight_utils.download_weights_from_hf(*args, force_remote=True) == str(
        tmp_path / "remote-snapshot"
    )
    assert download.call_args.args == ("Org/repo",)


@pytest.mark.parametrize(
    "source,draft_source", [("hf", "modelscope"), ("modelscope", "hf")]
)
@pytest.mark.parametrize("global_source", ["true", "false"])
def test_explicit_hf_source_tracks_resolved_model_and_draft(
    hubs, monkeypatch, source, draft_source, global_source
):
    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", global_source)
    args = ServerArgs(
        model_path=f"{source}://Org/target",
        speculative_draft_model_path=f"{draft_source}://Org/draft",
    )
    with patch("sglang.srt.arg_groups.model_path_hook.resolve_hf_gguf_model_path"):
        handle_model_source_paths(args)
    view = resolving_view(args)
    assert explicit_hf_source(args, view.model_path) == (
        ("Org/target", view.model_path, "") if source == "hf" else None
    )
    assert explicit_hf_source(args, view.speculative_draft_model_path) == (
        ("Org/draft", view.speculative_draft_model_path, "")
        if draft_source == "hf"
        else None
    )


def test_hf_weight_download_reuses_pinned_snapshot_and_loader_patterns(tmp_path):
    snapshot = tmp_path / "models--Org--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    def download(repo, cache_dir, patterns, revision, *, ignore_patterns, force_remote):
        assert (repo, revision, cache_dir) == ("Org/repo", "abc123", "/cache")
        assert patterns == ["*.safetensors"]
        assert ignore_patterns == ["unused*"]
        assert force_remote
        (snapshot / "model.safetensors").touch()
        return str(snapshot)

    result = download_hf_weights(
        str(snapshot),
        "Org/repo",
        "main",
        "/cache",
        ["*.safetensors"],
        ["unused*"],
        download,
    )
    assert result == str(snapshot)
    assert (snapshot / "model.safetensors").exists()
    assert not (snapshot / "pytorch_model.bin").exists()

    with pytest.raises(ValueError, match="snapshots do not match"):
        download_hf_weights(
            str(snapshot),
            "Org/repo",
            "main",
            "/cache",
            ["*.safetensors"],
            None,
            lambda *args, **kwargs: str(tmp_path / "other"),
        )


def test_hf_bundled_draft_downloads_its_subdirectory(hubs, tmp_path):
    snapshot = Path(hubs["hf"].return_value)
    draft = snapshot / "dspark"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    args = ServerArgs(model_path="hf://Org/repo")
    with patch("sglang.srt.arg_groups.model_path_hook.resolve_hf_gguf_model_path"):
        handle_model_source_paths(args)
    source = explicit_hf_source(args, str(draft))
    assert source == ("Org/repo", str(snapshot), "dspark")

    def download(repo, cache_dir, patterns, revision, *, ignore_patterns, force_remote):
        assert repo == "Org/repo"
        assert patterns == ["dspark/*.safetensors"]
        assert force_remote
        (draft / "model.safetensors").touch()
        return str(snapshot)

    result = download_hf_weights(
        str(snapshot),
        "Org/repo",
        None,
        None,
        ["*.safetensors"],
        None,
        download,
        "dspark",
    )
    assert result == str(draft)
    assert (draft / "model.safetensors").exists()


@pytest.mark.parametrize("subdir", ["", "dspark"])
def test_hf_weights_use_existing_loader_when_modelscope_is_global(
    hubs, monkeypatch, tmp_path, subdir
):
    from transformers import LlamaConfig

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader import loader as loader_module
    from sglang.srt.runtime_context import publish, reset_context

    snapshot = tmp_path / "models--Org--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=2048,
    ).save_pretrained(snapshot)
    weights_dir = snapshot / subdir
    weights_dir.mkdir(exist_ok=True)
    hubs["hf"].return_value = str(snapshot)
    monkeypatch.setenv("SGLANG_SORT_WEIGHT_FILES", "-1")
    args = ServerArgs(model_path="hf://Org/repo", device="cuda", revision="main")

    def download(repo, cache_dir, patterns, revision, *, ignore_patterns, force_remote):
        assert repo == "Org/repo"
        assert revision == "abc123"
        assert force_remote
        prefix = f"{subdir}/" if subdir else ""
        assert patterns[:2] == [f"{prefix}*.safetensors", f"{prefix}*.bin"]
        (weights_dir / "model.safetensors").touch()
        return str(snapshot)

    monkeypatch.setattr(loader_module, "download_weights_from_hf", download)
    try:
        publish(args, role="scheduler")
        loader = loader_module.DefaultModelLoader(LoadConfig())
        folder, files, safe = loader._prepare_weights(str(weights_dir), "main", True)
        assert folder == str(weights_dir)
        assert files == [str(weights_dir / "model.safetensors")]
        assert safe
        assert not (weights_dir / "pytorch_model.bin").exists()
    finally:
        reset_context()


@pytest.mark.parametrize(
    "revision,draft_format,load_format,calls",
    [
        ("v1", None, "auto", 1),
        ("v1", "pt", "auto", 1),
        ("v2", None, "auto", 2),
        ("v1", "dummy", "auto", 2),
        ("v1", None, "DUMMY", 1),
    ],
)
def test_shared_snapshots(hubs, revision, draft_format, load_format, calls):
    result = resolve_model_hub_paths(
        ServerArgs(
            model_path="hf://Org/repo",
            tokenizer_path="hf://Org/repo",
            speculative_draft_model_path="hf://Org/repo",
            served_model_name="custom",
            revision="v1",
            speculative_draft_model_revision=revision,
            speculative_draft_load_format=draft_format,
            load_format=load_format,
        )
    )
    assert result["tokenizer_path"] == result["model_path"]
    assert hubs["hf"].call_count == calls
    assert "served_model_name" not in result
    if load_format == "DUMMY":
        patterns = hubs["hf"].call_args.kwargs["allow_patterns"]
        assert "*.json" in patterns and "*.safetensors" not in patterns


@pytest.mark.parametrize("source", ["hf", "modelscope"])
def test_failure_preserves_exception_without_fallback(hubs, caplog, source):
    error = OSError("Hub unavailable")
    hubs[source].side_effect = error
    with pytest.raises(OSError) as caught:
        resolve_model_hub_paths(ServerArgs(model_path=f"{source}://Org/repo"))
    assert caught.value is error
    assert f"{source}://Org/repo" in caplog.text
    hubs["hf" if source == "modelscope" else "modelscope"].assert_not_called()


def test_modelscope_full_snapshot_preserves_bundled_weights_in_one_fetch(
    hubs, tmp_path
):
    source = "modelscope"
    folder = tmp_path / "snapshot"
    folder.mkdir()
    files = [
        "config.json",
        "pytorch_model.bin",
        "draft/model.safetensors",
        "original/model.safetensors",
    ]

    def download(repo, *, allow_patterns, **kwargs):
        for name in files:
            if allow_patterns is None or any(
                fnmatch.fnmatch(name, p) for p in allow_patterns
            ):
                path = folder / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text("{}")
        return str(folder)

    hubs[source].side_effect = download
    result = resolve_model_hub_paths(
        ServerArgs(model_path=f"{source}://Org/repo", revision="main")
    )
    assert result["model_path"] == str(folder)
    assert all((folder / name).is_file() for name in files)
    # A single SDK resolution cannot mix metadata from one branch tip with
    # weights downloaded after the branch has advanced to another commit.
    assert hubs[source].call_count == 1
    assert hubs[source].call_args.kwargs["allow_patterns"] is None


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize(
    "filename,config",
    [
        ("config.json", {"auto_map": {"AutoConfig": "Org/code--config.Custom"}}),
        (
            "tokenizer_config.json",
            {"auto_map": {"AutoTokenizer": ["Org/code--tok.Custom", None]}},
        ),
        ("tokenizer_config.json", {"auto_map": ["Org/code--tok.Custom", None]}),
        (
            "config.json",
            {"text_config": {"auto_map": {"AutoConfig": "Org/code--config.Custom"}}},
        ),
        ("dspark/config.json", {"auto_map": {"AutoConfig": "Org/code--config.Custom"}}),
    ],
)
def test_cross_repo_auto_map_rejected_without_code_fetch(hubs, trust, filename, config):
    path = Path(hubs["modelscope"].return_value) / filename
    path.parent.mkdir(exist_ok=True)
    original = json.dumps(config)
    path.write_text(original)
    with pytest.raises(ValueError, match="auto_map"):
        resolve_model_hub_paths(
            ServerArgs(model_path="modelscope://Org/repo", trust_remote_code=trust)
        )
    assert hubs["modelscope"].call_count == 1
    hubs["hf"].assert_not_called()
    assert path.read_text() == original


def test_local_auto_map_and_unrelated_json_are_left_unchanged(hubs):
    snapshot = Path(hubs["hf"].return_value)
    configs = {
        "config.json": json.dumps({"auto_map": {"AutoConfig": "configuration.Local"}}),
        "tokenizer_config.json": json.dumps({"auto_map": ["tokenization.Local", None]}),
        "other_config.json": "not valid JSON",
        "list_config.json": "[null, 1]",
        "scalar_config.json": "null",
    }
    for name, content in configs.items():
        (snapshot / name).write_text(content)
    result = resolve_model_hub_paths(
        ServerArgs(model_path="hf://Org/repo", trust_remote_code=True)
    )
    assert result["model_path"] == str(snapshot)
    assert all(
        (snapshot / name).read_text() == content for name, content in configs.items()
    )


def test_unsupported_loader_fails_before_download(hubs):
    with pytest.raises(ValueError, match="local snapshot"):
        resolve_model_hub_paths(
            ServerArgs(model_path="hf://Org/repo", load_format="gguf")
        )
    hubs["hf"].assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
