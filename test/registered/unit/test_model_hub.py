# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.model_path_hook import (
    handle_model_hub_paths,
    handle_model_source_paths,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.utils import parse_model_hub_reference

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
            handle_model_hub_paths(
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
        args = ServerArgs(model_path=path)
        handle_model_hub_paths(args)
        assert resolving_view(args).model_path == path
    for download in hubs.values():
        download.assert_not_called()


@pytest.mark.parametrize("global_source", ["true", "false"])
@pytest.mark.parametrize("hf_is_draft", [False, True])
def test_mixed_sources_reach_weight_loader(
    hubs, monkeypatch, tmp_path, global_source, hf_is_draft
):
    from transformers import LlamaConfig

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader import loader as loader_module
    from sglang.srt.runtime_context import get_context

    monkeypatch.setenv("SGLANG_USE_MODELSCOPE", global_source)
    monkeypatch.setenv("SGLANG_SORT_WEIGHT_FILES", "-1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Org" / "repo").mkdir(parents=True)
    snapshot = tmp_path / "cache" / "models--Org--repo" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    hubs["hf"].return_value = str(snapshot)
    LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(snapshot)
    ms = "modelscope://Org/other"
    hf = "hf://Org/repo"
    args = ServerArgs(
        model_path=ms if hf_is_draft else hf,
        speculative_draft_model_path=hf if hf_is_draft else ms,
        revision="target-tag",
        speculative_draft_model_revision="draft-tag",
        download_dir=str(tmp_path / "cache"),
        device="cpu",
    )
    handle_model_source_paths(args)
    cfg = resolving_view(args)
    assert cfg.tokenizer_path == cfg.model_path
    assert cfg.served_model_name == ("Org/other" if hf_is_draft else "Org/repo")
    assert hubs["hf"].call_args.kwargs["revision"] == (
        "draft-tag" if hf_is_draft else "target-tag"
    )
    assert hubs["modelscope"].call_args.kwargs["allow_patterns"] is None
    assert "*.safetensors" not in hubs["hf"].call_args.kwargs["allow_patterns"]

    config = ModelConfig.from_server_args(
        args,
        model_path=cfg.speculative_draft_model_path if hf_is_draft else cfg.model_path,
        model_revision="draft-tag" if hf_is_draft else "target-tag",
        is_draft_model=hf_is_draft,
    )
    loader = loader_module.DefaultModelLoader(
        LoadConfig(load_format="safetensors", download_dir=cfg.download_dir)
    )
    source = loader.Source.init_new(
        config, model=SimpleNamespace(fall_back_to_pt_during_load=False)
    )
    assert (source.model_or_path, source.revision) == (hf, snapshot.name)

    def fetch(repo, cache_dir, patterns, revision, *, ignore_patterns):
        assert (repo, cache_dir, patterns, revision) == (
            hf,
            cfg.download_dir,
            ["*.safetensors"],
            snapshot.name,
        )
        (snapshot / "model.safetensors").touch()
        return str(snapshot)

    monkeypatch.setattr(loader_module, "download_weights_from_hf", fetch)
    with (
        get_context().override_server_args(),
        patch.object(
            loader, "_maybe_download_from_modelscope", side_effect=AssertionError
        ),
        patch.object(loader_module, "download_safetensors_index_file_from_hf") as index,
    ):
        (resolved,) = loader.resolve_model_weights(
            config, SimpleNamespace(fall_back_to_pt_during_load=False)
        )
    assert resolved.weight_files == (str(snapshot / "model.safetensors"),)
    index.assert_called_once_with(
        "Org/repo", "model.safetensors.index.json", cfg.download_dir, snapshot.name
    )


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
    handle_model_hub_paths(args, default_modelscope=True)
    view = resolving_view(args)
    assert view.model_path == hubs["modelscope"].return_value
    assert view.speculative_draft_model_path == hubs["hf"].return_value
    assert hubs["modelscope"].call_args.args == ("Org/target",)
    assert hubs["hf"].call_args.args == ("Org/draft",)
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
    assert weight_utils.download_weights_from_hf("hf://Org/repo", *args[1:]) == str(
        tmp_path / "remote-snapshot"
    )
    assert download.call_args.args == ("Org/repo",)


def test_bundled_hf_draft_downloads_only_its_subfolder(hubs):
    from sglang.srt.arg_groups.speculative_hook import (
        _target_checkpoint_dspark_subfolder,
    )

    snapshot = Path(hubs["hf"].return_value)
    draft = snapshot / "dspark"
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    with patch("huggingface_hub.hf_hub_download") as probe:
        result = _target_checkpoint_dspark_subfolder(
            "hf://Org/repo", "pinned", "/cache"
        )
    assert result == str(draft)
    probe.assert_called_once_with(
        "Org/repo", "dspark/config.json", revision="pinned", cache_dir="/cache"
    )
    hubs["hf"].assert_called_once_with(
        "Org/repo", revision="pinned", cache_dir="/cache", allow_patterns=["dspark/*"]
    )


def test_tokenizer_reuses_full_draft_snapshot(hubs):
    args = ServerArgs(
        model_path="local",
        speculative_draft_model_path="modelscope://Org/repo",
        tokenizer_path="modelscope://Org/repo",
        revision="v1",
        speculative_draft_model_revision="v1",
    )
    handle_model_hub_paths(args)
    cfg = resolving_view(args)
    assert cfg.tokenizer_path == cfg.speculative_draft_model_path
    hubs["modelscope"].assert_called_once_with(
        "Org/repo",
        cache_dir=None,
        revision="v1",
        local_files_only=True,
        allow_patterns=None,
    )


def test_same_repo_different_tokenizer_revision(hubs):
    args = ServerArgs(
        model_path="local",
        speculative_draft_model_path="hf://Org/repo",
        tokenizer_path="hf://Org/repo",
        revision="v1",
        speculative_draft_model_revision="v2",
    )
    handle_model_hub_paths(args)
    assert [call.kwargs["revision"] for call in hubs["hf"].call_args_list] == [
        "v2",
        "v1",
    ]


@pytest.mark.parametrize("field", ["tokenizer_path", "model_path"])
def test_cross_repo_auto_map_rejected(hubs, field):
    folder = Path(hubs["hf"].return_value)
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": ["Other/code--tok.Custom", None]}})
    )
    with pytest.raises(ValueError, match="cross-repository"):
        handle_model_hub_paths(
            ServerArgs(**{"model_path": "local", field: "hf://Org/repo"})
        )


def test_unsupported_loader_fails_before_download(hubs):
    with pytest.raises(ValueError, match="local snapshot"):
        handle_model_hub_paths(
            ServerArgs(model_path="hf://Org/repo", load_format="gguf")
        )
    hubs["hf"].assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
