"""Tests for resolve_config_dict and its integration with config composition."""

import pytest
import yaml

from research_scaffold.config_tools import (
    resolve_config_dict,
    load_config,
    load_and_compose_config_steps,
    load_meta_config,
    process_meta_config,
)
from research_scaffold.types import Config


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temp directory with reusable YAML config files."""

    def write_yaml(name, data):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(data))
        return str(p)

    return tmp_path, write_yaml


# ── resolve_config_dict: plain configs ──────────────────────────────


class TestResolveConfigDictPlain:

    def test_plain_dict_passthrough(self):
        d = {"name": "test", "function_name": "f", "function_kwargs": {"lr": 0.01}}
        assert resolve_config_dict(d) == d

    def test_plain_dict_no_mutation(self):
        d = {"name": "test", "function_name": "f"}
        original = d.copy()
        resolve_config_dict(d)
        assert d == original

    def test_plain_yaml_file(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        path = write_yaml("plain.yaml", {
            "name": "from_file",
            "function_name": "train",
            "function_kwargs": {"lr": 0.1},
        })
        result = resolve_config_dict(path)
        assert result["name"] == "from_file"
        assert result["function_kwargs"]["lr"] == 0.1

    def test_empty_dict(self):
        assert resolve_config_dict({}) == {}


# ── resolve_config_dict: single-output meta configs ─────────────────


class TestResolveConfigDictMeta:

    def test_inline_meta_single_config(self):
        meta = {
            "experiments": [
                {"config": {"name": "base", "function_name": "train", "function_kwargs": {"lr": 0.001}}}
            ]
        }
        result = resolve_config_dict(meta)
        assert result["name"] == "base"
        assert result["function_name"] == "train"
        assert result["function_kwargs"]["lr"] == 0.001

    def test_meta_from_yaml_file(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        cfg_path = write_yaml("base.yaml", {
            "name": "base",
            "function_name": "train",
            "function_kwargs": {"lr": 0.01},
        })
        meta_path = write_yaml("meta.yaml", {
            "experiments": [{"config": cfg_path}],
        })
        result = resolve_config_dict(meta_path)
        assert result["name"] == "base"
        assert result["function_kwargs"]["lr"] == 0.01

    def test_meta_with_common_root(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        root_path = write_yaml("root.yaml", {
            "name": "root",
            "function_name": "train",
            "function_kwargs": {"lr": 0.01},
        })
        patch_path = write_yaml("patch.yaml", {
            "function_kwargs": {"batch_size": 64},
        })
        meta_path = write_yaml("meta.yaml", {
            "common_root": root_path,
            "experiments": [{"config": patch_path}],
        })
        result = resolve_config_dict(meta_path)
        assert result["function_kwargs"]["lr"] == 0.01
        assert result["function_kwargs"]["batch_size"] == 64

    def test_meta_with_common_patch(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        base_path = write_yaml("base.yaml", {
            "name": "base",
            "function_name": "train",
            "function_kwargs": {"lr": 0.01},
        })
        patch_path = write_yaml("patch.yaml", {
            "function_kwargs": {"dropout": 0.5},
        })
        meta_path = write_yaml("meta.yaml", {
            "common_patch": patch_path,
            "experiments": [{"config": base_path}],
        })
        result = resolve_config_dict(meta_path)
        assert result["function_kwargs"]["dropout"] == 0.5

    def test_meta_with_bonus_dict(self):
        meta = {
            "experiments": [
                {"config": {"name": "base", "function_name": "train", "function_kwargs": {"lr": 0.01}}}
            ],
            "bonus_dict": {"function_kwargs": {"extra": True}},
        }
        result = resolve_config_dict(meta)
        assert result["function_kwargs"]["extra"] is True

    def test_meta_with_config_steps(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        step1 = write_yaml("step1.yaml", {
            "name": "step1",
            "function_name": "train",
            "function_kwargs": {"lr": 0.01},
        })
        step2 = write_yaml("step2.yaml", {
            "function_kwargs": {"batch_size": 32},
        })
        meta_path = write_yaml("meta.yaml", {
            "experiments": [{"config_steps": [step1, step2]}],
        })
        result = resolve_config_dict(meta_path)
        assert result["function_kwargs"]["lr"] == 0.01
        assert result["function_kwargs"]["batch_size"] == 32


# ── resolve_config_dict: rejection of multi-output metas ────────────


class TestResolveConfigDictRejectsMulti:

    def test_multiple_options_raises(self):
        meta = {
            "experiments": [
                {"config_options": [
                    {"name": "a", "function_name": "f"},
                    {"name": "b", "function_name": "g"},
                ]}
            ]
        }
        with pytest.raises(ValueError, match="exactly 1"):
            resolve_config_dict(meta)

    def test_multiple_experiments_raises(self):
        meta = {
            "experiments": [
                {"config": {"name": "a", "function_name": "f"}},
                {"config": {"name": "b", "function_name": "g"}},
            ]
        }
        with pytest.raises(ValueError, match="exactly 1"):
            resolve_config_dict(meta)

    def test_repeats_gt1_raises(self):
        meta = {
            "experiments": [
                {
                    "config": {"name": "a", "function_name": "f"},
                    "repeats": 3,
                }
            ]
        }
        with pytest.raises(ValueError, match="exactly 1"):
            resolve_config_dict(meta)

    def test_config_axes_product_raises(self):
        meta = {
            "experiments": [
                {"config_axes": [
                    [{"name": "a", "function_name": "f"}, {"name": "b", "function_name": "g"}],
                    [{"function_kwargs": {"x": 1}}, {"function_kwargs": {"x": 2}}],
                ]}
            ]
        }
        with pytest.raises(ValueError, match="exactly 1"):
            resolve_config_dict(meta)


# ── load_and_compose_config_steps with meta configs ─────────────────


class TestComposeWithMeta:

    def test_meta_as_first_step(self):
        meta = {
            "experiments": [
                {"config": {"name": "base", "function_name": "train", "function_kwargs": {"lr": 0.001}}}
            ]
        }
        patch = {"function_kwargs": {"batch_size": 32}}
        config = load_and_compose_config_steps([meta, patch])
        assert config.name == "base"
        assert config.function_kwargs["lr"] == 0.001
        assert config.function_kwargs["batch_size"] == 32

    def test_meta_as_last_step(self):
        base = {"name": "base", "function_name": "train", "function_kwargs": {"lr": 0.01}}
        meta_override = {
            "experiments": [
                {"config": {"name": "override", "function_name": "train", "function_kwargs": {"lr": 0.1}}}
            ]
        }
        config = load_and_compose_config_steps([base, meta_override])
        assert config.function_kwargs["lr"] == 0.1

    def test_meta_as_middle_step(self):
        base = {"name": "base", "function_name": "train"}
        meta_middle = {
            "experiments": [
                {"config": {"name": "mid", "function_name": "train", "function_kwargs": {"lr": 0.01}}}
            ]
        }
        patch = {"function_kwargs": {"batch_size": 64}}
        config = load_and_compose_config_steps([base, meta_middle, patch])
        assert config.function_kwargs["lr"] == 0.01
        assert config.function_kwargs["batch_size"] == 64

    def test_meta_does_not_overwrite_with_none(self):
        base = {"name": "base", "function_name": "train", "wandb_project": "my_project"}
        meta_step = {
            "experiments": [
                {"config": {"name": "meta", "function_name": "train", "function_kwargs": {"lr": 0.01}}}
            ]
        }
        config = load_and_compose_config_steps([base, meta_step])
        assert config.wandb_project == "my_project"

    def test_multi_output_meta_in_compose_raises(self):
        base = {"name": "base", "function_name": "train"}
        bad_meta = {
            "experiments": [
                {"config_options": [
                    {"name": "a", "function_name": "f", "function_kwargs": {"x": 1}},
                    {"name": "b", "function_name": "g", "function_kwargs": {"x": 2}},
                ]}
            ]
        }
        with pytest.raises(ValueError, match="exactly 1"):
            load_and_compose_config_steps([base, bad_meta])


# ── Meta config as base of another meta config (end-to-end) ────────


class TestMetaAsBaseOfMeta:

    def test_meta_as_common_root_of_meta(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        inner_base = write_yaml("inner_base.yaml", {
            "name": "inner",
            "function_name": "train",
            "function_kwargs": {"lr": 0.001},
        })
        inner_meta = write_yaml("inner_meta.yaml", {
            "experiments": [{"config": inner_base}],
        })
        outer_patch = write_yaml("outer_patch.yaml", {
            "function_kwargs": {"batch_size": 128},
        })
        outer_meta_dict = {
            "common_root": inner_meta,
            "experiments": [{"config": outer_patch}],
        }
        mc = load_meta_config(outer_meta_dict)
        configs = process_meta_config(mc)
        assert len(configs) == 1
        assert configs[0].function_kwargs["lr"] == 0.001
        assert configs[0].function_kwargs["batch_size"] == 128

    def test_meta_as_expt_root(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        inner_base = write_yaml("inner.yaml", {
            "name": "inner",
            "function_name": "train",
            "function_kwargs": {"lr": 0.01},
        })
        inner_meta = write_yaml("inner_meta.yaml", {
            "experiments": [{"config": inner_base}],
        })
        patch = write_yaml("patch.yaml", {
            "function_kwargs": {"wd": 0.1},
        })
        outer_meta_dict = {
            "experiments": [
                {
                    "config": patch,
                    "expt_root": inner_meta,
                }
            ],
        }
        mc = load_meta_config(outer_meta_dict)
        configs = process_meta_config(mc)
        assert len(configs) == 1
        assert configs[0].function_kwargs["lr"] == 0.01
        assert configs[0].function_kwargs["wd"] == 0.1

    def test_meta_as_config_step_in_meta(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        inner_base = write_yaml("inner.yaml", {
            "name": "inner",
            "function_name": "train",
            "function_kwargs": {"lr": 0.01},
        })
        inner_meta = write_yaml("inner_meta.yaml", {
            "experiments": [{"config": inner_base}],
        })
        patch = write_yaml("patch.yaml", {
            "function_kwargs": {"epochs": 10},
        })
        outer_meta_dict = {
            "experiments": [
                {"config_steps": [inner_meta, patch]}
            ],
        }
        mc = load_meta_config(outer_meta_dict)
        configs = process_meta_config(mc)
        assert len(configs) == 1
        assert configs[0].function_kwargs["lr"] == 0.01
        assert configs[0].function_kwargs["epochs"] == 10

    def test_nested_meta_three_levels(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        leaf = write_yaml("leaf.yaml", {
            "name": "leaf",
            "function_name": "train",
            "function_kwargs": {"lr": 0.001},
        })
        level1 = write_yaml("level1.yaml", {
            "experiments": [{"config": leaf}],
        })
        level2 = write_yaml("level2.yaml", {
            "experiments": [{"config": level1}],
        })
        result = resolve_config_dict(level2)
        assert result["name"] == "leaf"
        assert result["function_kwargs"]["lr"] == 0.001

    def test_multi_output_inner_meta_as_root_raises(self, tmp_config_dir):
        tmp_path, write_yaml = tmp_config_dir
        inner_meta = write_yaml("inner_meta.yaml", {
            "experiments": [
                {"config_options": [
                    {"name": "a", "function_name": "f"},
                    {"name": "b", "function_name": "g"},
                ]}
            ],
        })
        patch = write_yaml("patch.yaml", {
            "function_kwargs": {"x": 1},
        })
        outer_meta_dict = {
            "common_root": inner_meta,
            "experiments": [{"config": patch}],
        }
        mc = load_meta_config(outer_meta_dict)
        with pytest.raises(ValueError, match="exactly 1"):
            process_meta_config(mc)


# ── load_config with resolve_config_dict ────────────────────────────


class TestLoadConfigViaMeta:

    def test_load_config_from_resolved_meta(self):
        meta = {
            "experiments": [
                {"config": {"name": "test", "function_name": "f", "function_kwargs": {"a": 1}}}
            ]
        }
        config = load_config(resolve_config_dict(meta))
        assert isinstance(config, Config)
        assert config.name == "test"
        assert config.function_kwargs["a"] == 1

    def test_load_config_from_resolved_plain(self):
        d = {"name": "test", "function_name": "f"}
        config = load_config(resolve_config_dict(d))
        assert isinstance(config, Config)
        assert config.name == "test"
