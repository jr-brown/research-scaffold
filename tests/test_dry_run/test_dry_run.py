"""Tests for dry-run config resolution and reporting."""

import pytest
import yaml

from research_scaffold.config_tools import (
    build_configs,
    dry_run_report,
    execute_experiments,
    resolve_config_for_display,
)
from research_scaffold.types import Config, InstanceConfig


@pytest.fixture
def write_yaml(tmp_path):
    def _write(name, data):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.dump(data))
        return str(p)

    return _write


class TestBuildConfigs:

    def test_single_config(self):
        resolved = build_configs({"name": "a", "function_name": "f"})
        assert [c.name for c in resolved.configs] == ["a"]
        assert resolved.sweep_dicts == []

    def test_meta_config_via_c_flag(self):
        meta = {
            "experiments": [
                {"config_axes": [
                    [{"name": "a", "function_name": "f"}, {"name": "b", "function_name": "f"}],
                    [{"function_kwargs": {"lr": 0.1}}, {"function_kwargs": {"lr": 0.2}}],
                ]}
            ]
        }
        resolved = build_configs(config_path=meta)
        assert len(resolved.configs) == 4
        assert resolved.sweep_dicts == []

    def test_meta_config_via_m_flag(self):
        meta = {"experiments": [{"config": {"name": "a", "function_name": "f"}}]}
        assert [c.name for c in build_configs(meta_config_path=meta).configs] == ["a"]

    def test_composition_applied(self):
        meta = {
            "common_root": {"name": "root", "function_name": "f", "function_kwargs": {"lr": 0.01}},
            "common_patch": {"function_kwargs": {"batch_size": 64}},
            "experiments": [{"config": {"function_kwargs": {"lr": 0.5}}}],
        }
        configs = build_configs(meta_config_path=meta).configs
        assert configs[0].function_kwargs == {"lr": 0.5, "batch_size": 64}

    def test_sweep_config(self):
        sweep = {"method": "grid", "parameters": {"lr": {"values": [1, 2]}}}
        resolved = build_configs(sweep_config_path=sweep)
        assert resolved.configs == []
        assert resolved.sweep_dicts == [sweep]

    def test_sweep_detected_from_c_flag(self):
        sweep = {"method": "grid", "parameters": {"lr": {"values": [1, 2]}}}
        resolved = build_configs(config_path=sweep)
        assert resolved.configs == []
        assert len(resolved.sweep_dicts) == 1

    def test_meta_with_sweep_experiment(self):
        meta = {
            "experiments": [
                {"config": {"name": "a", "function_name": "f"}},
                {"sweep_config": {"method": "grid", "parameters": {"lr": {"values": [1]}}}},
            ]
        }
        resolved = build_configs(meta_config_path=meta)
        assert len(resolved.configs) == 1
        assert len(resolved.sweep_dicts) == 1

    def test_nothing_specified(self):
        resolved = build_configs()
        assert (resolved.configs, resolved.sweep_dicts) == ([], [])

    def test_parallel_flag_carried_from_meta(self):
        meta = {"parallel": True, "experiments": [{"config": {"name": "a", "function_name": "f"}}]}
        assert build_configs(meta_config_path=meta).parallel is True


class TestResolveForDisplay:

    def test_placeholders_resolved(self):
        config = Config(
            name="expt",
            function_name="f",
            time_stamp_group=True,
            log_file_path="outputs/RUN_GROUP/RUN_NAME.log",
        )
        d = resolve_config_for_display(config, launch_time_stamp="STAMP")
        assert d["log_file_path"] == "outputs/expt_STAMP/expt.log"
        assert d["wandb_group"] == "expt_STAMP"

    def test_unset_fields_dropped(self):
        d = resolve_config_for_display(Config(name="a", function_name="f"))
        assert d == {"name": "a", "function_name": "f", "wandb_group": "a"}

    def test_instance_included(self):
        config = Config(
            name="a",
            function_name="f",
            instance=InstanceConfig(sky_config="sky.yaml", managed=True),
        )
        d = resolve_config_for_display(config)
        assert d["instance"]["sky_config"] == "sky.yaml"
        assert d["instance"]["managed"] is True


class TestDryRunReport:

    def test_report_lists_all_configs(self):
        meta = {
            "experiments": [
                {"config_options": [
                    {"name": "a", "function_name": "f"},
                    {"name": "b", "function_name": "f"},
                ]}
            ]
        }
        report = dry_run_report(build_configs(meta_config_path=meta))
        assert "2 config(s), 0 sweep(s)" in report
        assert "Config 1/2" in report and "Config 2/2" in report
        assert [d["name"] for d in yaml.safe_load_all(report)] == ["a", "b"]

    def test_report_includes_sweep_and_base(self, write_yaml):
        base = write_yaml("base.yaml", {
            "name": "base",
            "function_name": "f",
            "function_kwargs": {"out": "runs/SWEEP_NAME"},
        })
        sweep = {
            "sweep_name": "my_sweep",
            "method": "grid",
            "parameters": {"lr": {"values": [1]}},
            "base_config": base,
        }
        report = dry_run_report(build_configs(sweep_config_path=sweep))
        assert "0 config(s), 1 sweep(s)" in report
        (doc,) = yaml.safe_load_all(report)
        assert doc["base_config"]["function_kwargs"]["out"] == "runs/my_sweep"
        assert doc["sweep"]["method"] == "grid"

    def test_whole_report_parses_as_multi_document_yaml(self):
        meta = {
            "experiments": [
                {"config": {"name": "a", "function_name": "f"}},
                {"sweep_config": {"method": "grid", "parameters": {"lr": {"values": [1]}}}},
            ]
        }
        docs = list(yaml.safe_load_all(dry_run_report(build_configs(meta_config_path=meta))))
        assert docs[0] == {"name": "a", "function_name": "f", "wandb_group": "a"}
        assert "sweep" in docs[1]


class TestDryRunExecution:

    def test_function_is_not_called(self, capsys, write_yaml):
        called = []
        path = write_yaml("cfg.yaml", {
            "name": "a",
            "function_name": "f",
            "function_kwargs": {"x": 1},
        })
        execute_experiments(
            function_map={"f": lambda **kw: called.append(kw)},
            config_path=path,
            dry_run=True,
        )
        assert called == []
        assert "name: a" in capsys.readouterr().out

    def test_remote_config_not_launched(self, capsys, write_yaml):
        path = write_yaml("remote.yaml", {
            "name": "a",
            "function_name": "f",
            "instance": {"sky_config": "sky.yaml"},
        })
        execute_experiments(function_map={}, config_path=path, dry_run=True)
        assert "sky_config: sky.yaml" in capsys.readouterr().out

    def test_no_config_specified(self, capsys):
        execute_experiments(function_map={}, dry_run=True)
        assert capsys.readouterr().out == ""

    def test_sweep_is_not_created(self, mock_wandb, capsys, write_yaml):
        """A dry run of a sweep config must never reach wandb.sweep/agent."""
        base = write_yaml("base.yaml", {"name": "b", "function_name": "f"})
        path = write_yaml("sweep.yaml", {
            "method": "bayes",
            "parameters": {"lr": {"values": [1, 2]}},
            "sweep_name": "my_sweep",
            "project": "proj",
            "base_config": base,
        })
        for kwargs in [{"config_path": path}, {"sweep_config_path": path}]:
            execute_experiments(function_map={"f": lambda: None}, dry_run=True, **kwargs)
            assert "1 sweep(s)" in capsys.readouterr().out

        mock_wandb["sweep"].assert_not_called()
        mock_wandb["agent"].assert_not_called()
        mock_wandb["init"].assert_not_called()

        # Positive control: the same config without dry_run does create the sweep,
        # so the assertions above cannot pass vacuously.
        execute_experiments(function_map={"f": lambda: None}, config_path=path)
        mock_wandb["sweep"].assert_called_once()

    def test_sweep_in_meta_config_is_not_created(self, mock_wandb, write_yaml):
        path = write_yaml("meta.yaml", {
            "experiments": [
                {"config": {"name": "a", "function_name": "f"}},
                {"sweep_config": {"method": "grid", "parameters": {"lr": {"values": [1]}}, "project": "p"}},
            ]
        })
        execute_experiments(function_map={"f": lambda: None}, meta_config_path=path, dry_run=True)
        mock_wandb["sweep"].assert_not_called()
        mock_wandb["agent"].assert_not_called()

    def test_unknown_function_name_warns(self, caplog, write_yaml):
        path = write_yaml("cfg.yaml", {"name": "a", "function_name": "typo"})
        execute_experiments(function_map={"real_f": lambda: None}, config_path=path, dry_run=True)
        assert "typo" in caplog.text and "not in function_map" in caplog.text

    def test_known_function_name_does_not_warn(self, caplog, write_yaml):
        path = write_yaml("cfg.yaml", {"name": "a", "function_name": "f"})
        execute_experiments(function_map={"f": lambda: None}, config_path=path, dry_run=True)
        assert "not in function_map" not in caplog.text

    def test_meta_config_type_detected_from_c_flag(self, capsys, write_yaml):
        path = write_yaml("meta.yaml", {
            "experiments": [{"config_options": [
                {"name": "a", "function_name": "f"},
                {"name": "b", "function_name": "f"},
            ]}]
        })
        execute_experiments(function_map={}, config_path=path, dry_run=True)
        out = capsys.readouterr().out
        assert "2 config(s)" in out
        assert "name: a" in out and "name: b" in out
