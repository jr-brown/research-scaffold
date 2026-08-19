"""Tests for remote execution, including managed jobs."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from sky.utils.common_utils import check_cluster_name_is_valid

from research_scaffold.types import InstanceConfig, Config
from research_scaffold.remote_execution import (
    _build_sky_task,
    launch_remote_job,
    launch_managed_job,
    start_managed_log_streaming,
    execute_config_remotely,
    execute_sweep_remotely,
    start_log_streaming,
    sanitize_cluster_name,
    start_sync_back,
    SYNC_AUTOSTOP_MINUTES,
)

TEST_DIR = Path(__file__).parent
SKY_CONFIG_PATH = str(TEST_DIR / "sky_config.yaml")


# --- Cluster name sanitizing ---


class TestSanitizeClusterName:
    @pytest.mark.parametrize("raw", [
        "my_experiment_2026-07-31_16-31-24",  # typical resolved run name
        "3b_model_run",                       # starts with a digit
        "run_",                               # ends with an underscore
        "lr=0.001,bs=32",                     # characters from a param sweep name
        "exp/v2 final",                       # slash and space
        "___",                                # nothing usable left
        "café_run",                           # non-ascii
        "a",                                  # single letter
    ])
    def test_output_passes_skypilots_own_validator(self, raw):
        """Checked against SkyPilot's real validator, so this fails if their rule changes."""
        check_cluster_name_is_valid(sanitize_cluster_name(raw))

    def test_already_valid_name_is_unchanged(self):
        name = "my_experiment_2026-07-31_16-31-24"
        assert sanitize_cluster_name(name) == name

    def test_distinct_names_stay_distinct(self):
        """Leading digits are prefixed, not stripped, so these must not collapse together."""
        assert sanitize_cluster_name("3b-model") != sanitize_cluster_name("b-model")


# --- InstanceConfig tests ---


class TestInstanceConfigManaged:
    def test_instance_config_managed_default(self):
        """InstanceConfig.managed defaults to False."""
        ic = InstanceConfig()
        assert ic.managed is False

    def test_instance_config_managed_true(self):
        """InstanceConfig.managed can be set to True."""
        ic = InstanceConfig(managed=True)
        assert ic.managed is True


# --- _build_sky_task tests ---


class TestBuildSkyTask:
    def test_build_sky_task(self, mock_sky, mock_git_info):
        """_build_sky_task loads config, applies patch, injects run command, returns Task."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)
        task = _build_sky_task(ic, "echo hello", "test-cluster")

        # Should have called Task.from_yaml with a temp file
        mock_sky['Task'].from_yaml.assert_called_once()
        assert task == mock_sky['task']

    def test_build_sky_task_injects_git_commit(self, mock_sky, mock_git_info):
        """GIT_COMMIT env var is set when git_commit is specified."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, git_commit="abc12345")

        # We need to inspect the yaml written to the temp file
        # Patch tempfile to capture what gets written
        import yaml

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
            _build_sky_task(ic, "echo test", "test-cluster")

        assert len(written_configs) == 1
        sky_config = written_configs[0]
        assert sky_config['envs']['GIT_COMMIT'] == "abc12345"

    def test_build_sky_task_no_git_commit(self, mock_sky, mock_git_info):
        """GIT_COMMIT env var is NOT set when git_commit is None."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)

        import yaml

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
            _build_sky_task(ic, "echo test", "test-cluster")

        sky_config = written_configs[0]
        assert 'envs' not in sky_config or 'GIT_COMMIT' not in sky_config.get('envs', {})

    def test_build_sky_task_appends_run_command(self, mock_sky, mock_git_info):
        """Run command is appended to existing run block."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)

        import yaml

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
            _build_sky_task(ic, "echo injected", "test-cluster")

        sky_config = written_configs[0]
        assert "echo injected" in sky_config['run']
        # The original run content should still be present
        assert "echo" in sky_config['run']

    def test_build_sky_task_sets_workdir(self, mock_sky, mock_git_info):
        """workdir is set to the repo root."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)

        import yaml

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
            _build_sky_task(ic, "echo test", "test-cluster")

        sky_config = written_configs[0]
        assert sky_config['workdir'] == mock_git_info['repo_root']

    def test_build_sky_task_falls_back_to_sky_path_env(self, mock_sky, mock_git_info):
        """Falls back to SKY_PATH env var when sky_config is None."""
        ic = InstanceConfig(sky_config=None)

        with patch.dict(os.environ, {'SKY_PATH': SKY_CONFIG_PATH}):
            task = _build_sky_task(ic, "echo test", "test-cluster")

        mock_sky['Task'].from_yaml.assert_called_once()

    def test_build_sky_task_raises_without_config(self, mock_sky, mock_git_info):
        """Raises RuntimeError when no sky_config and no SKY_PATH."""
        ic = InstanceConfig(sky_config=None)

        with patch.dict(os.environ, {}, clear=True):
            # Remove SKY_PATH if present
            os.environ.pop('SKY_PATH', None)
            with pytest.raises(RuntimeError, match="No sky_config specified"):
                _build_sky_task(ic, "echo test", "test-cluster")

    def test_build_sky_task_applies_patch(self, mock_sky, mock_git_info, tmp_path):
        """Patch is applied to sky config."""
        import yaml

        # Create a patch file
        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text(yaml.dump({"resources": {"accelerators": "A100:1"}}))

        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, patch=str(patch_file))

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
            _build_sky_task(ic, "echo test", "test-cluster")

        sky_config = written_configs[0]
        assert sky_config['resources']['accelerators'] == "A100:1"


class TestVastFilters:
    @staticmethod
    def _build_capturing(ic, cluster_name, home):
        import yaml

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch.dict(os.environ, {'HOME': str(home)}):
            with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
                _build_sky_task(ic, "echo test", cluster_name)

        return written_configs[0]

    def test_vast_filters_written_and_stripped(self, mock_sky, mock_git_info, tmp_path):
        """vast_filters is removed from the task YAML and written to the registry."""
        import yaml

        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text(yaml.dump({"vast_filters": "cuda_max_good>=13.0 gpu_ram>=130"}))
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, patch=str(patch_file))

        sky_config = self._build_capturing(ic, "my-run", tmp_path)

        assert 'vast_filters' not in sky_config
        registry = tmp_path / ".sky" / "vast_filters" / "my-run"
        assert registry.read_text() == "cuda_max_good>=13.0 gpu_ram>=130"

    def test_patch_overrides_sky_config_filters(self, mock_sky, mock_git_info, tmp_path):
        """A patch's vast_filters replaces the value from the shared sky config."""
        import yaml

        base = tmp_path / "sky.yaml"
        base.write_text(yaml.dump({"run": "echo base", "vast_filters": "gpu_ram>=40"}))
        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text(yaml.dump({"vast_filters": "gpu_ram>=130"}))
        ic = InstanceConfig(sky_config=str(base), patch=str(patch_file))

        self._build_capturing(ic, "my-run", tmp_path)

        assert (tmp_path / ".sky" / "vast_filters" / "my-run").read_text() == "gpu_ram>=130"

    def test_empty_vast_filters_writes_nothing(self, mock_sky, mock_git_info, tmp_path):
        """An empty string is treated like an absent key."""
        import yaml

        patch_file = tmp_path / "patch.yaml"
        patch_file.write_text(yaml.dump({"vast_filters": ""}))
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, patch=str(patch_file))

        sky_config = self._build_capturing(ic, "my-run", tmp_path)

        assert 'vast_filters' not in sky_config
        assert not (tmp_path / ".sky").exists()


# --- launch_remote_job tests ---


class TestLaunchRemoteJob:
    def test_launch_remote_job_uses_sky_launch(self, mock_sky, mock_git_info):
        """launch_remote_job calls sky.launch() with correct args."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)

        cluster_name, request_id, was_running = launch_remote_job(ic, "test-job", "echo hello")

        mock_sky['launch'].assert_called_once()
        call_kwargs = mock_sky['launch'].call_args
        assert call_kwargs.kwargs['down'] is True
        assert call_kwargs.kwargs['cluster_name'] == cluster_name
        assert was_running is False
        assert request_id is not None

    def test_launch_remote_job_blocks_with_sky_get(self, mock_sky, mock_git_info):
        """launch_remote_job calls sky.get() to block until cluster is UP."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)

        launch_remote_job(ic, "test-job", "echo hello")

        # sky.get should be called with the request_id from sky.launch
        mock_sky['get'].assert_called_once_with("req-launch-123")

    def test_launch_remote_job_generates_cluster_name(self, mock_sky, mock_git_info):
        """Generates a uuid-based cluster name when instance_config.name is None."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, name=None)

        cluster_name, _, _ = launch_remote_job(ic, "test-job", "echo hello")

        assert cluster_name.startswith("c")
        assert len(cluster_name) == 9  # c + 8 hex chars

    def test_launch_remote_job_uses_instance_name(self, mock_sky, mock_git_info):
        """Uses instance_config.name as cluster name."""
        # Mock get_existing_cluster to return None (no existing cluster)
        with patch('research_scaffold.remote_execution.get_existing_cluster', return_value=None):
            ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, name="my-cluster")

            cluster_name, _, _ = launch_remote_job(ic, "test-job", "echo hello")

            assert cluster_name == "my-cluster"


# --- launch_managed_job tests ---


class TestLaunchManagedJob:
    def test_launch_managed_job_calls_sky_jobs_launch(self, mock_sky, mock_git_info):
        """launch_managed_job calls sky.jobs.launch() (not sky.launch())."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True)

        managed_name, request_id = launch_managed_job(ic, "test-job", "echo hello")

        mock_sky['jobs_launch'].assert_called_once()
        mock_sky['launch'].assert_not_called()
        assert request_id == "req-managed-456"

    def test_launch_managed_job_no_blocking(self, mock_sky, mock_git_info):
        """launch_managed_job does NOT call sky.get()."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True)

        launch_managed_job(ic, "test-job", "echo hello")

        mock_sky['get'].assert_not_called()

    def test_launch_managed_job_uses_instance_name(self, mock_sky, mock_git_info):
        """Uses instance_config.name as managed job name when set."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name="my-managed-job")

        managed_name, _ = launch_managed_job(ic, "test-job", "echo hello")

        assert managed_name == "my-managed-job"
        # Verify the name was passed to sky.jobs.launch
        call_kwargs = mock_sky['jobs_launch'].call_args
        assert call_kwargs.kwargs['name'] == "my-managed-job"

    def test_launch_managed_job_generates_name(self, mock_sky, mock_git_info):
        """Generates a uuid-based name when instance_config.name is None."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name=None)

        managed_name, _ = launch_managed_job(ic, "test-job", "echo hello")

        assert managed_name.startswith("j")
        assert len(managed_name) == 9  # j + 8 hex chars

    def test_launch_managed_job_injects_git_commit(self, mock_sky, mock_git_info):
        """GIT_COMMIT env var is set when git_commit is specified in managed mode."""
        import yaml

        ic = InstanceConfig(
            sky_config=SKY_CONFIG_PATH,
            managed=True,
            git_commit="deadbeef12345678",
        )

        written_configs = []
        original_dump = yaml.dump

        def capture_dump(data, stream, **kwargs):
            written_configs.append(data)
            return original_dump(data, stream, **kwargs)

        with patch('research_scaffold.remote_execution.yaml.dump', side_effect=capture_dump):
            launch_managed_job(ic, "test-job", "echo hello")

        sky_config = written_configs[0]
        assert sky_config['envs']['GIT_COMMIT'] == "deadbeef12345678"


# --- start_managed_log_streaming tests ---


class TestManagedLogStreaming:
    def test_managed_log_streaming(self, tmp_path):
        """Subprocess is spawned with correct sky.jobs.tail_logs script."""
        import subprocess
        real_popen_cls = subprocess.Popen

        log_path = str(tmp_path / "logs" / "managed.log")

        with patch('research_scaffold.remote_execution.subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock(spec=real_popen_cls)
            process = start_managed_log_streaming("my-job", log_path)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        # The script should reference sky.jobs.tail_logs with the job name
        script = call_args[0][0][2]  # [sys.executable, '-c', script]
        assert "sky.jobs.tail_logs" in script
        assert "my-job" in script
        assert log_path in script

        # Log directory should be created
        assert os.path.isdir(str(tmp_path / "logs"))


# --- execute_config_remotely tests ---


class TestExecuteConfigRemotely:
    def _make_config(self, **kwargs):
        defaults = {
            "name": "test-experiment",
            "function_name": "train",
        }
        defaults.update(kwargs)
        return Config(**defaults)

    def test_execute_config_remotely_managed(self, mock_sky, mock_git_info, tmp_path):
        """Managed path: uses launch_managed_job, not launch_remote_job."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name="managed-exp")
        config = self._make_config()
        resolved_names = {"name": "test-experiment", "name_base": "test-experiment", "group": "", "sweep_name": ""}

        with patch('research_scaffold.remote_execution.get_existing_cluster', return_value=None):
            result = execute_config_remotely(ic, config, resolved_names)

        # Should use managed jobs
        mock_sky['jobs_launch'].assert_called_once()
        mock_sky['launch'].assert_not_called()
        mock_sky['get'].assert_not_called()
        assert result == "managed-exp"

    def test_execute_config_remotely_non_managed(self, mock_sky, mock_git_info, tmp_path):
        """Non-managed path: uses launch_remote_job with sky.launch + sky.get blocking."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=False)
        config = self._make_config()
        resolved_names = {"name": "test-experiment", "name_base": "test-experiment", "group": "", "sweep_name": ""}

        result = execute_config_remotely(ic, config, resolved_names)

        # Should use standard launch
        mock_sky['launch'].assert_called_once()
        # sky.get now also resolves the cluster-existence query, so assert the blocking
        # call on the launch request specifically rather than a bare call count
        mock_sky['get'].assert_any_call("req-launch-123")
        mock_sky['jobs_launch'].assert_not_called()

    def _resolved(self, name):
        return {"name": name, "name_base": name, "group": "", "sweep_name": ""}

    def test_cluster_name_defaults_to_run_name(self, mock_sky, mock_git_info):
        """With no instance name set, the cluster is named after the resolved run."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True)

        result = execute_config_remotely(ic, self._make_config(), self._resolved("expt_a"))

        assert result == "expt_a"

    def test_batch_configs_get_distinct_clusters(self, mock_sky, mock_git_info):
        """The regression this default exists for: distinct configs must not share a cluster
        name, or all but the first are silently skipped."""
        names = []
        for run_name in ["expt_variant_a", "expt_variant_b"]:
            ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True)
            names.append(execute_config_remotely(ic, self._make_config(), self._resolved(run_name)))

        assert names == ["expt_variant_a", "expt_variant_b"]
        assert len(set(names)) == 2

    def test_skipped_launch_warns_that_config_did_not_start(self, mock_sky, mock_git_info, caplog):
        """A skip must be loud: it means this config did not run at all."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=False)

        with patch('research_scaffold.remote_execution.get_existing_cluster', return_value={'status': 'UP'}):
            result = execute_config_remotely(ic, self._make_config(), self._resolved("expt_a"))

        assert result == "expt_a"
        mock_sky['launch'].assert_not_called()
        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("not launched" in m for m in warnings), warnings

    def test_explicit_name_wins_and_substitutes_run_name(self, mock_sky, mock_git_info):
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name="gpu-RUN_NAME")

        result = execute_config_remotely(ic, self._make_config(), self._resolved("expt_a"))

        assert result == "gpu-expt_a"

    def test_derived_name_is_sanitized(self, mock_sky, mock_git_info):
        """A run name that is not a legal cluster name must not reach SkyPilot raw."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True)

        result = execute_config_remotely(ic, self._make_config(), self._resolved("3b/model run"))

        check_cluster_name_is_valid(result)
        assert result == "c-3b-model-run"

    def test_execute_config_remotely_managed_with_log_streaming(self, mock_sky, mock_git_info, tmp_path):
        """Managed path streams logs via start_managed_log_streaming."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name="managed-exp")
        config = self._make_config(log_file_path=str(tmp_path / "output.log"))
        resolved_names = {"name": "test-experiment", "name_base": "test-experiment", "group": "", "sweep_name": ""}

        with patch('research_scaffold.remote_execution.start_managed_log_streaming') as mock_stream:
            execute_config_remotely(ic, config, resolved_names)

            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args[0][0] == "managed-exp"  # managed_job_name


# --- sync-back tests ---


class TestSyncBack:
    def _make_config(self, **kwargs):
        defaults = {"name": "test-experiment", "function_name": "train"}
        defaults.update(kwargs)
        return Config(**defaults)

    def _resolved(self, name):
        return {"name": name, "name_base": name, "group": "", "sweep_name": ""}

    def test_start_sync_back_script_is_valid_and_complete(self, tmp_path):
        import subprocess
        real_popen_cls = subprocess.Popen
        with patch('research_scaffold.remote_execution.subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock(spec=real_popen_cls)
            start_sync_back("my-cluster", ["outputs"], str(tmp_path), "", str(tmp_path / "sync.log"))

        script = mock_popen.call_args[0][0][2]
        compile(script, '<sync-script>', 'exec')  # must be syntactically valid python
        assert "sky.tail_logs" in script
        assert "rsync" in script
        assert "sky.down" in script
        assert "'my-cluster'" in script
        assert "sky_workdir/outputs" in script
        assert str(tmp_path / "outputs") in script

    def test_start_sync_back_respects_rel_cwd(self, tmp_path):
        import subprocess
        real_popen_cls = subprocess.Popen
        with patch('research_scaffold.remote_execution.subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock(spec=real_popen_cls)
            start_sync_back("my-cluster", ["outputs"], str(tmp_path), "example", str(tmp_path / "sync.log"))

        script = mock_popen.call_args[0][0][2]
        assert "sky_workdir/example/outputs" in script

    def test_sync_launches_with_autostop_and_starts_watcher(self, mock_sky, mock_git_info):
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, sync=["outputs/RUN_NAME"])

        with patch('research_scaffold.remote_execution.start_sync_back') as mock_sync:
            execute_config_remotely(ic, self._make_config(), self._resolved("expt_a"))

        assert mock_sky['launch'].call_args.kwargs['idle_minutes_to_autostop'] == SYNC_AUTOSTOP_MINUTES
        mock_sync.assert_called_once()
        assert mock_sync.call_args[0][1] == ["outputs/expt_a"]  # placeholder substituted

    def test_no_sync_launches_without_autostop(self, mock_sky, mock_git_info):
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH)

        with patch('research_scaffold.remote_execution.start_sync_back') as mock_sync:
            execute_config_remotely(ic, self._make_config(), self._resolved("expt_a"))

        assert mock_sky['launch'].call_args.kwargs['idle_minutes_to_autostop'] is None
        mock_sync.assert_not_called()

    def test_sync_with_managed_raises(self, mock_sky, mock_git_info):
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, sync=["outputs"])

        with pytest.raises(RuntimeError, match="not supported with managed jobs"):
            execute_config_remotely(ic, self._make_config(), self._resolved("expt_a"))

    def test_sweep_sync_starts_watcher(self, mock_sky, mock_git_info):
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, sync=["outputs"])
        sweep_dict = {"method": "random", "parameters": {"lr": {"values": [0.01]}}}

        with patch('research_scaffold.remote_execution.start_sync_back') as mock_sync:
            execute_sweep_remotely(ic, sweep_dict, "test-sweep", resolved_names=self._resolved("test-sweep"))

        mock_sync.assert_called_once()


# --- execute_sweep_remotely tests ---


class TestExecuteSweepRemotely:
    def test_execute_sweep_remotely_managed(self, mock_sky, mock_git_info):
        """Managed path for sweeps: uses sky.jobs.launch, not sky.launch."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name="sweep-job")
        sweep_dict = {"method": "random", "parameters": {"lr": {"values": [0.01, 0.001]}}}
        resolved_names = {"name": "test-sweep", "name_base": "test-sweep", "group": "", "sweep_name": "test-sweep"}

        result = execute_sweep_remotely(ic, sweep_dict, "test-sweep", resolved_names=resolved_names)

        mock_sky['jobs_launch'].assert_called_once()
        mock_sky['launch'].assert_not_called()
        mock_sky['get'].assert_not_called()
        assert result == "sweep-job"

    def test_execute_sweep_remotely_non_managed(self, mock_sky, mock_git_info):
        """Non-managed sweep: uses standard sky.launch + blocking."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=False)
        sweep_dict = {"method": "random", "parameters": {"lr": {"values": [0.01, 0.001]}}}
        resolved_names = {"name": "test-sweep", "name_base": "test-sweep", "group": "", "sweep_name": "test-sweep"}

        execute_sweep_remotely(ic, sweep_dict, "test-sweep", resolved_names=resolved_names)

        mock_sky['launch'].assert_called_once()
        # sky.get now also resolves the cluster-existence query, so assert the blocking
        # call on the launch request specifically rather than a bare call count
        mock_sky['get'].assert_any_call("req-launch-123")
        mock_sky['jobs_launch'].assert_not_called()

    def test_execute_sweep_remotely_managed_with_log_streaming(self, mock_sky, mock_git_info, tmp_path):
        """Managed sweep streams logs via start_managed_log_streaming."""
        ic = InstanceConfig(sky_config=SKY_CONFIG_PATH, managed=True, name="sweep-job")
        sweep_dict = {"method": "random", "parameters": {"lr": {"values": [0.01]}}}
        resolved_names = {"name": "test-sweep", "name_base": "test-sweep", "group": "", "sweep_name": "test-sweep"}
        log_path = str(tmp_path / "sweep.log")

        with patch('research_scaffold.remote_execution.start_managed_log_streaming') as mock_stream:
            execute_sweep_remotely(
                ic, sweep_dict, "test-sweep",
                resolved_names=resolved_names,
                log_file_path=log_path,
            )

            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args[0][0] == "sweep-job"
