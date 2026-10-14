# Changelog

One line per user-visible change: config keys added or renamed, behaviour changes, breaking changes.

## Unreleased

- `instance.keep_alive_minutes`: a named cluster that is UP takes the job via `sky.exec` (no provisioning or setup) instead of skipping the config; a newly launched one is torn down only after that many idle minutes, and `sync` watchers follow their own job id and leave it up

- Config paths starting with `//` resolve against `RESEARCH_SCAFFOLD_CONFIG_ROOT` (or the working directory if unset) instead of the working directory

- `output_dir` config field: when set, `log_file_path` (default `logs.txt`) and `save_config_path` (default `config.yaml`) resolve relative to it

## 0.2.0 - 2026-09-07

- SkyPilot and vastai-sdk are no longer installed by default. Remote execution (`instance` block) now needs the `remote` extra: `research-scaffold[remote]`.

## 0.1.0 - 2026-09-03

First tagged release. Recent notable features:

- `<replace>: true` marker replaces a config block instead of merging into it
- `-d` dry-run flag prints fully composed configs without executing
- `parallel`, `max_concurrent`, `start_method` meta-config fields for local parallel execution
- `instance.sync` rsyncs result folders back after remote jobs; watcher only tears down on a confirmed terminal job status
- `vast_filters` sky-config key for provisioning-side offer filtering
- Cluster names default to the sanitized run name
