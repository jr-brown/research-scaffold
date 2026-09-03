# Changelog

One line per user-visible change: config keys added or renamed, behaviour changes, breaking changes.

## Unreleased

## 0.1.0 - 2026-09-03

First tagged release. Recent notable features:

- `<replace>: true` marker replaces a config block instead of merging into it
- `-d` dry-run flag prints fully composed configs without executing
- `parallel`, `max_concurrent`, `start_method` meta-config fields for local parallel execution
- `instance.sync` rsyncs result folders back after remote jobs; watcher only tears down on a confirmed terminal job status
- `vast_filters` sky-config key for provisioning-side offer filtering
- Cluster names default to the sanitized run name
