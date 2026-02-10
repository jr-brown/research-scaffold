# research_scaffold SkyPilot API Compatibility Issues

## Problem

`research_scaffold/remote_execution.py` was written against a SkyPilot API that
uses an async request/response pattern (`sky.get(request_id)`). SkyPilot 0.11.1
has a synchronous API — `sky.get()` does not exist.

## Affected Functions

### 1. `get_existing_cluster()` (lines 290-309)

**Current code:**
```python
request_id = sky.status()
clusters = sky.get(request_id)
for cluster in clusters:
    if cluster.name == cluster_name:
        return cluster
```

**Issue:** `sky.status()` in 0.11.1 returns `List[Dict[str, Any]]` directly, not
a request_id. `sky.get()` does not exist.

**Fix:** Call `sky.status()` directly and handle dict-based cluster records:
```python
clusters = sky.status()
for cluster in clusters:
    if cluster['name'] == cluster_name:
        return cluster
```

`sky.status()` also accepts `cluster_names=` to filter:
```python
clusters = sky.status(cluster_names=[cluster_name])
if clusters:
    return clusters[0]
```

### 2. `launch_remote_job()` (lines 404-421)

**Current code:**
```python
request_id = sky.launch(
    task,
    cluster_name=cluster_name,
    down=True,
)
# ...
sky.get(request_id)  # Blocks until cluster is UP and job is submitted
```

**Issue:** `sky.launch()` in 0.11.1 returns `Tuple[Optional[int], Optional[ResourceHandle]]`
(job_id, handle), not a request_id. The call already blocks by default
(`stream_logs=True`), so the `sky.get()` call is redundant.

**Fix:** Destructure the return value and remove the `sky.get()` call:
```python
job_id, handle = sky.launch(
    task,
    cluster_name=cluster_name,
    down=True,
)
# sky.launch() already blocks until cluster is UP and job is submitted
# (stream_logs=True by default), so no need for sky.get()
```

Update the return statement and downstream code to use `job_id` (int) instead of
`request_id` (str).

### 3. `start_log_streaming()` (lines 20-64)

**Current code (spawned subprocess):**
```python
sky.stream_and_get("{request_id}", output_stream=f)
sky.tail_logs("{cluster_name}", job_id=None, follow=True, output_stream=f)
```

**Issue:** `sky.stream_and_get()` does not exist in 0.11.1. Since `sky.launch()`
now blocks and streams logs directly, this function may need rethinking.

**Fix options:**
- Use `sky.tail_logs(cluster_name, job_id=job_id, follow=True)` directly
  (skipping the launch-log streaming which `sky.launch` already handles)
- Or use `sky.download_logs(cluster_name, job_ids=[job_id])` after completion

### 4. `execute_config_remotely()` (line 501)

Uses `request_id` from `launch_remote_job()` to call `start_log_streaming()`.
Update to pass `job_id` instead.

## SkyPilot 0.11.1 API Reference

Key functions and their signatures:

```python
sky.status(
    cluster_names: Optional[Union[str, List[str]]] = None,
    refresh: bool = False,
) -> List[Dict[str, Any]]
# Returns list of cluster dicts with keys: 'name', 'status', 'handle', etc.

sky.launch(
    task: Union[sky.Task, sky.Dag],
    cluster_name: Optional[str] = None,
    retry_until_up: bool = False,
    idle_minutes_to_autostop: Optional[int] = None,
    dryrun: bool = False,
    down: bool = False,
    stream_logs: bool = True,  # Blocks until job submitted when True
    detach_setup: bool = False,
    detach_run: bool = False,
    # ...
) -> Tuple[Optional[int], Optional[ResourceHandle]]
# Returns (job_id, handle). Blocks by default (stream_logs=True).

sky.tail_logs(
    cluster_name: str,
    job_id: Optional[int] = None,
    follow: bool = True,
) -> None

sky.download_logs(
    cluster_name: str,
    job_ids: Optional[List[int]] = None,
) -> Dict[str, str]
# Returns {job_id: local_log_dir}
```

## Second Issue: Python Version on Remote

The `runpod/base:0.0.2` image used in `sky_config.yaml` ships Python 3.10.13.
The project requires `>=3.11` (in `pyproject.toml`), so `pip install -e .` fails
on the remote with:

```
ERROR: Package 'llm-sft-alignment-research' requires a different Python: 3.10.13 not in '>=3.11'
```

**Fix options:**
1. Use a Docker image with Python 3.11+ (e.g., `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
2. Install Python 3.11+ in the `setup:` block before `pip install -e .`
3. Use `uv` on the remote (can target a specific Python version)

To set the image in `sky_config.yaml`:
```yaml
resources:
  cloud: runpod
  accelerators: A40:1
  image_id: docker:runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```
