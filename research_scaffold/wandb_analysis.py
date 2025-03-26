import logging

import numpy as np

from tqdm import tqdm
from pprint import pformat
from typing import Optional, Any

from research_scaffold.util import (
    nones_to_empty_dicts,
    recursive_dict_update,
)
from .wandb_run_processing import (
    get_runs_from_wandb,
    get_run_metrics,
    filter_runs_by_regexes,
)
from .plotting import smooth, get_plot_configs, plot_graph


Array = np.ndarray


log = logging.getLogger(__name__)


def average_runs(
    runs_data: list,  # Vaguely list[tuple[list, list]] but actually arrays not tuples
    source: str="unknown",
    samples=500,
    smoothing_factor: float=1.0,
    std_err_n_override: bool=True,
    scale_smoothing_on_longest_src_not_samples: bool=True,
):
    if runs_data == []:
        log.warning(f"No runs for {source=}")
        return None

    if sum(len(x) for x in runs_data) == 0:
        log.warning(f"Runs empty for {source}")
        return None

    min_time = min(min(v[0]) for v in runs_data)
    max_time = max(max(v[0]) for v in runs_data)

    ts = np.linspace(min_time, max_time, samples)

    if smoothing_factor < 1:
        log.warning(f"{smoothing_factor=} < 1 which may lead to weird results")

    elif smoothing_factor > 3 and not std_err_n_override:
        log.warning(f"{smoothing_factor=} > 3 and {std_err_n_override=} which may lead to compressed std error bounds to traditional methods")

    if scale_smoothing_on_longest_src_not_samples:
        shape_scaling = smoothing_factor * (max_time - min_time) / max(v.shape[1] for v in runs_data)
    else:
        shape_scaling = smoothing_factor * (max_time - min_time) / samples

    main_plot_data = smooth(
        *runs_data, desired_time_samples=ts, shape_scaling=shape_scaling,
        std_err_n_override=(len(runs_data) if std_err_n_override else None), output_as_dict=True
    )
    assert isinstance(main_plot_data, dict)

    raw_xs = [rd[1] for rd in runs_data]
    raw_ts = [rd[0] for rd in runs_data]
    main_plot_data["raw_mean"] = np.mean(raw_xs, axis=0)
    main_plot_data["raw_std_error"] = np.std(raw_xs, axis=0) / np.sqrt(len(raw_xs))
    main_plot_data["raw_time"] = np.mean(raw_ts, axis=0)
    main_plot_data["indices"] = np.array(range(len(runs_data[0][1])))  # Previously raw_time

    return main_plot_data


def plot_runs(
    name_regexes: dict[str, str],
    get_run_metrics_kwargs: dict[str, Any],
    runs: Optional[list] = None,
    get_runs_from_wandb_kwargs: dict[str, Any] | None = None,
    average_runs_kwargs: dict[str, Any] | None = None,
    get_plot_configs_kwargs: dict[str, Any] | None = None,
    plot_graph_kwargs: dict[str, Any] | None = None,
):
    (
        get_runs_from_wandb_kwargs,
        average_runs_kwargs,
        get_plot_configs_kwargs,
        plot_graph_kwargs,
    ) = nones_to_empty_dicts(
        get_runs_from_wandb_kwargs,
        average_runs_kwargs,
        get_plot_configs_kwargs,
        plot_graph_kwargs,
    )

    runs = runs or get_runs_from_wandb(**get_runs_from_wandb_kwargs)
    run_groups = filter_runs_by_regexes(runs, name_regexes)

    run_groups_and_sizes_str = "\n".join(
        [f"{k}: {len(runs)}" for k, runs in run_groups.items()]
    )
    log.info(f"Run groups and sizes:\n{run_groups_and_sizes_str}")

    group_iter = lambda x: tqdm(x, desc="Iterating through groups", leave=False)
    run_group_metrics = {
        k: get_run_metrics(
            rs,
            get_keys_separately_and_combine=True,
            **get_run_metrics_kwargs
        )
        for k, rs in group_iter(run_groups.items())
    }
    group_plot_data = {
        k: pd
        for k, rs in group_iter(run_group_metrics.items())
        if (pd := average_runs(rs, k, **average_runs_kwargs)) is not None
    }
    group_final_scores = {
        k: (pd["raw_mean"][-1], pd["raw_std_error"][-1])
        for k, pd in group_plot_data.items()
    }
    plot_name = plot_graph_kwargs.get(
        "title", plot_graph_kwargs.get("save_name", "unknown")
    )
    log.info(f"Group final scores for {plot_name}:\n{pformat(group_final_scores)}")

    # print(list(group_plot_data.values())[0])

    if group_final_scores == {}:
        log.info(f"{group_final_scores=}, will not plot")
        return

    plots = [
        pd
        for k in group_plot_data.keys()
        for pd in get_plot_configs(k, **get_plot_configs_kwargs)
    ]
    plot_graph(data=group_plot_data, plots=plots, **plot_graph_kwargs)


def plot_many_runs(
    plot_runs_kwargs_list: list[dict],
    runs: list | None = None,
    get_runs_from_wandb_kwargs: dict[str, Any] | None = None,
    plot_runs_kwargs_base: dict[str, Any] | None = None,
):
    (
        get_runs_from_wandb_kwargs,
        plot_runs_kwargs_base,
    ) = nones_to_empty_dicts(get_runs_from_wandb_kwargs, plot_runs_kwargs_base)

    runs = runs or get_runs_from_wandb(**get_runs_from_wandb_kwargs)

    for plot_runs_kwargs in tqdm(
        plot_runs_kwargs_list, desc="Making plots", leave=False
    ):
        plot_runs(
            runs=runs, **recursive_dict_update(plot_runs_kwargs_base, plot_runs_kwargs)
        )


def plot_final_run_statistics(
    name_regexes: dict[str, str],
    get_run_metrics_kwargs: dict[str, Any],
    runs: list | None = None,
    get_runs_from_wandb_kwargs: dict[str, Any] | None = None,
    average_runs_kwargs: dict[str, Any] | None = None,
    get_plot_configs_kwargs: dict[str, Any] | None = None,
    plot_graph_kwargs: dict[str, Any] | None = None,
):
    (
        get_runs_from_wandb_kwargs,
        average_runs_kwargs,
        get_plot_configs_kwargs,
        plot_graph_kwargs,
    ) = nones_to_empty_dicts(
        get_runs_from_wandb_kwargs,
        average_runs_kwargs,
        get_plot_configs_kwargs,
        plot_graph_kwargs,
    )

    runs = runs or get_runs_from_wandb(**get_runs_from_wandb_kwargs)
    run_groups = filter_runs_by_regexes(runs, name_regexes)

    run_groups_and_sizes_str = "\n".join(
        [f"{k}: {len(runs)}" for k, runs in run_groups.items()]
    )
    log.info(f"Run groups and sizes:\n{run_groups_and_sizes_str}")

    def process_metrics_to_plot_data(x_y_pairs):
        xys = np.stack(sorted(np.stack(x_y_pairs), key=lambda xs: xs[0])).T
        return average_runs([xys], **average_runs_kwargs)

    group_iter = lambda x: tqdm(x, desc="Iterating through groups", leave=False)
    group_plot_data = {
        k: process_metrics_to_plot_data(
            get_run_metrics(
                rs,
                get_keys_separately_and_combine=True,
                processing_fn=lambda xs: xs[::, -1],
                **get_run_metrics_kwargs,
            )
        )
        for k, rs in group_iter(run_groups.items())
    }
    plots = [
        pd
        for k in group_plot_data.keys()
        for pd in get_plot_configs(k, **get_plot_configs_kwargs)
    ]
    plot_graph(data=group_plot_data, plots=plots, **plot_graph_kwargs)

