import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from os import path
from math import floor
from typing import Optional, Any, Union

from .util import nones_to_empty_dicts, merge_dicts
from .file_io import load


log = logging.getLogger(__name__)


Array = np.ndarray

COLOURS = ["r", "b", "g", "y", "k", "m", "c"]
LINE_STYLES = ["-", "--", ":", "-."]


def get_format_func(format_type: str):
    if format_type == "k":
        return tck.FuncFormatter(lambda val, _: f"{int(val/1_000)}k")

    elif format_type == "M":
        return tck.FuncFormatter(lambda val, _: f"{int(val/1_000_000)}M")

    elif format_type == "B":
        return tck.FuncFormatter(lambda val, _: f"{int(val/1_000_000_000)}B")

    else:
        raise ValueError(f"{format_type=} not valid.")


def enforce_str_list(xs: None | str | list) -> Optional[list[str]]:
    return ([xs] if isinstance(xs, str) else xs) if xs is not None else None

def multi_enforce_str_lists(*strs_or_lists: None | str | list) -> list[Optional[list[str]]]:
    return [enforce_str_list(x) for x in strs_or_lists]


def key_list_get(_dict: dict, keys: list) -> Any:
    try:
        if len(keys) == 0:
            raise Exception("Empty keys")

        elif len(keys) == 1:
            return _dict[keys[0]]

        else:
            k = keys[0]
            next_keys = keys[1:]
            return key_list_get(_dict[k], next_keys)

    except TypeError as e:
        log.error(f"TypeError in key_list_get where {keys=} and {_dict=}")
        raise e


def get_plot_data(
    data: dict,
    keys_to_relevant: Union[None, str, list[str]]=None,
    x_key_chain_to_list: Union[None, str, list[str]]=None,
    y_key_chain_to_list: Union[None, str, list[str]]=None,
    err_key_chain_to_list:  Union[None, str, list[str]]=None,
    x_key_chain_from_list_to_val: Union[None, str, list[str]]=None,
    y_key_chain_from_list_to_val: Union[None, str, list[str]]=None,
    err_key_chain_from_list_to_val: Union[None, str, list[str]]=None,
    x_is_idx: bool=False,
    order_x_vals: bool=True,
    normalise: bool=False,
    is_scatter: bool=False,
    trim_start_idx: Optional[int]=None,
    trim_end_idx: Optional[int]=None,
    plot_style_num: Optional[int]=None,
    plot_kwargs: dict[str, Any] | None=None,
    label_join: str=' ',
):
    """
    Take data and some configuration and produce data to be given directly to plt.plot

    :param data: The source for getting data from
    :param x_vals: From each list, the path to the x_vals for the plot, None for index
    :param y_vals: From each list, the path to the y_vals for the plot, None for list element
    :param entries: Which entries from the dict to plot with
    :param keys_to_list: The path of keys to find the inner list of plot data in each dict
    :normalise: Whether to normalise y values
    :ignore_key_errors: Silently ignore key errors

    Having kwargs in function call allows for easy unwrapping of config dicts and outputting
        directly into plt.plot
    """

    # Arg pre-processing
    plot_kwargs, = nones_to_empty_dicts(plot_kwargs)
    (
        keys_to_relevant, x_key_chain_to_list, y_key_chain_to_list, err_key_chain_to_list,
        x_key_chain_from_list_to_val, y_key_chain_from_list_to_val, err_key_chain_from_list_to_val,
    ) = multi_enforce_str_lists (
        keys_to_relevant, x_key_chain_to_list, y_key_chain_to_list, err_key_chain_to_list,
        x_key_chain_from_list_to_val, y_key_chain_from_list_to_val, err_key_chain_from_list_to_val,
    )

    # Pre-process if pre_entries_key_chain
    if keys_to_relevant is not None:
        data = key_list_get(data, keys_to_relevant)

    if ("label" not in plot_kwargs) and (keys_to_relevant is not None):
        plot_kwargs["label"] = label_join.join(keys_to_relevant)

    x_list = (key_list_get(data, x_key_chain_to_list)
              if x_key_chain_to_list is not None
              else data)
    y_list = (key_list_get(data, y_key_chain_to_list)
              if y_key_chain_to_list is not None
              else data)

    if x_is_idx:
        assert x_key_chain_from_list_to_val is None
        x_points = range(len(x_list))

    else:
        x_points = (x_list if x_key_chain_from_list_to_val is None
                    else [key_list_get(v, x_key_chain_from_list_to_val) for v in x_list])

    y_points = (y_list if y_key_chain_from_list_to_val is None
                else [key_list_get(v, y_key_chain_from_list_to_val) for v in y_list])

    if err_key_chain_to_list is None and err_key_chain_from_list_to_val is None:
        err_vals = None
    else:
        err_list = (key_list_get(data, err_key_chain_to_list)
                    if err_key_chain_to_list is not None
                    else data)
        err_vals = (err_list if err_key_chain_from_list_to_val is None
                    else [key_list_get(v, err_key_chain_from_list_to_val) for v in err_list])

    if order_x_vals:
        sort_idxs = [i for i, _ in sorted(enumerate(x_points), key=lambda ix: ix[1])]
        x_points = [x_points[i] for i in sort_idxs]
        y_points = [y_points[i] for i in sort_idxs]
        err_vals = None if err_vals is None else [err_vals[i] for i in sort_idxs]

    if normalise:
        ys = np.array(y_points)
        mu, std = np.mean(ys), np.std(ys)
        y_points = list((ys - mu) / std)
        plot_kwargs["label"] = f"{plot_kwargs['label']} mean={mu:.4} std={std:.4}"

    if trim_start_idx is not None:
        x_points = x_points[trim_start_idx:]
        y_points = y_points[trim_start_idx:]
        err_vals = None if err_vals is None else err_vals[trim_start_idx:]

    if trim_end_idx is not None:
        x_points = x_points[:trim_end_idx]
        y_points = y_points[:trim_end_idx]
        err_vals = None if err_vals is None else err_vals[:trim_end_idx]

    if plot_style_num is not None:
        if plot_style_num >= len(COLOURS) * len(LINE_STYLES):
            log.warning("Lines possibly generated with same style!")

        auto_colour = COLOURS[plot_style_num % len(COLOURS)]
        auto_ls = LINE_STYLES[floor(plot_style_num / len(COLOURS)) % len(LINE_STYLES)]
        plot_style_num += 1

    else:
        auto_colour, auto_ls = None, None

    plot_kwargs["color"] = plot_kwargs.get("color", auto_colour)
    plot_kwargs["ls"] = plot_kwargs.get("ls", auto_ls)

    assert len(x_points) == len(y_points)
    if err_vals is not None:
        assert len(y_points) == len(err_vals)

    return (x_points, y_points, err_vals, plot_kwargs, is_scatter), plot_style_num


def plot_graph(
    data: dict,
    plots: Union[list[dict], dict, None]=None,
    keys_to_relevant: Union[None, str, list[str]]=None,
    show_plot: bool=True,
    title: Optional[str]=None,
    save_name: Optional[str]=None,
    save_folder: Optional[str]=None,
    xlabel: Optional[str]=None,
    ylabel: Optional[str]=None,
    horizontal_lines: Optional[list[dict]]=None,
    vertical_lines: Optional[list[dict]]=None,
    xlim: Optional[tuple[float, float]]=None,
    ylim: Optional[tuple[float, float]]=None,
    xscale: Optional[str]=None,
    yscale: Optional[str]=None,
    xformat: Optional[str]=None,
    yformat: Optional[str]=None,
    need_legend: bool=False,
    plt_save_width: int=12,
    plt_save_height: int=6,
    high_contrast_colours: bool=True,
    all_plots_kwargs: dict[str, Any] | None=None,
    legend_kwargs: dict[str, Any] | None=None,
    legend_lines_kwargs: dict[str, Any] | None=None,
    legend_text_kwargs: dict[str, Any] | None=None,
    grid_kwargs: dict[str, Any] | None=None,
):
    """
    Plot a graph using some data

    :param data: Data to use for plotting
    :param plots: Configuration for each plot
    :param show_plot: Whether to display the plot once made
    :param output_path: File to save to, None means no saving
    :param title: Plot title
    :param entries: Which entries from the dict to plot with
    :param keys_to_list: The path of keys to find the inner list of plot data in each dict
    """

    all_plots_kwargs, grid_kwargs = nones_to_empty_dicts(all_plots_kwargs, grid_kwargs)
    keys_to_relevant = enforce_str_list(keys_to_relevant)

    if keys_to_relevant is not None:
        data = key_list_get(data, keys_to_relevant)

    assert len(data) > 0, "No data for graph!"

    assert (save_folder is not None) or show_plot

    if plots is None:
        plots = {}

    if not isinstance(plots, list):
        plots = [plots]

    plot_style_num = 0 if high_contrast_colours else None
    plot_data = []

    for plot_cfg in plots:
        pdats, plot_style_num = get_plot_data(data, **plot_cfg, **all_plots_kwargs,
                                              plot_style_num=plot_style_num)
        plot_data.append(pdats)

    for xs, ys, error_vals, kwargs, is_scatter in plot_data:
        if is_scatter:
            plt.scatter(xs, ys, **kwargs)
            if error_vals is not None:
                raise NotImplementedError
        else:
            plt.plot(xs, ys, **kwargs)
            if error_vals is not None:
                plt.fill_between(xs, np.array(ys)+np.array(error_vals),
                                 np.array(ys)-np.array(error_vals), alpha=0.08,
                                 color=kwargs["color"])

    if horizontal_lines is not None:
        for hline_kwargs in horizontal_lines:
            plt.axhline(**hline_kwargs)

    if vertical_lines is not None:
        for vline_kwargs in vertical_lines:
            plt.axvline(**vline_kwargs)

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)

    if xformat is not None:
        plt.gca().xaxis.set_major_formatter(get_format_func(xformat))

    if yformat is not None:
        plt.gca().yaxis.set_major_formatter(get_format_func(yformat))

    if need_legend:
        (legend_kwargs,) = nones_to_empty_dicts(legend_kwargs)

        legend = plt.legend(**legend_kwargs)

        if legend_lines_kwargs is not None:
            plt.setp(legend.get_lines(), **legend_lines_kwargs)

        if legend_text_kwargs is not None:
            plt.setp(legend.get_texts(), **legend_text_kwargs)

    if title is not None:
        plt.title(title)
    else:
        title = "plot"

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.grid(**grid_kwargs)

    if save_folder is not None:
        fig = plt.gcf()
        fig.set_size_inches(plt_save_width, plt_save_height)
        plt.tight_layout()
        save_name = save_name if save_name is not None else title

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.savefig(path.join(save_folder, save_name), dpi=100)

    if show_plot:
        plt.show()

    else:
        plt.clf()


def make_graphs(
    graph_cfgs: list[dict[str, Any]],
    data: Optional[dict]=None,
    data_file_path: Optional[str]=None,
    data_multipath_dict: Optional[dict[str, str]]=None,
    file_type: str="pickle",
    **all_graph_kwargs,
):
    if data_multipath_dict is not None:
        assert data is None and data_file_path is None, "Too many data sources provided"

        data = {k: load(file_type, v) if isinstance(v, str) else v
                for k, v in data_multipath_dict.items()}

    elif data_file_path is not None:
        assert data is None, "Too many data sources provided"
        data = load(file_type, data_file_path)

    assert data is not None, "No data source provided"

    for graph_cfg in graph_cfgs:
        plot_graph(data, **graph_cfg, **all_graph_kwargs)


def smooth(
    *time_series_data: Array,
    desired_time_samples: Optional[Array]=None,
    shape_exponent: float=2.0,
    shape_scaling: float=1.0,
    mode="exponential",
    std_err_n_override: Optional[float | int]=None,
    output_as_dict: bool=False,
) -> tuple[Array, Array, Array, Array] | dict[str, Array]:

    clean_data = []

    for i, data in enumerate(time_series_data):
        if len(data.shape) == 1:
            data = np.array([range(len(data)), data])

        assert len(data.shape) == 2 and data.shape[0] == 2, f"{time_series_data[i].shape=} incorrect"

        clean_data.append(data)

    xts = np.concatenate(clean_data, axis=1)
    ts, xs = xts[:, np.argsort(xts[0])]

    if mode.lower() in ["homographic", "homo", "h"]:
        def shape_fn(dist):
            return (1 / (1 + (dist / shape_scaling) ** shape_exponent))

    elif mode.lower() in ["exponential", "exp", "e"]:
        def shape_fn(dist):
            return np.exp(-(dist / shape_scaling) ** shape_exponent)

    else:
        raise ValueError("Mode must be one of: '((e)xp)onential', or '((h)omo)graphic'")

    sample_ts = np.array(ts if desired_time_samples is None else desired_time_samples)
    ts_rows, sample_ts_cols = np.meshgrid(ts, sample_ts)

    weights = shape_fn(abs(ts_rows - sample_ts_cols))  # Rows are weights for each sample_t
    total_weights = np.sum(weights, axis=1) + 1e-12  # To avoid later div by 0

    smoothed_xs = np.matmul(weights, xs) / total_weights
    xs_rows, val_cols = np.meshgrid(xs, smoothed_xs)

    xs_vars = np.sum(weights * ((xs_rows - val_cols) ** 2), axis=1) / total_weights
    xs_std_devs = xs_vars ** 0.5
    std_err_div = std_err_n_override if std_err_n_override is not None else total_weights
    xs_std_errs = xs_std_devs / (std_err_div ** 0.5)

    if output_as_dict:
        return dict(zip(
            ["time", "mean", "std_devs", "std_errs"],
            [sample_ts, smoothed_xs, xs_std_devs, xs_std_errs],
        ))

    else:
        return sample_ts, smoothed_xs, xs_std_devs, xs_std_errs


def get_plot_configs(
    k,
    plot_raw: bool=True,
    plot_smoothed: bool=False,
    plot_std_error: bool=False,
    plot_smoothed_std_error: bool=False,
    plot_kwargs: dict[str, Any] | None=None,
):
    assert plot_raw or plot_smoothed, "Need to plot at least one of smoothed and raw"

    plot_kwargs, = nones_to_empty_dicts(plot_kwargs)

    plots = []

    if plot_raw:
        pre_compiled_plot_kwargs = merge_dicts(
            plot_kwargs.get("all", {}),
            plot_kwargs.get("all_raw", {}),
            plot_kwargs.get(k, {}),
            {"label": (
                f"_{plot_kwargs[k]['label']}"
                if (k in plot_kwargs.keys()) and ("label" in plot_kwargs[k].keys()) else
                None
            )} if plot_smoothed else {},
        )
        is_scatter = (
            pre_compiled_plot_kwargs.pop("is_scatter")
            if "is_scatter" in pre_compiled_plot_kwargs.keys() else
            False
        )
        # Require certain defaults based on whether plotting smooth or as a scatter
        compiled_plot_kwargs = merge_dicts(
            {
                "ls": "--" if plot_smoothed and not is_scatter else "-",
                "alpha": 0.4 if plot_smoothed and not is_scatter else 1.0,
            },
            pre_compiled_plot_kwargs,
        )
        plots.append({
            "keys_to_relevant": k,
            "x_key_chain_to_list": "raw_time",
            "y_key_chain_to_list": "raw_mean",
            "err_key_chain_to_list": "std_errs" if plot_std_error else None,
            "is_scatter": is_scatter,
            "plot_kwargs": compiled_plot_kwargs,
        })

    if plot_smoothed:
        compiled_plot_kwargs = merge_dicts(
            {"ls": "-"},
            plot_kwargs.get("all", {}),
            plot_kwargs.get("all_smooth", {}),
            plot_kwargs.get(k, {}),
        )
        is_scatter = (
            compiled_plot_kwargs.pop("is_scatter")
            if "is_scatter" in compiled_plot_kwargs.keys() else
            False
        )
        plots.append({
            "keys_to_relevant": k,
            "x_key_chain_to_list": "time",
            "y_key_chain_to_list": "mean",
            "err_key_chain_to_list": "std_errs" if plot_smoothed_std_error else None,
            "is_scatter": is_scatter,
            "plot_kwargs": compiled_plot_kwargs,
        })

    return plots

