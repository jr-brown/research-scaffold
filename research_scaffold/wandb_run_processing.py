import wandb
import logging

import numpy as np

from re import search
from tqdm import tqdm
from typing import Optional, Any, Callable


Array = np.ndarray


log = logging.getLogger(__name__)


def filter_only_finished(runs):
    return [
        r
        for r in tqdm(runs, desc="Filtering for finished runs", leave=False)
        if r.state == "finished"
    ]


def filter_runs_by_regex(
    runs: list,
    regex: str,
    tqdm_desc_override: str | None = None,
):
    desc = (
        tqdm_desc_override
        if tqdm_desc_override is not None
        else f"Filtering runs, {regex=}"
    )
    return [r for r in tqdm(runs, desc=desc, leave=False) if search(regex, r.name)]


def filter_runs_by_regexes(
    runs: list,
    name_regexes: dict[str, str],
) -> dict[str, list]:
    return {
        k: filter_runs_by_regex(
            runs,
            regex,
            tqdm_desc_override=f"Filtering runs for {k}",
        )
        for k, regex in name_regexes.items()
    }


def get_runs_from_wandb(
    wandb_path: str,
    wandb_filters: dict[str, Any] | None = None,
    max_num_for_testing: Optional[int] = None,
    only_finished: bool = True,
    name_regex_filter: str | None = None,
) -> list:

    log.info("Loading WandB API and runs")
    api = wandb.Api()
    runs = api.runs(wandb_path, filters=wandb_filters)

    if max_num_for_testing is not None:
        runs = runs[:max_num_for_testing]

    if only_finished:
        runs = filter_only_finished(runs)

    if name_regex_filter is not None:
        runs = filter_runs_by_regex(runs, name_regex_filter)

    return list(runs)


def get_run_metrics(
    runs: list,
    y_key: str,
    x_key: Optional[str] = "_step",
    n_samples: Optional[int] = None,
    get_keys_separately_and_combine: bool = False,
    processing_fn: Optional[Callable[[Array], Array]] = None,
) -> list:

    if processing_fn is None:
        processing_fn = lambda xs: xs

    if n_samples is None:
        history_inner_fn = lambda r, *ks: r.history(keys=["_step", *ks])
    else:
        history_inner_fn = lambda r, *ks: r.history(
            keys=["_step", *ks], samples=n_samples
        )

    def history_fn(r, x_key, y_key) -> Array:
        if get_keys_separately_and_combine:
            # No need to handle if x is "_step" here as always taking last array element
            xs = history_inner_fn(r, x_key).values.T[-1]
            ys = history_inner_fn(r, y_key).values.T[-1]

            # Account for mismatching size - resample at size of smallest
            if len(xs) < len(ys):
                ys = r.history(keys=["_step", y_key], samples=len(xs)).values.T[-1]

            elif len(xs) > len(ys):
                xs = r.history(keys=["_step", x_key], samples=len(ys)).values.T[-1]

            return np.stack([xs, ys])

        else:
            # "_step" always generated so if we don't want it we need to slice it out
            vals = history_inner_fn(r, x_key, y_key).values.T
            return vals[1:3] if x_key != "_step" else vals

    return [
        processing_fn(history_fn(r, x_key, y_key))
        for r in tqdm(runs, desc="Fetching run histories", leave=False)
    ]

