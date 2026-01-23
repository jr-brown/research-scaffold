
import wandb
import logging
import numpy as np

from tqdm import tqdm
from pprint import pformat
from typing import TypeVar, Generic
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from wandb.apis.public.runs import Run

from .util import key_list_get, dmap


log = logging.getLogger(__name__)


Array = np.ndarray


@dataclass
class GeneralParameterSubset:
    pmax: float
    pmin: float
    is_log_domain: bool=False

    def __str__(self) -> str:
        if self.is_log_domain:
            return f"GPS({self.pmin:.5g} to {self.pmax:.5g} ({np.exp(self.pmin):.5g} to {np.exp(self.pmax):.5g}), is_log_domain={self.is_log_domain})"
        else:
            return f"GPS({self.pmin:.5g} to {self.pmax:.5g}, is_log_domain={self.is_log_domain})"

@dataclass
class CategoricalParameterSubset:
    vals: list

    def __str__(self) -> str:
        return f"CPS(vals={self.vals})"


ParameterBounds = GeneralParameterSubset | CategoricalParameterSubset
PB = TypeVar('PB', GeneralParameterSubset, CategoricalParameterSubset)


@dataclass
class ParameterOptions(Generic[PB]):
    total: PB
    top: PB
    mid: PB
    bot: PB

    def unpack(self) -> tuple[PB, PB, PB, PB]:
        return self.total, self.top, self.mid, self.bot


@dataclass
class ScoredParameterOptions(Generic[PB]):
    total: PB
    top: PB
    top_score: float
    mid: PB
    mid_score: float
    bot: PB
    bot_score: float


def _best_bounds(scored_opt: ScoredParameterOptions[PB]) -> PB:
    score_pairs = [
        (scored_opt.top_score, scored_opt.top),
        (scored_opt.mid_score, scored_opt.mid),
        (scored_opt.bot_score, scored_opt.bot),
    ]
    _, best_bounds = sorted(score_pairs, key=lambda kv: kv[0])[-1]
    return best_bounds


def _best_score(scored_opt: ScoredParameterOptions) -> float:
    return max(scored_opt.top_score, scored_opt.mid_score, scored_opt.bot_score)


def _check_parameter_val(
    param_bounds: ParameterBounds,
    val
) -> bool:

    if isinstance(param_bounds, CategoricalParameterSubset):
        return val in param_bounds.vals

    elif isinstance(param_bounds, GeneralParameterSubset):
        if param_bounds.is_log_domain:
            val = np.log(val)

        # Cast to avoid return type error if val is numpy error
        return bool(param_bounds.pmin <= val <= param_bounds.pmax)

    else:
        raise ValueError(f"{param_bounds=} bad type")


def _check_run_parameter(
    run: Run,
    composite_key: str,
    param_bounds: ParameterBounds
) -> bool:
    return _check_parameter_val(param_bounds, key_list_get(run.config, composite_key.split('/')))


def _filter_runs(
    runs: dict[str, tuple[Run, float]],
    bounds: dict[str, ParameterBounds]
) -> dict[str, tuple[Run, float]]:

    return {
        r_id: (run, r_score)
        for r_id, (run, r_score) in tqdm(runs.items(), desc="Filtering runs", leave=False)
        if all(_check_run_parameter(run, k, v) for k, v in bounds.items())
    }


def _get_normed_total_run_score(
    runs: dict[str, tuple[Run, float]],
    mean_score: float=0.0,
) -> float:

    if len(runs) > 0:
        return np.sum([score - mean_score for _, score in runs.values()]).item()
    else:
        return 0.0


def _avg_run_score(
    runs: dict[str, tuple[Run, float]],
) -> float:
    return np.mean([score for _, score in runs.values()]).item()


def _score_options(
    runs: dict[str, tuple[Run, float]],
    composite_key: str,
    param_opts: ParameterOptions[PB],
) -> ScoredParameterOptions[PB]:

    total_b, top_b, mid_b, bot_b = param_opts.unpack()

    top_runs = _filter_runs(runs, {composite_key: top_b})
    mid_runs = _filter_runs(runs, {composite_key: mid_b})
    bot_runs = _filter_runs(runs, {composite_key: bot_b})

    avg_score = _avg_run_score(runs)
    top_score = _get_normed_total_run_score(top_runs, avg_score)
    mid_score = _get_normed_total_run_score(mid_runs, avg_score)
    bot_score = _get_normed_total_run_score(bot_runs, avg_score)

    return ScoredParameterOptions(
        total_b,
        top_b, top_score,
        mid_b, mid_score,
        bot_b, bot_score,
    )


def _trim_redundant_leading_key_parts(_dict):
    all_leading_key_fragments = [k.split('/')[0] for k in _dict.keys()]

    if len(set(all_leading_key_fragments)) == 1:
        return _trim_redundant_leading_key_parts({
            '/'.join(k.split('/')[1:]): v
            for k, v in _dict.items()
        })

    else:
        return _dict


def _best_scores_to_str(best_scores: dict[str, float]) -> str:
    return '\n'.join(f"  {k}:\n    {v:.5g}" for k, v in _trim_redundant_leading_key_parts(best_scores).items())


def _take_best(scored_opts: dict[str, ScoredParameterOptions]) -> dict[str, ParameterBounds]:
    best_scores = dmap(_best_score, scored_opts)
    log.info(f"Best parameter scores:\n{_best_scores_to_str(best_scores)}")
    best_param = sorted(best_scores.items(), key=lambda kv: kv[1])[-1][0]
    to_return = {k: _best_bounds(v) if k == best_param else v.total
                 for k, v in scored_opts.items()}
    log.info(f"Picked {best_param=} with score={_best_score(scored_opts[best_param]):.5g} and bounds={to_return[best_param]}")
    return to_return


def _generate_options(param_bounds: PB) -> ParameterOptions[PB]:
    if isinstance(param_bounds, GeneralParameterSubset):
        q4 = param_bounds.pmax
        q0 = param_bounds.pmin
        q2 = (q0 + q4) / 2
        q3 = (q2 + q4) / 2
        q1 = (q0 + q2) / 2
        return ParameterOptions(
            param_bounds,
            GeneralParameterSubset(q4, q2, param_bounds.is_log_domain),
            GeneralParameterSubset(q3, q1, param_bounds.is_log_domain),
            GeneralParameterSubset(q2, q0, param_bounds.is_log_domain),
        )

    elif isinstance(param_bounds, CategoricalParameterSubset):
        vals = param_bounds.vals
        q2_ub = (len(vals) + 1) // 2
        q2_lb = len(vals) // 2
        q1 = q2_lb // 2
        q3 = len(vals) - q1
        return ParameterOptions(
            param_bounds,
            CategoricalParameterSubset(vals[q2_lb:]),
            CategoricalParameterSubset(vals[q1:q3]),
            CategoricalParameterSubset(vals[:q2_ub]),
        )

    else:
        raise ValueError


def _fit_search_space(
    search_space: dict[str, ParameterBounds],
    runs: dict[str, tuple[Run, float]]
) -> dict[str, ParameterBounds]:

    new_space = {}

    for key, bounds in search_space.items():
        # Extract the parameter values from all runs
        param_values = []
        for run, _ in runs.values():
            try:
                val = key_list_get(run.config, key.split('/'))
                param_values.append(val)
            except (KeyError, TypeError):
                # Skip if the parameter doesn't exist in the run
                pass

        if param_values == []:
            # If no valid parameter values found, keep the original bounds
            new_space[key] = bounds
            continue

        # Handle different parameter types
        if isinstance(bounds, CategoricalParameterSubset):
            # Only keep values that appear in the runs
            new_space[key] = CategoricalParameterSubset(list(set(param_values)))

        elif isinstance(bounds, GeneralParameterSubset):
            # Find the min and max values from the runs
            if bounds.is_log_domain:
                # Convert to log domain for comparison
                param_values = [np.log(val) for val in param_values]

            min_val = min(param_values)
            max_val = max(param_values)

            # Constrain within original bounds and min/max with a 0.1% epsilon for numerical stability
            new_min = max(min_val - (1e-3 * abs(min_val)), bounds.pmin)
            new_max = min(max_val + (1e-3 * abs(max_val)), bounds.pmax)

            new_space[key] = GeneralParameterSubset(new_max, new_min, bounds.is_log_domain)

    return new_space


def _parse_parameter_config(cfg: dict) -> ParameterBounds:
    distribution = cfg.get("distribution", "categorical")
    is_log = "log" in distribution.split("_")

    if distribution == "categorical":
        return CategoricalParameterSubset(cfg["values"])

    elif distribution == "constant":
        return CategoricalParameterSubset([cfg["value"]])

    elif distribution in ["log_uniform_values", "q_log_uniform_values"]:
        return GeneralParameterSubset(np.log(cfg["max"]), np.log(cfg["min"]), True)

    elif distribution in ["inv_log_uniform", "inv_log_uniform_values"]:
        # TODO: Modify GeneralParameterSubset to handle inv_log values
        raise NotImplementedError
        # return GeneralParameterSubset(cfg["max"], cfg["min"], True)
        # return GeneralParameterSubset(np.log(1/cfg["min"]), np.log(1/cfg["max"]), True)

    else:
        return GeneralParameterSubset(cfg.get("max", np.inf), cfg.get("min", -np.inf), is_log)


def _search_space_to_str(search_space: dict[str, ParameterBounds]) -> str:

    return '\n'.join(f"  {k}:\n    {v}" for k, v in _trim_redundant_leading_key_parts(search_space).items())


def _recursive_parameter_parse(
    cfg: dict,
    ignored_keys: list[str] | None = None,
) -> dict[str, ParameterBounds]:

    out_d = {}

    for k1, v1 in cfg.items():
        if isinstance(v1, dict) and "parameters" in v1.keys():
            out_d.update({f"{k1}/{k2}": v2
                          for k2, v2 in _recursive_parameter_parse(v1["parameters"]).items()})
        else:
            out_d[k1] = _parse_parameter_config(v1)

    if ignored_keys is not None:
        for k in ignored_keys:
            out_d.pop(k, None)

    return out_d


def parameter_subset_search(
    wandb_entity: str,
    wandb_project: str,
    sweep_id: str,
    iters: int,
    metric_cfg_name_override: str | None = None,
    metric_cfg_goal_override: str | None = None,
    ignored_keys: list[str] | None = None,
    max_workers: int = 8,
):
    log.info(f"Fetching sweep")
    api = wandb.Api()
    sweep = api.sweep(f"{wandb_entity}/{wandb_project}/{sweep_id}")

    def metric_fn(run) -> float:
        metric_cfg = sweep.config["metric"]
        metric_name = metric_cfg_name_override or metric_cfg["name"]
        metric_goal = metric_cfg_goal_override or metric_cfg["goal"]

        if metric_goal == "maximize":
            mult = 1.0
        elif metric_goal == "minimize":
            mult = -1.0
        else:
            raise ValueError

        return mult * run.summary[metric_name]

    log.info(f"Parsing parameter space")
    search_space = _recursive_parameter_parse(
        sweep.config["parameters"],
        ignored_keys=ignored_keys,
    )

    def fetch_run(r):
        if r.state == "finished":
            return r.id, (r, metric_fn(r))
        return None

    runs = {}
    sweep_runs = list(sweep.runs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_run, r): r for r in sweep_runs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching runs"):
            try:
                result = future.result()
                if result is not None:
                    run_id, run_data = result
                    runs[run_id] = run_data
            except Exception as e:
                run = futures[future]
                log.warning(f"Failed to fetch run {run.name}: {e}")
    runs = _filter_runs(runs, search_space)
    search_space = _fit_search_space(search_space, runs)
    current_avg_score = _avg_run_score(runs)

    log.info(f"""Start stats:
{len(runs)=}
{current_avg_score=}
search_space:
{_search_space_to_str(search_space)}""")

    for _ in tqdm(range(iters), desc="Iteration", leave=False):
        if len(runs) == 1:
            log.info("Only one run left, terminating...")
            break

        max_score = np.sum([max(score - current_avg_score, 0) for _, score in runs.values()])

        log.info(f"Best possible score: {max_score:.5g}")
        options = dmap(_generate_options, search_space)
        scored_opts = {
            k: _score_options(runs, k, v)
            for k, v in tqdm(options.items(), desc="Scoring options", leave=False)
        }
        log.debug(pformat(scored_opts))
        search_space = _take_best(scored_opts)
        runs = _filter_runs(runs, search_space)
        current_avg_score = _avg_run_score(runs)
        search_space = _fit_search_space(search_space, runs)

        log.info(f"""New stats:
{len(runs)=}
{current_avg_score=}
search_space:
{_search_space_to_str(search_space)}""")

    log.info("Sweep finished")
    log.info(f"Final space:\n{_search_space_to_str(search_space)}")

