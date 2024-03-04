import click
import yaml
import copy
import typing
import re
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from wasabi import color
from functools import partial
import wasabi
import enum
import itertools
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
from collections import defaultdict

from gpucachesim import REPO_ROOT_DIR
import gpucachesim.stats.stats
import gpucachesim.stats.native
import gpucachesim.stats.accelsim
import gpucachesim.stats.playground
import gpucachesim.stats.common as common
import gpucachesim.benchmarks as benchmarks
import gpucachesim.stats.parallel_table
import gpucachesim.stats.speed_table
import gpucachesim.stats.result_table
import gpucachesim.stats.generate
import gpucachesim.stats.timings
import gpucachesim.stats.view

import gpucachesim.plot as plot
import gpucachesim.utils as utils
from gpucachesim.stats.load import load_stats

from gpucachesim.benchmarks import (
    Target,
    Benchmarks,
    GPUConfig,
    DEFAULT_BENCH_FILE,
)


# suppress scientific notation by setting float_format
# pd.options.display.float_format = "{:.3f}".format
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 500)
# pd.set_option("future.no_silent_downcasting", True)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.set_printoptions(suppress=True, formatter={"float_kind": "{:f}".format})
np.seterr(all="raise")
print("pandas version: {}".format(pd.__version__))

DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.original.config.yml"


@click.group()
def main():
    # ctx.ensure_object(dict)
    pass


def aggregate_benchmark_results(
    selected_df: pd.DataFrame,
    targets=None,
    mode="serial",
    memory_only=False,
    per_kernel=False,
    inspect=False,
    mean=False,
    cores_per_cluster=int(benchmarks.BASELINE["cores_per_cluster"]),
    num_clusters=int(benchmarks.BASELINE["num_clusters"]),
) -> typing.Tuple[pd.DataFrame, typing.List[str]]:
    """View results for a benchmark"""
    for col in benchmarks.SIMULATE_INPUT_COLS:
        if col not in selected_df:
            selected_df[col] = np.nan

    # non_gpucachesim = selected_df["input_mode"].isnull()
    non_gpucachesim = selected_df["target"] != Target.Simulate.value

    gold_gpucachesim = selected_df["input_mode"] == mode
    gold_gpucachesim &= selected_df["input_memory_only"] == memory_only
    gold_gpucachesim &= selected_df["input_cores_per_cluster"] == cores_per_cluster
    gold_gpucachesim &= selected_df["input_num_clusters"] == num_clusters
    assert gold_gpucachesim.sum() > 0
    print(
        "gpucachesim gold input ids:",
        sorted(selected_df.loc[gold_gpucachesim, "input_id"].unique().tolist()),
    )
    print(
        selected_df[gold_gpucachesim][
            ["kernel_name_mangled", "kernel_name"]
        ].drop_duplicates()
    )

    kernels = selected_df[gold_gpucachesim]["kernel_name"].unique().tolist()
    print("kernels:", kernels)

    # only keep gold gpucachesim and other targets
    no_kernel = selected_df["kernel_name"].isna() | (selected_df["kernel_name"] == "")
    valid_kernel = selected_df["kernel_name"].isin(kernels)
    selected_df = selected_df[
        (gold_gpucachesim | non_gpucachesim) & (valid_kernel | no_kernel)
    ]

    # filter targets to keep
    if isinstance(targets, list):
        selected_df = selected_df[selected_df["target"].isin(targets)]

    # input_cols = benchmarks.ALL_BENCHMARK_INPUT_COLS
    # input_cols = sorted(list([col for col in input_cols if col in selected_df]))
    #
    # group_cols = (
    #     benchmarks.BENCH_TARGET_INDEX_COLS + ["input_id", "run", "kernel_name"]
    #     # ["kernel_name", "run"] + input_cols
    # )
    #
    # aggregations = {
    #     **{c: "mean" for c in set(selected_df.columns)},
    #     **benchmarks.NON_NUMERIC_COLS,
    # }
    # aggregations = {
    #     col: agg
    #     for col, agg in aggregations.items()
    #     if col in selected_df and col not in group_cols
    # }
    # # pprint(aggregations)
    #
    # per_kernel = (
    #     selected_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    # )

    # per_config = sum_per_config_kernel_metrics(selected_df)
    # per_config, group_cols = aggregate_mean_input_config_stats(

    # group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + input_cols
    # grouped = per_kernel.groupby(group_cols, dropna=False)
    # aggregations = {
    #     **{c: "mean" for c in set(per_kernel.columns)},
    #     **benchmarks.NON_NUMERIC_COLS,
    # }
    # aggregations = {
    #     col: agg
    #     for col, agg in aggregations.items()
    #     if col in per_kernel and not col in group_cols
    # }
    # # pprint(aggregations)
    #
    # per_config = grouped.agg(aggregations).reset_index()
    # return None, per_config
    # return per_config, group_cols

    return aggregate_mean_input_config_stats(
        selected_df, per_kernel=per_kernel, inspect=inspect, mean=mean
    )


# def sum_per_config_kernel_metrics(df, per_kernel=False):
#     input_cols = copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS)
#     input_cols = sorted(list([col for col in input_cols if col in df]))
#     # we group by target and benchmark
#     group_cols = copy.deepcopy(benchmarks.BENCH_TARGET_INDEX_COLS)
#     # we group by the input id and each run, such that we can compute mean and stddev
#     # because we aggregate statistics for each run
#     group_cols += ["input_id", "run"]
#
#     if per_kernel:
#         # instead of grouping by kernel launch id, we group by kernel name
#         # this aggregates statistics for repeated launches of the same kernel
#         # also, it does not average statistics when the kernel name is nan
#         group_cols += ["kernel_name", "kernel_name_mangled"]
#
#     pprint(group_cols)
#     # pprint(benchmarks.NON_NUMERIC_COLS)
#     # pprint(sorted(list(set(df.columns) - set(benchmarks.NON_NUMERIC_COLS))))
#
#     preview_cols = ["target", "benchmark", "input_id", "run", "kernel_launch_id", "kernel_name", "kernel_name_mangled", "exec_time_sec"]
#     print(df.loc[:,preview_cols][:])
#
#     grouped = df.groupby(group_cols, dropna=False)
#     aggregations = {
#         **{c: "sum" for c in set(df.columns)},
#         **{c: "mean" for c in benchmarks.RATE_COLUMNS},
#         **benchmarks.NON_NUMERIC_COLS,
#     }
#     aggregations = {
#         col: agg
#         for col, agg in aggregations.items()
#         if col in df and not col in group_cols
#     }
#     def _inspect(df):
#         print("\nINSPECT: per config kernel metrics")
#         print(df.loc[:,preview_cols][:10])
#         pass
#
#     grouped.apply(_inspect)
#     return grouped.agg(aggregations).reset_index()


def different_cols(df):
    return [col for col in df.columns if len(df[col].value_counts()) > 1]


@main.command(name="speed-table")
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "--mean-time",
    "include_mean_time",
    type=bool,
    is_flag=True,
    help="include mean time",
)
@click.option(
    "--combined-only",
    "combined_only",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option(
    "--no-combined",
    "no_combined",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option(
    "--large",
    "large",
    type=bool,
    is_flag=True,
    help="only consider large inputs when computing the average speedup",
)
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--inspect", "inspect", type=bool, is_flag=True, help="inspect")
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_speed_table(
    bench_name,
    path,
    nsight,
    verbose,
    include_mean_time,
    large,
    combined_only,
    no_combined,
    inspect,
    png,
):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name, profiler=profiler, path=path)
    gpucachesim.stats.speed_table.speed_table(
        selected_df,
        bench_name=bench_name,
        include_mean_time=include_mean_time,
        large=large,
        verbose=verbose,
        combined_only=combined_only,
        no_combined=no_combined,
        inspect=inspect,
        png=png,
    )


@main.command(name="all-speed-table")
@click.option("-p", "--path", "path", help="Path to materialized benchmark config")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "--mean-time",
    "include_mean_time",
    type=bool,
    is_flag=True,
    help="include mean time",
)
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def all_speed_table(path, nsight, include_mean_time, verbose, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=None, profiler=profiler, path=path)

    options = dict(
        include_mean_time=include_mean_time,
        verbose=verbose,
        png=png,
    )

    # combined metrics
    for combined_only, no_combined, large in itertools.product(
        [True, False], [True, False], [True, False]
    ):
        gpucachesim.stats.speed_table.speed_table(
            selected_df,
            bench_name=None,
            large=large,
            combined_only=combined_only,
            no_combined=no_combined,
            **options,
        )

    # per benchmark
    benches = sorted(list(selected_df["benchmark"].unique()))
    for bench_name, large in itertools.product(benches, [True, False]):
        gpucachesim.stats.speed_table.speed_table(
            selected_df,
            bench_name=bench_name,
            large=large,
            combined_only=False,
            # equivalent to True here
            no_combined=False,
            **options,
        )


@main.command(name="result-table")
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name", help="Benchmark name")
@click.option("-m", "--metric", "metric", type=str, help="metric")
@click.option(
    "--combined-only",
    "combined_only",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option(
    "--no-combined",
    "no_combined",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option(
    "--large",
    "large",
    type=bool,
    is_flag=True,
    help="only consider large inputs when computing the average speedup",
)
@click.option(
    "--scaled-clusters",
    "scaled_clusters",
    type=bool,
    is_flag=True,
    help="scale clusters",
)
@click.option(
    "--scaled-cores",
    "scaled_cores",
    type=bool,
    is_flag=True,
    help="scale cores",
)
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_result_table(
    path,
    bench_name,
    metric,
    combined_only,
    no_combined,
    large,
    scaled_clusters,
    scaled_cores,
    nsight,
    verbose,
    png,
):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name, profiler=profiler, path=path)
    gpucachesim.stats.result_table.result_table(
        selected_df,
        bench_name=bench_name,
        metrics=[metric],
        large=large,
        combined_only=combined_only,
        scaled_cores=scaled_cores,
        scaled_clusters=scaled_clusters,
        no_combined=no_combined,
        verbose=verbose,
        png=png,
    )


@main.command(name="all-result-table")
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-m", "--metric", "metric", type=str, help="metric")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def all_result_table(path, metric, nsight, verbose, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=None, profiler=profiler, path=path)

    all_benches = sorted(list(selected_df["benchmark"].unique()))
    metrics: typing.List[typing.Optional[str]]
    if metric is None:
        metrics = [
            "dramreads",
            "dramwrites",
            "l2accesses",
            "l2dhitrate",
            "l1accesses",
            "l1dhitrate",
            "cycles",
        ]
    elif isinstance(metric, str):
        metrics = [metric]
    elif metric is None:
        metrics = [None]
    else:
        raise ValueError("metric must be a string, got {}".format(type(metric)))

    options = dict(verbose=verbose, png=png)
    # all_benches = ["babelstream"]

    # all metrics per benchmark
    # scaled_clusters, scaled_cores does not make sense for results,
    # we cannot compare results for these configurations with hardware.
    for bench_name, large in itertools.product(all_benches, [True, False]):
        # if scaled_clusters and scaled_cores:
        #     continue
        print("{:<30} large={:<10}".format(bench_name, str(large)))
        gpucachesim.stats.result_table.result_table(
            selected_df.copy(),
            bench_name=bench_name,
            large=large,
            scaled_clusters=False,
            scaled_cores=False,
            metrics=metrics,
            **options,
        )

    # all metrics for all benchmarks combined
    for large, combined_only in itertools.product([True, False], [True, False]):
        gpucachesim.stats.result_table.result_table(
            selected_df.copy(),
            bench_name=None,
            large=large,
            combined_only=combined_only,
            metrics=metrics,
            **options,
        )

    return
    # single metrics for each benchmark and combined
    for valid_metric, combined_only, large in itertools.product(
        metrics, [True, False], [True, False]
    ):
        gpucachesim.stats.result_table.result_table(
            selected_df.copy(),
            bench_name=None,
            large=large,
            metrics=valid_metric,
            combined_only=combined_only,
            **options,
        )


@main.command(name="parallel-table")
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "--scale-clusters",
    "scale_clusters",
    type=bool,
    default=True,
    help="scale clusters instead of cores per cluster",
)
@click.option(
    "--combined-only",
    "combined_only",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option(
    "--no-combined",
    "no_combined",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option(
    "--large",
    "large",
    type=bool,
    is_flag=True,
    help="only consider large inputs when computing the average speedup",
)
@click.option(
    "--verbose",
    "verbose",
    type=bool,
    default=True,
    help="verbose output",
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_parallel_table(
    bench_name,
    path,
    nsight,
    scale_clusters,
    combined_only,
    no_combined,
    large,
    verbose,
    png,
):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name, profiler=profiler, path=path)
    gpucachesim.stats.parallel_table.parallel_table(
        selected_df,
        bench_name=bench_name,
        scale_clusters=scale_clusters,
        combined_only=combined_only,
        no_combined=no_combined,
        large=large,
        verbose=verbose,
        png=png,
    )


@main.command(name="all-parallel-table")
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def all_parallel_table(path, nsight, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=None, profiler=profiler, path=path)

    bench_names = sorted(list(selected_df["benchmark"].unique()))
    configs = list(itertools.product([True, False], [True, False]))

    total = len(bench_names) * len(configs)

    options = dict(batch=True, verbose=False, png=png)

    done = 0
    for scale_clusters, large in configs:
        print("========= {:>4}/{:<4} =======".format(done, total))

        # average for all benchmarks
        gpucachesim.stats.parallel_table.parallel_table(
            selected_df,
            bench_name=None,
            scale_clusters=scale_clusters,
            large=large,
            **options,
        )

        # for each benchmarks
        for bench_name, combined_only, no_combined in itertools.product(
            bench_names, [True, False], [True, False]
        ):
            if combined_only and no_combined:
                continue
            mask = selected_df["benchmark"] == bench_name
            bench_df = selected_df[mask].copy()
            gpucachesim.stats.parallel_table.parallel_table(
                bench_df,
                bench_name=bench_name,
                scale_clusters=scale_clusters,
                combined_only=combined_only,
                no_combined=no_combined,
                **options,
            )
        done += 1


@main.command()
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name_arg", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
def correlation_plots(path, bench_name_arg, nsight):
    profiler = "nsight" if nsight else "nvprof"
    stats = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)
    print(stats.shape)

    stat_cols = gpucachesim.stats.native.stat_cols_for_profiler(profiler)
    if True:
        stat_cols = []
        stat_cols += ["cycles"]
        stat_cols += ["instructions", "l2_accesses", "dram_reads", "dram_writes"]

    for stat_col in stat_cols:
        stat_config = STAT_CONFIGS.get(stat_col) or StatConfig(
            **{**DEFAULT_STAT_CONFIG._asdict(), **dict(label=stat_col)}
        )
        xlabel = "{} (native)".format(stat_config.label)
        ylabel = "{} (simulated)".format(stat_config.label)
        fontsize = plot.FONT_SIZE_PT
        font_family = "Helvetica"

        # create one plot per benchmark
        for bench_name, bench_df in stats.groupby("benchmark"):
            print(bench_name)

            bench_input_cols = copy.deepcopy(
                benchmarks.BENCHMARK_INPUT_COLS[bench_name]
            )
            bench_df = bench_df.set_index(
                ["target"] + benchmarks.SIMULATE_INPUT_COLS
            ).sort_index()

            def gpucachesim_baseline(target, memory_only=False):
                # "input_mode", "input_threads", "input_run_ahead",
                # "input_memory_only", "input_num_clusters", "input_cores_per_cluster",
                return (
                    target,
                    "serial",
                    4,
                    5,
                    memory_only,
                    int(benchmarks.BASELINE["num_clusters"]),
                    int(benchmarks.BASELINE["cores_per_cluster"]),
                )

            group_cols = bench_input_cols

            aggregations = {
                **{c: "mean" for c in bench_df},
                **benchmarks.NON_NUMERIC_COLS,
            }
            aggregations = {
                col: agg
                for col, agg in aggregations.items()
                if col in bench_df and not col in group_cols
            }

            native_df = bench_df.loc[Target.Profile.value]
            native_df = native_df.groupby(bench_input_cols).agg(aggregations)

            accelsim_df = bench_df.loc[Target.AccelsimSimulate.value]
            accelsim_df = accelsim_df.groupby(bench_input_cols).agg(aggregations)

            gpucachesim_df = bench_df.loc[
                gpucachesim_baseline(target=Target.Simulate.value, memory_only=False)
            ]
            gpucachesim_df = gpucachesim_df.groupby(bench_input_cols).agg(aggregations)

            gpucachesim_memory_only_df = bench_df.loc[
                gpucachesim_baseline(Target.Simulate.value, memory_only=True)
            ]
            gpucachesim_memory_only_df = gpucachesim_memory_only_df.groupby(
                bench_input_cols
            ).agg(aggregations)

            gpucachesim_trace_reconstruction_df = bench_df.loc[
                Target.ExecDrivenSimulate.value
            ]
            gpucachesim_trace_reconstruction_df = (
                gpucachesim_trace_reconstruction_df.groupby(bench_input_cols).agg(
                    aggregations
                )
            )

            print("native                    ", native_df.shape)
            print("accelsim                  ", accelsim_df.shape)
            print("gpucachesim               ", gpucachesim_df.shape)
            print("gpucachesim (mem only)    ", gpucachesim_memory_only_df.shape)
            print(
                "gpucachesim (exec driven) ", gpucachesim_trace_reconstruction_df.shape
            )

            targets = [
                (("native", "native", "o"), native_df),
                (("AccelSim", "accelsim", "o"), accelsim_df),
                (("gpucachesim", "gpucachesim", "o"), gpucachesim_df),
                (
                    ("gpucachesim (memory only)", "gpucachesim", "x"),
                    gpucachesim_memory_only_df,
                ),
                (
                    ("gpucachesim (exec driven)", "gpucachesim", "D"),
                    gpucachesim_trace_reconstruction_df,
                ),
            ]
            assert all(
                [len(target_df) == len(targets[0][1]) for _, target_df in targets]
            )

            plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
            fig = plt.figure(
                figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.5 * plot.DINA4_WIDTH_INCHES),
                layout="constrained",
            )
            ax = plt.axes()

            marker_size = {
                "native": 30,
                "accelsim": 20,
                "gpucachesim": 10,
            }

            for (target_name, target, marker), target_df in targets:
                target_df.sort_values(bench_input_cols)
                ax.scatter(
                    native_df[stat_col],
                    target_df[stat_col],
                    marker_size[target],
                    # marker=plot.SIM_MARKER[target],
                    marker=marker,
                    facecolor=plot.plt_rgba(*plot.SIM_RGB_COLOR[target], 0.5),
                    edgecolor="none" if marker != "x" else None,
                    label=target_name,
                )

            all_targets_df = pd.concat([target_df for _, target_df in targets])
            print(all_targets_df.shape)

            stat_col_min = all_targets_df[stat_col].min()
            stat_col_max = all_targets_df[stat_col].max()

            if stat_config.log_y_axis:
                log_stat_col_max = np.ceil(np.log10(stat_col_max))
                stat_col_max = 10**log_stat_col_max
                log_stat_col_min = np.floor(np.log10(stat_col_min))
                stat_col_min = 10**log_stat_col_min
                tick_values = np.arange(
                    log_stat_col_min,
                    log_stat_col_max,
                    step=int(np.ceil(log_stat_col_max / 6)),
                )
                tick_values = np.power(10, tick_values)
                xyrange = np.arange(1, stat_col_max)

                ax.set_yscale("log", base=10)
                ax.set_xscale("log", base=10)
            else:
                xyrange = np.arange(stat_col_min, stat_col_max, step=1)
                tick_values = np.linspace(stat_col_min, stat_col_max, 6)

            ax.plot(
                xyrange,
                xyrange,
                color="gray",
                linewidth=1,
            )

            ax.grid(
                stat_config.grid,
                axis="both",
                linestyle="-",
                linewidth=1,
                color="black",
                alpha=0.1,
                zorder=1,
            )

            tick_labels = [
                plot.human_format_thousands(v, round_to=0) for v in tick_values
            ]

            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_xticks(tick_values, tick_labels)
            ax.set_yticks(tick_values, tick_labels)
            ax.set_xlim(stat_col_min, stat_col_max)
            ax.set_ylim(stat_col_min, stat_col_max)
            ax.legend(loc="upper left")
            filename = plot.PLOT_DIR / "correlations/{}.{}.{}.pdf".format(
                profiler, bench_name, stat_col
            )
            print("writing to ", filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filename)

        # create one plot for all benchmarks
        if bench_name_arg is not None:
            bench_df = stats.set_index(
                ["target"] + benchmarks.SIMULATE_INPUT_COLS
            ).sort_index()


class StatConfig(typing.NamedTuple):
    label: str
    log_y_axis: bool
    grid: bool
    percent: bool


DEFAULT_STAT_CONFIG = StatConfig(
    label="",
    log_y_axis=False,
    grid=True,
    percent=False,
)

STAT_CONFIGS = {
    "instructions": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="Instructions", log_y_axis=True),
        }
    ),
    "num_blocks": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="Block count", log_y_axis=False),
        }
    ),
    "mean_blocks_per_sm": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="Average blocks per SM", log_y_axis=False),
        }
    ),
    # "input_id": StatConfig(
    #     **{
    #         **DEFAULT_STAT_CONFIG._asdict(),
    #         **dict(label="Input ID", log_y_axis=False),
    #     }
    # ),
    "cycles": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="Cycles", log_y_axis=True),
        }
    ),
    "dram_reads": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="DRAM reads", log_y_axis=True),
        }
    ),
    "dram_writes": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="DRAM writes", log_y_axis=True),
        }
    ),
    "exec_time_sec": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="Execution time (s)", log_y_axis=True),
        }
    ),
    "l1_global_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L1 global hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l1_local_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L1 local hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l1_accesses": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="L1 accesses", log_y_axis=True),
        }
    ),
    "l2_accesses": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label="L2 accesses", log_y_axis=True),
        }
    ),
    "l1_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"Unified L1 hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l2_read_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 read hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l1_read_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L1 read hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l2_write_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 write hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l1_write_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L1 write hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
    "l2_writes": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 writes", log_y_axis=True, percent=False),
        }
    ),
    "l2_write_hits": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 write hits", log_y_axis=True, percent=False),
        }
    ),
    "l2_reads": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 reads", log_y_axis=True, percent=False),
        }
    ),
    "l2_read_hits": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 read hits", log_y_axis=True, percent=False),
        }
    ),
    "l2_hit_rate": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 hit rate (%)", log_y_axis=False, percent=True),
        }
    ),
}


def compute_label_for_benchmark_df(df, per_kernel=False):
    assert isinstance(df, pd.Series)

    benchmark = df["benchmark"]
    bench_input_cols = copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[benchmark])
    assert all([c in df for c in bench_input_cols])

    kernel_name = df["kernel_name"].replace("_kernel", "").strip()

    match benchmark.lower():
        case "vectoradd":
            label = "VectorAdd\n"
            if per_kernel:
                label += "{}\n".format(kernel_name)
            label += "f{:<2} {}".format(
                int(df["input_dtype"]),
                plot.human_format_thousands(
                    int(df["input_length"]), round_to=0, variable_precision=True
                ),
            )
        case "matrixmul":
            label = "MatrixMul\n"
            if per_kernel:
                label += "{}\n".format(kernel_name)
            label += "f{:<2} {}x{}x{}".format(
                int(df["input_dtype"]),
                int(df["input_rows"]),
                int(df["input_rows"]),
                int(df["input_rows"]),
            )
        case "simple_matrixmul":
            label = "Naive MatrixMul\n"
            if per_kernel:
                label += "{}\n".format(kernel_name)
            label += "f{:<2} {}x{}x{}".format(
                int(df["input_dtype"]),
                int(df["input_m"]),
                int(df["input_n"]),
                int(df["input_p"]),
            )
        case "transpose":
            label = "Transpose\n"
            label += "{}\n".format(df["input_variant"])
            if per_kernel:
                label += "{}\n".format(kernel_name)
            label += "{}x{}".format(
                int(df["input_dim"]),
                int(df["input_dim"]),
            )
        case "babelstream":
            label = ""
            if per_kernel:
                label += "BStream\n"
                label += "{}\n".format(kernel_name)
            else:
                label += "BabelStream\n"
            label += "{}".format(int(df["input_size"]))
        case other:
            label = str(other)

    return label


@main.command(name="view")
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name", help="Benchmark name")
@click.option("--plot", "should_plot", type=bool, default=True, help="generate plots")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option("--memory-only", "mem_only", type=bool, is_flag=True, help="memory only")
@click.option(
    "-v", "--verbose", "verbose", type=bool, is_flag=True, help="verbose output"
)
@click.option(
    "--strict", "strict", type=bool, default=True, help="fail on missing results"
)
@click.option(
    "--tr",
    "trace_reconstruction",
    type=bool,
    default=True,
    help="show trace reconstruction in table",
)
@click.option(
    "--plot-tr",
    "plot_trace_reconstruction",
    type=bool,
    is_flag=True,
    help="plot trace reconstruction",
)
@click.option(
    "--play", "playground", type=bool, default=False, help="show playground in table"
)
@click.option("--per-kernel", "per_kernel", type=bool, is_flag=True, help="per kernel")
@click.option("--normalized", "normalized", type=bool, is_flag=True, help="normalized")
@click.option("--stats", "stat_names_arg", type=str, help="stat names")
@click.option(
    "--inspect", "inspect", type=bool, default=False, help="inspet aggregations"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_view(
    path,
    bench_name,
    should_plot,
    nsight,
    mem_only,
    trace_reconstruction,
    plot_trace_reconstruction,
    playground,
    stat_names_arg,
    verbose,
    strict,
    per_kernel,
    normalized,
    inspect,
    png,
):
    if stat_names_arg is None:
        stat_names = []
    elif isinstance(stat_names_arg, str):
        stat_names = [s.strip().lower() for s in stat_names_arg.split(",")]
    elif isinstance(stat_names_arg, list):
        stat_names = [str(s).strip().lower() for s in stat_names_arg]
    else:
        raise ValueError("bad stat names")

    gpucachesim.stats.view.view(
        path=path,
        bench_name=bench_name,
        should_plot=should_plot,
        nsight=nsight,
        mem_only=mem_only,
        trace_reconstruction=trace_reconstruction,
        plot_trace_reconstruction=plot_trace_reconstruction,
        playground=playground,
        stat_names=stat_names,
        verbose=verbose,
        strict=strict,
        per_kernel=per_kernel,
        normalized=normalized,
        inspect=inspect,
        png=png,
    )


@main.command(name="generate")
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option(
    "--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config"
)
@click.option("-b", "--bench", "bench_name", help="Benchmark name")
@click.option("-i", "--input", "input_idx", type=int, help="Input index")
@click.option(
    "--baseline",
    "--quick",
    "quick",
    type=bool,
    is_flag=True,
    help="Fast mode: only collect baseline benchmark configurations",
)
@click.option(
    "-t",
    "--target",
    "target",
    type=str,
    help="target",
)
@click.option(
    "-v", "--verbose", "verbose", type=bool, is_flag=True, help="verbose output"
)
@click.option(
    "--strict", "strict", type=bool, default=True, help="fail on missing results"
)
@click.option("--nvprof", "nvprof", type=bool, default=True, help="use nvprof")
@click.option("--nsight", "nsight", type=bool, default=False, help="use nsight")
@click.option("--out", "output_path", help="Output path for combined stats")
def run_generate(
    path,
    config_path,
    bench_name,
    input_idx,
    quick,
    target,
    verbose,
    strict,
    nvprof,
    nsight,
    output_path,
):
    gpucachesim.stats.generate.generate(
        path=path,
        config_path=config_path,
        bench_name=bench_name,
        input_idx=input_idx,
        quick=quick,
        target=target,
        verbose=verbose,
        strict=strict,
        nvprof=nvprof,
        nsight=nsight,
        output_path=output_path,
    )


@main.command(name="timings")
@click.option(
    "-p",
    "--path",
    default=DEFAULT_BENCH_FILE,
    help="Path to materialized benchmark config",
)
@click.option("-b", "--bench", "bench_name", help="Benchmark name")
@click.option("--baseline", type=bool, default=True, help="Baseline configurations")
@click.option("--strict", type=bool, default=True, help="strict mode")
@click.option("--validate", type=bool, is_flag=True, help="validate")
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_timings(path, bench_name, baseline, strict, validate, png):
    gpucachesim.stats.timings.timings(
        path=path,
        bench_name=bench_name,
        baseline=baseline,
        strict=strict,
        validate=validate,
        png=png,
    )


if __name__ == "__main__":
    main()
    # main(ctx={})
