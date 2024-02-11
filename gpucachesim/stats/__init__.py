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
pd.set_option("future.no_silent_downcasting", True)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.set_printoptions(suppress=True, formatter={"float_kind": "{:f}".format})
np.seterr(all="raise")
print("pandas version: {}".format(pd.__version__))

DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.original.config.yml"


@click.group()
# @click.pass_context
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
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_speed_table(bench_name, path, nsight, verbose, include_mean_time, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name, profiler=profiler, path=path)
    gpucachesim.stats.speed_table.speed_table(
        selected_df,
        bench_name,
        include_mean_time=include_mean_time,
        verbose=verbose,
        png=png,
    )


@main.command(name="result-table")
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--metric", "metric", type=str, help="metric")
@click.option(
    "--combined-only",
    "combined_only",
    type=bool,
    is_flag=True,
    help="only output combined metrics",
)
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_result_table(path, bench_name, metric, combined_only, nsight, verbose, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name, profiler=profiler, path=path)
    gpucachesim.stats.result_table.result_table(
        selected_df,
        bench_name=bench_name,
        metrics=[metric],
        combined_only=combined_only,
        verbose=verbose,
        png=png,
    )


@main.command(name="all-result-table")
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--metric", "metric", type=str, help="metric")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def all_result_table(path, bench_name, metric, nsight, verbose, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=None, profiler=profiler, path=path)

    all_benches = sorted(list(selected_df["benchmark"].unique()))
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
    else:
        metrics = [metric]

    options = dict(verbose=verbose, png=png)

    for bench_name in all_benches:
        gpucachesim.stats.result_table.result_table(
            selected_df.copy(), bench_name=bench_name, metrics=metrics, **options
        )

    for combined_only in [True, False]:
        gpucachesim.stats.result_table.result_table(
            selected_df.copy(),
            bench_name=None,
            metrics=metrics,
            combined_only=combined_only,
            **options,
        )


@main.command(name="parallel-table")
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "-b", "bench_name", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "--scale-clusters",
    "scale_clusters",
    type=bool,
    default=True,
    help="scale clusters instead of cores per cluster",
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
def run_parallel_table(bench_name, path, nsight, scale_clusters, large, verbose, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name, profiler=profiler, path=path)
    gpucachesim.stats.parallel_table.parallel_table(
        selected_df,
        bench_name=bench_name,
        scale_clusters=scale_clusters,
        large=large,
        verbose=verbose,
        png=png,
    )


@main.command(name="all-parallel-table")
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option("--png", "png", type=bool, is_flag=True, help="convert to png")
def run_all_parallel_table(path, nsight, png):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=None, profiler=profiler, path=path)

    bench_names = sorted(list(selected_df["benchmark"].unique()))
    configs = list(itertools.product([True, False], [True, False]))

    total = len(bench_names) * len(configs)

    options = dict(batch=True, verbose=False, png=png)

    done = 0
    for scale_clusters, large in configs:
        print("========= {:>4}/{:<4} =======".format(done, total))
        gpucachesim.stats.parallel_table.parallel_table(
            selected_df,
            bench_name=None,
            scale_clusters=scale_clusters,
            large=large,
            **options,
        )

        # for all benchmarks
        for bench_name in bench_names:
            mask = selected_df["benchmark"] == bench_name
            bench_df = selected_df[mask].copy()
            gpucachesim.stats.parallel_table.parallel_table(
                bench_df,
                bench_name=bench_name,
                scale_clusters=scale_clusters,
                **options,
            )
            done += 1


@main.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name_arg", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
def correlation_plots(path, bench_name_arg, nsight):
    profiler = "nsight" if nsight else "nvprof"
    stats = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)
    print(stats.shape)

    stat_cols = stat_cols_for_profiler(profiler)
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


def stat_cols_for_profiler(profiler: str) -> typing.Sequence[str]:
    stat_cols = [
        # GPU load and debug information
        "num_blocks",
        "mean_blocks_per_sm",
        "input_id",
        # execution time
        "exec_time_sec",
        # cycles
        "cycles",
        # instructions
        "instructions",
        # dram stats
        "dram_reads",
        "dram_writes",
        # l2 stats
        "l2_accesses",
        "l2_reads",
        "l2_writes",
    ]

    if profiler == "nvprof":
        # nvprof
        stat_cols += [
            "l2_read_hit_rate",
            "l2_write_hit_rate",
            "l2_read_hits",
            "l2_write_hits",
            "l2_hit_rate",
            # "l2_hits",
            # "l2_misses",
            "l1_accesses",
            # "l1_reads",
            # "l1_misses",
            "l1_hit_rate",
            "l1_global_hit_rate",
            "l1_local_hit_rate",
        ]
    else:
        # nsight
        stat_cols += [
            "l2_hits",
            "l2_misses",
            "l2_hit_rate",
            "l1_hit_rate",
        ]
    return stat_cols


def table_stat_cols_for_profiler(profiler: str) -> typing.Sequence[str]:
    stat_cols = [
        # "num_blocks",
        # "mean_blocks_per_sm",
        # "input_id",
        # execution time
        "exec_time_sec",
        # cycles
        "cycles",
        # instructions
        "instructions",
    ]

    if profiler == "nvprof":
        # nvprof
        stat_cols += [
            # l1 accesses
            "l1_accesses",
            "l1_global_hit_rate",
            # l2 reads
            "l2_read_hits",
            "l2_read_hit_rate",
            # l2 writes
            "l2_write_hits",
            "l2_write_hit_rate",
            # "l2_hit_rate",
            # "l2_hits",
            # "l2_misses",
            # "l1_reads",
            # "l1_misses",
            # "l1_hit_rate",
            # "l1_local_hit_rate",
        ]
    else:
        # nsight
        stat_cols += [
            "l1_hit_rate",
            "l2_hits",
            "l2_misses",
            "l2_hit_rate",
        ]

    stat_cols += [
        # dram stats
        "dram_reads",
        "dram_writes",
        # l2 stats
        # "l2_accesses",
        # "l2_reads",
        # "l2_writes",
    ]

    new_cols = set(stat_cols) - set(stat_cols_for_profiler(profiler))
    assert len(new_cols) == 0
    return stat_cols


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


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name_arg", help="Benchmark name")
@click.option("--plot", "should_plot", type=bool, default=True, help="generate plots")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option("--memory-only", "mem_only", type=bool, is_flag=True, help="memory only")
@click.option(
    "-v", "--verbose", "verbose", type=bool, is_flag=True, help="verbose output"
)
@click.option(
    "--strict", "strict", type=bool, default=True, help="fail on missing results"
)
@click.option("--per-kernel", "per_kernel", type=bool, default=False, help="per kernel")
@click.option(
    "--inspect", "inspect", type=bool, default=False, help="inspet aggregations"
)
def view(
    path,
    bench_name_arg,
    should_plot,
    nsight,
    mem_only,
    verbose,
    strict,
    per_kernel,
    inspect,
):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)

    # gpucachesim stats include "no kernel" (e.g. memcopies) stats
    assert selected_df["kernel_name"].isna().sum() > 0

    target_bench_input_count_hist = (
        selected_df[["target", "benchmark", "input_id"]]
        .drop_duplicates()
        .value_counts(["target", "benchmark"], dropna=False)
        .sort_index()
    )
    print(target_bench_input_count_hist)

    print(
        "num targets={} num benchmarks={}".format(
            len(selected_df["target"].unique()), len(selected_df["benchmark"].unique())
        )
    )

    print(
        "num clusters={} cores per cluster={}".format(
            selected_df["num_clusters"].unique().tolist(),
            selected_df["cores_per_cluster"].unique().tolist(),
        )
    )

    if verbose:
        print(
            selected_df[
                [
                    "target",
                    "benchmark",
                    "input_id",
                    "kernel_name",
                    "run",
                    "num_clusters",
                    "input_mode",
                    "input_num_clusters",
                ]
            ].drop_duplicates()
        )

    stat_cols = stat_cols_for_profiler(profiler)

    # pprint(per_config_group_cols)

    # print(selected_df.loc[
    #     (selected_df["input_id"] == 0) & (selected_df["target"] == Target.Simulate.value),
    #     ["target", "benchmark", "input_id", "kernel_name", "kernel_name_mangled", "run"]
    #     + all_input_cols + benchmarks.SIMULATE_INPUT_COLS + ["l2_hit_rate"],
    # ].T)

    # remove "no kernel" stats
    # NOTE: the execution time for gpucachesim no kernel is already added
    # inside the load stats function
    no_kernel_mask = selected_df["kernel_name"].isna()
    no_kernel_mask &= selected_df["kernel_name_mangled"].isna()
    selected_df = selected_df[~no_kernel_mask]

    per_config, _ = aggregate_benchmark_results(
        selected_df,
        memory_only=mem_only,
        per_kernel=per_kernel,
        inspect=inspect,
        mean=False,
    )

    print(
        per_config[
            ["target", "benchmark"]
            + (["input_id", "kernel_name"] if per_kernel else [])
        ].drop_duplicates()
    )

    all_input_cols = list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    if per_kernel:
        all_input_cols += ["kernel_launch_id", "kernel_name"]
    all_input_cols = [col for col in all_input_cols if col in selected_df]
    all_input_cols = sorted(all_input_cols)

    # # all_input_cols = sorted(list([col for col in all_input_cols if col in per_config]))
    #     all_input_cols = [col for col in all_input_cols if col in per_config]

    per_config_group_cols = utils.dedup(
        [
            "target",
            "benchmark",
            "input_id",
            "kernel_name",
            "kernel_name_mangled",
            "run",
        ]
        + benchmarks.SIMULATE_INPUT_COLS
        + all_input_cols
    )

    def _inspect(df):
        if len(df) > 1:
            print(df[per_config_group_cols].T)
            print(df.T)
            raise ValueError("must have exactly one row per config/run")

    rows_per_config_grouper = per_config.groupby(
        per_config_group_cols,
        as_index=False,
        dropna=False,
    )
    rows_per_config_grouper[per_config.columns].apply(_inspect)
    rows_per_config = rows_per_config_grouper.size()

    if False:
        # print(rows_per_config)
        print(rows_per_config[rows_per_config["size"] > 1].shape)
        # print(rows_per_config.loc[
        #     rows_per_config["size"] > 1,per_config_group_cols].sort_values(by=per_config_group_cols)[:5].T)
        print(rows_per_config[rows_per_config["size"] > 1][:1].T)
    assert (
        rows_per_config["size"] == 1
    ).all(), "must have exactly one row per config/run"

    # per_config = per_config.reset_index()
    # print(per_config.loc[
    #     (per_config["input_id"] == 0) & (per_config["target"] == Target.Simulate.value),
    #     ["target", "benchmark", "input_id", "kernel_name", "kernel_name_mangled", "run"]
    #     + all_input_cols + benchmarks.SIMULATE_INPUT_COLS + ["l2_hit_rate"],
    # ].T)

    # make sure kernels per input have been summed but we keep repetitions (runs) for
    # computing statistical properties (e.g. stddev)
    assert len(
        per_config[
            [
                "target",
                "benchmark",
                "input_id",
                "kernel_launch_id",
                "kernel_name",
                "run",
            ]
        ].drop_duplicates()
    ) == len(per_config)

    # group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + all_input_cols
    # if per_kernel:
    #     group_cols += ["kernel_launch_id", "kernel_name"]

    group_cols = ["target", "benchmark"] + all_input_cols
    print("group cols:", group_cols)

    # print("all input cols", all_input_cols)
    # print("BENCH_TARGET_INDEX_COLS", benchmarks.BENCH_TARGET_INDEX_COLS)
    # pprint(group_cols)
    # return
    per_config_grouped = per_config.groupby(group_cols, dropna=False)

    preview_cols = [
        "target",
        "benchmark",
        "input_id",
        "run",
        "kernel_launch_id",
        "kernel_name",
        "kernel_name_mangled",
    ]
    preview_cols += ["exec_time_sec"]
    preview_cols += ["cycles"]

    def _inspect(df):
        print("\nINSPECT")
        print(df.loc[:, preview_cols][:10])
        pass

    if inspect:
        per_config_grouped[per_config.columns].apply(_inspect)

    # average over runs
    aggregations = {
        **{c: "mean" for c in set(per_config.columns)},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {
        col: agg
        for col, agg in aggregations.items()
        if col in per_config and not col in group_cols
    }
    per_config_pivoted = per_config_grouped.agg(aggregations).reset_index()
    per_config_pivoted = per_config_pivoted.pivot(
        index=[col for col in group_cols if col not in ["target"]],
        # index=["benchmark"] + all_input_cols,
        columns="target",
    )

    print(" === {} === ".format(profiler))
    assert len(per_config_pivoted) > 0
    preview_per_config_pivoted = per_config_pivoted.T.copy()
    preview_target_name = {
        Target.Simulate.value.lower(): "Ours",
        Target.AccelsimSimulate.value.lower(): "Accel",
        Target.PlaygroundSimulate.value.lower(): "Play",
        Target.Profile.value.lower(): "Native",
    }
    print(preview_per_config_pivoted.index)
    preview_per_config_pivoted.index = preview_per_config_pivoted.index.set_levels(
        [
            preview_target_name[target.lower()]
            for target in preview_per_config_pivoted.index.levels[1].values
        ],
        level=1,
    )
    print(preview_per_config_pivoted.index)
    print(preview_per_config_pivoted.loc[pd.IndexSlice[stat_cols, :], :])

    def build_per_config_table(df):
        assert len(df) > 0

        num_bench_configs = len(df.index)

        # benchmark, inputs_cols
        table = r"{\renewcommand{\arraystretch}{1.5}%" + "\n"
        table += r"\begin{tabularx}{\textwidth}"
        table += "{ZZ|" + ("z" * num_bench_configs) + "}\n"

        def dedup_and_count(l):
            assert None not in l
            last_value = None
            count = 0
            out = []
            for ll in l:
                if last_value is None:
                    last_value = ll
                if ll == last_value:
                    count += 1
                else:
                    # add to output
                    out.append((last_value, count))
                    # update last value and count
                    last_value = ll
                    count = 1
            if last_value is not None:
                out.append((last_value, count))
            return out

        # benchmark index levels
        for index_col in df.index.names:
            index_values = df.index.get_level_values(index_col)
            index_values_reduced = dedup_and_count(index_values.values)

            index_col_label = benchmarks.BENCHMARK_INPUT_COL_LABELS[index_col]
            table += r"\multicolumn{2}{r}{" + str(index_col_label) + "}"
            for value, count in index_values_reduced:
                if isinstance(value, str):
                    value = str(value).replace("_", " ")
                else:
                    value = plot.human_format_thousands(
                        value, round_to=2, variable_precision=True
                    )
                table += r" & \multicolumn{" + str(count) + "}{|l}{"
                table += value + r"}"
            table += r"\\" + "\n"

        # table += r" & benchmark & \multicolumn{6}{l}{vectoradd} \\"
        # table += r" & data type & \multicolumn{3}{l}{32} & \multicolumn{3}{l}{64} \\"
        # table += r" & length & \multicolumn{1}{l}{100} & 1K & 500K & 100 & 1K & 500K "
        table += r"\hline\hline" + "\n"

        stat_cols = df.columns.get_level_values(0)
        stat_cols = dedup_and_count(stat_cols.values)
        # print("stat cols:", stat_cols)

        round_to = 1

        for stat_col_idx, (stat_col, _) in enumerate(stat_cols):
            stat_config = STAT_CONFIGS[stat_col]
            stat_col_label = str(stat_config.label)
            stat_col_label = stat_col_label.replace("_", " ")
            stat_col_label = re.sub(r"(?<!\\)%", r"\%", stat_col_label)

            # native
            native_values = df[stat_col, Target.Profile.value]
            assert len(native_values) == num_bench_configs
            if stat_col_idx % 2 == 0:
                table += r"\rowcolor{gray!10} "

            table += r" & Native"
            for value in native_values:
                if stat_config.percent:
                    assert 0.0 <= value <= 1.0
                    table += r" & ${}\%$".format(
                        plot.human_format_thousands(
                            value * 100.0,
                            round_to=2,
                            variable_precision=True,
                        )
                    )
                else:
                    table += " & ${}$".format(
                        plot.human_format_thousands(
                            value,
                            round_to=round_to,
                            variable_precision=True,
                        )
                    )
            table += r"\\" + "\n"

            # gpucachesim
            sim_values = df[stat_col, Target.Simulate.value]
            assert len(sim_values) == num_bench_configs
            # table += r" & \textsc{Gpucachesim}"
            if stat_col_idx % 2 == 0:
                table += r"\rowcolor{gray!10} "
            table += r" & Ours"
            for value in sim_values:
                if stat_config.percent:
                    assert 0.0 <= value <= 1.0
                    table += r" & ${}\%$".format(
                        plot.human_format_thousands(
                            value * 100.0,
                            round_to=2,
                            variable_precision=True,
                        )
                    )
                else:
                    table += " & ${}$".format(
                        plot.human_format_thousands(
                            value, round_to=round_to, variable_precision=True
                        )
                    )
            table += r"\\" + "\n"

            # accelsim
            accelsim_values = df[stat_col, Target.AccelsimSimulate.value]
            assert len(accelsim_values) == num_bench_configs
            if stat_col_idx % 2 == 0:
                table += r"\rowcolor{gray!10} "

            table += r"\multirow[r]{-3}{1.5cm}{\raggedleft "
            # table += r"\parbox{1.5cm}{"
            table += stat_col_label
            # table += r"}"
            table += r"}"
            table += r" & \textsc{Accelsim}"
            for value in accelsim_values:
                if stat_config.percent:
                    assert 0.0 <= value <= 1.0
                    table += r" & ${}\%$".format(
                        plot.human_format_thousands(
                            value * 100.0,
                            round_to=2,
                            variable_precision=True,
                        )
                    )
                else:
                    table += " & ${}$".format(
                        plot.human_format_thousands(
                            value,
                            round_to=round_to,
                            variable_precision=True,
                        )
                    )
            table += r"\\ \hline" + "\n"

            table += "%\n"
            table += "%\n"

        table += r"\end{tabularx}}" + "\n"
        table += r"\end{table}" + "\n"
        return table

    table_stat_cols = table_stat_cols_for_profiler(profiler)
    # table_stat_cols = [
    #     col
    #     for col in table_stat_cols_for_profiler(profiler)
    #     if col not in ["input_id", "mean_blocks_per_sm", "l1_local_hit_rate", "l1_hit_rate"]
    # ]

    # filter benchmarks that should be in the table
    selected_table_benchmarks = [
        # babelstream
        pd.DataFrame.from_records(
            (
                [
                    ("babelstream", 10240.0),
                    ("babelstream", 102400.0),
                ]
                if per_kernel
                else [
                    ("babelstream", 1024.0),
                    ("babelstream", 10240.0),
                    ("babelstream", 102400.0),
                ]
            ),
            columns=["benchmark", "input_size"],
        ),
        # transpose
        pd.DataFrame.from_records(
            [
                ("transpose", 128.0, "naive"),
                ("transpose", 128.0, "coalesced"),
                ("transpose", 256.0, "naive"),
                ("transpose", 256.0, "coalesced"),
                ("transpose", 512.0, "naive"),
                ("transpose", 512.0, "coalesced"),
            ],
            columns=["benchmark", "input_dim", "input_variant"],
        ),
        # simple matrixmul
        pd.DataFrame.from_records(
            [
                ("simple_matrixmul", 32, 32, 32, 32),
                ("simple_matrixmul", 32, 128, 128, 128),
                ("simple_matrixmul", 32, 32, 64, 128),
                ("simple_matrixmul", 32, 128, 32, 32),
                # extra configs
                ("simple_matrixmul", 32, 128, 512, 128),
                ("simple_matrixmul", 32, 512, 32, 512),
            ],
            columns=["benchmark", "input_dtype", "input_m", "input_n", "input_p"],
        ),
        # matrixmul
        pd.DataFrame.from_records(
            [
                ("matrixmul", 32, 32),
                ("matrixmul", 32, 64),
                ("matrixmul", 32, 128),
                ("matrixmul", 32, 256),
                ("matrixmul", 32, 512),
            ],
            columns=["benchmark", "input_dtype", "input_rows"],
        ),
        # vectoradd
        pd.DataFrame.from_records(
            [
                ("vectorAdd", 32, 100),
                ("vectorAdd", 32, 1000),
                # ("vectorAdd", 32, 10_000),
                # ("vectorAdd", 32, 20_000),
                ("vectorAdd", 32, 100_000),
                ("vectorAdd", 32, 500_000),
                # 64 bit
                ("vectorAdd", 64, 100),
                ("vectorAdd", 64, 1000),
                # ("vectorAdd", 64, 10_000),
                # ("vectorAdd", 64, 20_000),
                ("vectorAdd", 64, 100_000),
                ("vectorAdd", 64, 500_000),
            ],
            columns=["benchmark", "input_dtype", "input_length"],
        ),
    ]

    # choose subset of bench configs for the table
    selected_table_benchmarks = pd.concat(selected_table_benchmarks)
    selected_table_benchmarks = selected_table_benchmarks.set_index("benchmark")
    # selected_table_benchmarks = selected_table_benchmarks.loc[
    #     :,[col for col in per_config_pivoted.index.names if col in selected_table_benchmarks]
    #     # :,[col for col in per_config_pivoted.index.names if col in selected_table_benchmarks]
    # ]
    # print(sorted(per_config_pivoted.index.names))
    # print(sorted(selected_table_benchmarks.columns))
    # assert sorted(per_config_pivoted.index.names) == sorted(selected_table_benchmarks.columns)
    table_index = (
        per_config_pivoted.index.to_frame()
        .reset_index(drop=True)
        .merge(selected_table_benchmarks, how="inner")
    )
    table_index = pd.MultiIndex.from_frame(table_index)
    assert len(table_index) == len(table_index.drop_duplicates())

    # print(table_index)
    # print(per_config_pivoted.index)

    # build table
    table_per_config_pivoted = per_config_pivoted.loc[table_index, :]
    table = build_per_config_table(table_per_config_pivoted[table_stat_cols])
    print("\n\n\n")
    print(table)
    utils.copy_to_clipboard(table)
    print("copied table to clipboard")

    if not should_plot:
        return

    # remove some stat_cols that should not be plotted
    plot_stat_cols = sorted(list(set(stat_cols) - set(["num_blocks", "input_id"])))

    # add plot labels
    per_config.loc[:, "label"] = per_config.apply(
        partial(compute_label_for_benchmark_df, per_kernel=per_kernel), axis=1
    )
    per_config.loc[per_config["target"] == Target.Simulate.value, "target_name"] = (
        "gpucachesim"
    )
    per_config.loc[
        per_config["target"] == Target.AccelsimSimulate.value, "target_name"
    ] = "AccelSim"
    per_config.loc[per_config["target"] == Target.Profile.value, "target_name"] = (
        per_config.loc[~per_config["device"].isna(), "device"].apply(
            gpucachesim.stats.native.normalize_nvprof_device_name
        )
    )

    # targets = sorted(per_config["target"].unique().tolist())
    benches = sorted(per_config["benchmark"].unique().tolist())
    plot_targets = ["Profile", "Simulate", "AccelsimSimulate"]

    # compute plot index
    # print(per_config.index.to_frame())

    # plot_index_cols = ["target"] + [col for col in selected_table_benchmarks.columns if col in per_config]
    # per_config = per_config.set_index(plot_index_cols)

    # plot_index = (
    #     per_config[[col for col in selected_table_benchmarks.columns if col in per_config]]
    #     # per_config
    #     .reset_index(drop=True)
    #     .merge(selected_table_benchmarks, how="inner")
    # )
    # plot_index = pd.MultiIndex.from_frame(plot_index).drop_duplicates()
    # print(plot_index)

    # only keep selected benchmarks
    plot_per_config = per_config.reset_index(drop=True).merge(
        selected_table_benchmarks, how="inner"
    )
    assert len(plot_per_config) <= len(per_config)
    assert "input_size" in plot_per_config

    plot_per_config = plot_per_config.set_index(
        ["target", "benchmark"] + list(selected_table_benchmarks.columns),
        # group_cols,
    )
    # print(sorted(group_cols))
    # print(sorted(["target", "benchmark"] + list(selected_table_benchmarks.columns)))
    assert "input_size" in plot_per_config.index.names

    # print(plot_per_config)

    # plot_per_config = per_config.loc[plot_index,:] #.reset_index()
    # print(plot_per_config)

    # plot_targets = [
    #     target
    #     for target in targets
    #     if target in ["Profile", "Simulate", "AccelsimSimulate"]
    # ]

    for stat_col, benchmark in itertools.product(plot_stat_cols, benches):
        print(stat_col, benchmark)
        stat_config = STAT_CONFIGS.get(stat_col) or StatConfig(
            **{**DEFAULT_STAT_CONFIG._asdict(), **dict(label=stat_col)}
        )
        ylabel = stat_config.label
        fontsize = plot.FONT_SIZE_PT - 4
        font_family = "Helvetica"

        bar_width = 10
        spacing = 2
        group_spacing = 2 * bar_width

        group_width = len(plot_targets) * (bar_width + spacing) + group_spacing

        plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

        figsize = (
            1.0 * plot.DINA4_WIDTH_INCHES,
            0.10 * plot.DINA4_HEIGHT_INCHES,
        )
        fig = plt.figure(
            figsize=figsize,
            layout="constrained",
        )
        ax = plt.axes()

        ax.grid(
            stat_config.grid,
            axis="y",
            linestyle="-",
            linewidth=1,
            color="black",
            alpha=0.1,
            zorder=1,
        )

        # bench_input_cols = copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[benchmark])
        # group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + bench_input_cols

        # bench_input_values = per_config.loc[
        #     per_config["benchmark"] == benchmark, all_input_cols
        #     # per_config["benchmark"] == benchmark, bench_input_cols
        # ]
        #
        # if True:
        #     # reuse table_index
        #     # table_index
        #
        #     # filter benchmarks that should be plotted
        #     # TODO: dedup this with the same logic like the table above
        #     match benchmark:
        #         case "simple_matrixmul":
        #             subset = pd.DataFrame.from_records(
        #                 [
        #                     (32, 32, 32),
        #                     (128, 128, 128),
        #                     (32, 64, 128),
        #                     (128, 32, 32),
        #                     (128, 512, 128),
        #                     (512, 32, 512),
        #                 ],
        #                 columns=["input_m", "input_n", "input_p"],
        #             )
        #             bench_input_values = bench_input_values.merge(subset, how="inner")
        #         case "vectorAdd":
        #             subset = pd.DataFrame.from_records(
        #                 [
        #                     (32, 100),
        #                     (32, 1000),
        #                     # (32, 10_000),
        #                     (32, 20_000),
        #                     (32, 100_000),
        #                     (32, 500_000),
        #                     (64, 100),
        #                     (64, 1000),
        #                     # (64, 10_000),
        #                     (64, 20_000),
        #                     (64, 100_000),
        #                     (64, 500_000),
        #                 ],
        #                 columns=["input_dtype", "input_length"],
        #             )
        #             bench_input_values = bench_input_values.merge(subset, how="inner")
        #
        #     bench_input_values = bench_input_values.drop_duplicates().reset_index()

        # target_configs = list(
        #     itertools.product(targets, list(bench_input_values.iterrows()))
        # )

        # bench_configs = selected_table_benchmarks.loc[benchmark,:].reset_index(drop=True)
        # print(bench_configs)
        #
        # target_bench_configs = list(
        #     itertools.product(list(enumerate(plot_targets)), list(bench_configs.iterrows()))
        # )

        # for (target_idx, target), (input_idx, input_values) in target_bench_configs:
        for target_idx, target in enumerate(plot_targets):
            # print(table_per_config_pivoted)

            # print(per_config.loc[table_index, :])
            # for target, target_df in table_per_config_pivoted.groupby(["target"]):

            # bench_configs = plot_index[benchmark]
            # print(bench_configs)

            # for target in plot_targets:
            # print(target)
            # target_configs = plot_per_config[target, benchmark,:]
            # target_configs = plot_per_config.loc[pd.IndexSlice[target, benchmark], :]
            # .loc[plot_per_config["benchmark"] ==
            # target_configs = plot_per_config.loc[plot_per_config["benchmark"] ==
            # for input_idx, input_values in target_configs.iterrows()
            # target_df = per_config
            # print(target_df)

            target_df = plot_per_config.loc[(target, benchmark), :]
            assert target_df["run"].nunique() > 1
            assert "input_size" in target_df.index.names

            target_df = target_df.reset_index()
            # target_df = target_df.reset_index(drop=True)
            assert "input_size" in target_df
            # print(target_df)
            # print(target_df[[c for c in preview_cols if c in target_df]])

            # print(target_df[preview_cols])

            # target_df=target_df.reset_index(drop=True)

            # if len(target_df) < 1:
            #     print(
            #         color(
            #             "missing {} {} [{}]".format(
            #                 target, benchmark, input_values.values.tolist()
            #             ),
            #             fg="red",
            #         )
            #     )
            #     if strict:
            #         return
            #     continue

            # for input_idx, input_values_df in target_df.iterrows():
            for input_idx, (_, input_values_df) in enumerate(
                target_df.groupby([col for col in group_cols if col in target_df])
            ):
                # for input_idx, (_input_id, input_values_df) in enumerate(target_df.groupby("input_id")):

                # key = (target, benchmark) + tuple(input_values.values)

                # print(input_idx, input_values)
                input_values = (
                    input_values_df[
                        [col for col in all_input_cols if col in input_values_df]
                    ]
                    .drop_duplicates()
                    .dropna()
                )
                assert len(input_values) == 1
                input_values = dict(input_values.iloc[0])
                print(target, input_idx, input_values)
                # print(input_values_df[[c for c in preview_cols if c in input_values_df]])

                # print(key)
                # target_df = plot_per_config.loc[pd.IndexSlice[key], :]
                # target_df=target_df.reset_index(drop=True)
                # print(target_df[[c for c in preview_cols if c in target_df]])

                # target_df_mask = per_config["target"] == target
                # target_df_mask &= per_config["benchmark"] == benchmark
                # for col in bench_input_cols:
                #     target_df_mask &= per_config[col] == input_values[col]
                # target_df = per_config.loc[target_df_mask, :]

                # if len(target_df) < 1:
                #     print(
                #         color(
                #             "missing {} {} [{}]".format(
                #                 target, benchmark, input_values.values.tolist()
                #             ),
                #             fg="red",
                #         )
                #     )
                #     if strict:
                #         return
                #     continue

                # # if stat_col == "l2_hit_rate":
                # if stat_col == "exec_time_sec":
                #     print(target_df[preview_cols])
                #     print(target_df[stat_col])

                # target_df = target_df.groupby([col for col in group_cols if col in target_df], dropna=False)

                # target_idx = targets.index(target)
                # print(input_idx, group_width, target_idx + 0.5, bar_width + spacing)
                idx = input_idx * group_width + (target_idx + 0.5) * (
                    bar_width + spacing
                )

                # target = target_df["target"].first().values[0]
                # assert target == target_df["target"].first().values[0]
                assert input_values_df["target_name"].nunique() == 1
                target_name = input_values_df["target_name"].iloc[0]
                # target_name = target_df["target_name"].first().values[0]

                x = [idx]
                raw_y = input_values_df[stat_col]  # .fillna(0.0)
                # print("raw_y")
                # print(raw_y)
                # assert len(raw_y.mean()) ==1

                # print((target_name, stat_col), x, raw_y.mean())

                # raise ValueError("test")
                if verbose:
                    print(
                        "{:>15} {:<10} {:>15} [{:<3}]  {:<35}  {:<3} {:<4} = {:<8.2f} {:<8.2f}".format(
                            benchmark,
                            stat_col,
                            target_name,
                            target_idx,
                            "todo",
                            # str(input_values[bench_input_cols].tolist()),
                            input_idx,
                            idx,
                            raw_y.mean(),
                            raw_y.std(),
                        )
                    )

                if stat_config.percent:
                    y = raw_y.median() * 100.0
                else:
                    y = raw_y.mean()  # .fillna(0.0)

                ystd = raw_y.std()  # .fillna(0.0)

                y = np.nan_to_num(y, nan=0.0)
                ystd = np.nan_to_num(ystd, nan=0.0)

                bar_color = plot.plt_rgba(*plot.SIM_RGB_COLOR[target.lower()], 1.0)
                hatch = plot.SIM_HATCH[target.lower()]

                ax.bar(
                    x,
                    y,
                    color=bar_color,
                    hatch=hatch,
                    width=bar_width,
                    linewidth=1,
                    edgecolor="black",
                    zorder=2,
                    label=target_name if input_idx == 0 else None,
                )

                ax.errorbar(
                    x,
                    y,
                    yerr=ystd,
                    linewidth=1,
                    ecolor="black",
                    capsize=0.5 * bar_width,
                    linestyle="-",
                )

        ax.set_ylabel(ylabel)
        ax.axes.set_zorder(10)

        # simulate_df_mask = per_config["target"] == Target.Simulate.value
        # simulate_df_mask &= per_config["benchmark"] == benchmark
        # simulate_df = per_config.loc[simulate_df_mask, :]
        # simulate_df = simulate_df.merge(bench_input_values, how="inner")
        simulate_df = plot_per_config.loc[(Target.Simulate.value, benchmark), :]

        # print(simulate_df.head(n=100))
        # simulate_df = simulate_df.drop_duplicates().reset_index()
        assert len(simulate_df) > 0

        # these should be unique over runs (e.g. take first)
        # note: no bench input cols!
        bar_group_cols = [
            # "benchmark",
            "input_id",
            "kernel_launch_id",
        ]

        # print(simulate_df)
        # pprint(bar_group_cols)
        # print(simulate_df[bar_group_cols + ["label"]])
        # pprint(group_cols)
        # pprint(bar_group_cols)
        simulate_grouped = simulate_df.groupby(bar_group_cols, dropna=False)
        # simulate_grouped = simulate_df.groupby([col for col in bar_group_cols if col in simulate_df], dropna=False)

        # print(simulate_grouped["label"].first())
        # print(simulate_grouped["label"].apply(lambda df: print(df)))

        # labels = simulate_grouped["label"].to_numpy()
        labels = simulate_grouped["label"].first().to_numpy()
        num_blocks = simulate_grouped["num_blocks"].max().to_numpy()
        # num_blocks = simulate_df["num_blocks"].values
        # print(labels.tolist())
        assert len(labels) == len(num_blocks)
        assert len(labels) > 0
        # print(num_blocks)
        # print(labels)

        # all_values_mask = per_config["benchmark"] == benchmark
        # all_values_df = per_config.loc[all_values_mask, :]
        # all_values_df = all_values_df.merge(bench_input_values, how="inner")
        all_values_df = plot_per_config.loc[pd.IndexSlice[:, benchmark], :]
        assert len(all_values_df) > 0

        ymax = all_values_df[stat_col].max()

        if stat_config.log_y_axis:
            assert not stat_config.percent
            ymax_log = np.ceil(np.log10(ymax))
            ytick_values = np.arange(0, ymax_log + 1, step=int(np.ceil(ymax_log / 6)))
            ytick_values = np.power(10, ytick_values)
            print(stat_col, ymax_log, ytick_values)
            ax.set_yscale("log", base=10)
            ax.set_ylim(0.01, max(10 * ymax, 10))
        else:
            if stat_config.percent:
                ymax *= 100.0
                assert ymax <= 101.0
                ymax = utils.round_to_multiple_of(1.5 * ymax, multiple_of=25.0)
                ymax = np.clip(ymax, a_min=25.0, a_max=100.0)
                ax.set_ylim(0, ymax + 10.0)
            else:
                ymax = max(2 * ymax, 1)
                ax.set_ylim(0, ymax)
            ytick_values = np.linspace(0, ymax, 6)

        ytick_labels = [
            plot.human_format_thousands(v, round_to=0).rjust(6, " ")
            for v in ytick_values
        ]
        ax.set_yticks(ytick_values, ytick_labels)

        plot_dir = plot.PLOT_DIR / "validation"
        plot_dir.parent.mkdir(parents=True, exist_ok=True)

        xtick_labels = [
            "{}\n{} {}".format(label, int(blocks), "blocks" if blocks > 1 else "block")
            for label, blocks in zip(labels, num_blocks)
        ]
        assert len(xtick_labels) == len(labels)
        assert len(xtick_labels) > 0

        xtick_values = np.arange(0, len(xtick_labels), dtype=np.float64)
        xtick_values *= group_width
        xtick_values += 0.5 * float((group_width - group_spacing))
        # print("xvalues", xtick_values)
        # print("xlables", xtick_labels)
        xmargin = 0.5 * group_spacing
        ax.set_xlim(-xmargin, len(xtick_labels) * group_width - xmargin)

        new_width, height = figsize[0], figsize[1]

        # if per_kernel:
        if len(labels) >= 12:
            new_width = 1.5 * plot.DINA4_WIDTH_INCHES
        if len(labels) >= 18:
            new_width = 2.0 * plot.DINA4_WIDTH_INCHES

        # plot without xticks
        fig.set_size_inches(new_width, height)

        # invisible text at the top left to make the plots align under
        # each other
        ax.text(
            -0.12,
            1.0,
            "H",
            fontsize=7,
            color="red",
            alpha=0.0,
            # xy=(1.0, 1.0),
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        # plot without legend or xticks (middle)
        ax.set_xticks(xtick_values, ["" for _ in range(len(xtick_values))], rotation=0)
        fig.set_size_inches(new_width, height)

        filename = plot_dir / "{}.{}.{}_no_xticks_no_legend.pdf".format(
            profiler, benchmark, stat_col
        )
        fig.savefig(filename)

        # plot with xticks but without legend (bottom)
        ax.set_xticks(xtick_values, xtick_labels, rotation=0)
        fig.set_size_inches(new_width, height)

        filename = plot_dir / "{}.{}.{}_with_xticks_no_legend.pdf".format(
            profiler, benchmark, stat_col
        )
        fig.savefig(filename)

        # plot with legend and xticks (default)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            borderpad=0.1,
            labelspacing=0.2,
            columnspacing=2.0,
            edgecolor="none",
            frameon=False,
            fancybox=False,
            shadow=False,
            ncols=4,
        )

        fig.set_size_inches(new_width, height)

        filename = plot_dir / "{}.{}.{}.pdf".format(profiler, benchmark, stat_col)
        fig.savefig(filename)
        print(color("wrote {}".format(filename), fg="cyan"))

        # plot with legend but without xticks (top)
        ax.set_xticks(xtick_values, ["" for _ in range(len(xtick_values))], rotation=0)
        fig.set_size_inches(new_width, height)

        filename = plot_dir / "{}.{}.{}_no_xticks_with_legend.pdf".format(
            profiler, benchmark, stat_col
        )
        fig.savefig(filename)


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option(
    "--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config"
)
@click.option("-b", "--bench", "bench_name_arg", help="Benchmark name")
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
def generate(
    path,
    config_path,
    bench_name_arg,
    input_idx,
    quick,
    target,
    verbose,
    strict,
    nvprof,
    nsight,
    output_path,
):
    b = Benchmarks(path)
    results_dir = Path(b.config["results_dir"])

    if target is not None:
        targets = [t for t in Target if t.value.lower() == target.lower()]
    else:
        targets = [
            Target.Profile,
            Target.Simulate,
            Target.ExecDrivenSimulate,
            Target.AccelsimSimulate,
            Target.PlaygroundSimulate,
        ]

    print("targets: {}".format([str(t) for t in targets]))
    print("benchmarks: {}".format(bench_name_arg))

    benches = defaultdict(list)
    for target in targets:
        if bench_name_arg is None:
            bench_names = b.benchmarks[target.value].keys()
        elif isinstance(bench_name_arg, str):
            bench_names = [bench_name_arg]
        elif isinstance(bench_name_arg, list):
            bench_names = bench_name_arg
        else:
            raise ValueError

        for bench_name in bench_names:
            benches[bench_name].extend(b.benchmarks[target.value][bench_name])

    print(
        "processing {} benchmark configurations ({} targets)".format(
            sum([len(b) for b in benches.values()]), len(targets)
        )
    )

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    profilers = []
    if nvprof:
        profilers += ["nvprof"]
    if nsight:
        profilers += ["nsight"]

    for profiler in profilers:
        for bench_name, bench_configs in benches.items():
            all_stats = []
            for bench_config in bench_configs:
                name = bench_config["name"]
                target = bench_config["target"]
                input_idx = bench_config["input_idx"]
                input_values = bench_config["values"]
                target_name = f"[{target}]"

                if quick:
                    if input_values.get("mode") not in ["serial", None]:
                        continue
                    # if input_values.get("memory_only") not in [False, None]:
                    #     continue
                    if input_values.get("cores_per_cluster") not in [
                        int(benchmarks.BASELINE["cores_per_cluster"]),
                        None,
                    ]:
                        continue
                    if input_values.get("num_clusters") not in [
                        int(benchmarks.BASELINE["num_clusters"]),
                        None,
                    ]:
                        continue

                current_bench_log_line = " ===> {:>20} {:>15}@{:<4} {}".format(
                    target_name, name, input_idx, input_values
                )

                try:
                    match (target.lower(), profiler):
                        case ("profile", "nvprof"):
                            target_name += "[nvprof]"
                            bench_stats = gpucachesim.stats.native.NvprofStats(
                                config, bench_config
                            )
                        case ("profile", "nsight"):
                            target_name += "[nsight]"
                            bench_stats = gpucachesim.stats.native.NsightStats(
                                config, bench_config
                            )
                        case ("simulate", _):
                            bench_stats = gpucachesim.stats.stats.Stats(
                                config, bench_config
                            )
                        case ("execdrivensimulate", _):
                            bench_stats = gpucachesim.stats.stats.ExecDrivenStats(
                                config, bench_config
                            )
                        case ("accelsimsimulate", _):
                            bench_stats = gpucachesim.stats.accelsim.Stats(
                                config, bench_config
                            )
                        case ("playgroundsimulate", _):
                            bench_stats = gpucachesim.stats.playground.Stats(
                                config, bench_config
                            )
                        case other:
                            print(
                                color(
                                    f"WARNING: {name} has unknown target {other}",
                                    fg="red",
                                )
                            )
                            continue
                    print(current_bench_log_line)
                except Exception as e:
                    # allow babelstream for exec driven to be missing
                    if (target.lower(), name.lower()) == (
                        "execdrivensimulate",
                        "babelstream",
                    ):
                        continue

                    print(color(current_bench_log_line, fg="red"))
                    if strict:
                        raise e
                    continue

                values = pd.DataFrame.from_records([bench_config["values"]])
                values.columns = ["input_" + c for c in values.columns]

                # this will be the new index
                values["target"] = target
                values["benchmark"] = name
                values["input_id"] = input_idx

                values = bench_stats.result_df.merge(values, how="cross")
                assert "run" in values.columns

                if verbose:
                    print(values.T)
                all_stats.append(values)

            all_stats = pd.concat(all_stats)
            if verbose:
                print(all_stats)

            stats_output_path = (
                results_dir / f"combined.stats.{profiler}.{bench_name}.csv"
            )

            if output_path is not None:
                stats_output_path = Path(output_path)

            print(color(f"saving to {stats_output_path}", fg="cyan"))
            stats_output_path.parent.mkdir(parents=True, exist_ok=True)
            all_stats.to_csv(stats_output_path, index=False)


TIMING_COLS_SUMMING_TO_FULL_CYCLE = [
    "cycle::core",
    "cycle::dram",
    "cycle::interconn",
    "cycle::issue_block_to_core",
    "cycle::l2",
    "cycle::subpartitions",
]


def _build_timings_pie(ax, timings_df, sections, colors, title=None, validate=False):
    if validate:
        # for _, df in timings_df.groupby(["benchmark", "input_id", "target", "run"]):
        #     print(df)
        #     exec_time_sec = df["exec_time_sec"]
        #     print(exec_time_sec)
        #     assert len(exec_time_sec.unique()) == 1
        #
        #     total = df.loc[cols_summing_to_full_cycle, "total"].sum()
        #     # total = df.T[cols_summing_to_full_cycle].T["total"] # [cols_summing_to_full_cycle].sum()
        #     print(total)
        #     df["abs_diff"] = (total - exec_time_sec).abs()
        #     # abs_diff = (total - exec_time_sec).abs()
        #     df["rel_diff"] = (1 - (total / exec_time_sec)).abs()
        #     # rel_diff = (1 - (total / exec_time_sec)).abs()
        #
        #     valid_rel = df["rel_diff"] <= 0.2
        #     valid_abs = df["abs_diff"] <= 0.1
        #     # print(timings_df[timings_df["total"] > timings_df["exec_time_sec"]])
        #     # print(timings_df[~(valid_rel | valid_abs)])
        #     if not (valid_rel | valid_abs).all():
        #         invalid = ~(valid_rel | valid_abs)
        #         print(df.loc[invalid, ["total", "exec_time_sec", "abs_diff", "rel_diff"]])
        #
        #     assert (valid_rel | valid_abs).all()
        pass

    def stderr(df):
        return df.std() / np.sqrt(len(df))

    averaged = timings_df.groupby("name")[
        [
            "total_sec",
            "share",
            "mean_sec",
            "mean_micros",
            "mean_millis",
            "exec_time_sec",
            # "total_cores",
        ]
    ].agg(["min", "max", "mean", "median", "std", "sem", stderr])

    # make sure sem is correct
    all_sem = averaged.iloc[:, averaged.columns.get_level_values(1) == "sem"]
    all_sem.columns = all_sem.columns.droplevel(1)
    all_stderr = averaged.iloc[:, averaged.columns.get_level_values(1) == "stderr"]
    all_stderr.columns = all_stderr.columns.droplevel(1)
    assert ((all_sem - all_stderr).abs() > 0.001).sum().sum() == 0

    # total does not really say much, because we are averaging for different
    # benchmark configurations
    # print("\n\n=== TOTAL")
    # pd.options.display.float_format = "{:.2f}".format
    # print(averaged["total"])

    def compute_gustafson_speedup(p, n):
        s = 1 - p
        assert 1 + (n - 1) * p == s + p * n
        return 1 + (n - 1) * p

    def compute_amdahl_speedup(p, n):
        """p is the fraction of parallelizeable work. n is the speedup of that parallel part, i.e. number of processors."""
        return 1 / ((1 - p) + p / n)

    threads = 8
    parallel_frac = float(averaged.loc["cycle::core", ("share", "median")])
    amdahl_speedup = compute_amdahl_speedup(p=parallel_frac, n=threads)
    print(
        "AMDAHL SPEEDUP = {:>6.3f}x for {:>2} threads (p={:>5.2f})".format(
            amdahl_speedup, threads, parallel_frac
        )
    )

    gustafson_speedup = compute_gustafson_speedup(p=parallel_frac, n=threads)
    print(
        "GUSTAFSON SPEEDUP = {:>6.3f}x for {:>2} threads (p={:>5.2f})".format(
            gustafson_speedup, threads, parallel_frac
        )
    )

    print("\n\n=== MEAN MICROSECONS")
    pd.options.display.float_format = "{:.6f}".format
    print(averaged["mean_micros"])
    print("\n\n=== SHARE")
    pd.options.display.float_format = "{:.2f}".format
    print(averaged["share"] * 100.0)

    # validate averaged values
    total_cycle_share = averaged["share", "mean"].T["cycle::total"]

    computed_total_cycle_share = averaged["share", "mean"].T[
        TIMING_COLS_SUMMING_TO_FULL_CYCLE
    ]
    if computed_total_cycle_share.sum() > total_cycle_share:
        print(total_cycle_share, computed_total_cycle_share.sum())
    assert computed_total_cycle_share.sum() <= total_cycle_share

    unit = "mean_micros"
    agg = "median"
    idx = pd.MultiIndex.from_product((["share", unit], [agg, "std"]))
    # print(averaged[idx])

    # sort based on share
    shares = averaged.loc[sections, idx]
    shares = shares.sort_values([("share", agg)], ascending=False)

    # compute other
    other = 1.0 - shares["share", agg].sum()
    # print("other:", other)
    shares.loc["other", :] = 0.0
    shares.loc["other", ("share", agg)] = other
    print(shares)

    values = shares["share", agg].values * 100.0
    wedges, texts, autotexts = ax.pie(
        values,
        # labels=shares.index,
        # autopct=compute_label,
        autopct="",
        colors=[colors[s] for s in shares.index],
        # labeldistance=1.2,
        pctdistance=1.0,
    )
    # textprops=dict(color="w"))

    labels = shares.index
    # labels = [r"{} (${:4.1f}\%$)".format(label, values[i])
    #           for i, label in enumerate(shares.index)]
    # # labels = [label.removeprefix("cycle::").replace("_", " ").capitalize() for label in shares.index]
    # legend = ax.legend(wedges, labels,
    #       # title="Ingredients",
    #       loc="center left",
    #       bbox_to_anchor=(1, 0, 0.5, 1))
    #
    # bbox_extra_artists.append(legend)

    for i, a in enumerate(autotexts):
        share = values[i]
        col = shares.index[i].lower()

        # compute desired pct distance
        if share >= 40.0:
            label_dist = 0.5
        else:
            label_dist = 0.7
        xi, yi = a.get_position()
        ri = np.sqrt(xi**2 + yi**2)
        phi = np.arctan2(yi, xi)
        x = label_dist * ri * np.cos(phi)
        y = label_dist * ri * np.sin(phi)
        a.set_position((x, y))
        # print(col, share, label_dist)

        if share < 5.0 or col == "other":
            a.set_text("")
        else:
            label = r"${:>4.1f}\%".format(share)
            share_std = shares["share", "std"].iloc[i] * 100.0
            # label += r" \pm {:>4.1f}\%".format(share_std)
            label += "$"
            label += "\n"

            dur = shares[unit, agg].iloc[i]
            dur_std = shares[unit, "std"].iloc[i]
            label += r"${:4.1f}\mu s$".format(dur)
            # label += r"$({:4.1f}ms \pm {:4.2f}ms)$".format(dur_mean, dur_std)

            if col == "cycle::core":
                dur_per_sm = averaged.loc["core::cycle", (unit, agg)]
                # temp fix
                dur_per_sm = dur / 28
                label += "\n"
                label += r"${:4.1f}\mu s$ per core".format(dur_per_sm)
            a.set_text(label)

    # plt.setp(autotexts, size=fontsize, weight="bold")

    if title is not None:
        ax.set_title(title)

    return wedges, list(labels), texts, autotexts


@main.command()
@click.option(
    "-p",
    "--path",
    default=DEFAULT_BENCH_FILE,
    help="Path to materialized benchmark config",
)
@click.option("-b", "--bench", "bench_name_arg", help="Benchmark name")
@click.option("--baseline", type=bool, default=True, help="Baseline configurations")
@click.option("--strict", type=bool, default=True, help="strict mode")
@click.option("--validate", type=bool, is_flag=True, help="validate")
def timings(path, bench_name_arg, baseline, strict, validate):
    print("loading", path)
    b = Benchmarks(path)
    benches = b.benchmarks[Target.Simulate.value]

    if bench_name_arg is not None:
        selected_benchmarks = [
            bench_name
            for bench_name in benches.keys()
            if bench_name.lower() == bench_name_arg.lower()
        ]
    else:
        selected_benchmarks = list(benches.keys())

    timings_dfs = []
    for bench_name in selected_benchmarks:
        bench_configs = benches[bench_name]

        def is_baseline(config):
            return not baseline or all(
                [
                    config["values"].get("memory_only") in [False, None],
                    config["values"].get("num_clusters")
                    in [int(benchmarks.BASELINE["num_clusters"]), None],
                    config["values"].get("cores_per_cluster")
                    in [int(benchmarks.BASELINE["cores_per_cluster"]), None],
                    config["values"].get("mode") in ["serial", None],
                ]
            )

        baseline_bench_configs = [
            config for config in bench_configs if is_baseline(config)
        ]

        for bench_config in baseline_bench_configs:
            repetitions = int(bench_config["common"]["repetitions"])
            target_config = bench_config["target_config"].value
            stats_dir = Path(target_config["stats_dir"])
            assert bench_config["values"]["mode"] == "serial"

            num_clusters = bench_config["values"].get(
                "num_clusters", benchmarks.BASELINE["num_clusters"]
            )
            cores_per_cluster = bench_config["values"].get(
                "cores_per_cluster", benchmarks.BASELINE["cores_per_cluster"]
            )
            total_cores = num_clusters * cores_per_cluster
            assert total_cores == 28

            for r in range(repetitions):
                sim_df = pd.read_csv(
                    stats_dir / f"stats.sim.{r}.csv",
                    header=0,
                )
                sim_df["run"] = r
                grouped_sim_including_no_kernel = sim_df.groupby(
                    gpucachesim.stats.stats.INDEX_COLS, dropna=False
                )
                grouped_sim_excluding_no_kernel = sim_df.groupby(
                    gpucachesim.stats.stats.INDEX_COLS, dropna=True
                )

                timings_path = stats_dir / f"timings.{r}.csv"
                # timings_path = stats_dir / f"timings.csv"
                # print(timings_path)
                if not strict and not timings_path.is_file():
                    continue

                assert timings_path.is_file()

                timing_df = pd.read_csv(timings_path, header=0)
                timing_df = timing_df.rename(columns={"total": "total_sec"})
                timing_df["benchmark"] = bench_config["name"]
                timing_df["input_id"] = bench_config["input_idx"]
                timing_df["target"] = bench_config["target"]
                timing_df["run"] = r

                timing_df["total_cores"] = total_cores
                timing_df["mean_blocks_per_sm"] = (
                    grouped_sim_excluding_no_kernel["num_blocks"].mean().mean()
                    / total_cores
                )
                timing_df["exec_time_sec"] = (
                    grouped_sim_including_no_kernel["elapsed_millis"].sum().sum()
                )
                timing_df["exec_time_sec"] /= 1000.0

                timings_dfs.append(timing_df)

    timings_df = pd.concat(timings_dfs)
    timings_df = timings_df.set_index(["name"])
    # timings_df = timings_df.set_index(["target", "benchmark", "input_id", "run", "name"])
    index_cols = ["target", "benchmark", "input_id", "run"]
    timings_df["max_total_sec"] = timings_df.groupby(index_cols)["total_sec"].transform(
        "max"
    )

    def compute_exec_time_sec(df) -> float:
        time = df.loc[TIMING_COLS_SUMMING_TO_FULL_CYCLE, "total_sec"].sum()
        return time

    computed_exec_time_sec = (
        timings_df.groupby(index_cols)[timings_df.columns]
        .apply(compute_exec_time_sec)
        .rename("computed_exec_time_sec")
    )

    before = len(timings_df)
    timings_df = timings_df.reset_index().merge(
        computed_exec_time_sec, on=index_cols, how="left"
    )
    assert len(timings_df) == before

    if "computed_exec_time_sec" in timings_df:
        timings_df["abs_diff_to_real"] = (
            timings_df["computed_exec_time_sec"] - timings_df["exec_time_sec"]
        ).abs()
        timings_df["rel_diff_to_real"] = (
            1 - (timings_df["computed_exec_time_sec"] / timings_df["exec_time_sec"])
        ).abs()

    # exec time sec is usually more efficient when timing is disabled.
    # while its not quite the real thing, we normalize to max total timing
    timings_df["exec_time_sec"] = timings_df["max_total_sec"]

    # timings_df["exec_time_sec"] = timings_df[["max_total", "exec_time_sec"]].max(axis=1)
    timings_df["mean_sec"] = timings_df["total_sec"] / timings_df["count"]
    timings_df["mean_millis"] = timings_df["mean_sec"] * 1000.0
    timings_df["mean_micros"] = timings_df["mean_millis"] * 1000.0
    timings_df["share"] = timings_df["total_sec"] / timings_df["exec_time_sec"]
    timings_df = timings_df.set_index("name")

    # filter
    sufficient_size_mask = timings_df["mean_blocks_per_sm"] > 1.0
    sufficient_size_timings_df = timings_df[sufficient_size_mask]

    print(timings_df.head(n=10).T)
    print(timings_df.head(n=30))

    fontsize = plot.FONT_SIZE_PT - 4
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    figsize = (
        0.8 * plot.DINA4_WIDTH_INCHES,
        0.2 * plot.DINA4_HEIGHT_INCHES,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    sections = [
        "cycle::core",
        "cycle::dram",
        "cycle::interconn",
        "cycle::issue_block_to_core",
        "cycle::l2",
        "cycle::subpartitions",
    ]
    cmap = plt.get_cmap("tab20")
    # cmap = plt.get_cmap('tab20c')
    # cmap = plt.get_cmap('Set3')

    colors = cmap(np.linspace(0, 1.0, len(sections)))
    colors = [
        "lightskyblue",
        "gold",
        "yellowgreen",
        "lightcoral",
        "violet",
        "palegreen",
    ]
    assert len(colors) == len(sections)

    colors = {section: colors[i] for i, section in enumerate(sections)}
    colors["other"] = "whitesmoke"

    args = dict(sections=sections, colors=colors, validate=validate)

    print("=============== blocks/core <= 1 =============")
    title = r"$N_{\text{blocks}}$/SM $\leq 1$"
    samples = len(timings_df[index_cols].drop_duplicates())
    title += "\n({} benchmark configurations)".format(samples)

    total_micros = (
        timings_df.loc["cycle::total", :]
        .groupby(index_cols)["mean_micros"]
        .first()
        .median()
    )
    title += "\n" + r"${:4.1f}\mu s$ total".format(total_micros)

    wedges1, labels1, texts1, autotexts1 = _build_timings_pie(
        ax1, timings_df, title=title, **args
    )

    print("=============== blocks/core > 1 =============")
    title = r"$N_{\text{blocks}}$/SM $>1$"
    samples = len(sufficient_size_timings_df[index_cols].drop_duplicates())
    title += "\n({} benchmark configurations)".format(samples)

    total_micros = (
        sufficient_size_timings_df.loc["cycle::total", :]
        .groupby(index_cols)["mean_micros"]
        .first()
        .median()
    )
    title += "\n" + r"${:4.1f}\mu s$ total".format(total_micros)

    wedges2, labels2, texts2, autotexts2 = _build_timings_pie(
        ax2, sufficient_size_timings_df, title=title, **args
    )

    handles = wedges1 + wedges2
    labels = labels1 + labels2
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    legend = fig.legend(
        *zip(*unique),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        edgecolor="none",
        frameon=False,
        fancybox=False,
        shadow=False,
    )
    bbox_extra_artists = [legend]

    plot_dir = plot.PLOT_DIR
    plot_dir.parent.mkdir(parents=True, exist_ok=True)

    filename = plot_dir / "timings_pie.pdf"

    print(color("wrote {}".format(filename), fg="cyan"))
    fig.tight_layout()
    plt.tight_layout()
    fig.savefig(filename, bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")


if __name__ == "__main__":
    main()
    # main(ctx={})
