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

import gpucachesim.plot as plot
import gpucachesim.utils as utils

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
pd.set_option('future.no_silent_downcasting', True)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.set_printoptions(suppress=True, formatter={"float_kind": "{:f}".format})
np.seterr(all="raise")
print("pandas version: {}".format(pd. __version__))

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
    cores_per_cluster=int(common.BASELINE["cores_per_cluster"]),
    num_clusters=int(common.BASELINE["num_clusters"]),
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
            selected_df, per_kernel=per_kernel,
            inspect=inspect, mean=mean)



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


class ParallelTableRow(typing.NamedTuple):
    metric: str
    threads: int
    serial_value: typing.Optional[typing.Tuple[float, typing.Union[float, int, str]]]
    det_value: typing.Optional[typing.Tuple[float, typing.Union[float, int, str]]]
    nondet_values: typing.Sequence[typing.Tuple[float, typing.Union[float, int, str]]]

    def values(self):
        values = []
        if self.serial_value is not None:
            values.append(self.serial_value[0])
        if self.det_value is not None:
            values.append(self.det_value[0])
        values += [v[0] for v in self.nondet_values]
        return values


def build_parallel_table_rows(
    df: pd.DataFrame,
    num_bench_configs: int,
    thousands_round_to=1,
    variable_precision=True,
) -> typing.Sequence[ParallelTableRow]:
    # interleave_n = list(itertools.product([False, True], [5, 10]))
    run_ahead_values = [5, 10]
    table_rows: typing.Sequence[ParallelTableRow] = []

    assert num_bench_configs > 0
    multiple_bench_configs = num_bench_configs > 1

    for threads in [4, 8]:
        threads_mask = df["input_threads_parallel"] == threads
        det_mask = df["input_mode_parallel"] == "deterministic"
        nondet_mask = df["input_mode_parallel"] == "nondeterministic"

        preview_cols = (
            benchmarks.BENCH_TARGET_INDEX_COLS
            + ["kernel_name", "kernel_launch_id", "run"]
            + list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS))
            + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
            + [col + "_parallel" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
            + [
                "exec_time_sec_parallel",
                "input_id_parallel",
                "input_id_serial",
                "cycles_serial",
                "cycles_parallel",
                "cycles_mape",
                # "dram_reads_serial",
                # "dram_reads_parallel",
                # "dram_reads_rel_err",
                # "dram_writes_serial",
                # "dram_writes_parallel",
                # "dram_writes_rel_mape",
            ]
            # + different_cols(det)
        )
        preview_cols = [col for col in preview_cols if col in df]

        all_parallel = df[(nondet_mask | det_mask) & threads_mask]

        # diff = set(preview_cols) - set(list(all_parallel.columns))
        # print(diff)

        print("max speedup for {} threads is {}".format(
            threads, all_parallel["exec_time_sec_speedup"].max()))
        weird_mask = all_parallel["exec_time_sec_speedup"] > threads
        weird = all_parallel.loc[weird_mask,preview_cols]
        print("weird results for {} threads:".format(threads))
        if len(weird) > 0:
            print(color("WARNING", fg="red"))
            print(weird.T)
            print("===")
        # assert len(weird) == 0

        # nondet_no_interleave_mask = df["input_mode_parallel"] == "nondeterministic"
        # nondet_interleave_mask = (
        #     df["input_mode_parallel"] == "nondeterministic_interleave"
        # )
        # print([m.sum() for m in [
        #     mask, threads_mask, det_mask, nondet_no_interleave_mask, nondet_interleave_mask
        # ]])

        det = df[threads_mask & det_mask]
        if False:
            if num_bench_configs > 1:
                print(det.loc[det["benchmark"] == "vectorAdd", preview_cols].T)
            else:
                print(det.loc[:,preview_cols].T)
        print("===")
        all_nondet = df[threads_mask & nondet_mask]
        # nondet_no_interleave = df[threads_mask & nondet_no_interleave_mask]
        # nondet_interleave = df[threads_mask & nondet_interleave_mask]

        print("num det={} num benchmark configs={}".format(len(det), num_bench_configs))
        # print(det)
        assert len(det) == num_bench_configs
        assert len(all_nondet) == len(run_ahead_values) * num_bench_configs

        # assert len(nondet_no_interleave) == 2 * num_bench_configs
        # assert len(nondet_interleave) == 2 * num_bench_configs
        # assert (
        #     len(
        #         df[[
        #             "exec_time_sec_serial",
        #             "cycles_serial",
        #             "input_id_serial",
        #         ]].drop_duplicates()
        #     )
        #     == 1
        # )


        # exec time (speedup)
        serial_exec_time = df.loc[threads_mask, "exec_time_sec_serial"].mean()
        det_exec_time = det["exec_time_sec_parallel"].mean()
        det_speedup = det["exec_time_sec_speedup"].mean()
        if multiple_bench_configs:
            preview_cols = list(
                benchmarks.BENCH_TARGET_INDEX_COLS
                + benchmarks.INDEX_COLS
                + [c for c in benchmarks.SIMULATE_INPUT_COLS]
                + [c + "_parallel" for c in benchmarks.SIMULATE_INPUT_COLS]
                + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
            )
            preview_cols = [col for col in preview_cols if col in df]

            print("DETERMINISTIC {}".format(det.shape))
            print(det[preview_cols][:8].T)

            # make sure we aggregate a single functional config only
            assert det["input_cores_per_cluster"].nunique() == 1
            assert det["input_num_clusters"].nunique() == 1
            assert det["input_memory_only"].nunique() == 1

        nondet_values = []
        # for interleave, n in interleave_n:
        for run_ahead in run_ahead_values:
            # nondet = nondet_interleave if interleave else nondet_no_interleave
            # print("run ahead={}".format(run_ahead))
            nondet = all_nondet[all_nondet["input_run_ahead_parallel"] == run_ahead]
            # print(nondet.T)
            # assert len(nondet) == 1
            assert len(nondet) == num_bench_configs

            nondet_exec_time = nondet["exec_time_sec_parallel"].mean()
            nondet_speedup = nondet["exec_time_sec_speedup"].mean()
            if multiple_bench_configs:
                nondet_values.append(
                    (
                        nondet_speedup,
                        "${}x$".format(
                            plot.round_to_precision_str(
                                nondet_speedup,
                                round_to=1,
                                variable_precision=variable_precision,
                            )
                        ),
                    )
                )

            else:
                nondet_values.append(
                    (
                        nondet_exec_time,
                        "${:>3.1f}s~({}x)$".format(
                            nondet_exec_time,
                            plot.round_to_precision_str(
                                nondet_speedup,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if multiple_bench_configs
            else (serial_exec_time, "${:>3.1f}s$".format(serial_exec_time))
        )
        if multiple_bench_configs:
            det_value = (
                det_speedup,
                "${}x$".format(
                    plot.round_to_precision_str(
                        det_speedup, round_to=1, variable_precision=variable_precision
                    )
                ),
            )
        else:
            det_value = (
                det_exec_time,
                "${:>3.1f}s~({}x)$".format(
                    det_exec_time,
                    plot.round_to_precision_str(
                        det_speedup, round_to=1, variable_precision=variable_precision
                    ),
                ),
            )
        table_rows.append(
            ParallelTableRow(
                metric=r"exec\\time",
                threads=threads,
                serial_value=serial_value,
                det_value=det_value,
                nondet_values=nondet_values,
            )
        )

        # cycles (rel err)
        serial_cycles = int(df.loc[threads_mask, "cycles_serial"].mean())
        det_cycles = int(det["cycles_parallel"].mean())
        det_rel_err = det["cycles_mape"].mean()
        nondet_values = []
        # for interleave, n in interleave_n:
        for run_ahead in run_ahead_values:
            # nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = all_nondet[all_nondet["input_run_ahead_parallel"] == run_ahead]
            # assert len(nondet) == 1
            # assert len(nondet) == num_bench_configs

            nondet_cycles = int(nondet["cycles_parallel"].mean())
            nondet_rel_err = nondet["cycles_mape"].mean()
            if multiple_bench_configs:
                nondet_values.append(
                    (
                        nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            )
                        ),
                    )
                )
            else:
                nondet_values.append(
                    (
                        nondet_cycles,
                        "${} ({}\\%)$".format(
                            plot.human_format_thousands(
                                nondet_cycles,
                                round_to=thousands_round_to,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if multiple_bench_configs
            else (
                serial_cycles,
                "${}$".format(
                    plot.human_format_thousands(
                        serial_cycles,
                        round_to=thousands_round_to,
                        variable_precision=variable_precision,
                    )
                ),
            )
        )
        if multiple_bench_configs:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    )
                ),
            )
        else:
            det_value = (
                det_cycles,
                "${} ({}\\%)$".format(
                    plot.human_format_thousands(
                        det_cycles,
                        round_to=thousands_round_to,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        table_rows.append(
            ParallelTableRow(
                metric="cycles",
                threads=threads,
                serial_value=serial_value,
                det_value=det_value,
                nondet_values=nondet_values,
            )
        )

        # l1 data hit rate (rel err)
        serial_l1_hit_rate = df.loc[threads_mask, "l1_hit_rate_serial"].mean()
        det_l1_hit_rate = det["l1_hit_rate_parallel"].mean()
        det_rel_err = det["l1_hit_rate_mae"].mean()
        nondet_values = []
        # for interleave, n in interleave_n:
        for run_ahead in run_ahead_values:
            # nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = all_nondet[all_nondet["input_run_ahead_parallel"] == run_ahead]
            # assert len(nondet) == 1
            # assert len(nondet) == num_bench_configs

            nondet_l1_hit_rate = nondet["l1_hit_rate_parallel"].mean()
            nondet_rel_err = nondet["l1_hit_rate_mae"].mean()
            if multiple_bench_configs:
                nondet_values.append(
                    (
                        100.0 * nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )
            else:
                nondet_values.append(
                    (
                        100.0 * nondet_l1_hit_rate,
                        "${}\\%~({}\\%)$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_l1_hit_rate,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if multiple_bench_configs
            else (
                100.0 * serial_l1_hit_rate,
                "${:>2.1f}\\%$".format(100.0 * serial_l1_hit_rate),
            )
        )
        if multiple_bench_configs:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        else:
            det_value = (
                100.0 * det_l1_hit_rate,
                "${}\\%~({}\\%)$".format(
                    plot.round_to_precision_str(
                        100.0 * det_l1_hit_rate,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )

        table_rows.append(
            ParallelTableRow(
                metric=r"L1D\\hit rate",
                threads=threads,
                serial_value=serial_value,
                det_value=det_value,
                nondet_values=nondet_values,
            )
        )

        # l2 data hit rate (rel err)
        serial_l2_hit_rate = df.loc[threads_mask, "l2_hit_rate_serial"].mean()
        det_l2_hit_rate = det["l2_hit_rate_parallel"].mean()
        det_rel_err = det["l2_hit_rate_mae"].mean()
        nondet_values = []
        # for interleave, n in interleave_n:
        for run_ahead in run_ahead_values:
            # nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = all_nondet[all_nondet["input_run_ahead_parallel"] == run_ahead]
            # assert len(nondet) == 1
            # assert len(nondet) == num_bench_configs

            nondet_l2_hit_rate = nondet["l2_hit_rate_parallel"].mean()
            nondet_rel_err = nondet["l2_hit_rate_mae"].mean()
            if multiple_bench_configs:
                nondet_values.append(
                    (
                        100.0 * nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )
            else:
                nondet_values.append(
                    (
                        100.0 * nondet_l2_hit_rate,
                        "${}\\%~({}\\%)$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_l2_hit_rate,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if multiple_bench_configs
            else (
                100.0 * serial_l2_hit_rate,
                "${}\\%$".format(
                    plot.round_to_precision_str(
                        100.0 * serial_l2_hit_rate,
                        round_to=1,
                        variable_precision=variable_precision,
                    )
                ),
            )
        )
        if multiple_bench_configs:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        else:
            det_value = (
                100.0 * det_l2_hit_rate,
                "${}\\%~({}\\%)$".format(
                    plot.round_to_precision_str(
                        100.0 * det_l2_hit_rate,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        table_rows.append(
            ParallelTableRow(
                metric=r"L2D\\hit rate",
                threads=threads,
                serial_value=serial_value,
                det_value=det_value,
                nondet_values=nondet_values,
            )
        )

        # dram reads (rel err)
        serial_dram_reads = int(df.loc[threads_mask, "dram_reads_serial"].mean())
        det_dram_reads = int(det["dram_reads_parallel"].mean())
        det_rel_err = det["dram_reads_smape"].mean()
        nondet_values = []
        # for interleave, n in interleave_n:
        for run_ahead in run_ahead_values:
            # nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = all_nondet[all_nondet["input_run_ahead_parallel"] == run_ahead]
            # assert len(nondet) == 1
            # assert len(nondet) == num_bench_configs

            nondet_dram_reads = int(nondet["dram_reads_parallel"].mean())
            nondet_rel_err = nondet["dram_reads_smape"].mean()
            if multiple_bench_configs:
                nondet_values.append(
                    (
                        nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )
            else:
                nondet_values.append(
                    (
                        nondet_dram_reads,
                        "${} ({}\\%)$".format(
                            plot.human_format_thousands(
                                nondet_dram_reads,
                                round_to=thousands_round_to,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if multiple_bench_configs
            else (
                serial_dram_reads,
                "${}$".format(
                    plot.human_format_thousands(
                        serial_dram_reads,
                        round_to=thousands_round_to,
                        variable_precision=variable_precision,
                    )
                ),
            )
        )
        if multiple_bench_configs:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        else:
            det_value = (
                det_dram_reads,
                "${} ({}\\%)$".format(
                    plot.human_format_thousands(
                        det_dram_reads,
                        round_to=thousands_round_to,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )

        table_rows.append(
            ParallelTableRow(
                metric=r"DRAM\\reads",
                threads=threads,
                serial_value=serial_value,
                det_value=det_value,
                nondet_values=nondet_values,
            )
        )

        # dram writes (rel err)
        serial_dram_writes = int(df.loc[threads_mask, "dram_writes_serial"].mean())
        det_dram_writes = int(det["dram_writes_parallel"].mean())
        det_rel_err = det["dram_writes_smape"].mean()
        nondet_values = []
        # for interleave, n in interleave_n:
        for run_ahead in run_ahead_values:
            # nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = all_nondet[all_nondet["input_run_ahead_parallel"] == run_ahead]
            # assert len(nondet) == 1
            # assert len(nondet) == num_bench_configs

            nondet_dram_writes = int(nondet["dram_writes_parallel"].mean())
            nondet_rel_err = nondet["dram_writes_smape"].mean()
            if multiple_bench_configs:
                nondet_values.append(
                    (
                        100.0 * nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )
            else:
                nondet_values.append(
                    (
                        nondet_dram_writes,
                        "${} ({}\\%)$".format(
                            plot.human_format_thousands(
                                nondet_dram_writes,
                                round_to=thousands_round_to,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision_str(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if multiple_bench_configs
            else (
                serial_dram_writes,
                "${}$".format(
                    plot.human_format_thousands(
                        serial_dram_writes,
                        round_to=thousands_round_to,
                        variable_precision=variable_precision,
                    )
                ),
            )
        )
        if multiple_bench_configs:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        else:
            det_value = (
                det_dram_writes,
                "${} ({}\\%)$".format(
                    plot.human_format_thousands(
                        det_dram_writes,
                        round_to=thousands_round_to,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision_str(
                        100.0 * det_rel_err,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                ),
            )
        table_rows.append(
            ParallelTableRow(
                metric=r"DRAM\\writes",
                threads=threads,
                serial_value=serial_value,
                det_value=det_value,
                nondet_values=nondet_values,
            )
        )
    return table_rows


def slowdown(baseline, values):
    return values / baseline


def speedup(baseline, values):
    return baseline / values


def geo_mean(values: np.ndarray) -> np.ndarray:
    a = np.array(values)
    return a.prod() ** (1.0 / len(a))


# def geo_mean(values: np.narray):
#     return np.exp(np.log(values).mean())


def bounded_relative_absolute_error(
    true_values: np.ndarray, values: np.ndarray, **kwargs
) -> np.ndarray:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    correct = values == true_values

    # we only deal with positive numbers
    assert np.all(values >= 0.0)
    assert np.all(true_values >= 0.0)

    brae = values.abs() / (values.abs() + true_values.abs())
    brae = brae.fillna(0.0)
    # brae[brae] = 0.0
    brae[brae == 0.0] = 0.0
    return brae


def rel_err(
    true_values: np.ndarray, values: np.ndarray, eps: typing.Optional[float] = None
) -> np.ndarray:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    correct = values == true_values

    # we only deal with positive numbers
    assert np.all(values >= 0.0)
    assert np.all(true_values >= 0.0)

    # because we only use posive numbers, we can safely clip to a small positive epsilon
    # if eps is not None:
    #     values = values + eps
    #     true_values = true_values + eps
    #     # true_values = np.clip(true_values, a_min=eps, a_max=None)
    rel_err = (values - true_values).abs() / true_values
    # rel_err = values.abs() / (values.abs() + true_values.abs())

    # print(values)
    # print(true_values)
    # print(values == true_values)
    rel_err = rel_err.fillna(0.0)
    rel_err[correct] = 0.0
    rel_err[rel_err == 0.0] = 0.0

    return rel_err


def rpd(true_values: np.ndarray, values: np.ndarray):
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    pass
    # rel_err = (values - true_values).abs() / true_values
    # rel_err = rel_err.fillna(0.0)
    # rel_err[rel_err == 0.0] = 0.0
    # return rel_err


def mse(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return sklearn.metrics.mean_squared_error(true_values, values)


def rmse_real(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return ((values - true_values) ** 2).mean() ** 0.5


def rmse(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    diff = values - true_values
    scale = values.abs() + true_values.abs()
    return (diff / scale).mean()


def abs_err(true_values: np.ndarray, values: np.ndarray) -> np.array:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return (true_values - values).abs()
    # return sklearn.metrics.mean_absolute_error(true_values, values)


def smape(true_values: np.ndarray, values: np.ndarray) -> float:
    """SMAPE (symmetric)"""
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)

    smape = (values - true_values).abs() / (values.abs() + true_values.abs())
    return smape.mean()


def ermsle(true_values: np.ndarray, values: np.ndarray) -> float:
    """ERMSLE: Exponential root mean square log error"""
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    ratios = values / true_values
    ratios[values == true_values] = 1.0

    log_ratios = np.empty_like(ratios)
    valid_mask = np.isfinite(ratios) & ratios != 0

    # temp
    ratios[~valid_mask] = 1.0
    log_ratios = np.abs(np.log(ratios)) ** 2
    # undo temp
    log_ratios[~valid_mask] = np.nan
    # mean
    rmsle = np.sqrt(np.mean(log_ratios[valid_mask]))
    # exponential
    rmsle = np.abs(np.exp(rmsle))
    return rmsle


def emale(true_values: np.ndarray, values: np.ndarray) -> float:
    """EMALE: Exponential mean absolute log error"""
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    ratios = values / true_values
    ratios[values == true_values] = 1.0

    log_ratios = np.empty_like(ratios)
    valid_mask = np.isfinite(ratios) & ratios != 0

    # temp
    ratios[~valid_mask] = 1.0
    log_ratios = np.abs(np.log(ratios))
    # undo temp
    log_ratios[~valid_mask] = np.nan
    # mean
    male = np.mean(log_ratios[valid_mask])
    # exponential
    emale = np.abs(np.exp(male))
    return emale


def mape(true_values: np.ndarray, values: np.ndarray) -> np.array:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return sklearn.metrics.mean_absolute_percentage_error(true_values, values)


def correlation(true_values: np.ndarray, values: np.ndarray, atol=None) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    # print("true values", true_values)
    # print("values", values)
    # print("values sum", values.sum())
    # print("values stddev", values.std())
    # print("true values stddev", true_values.std())
    # if values.sum() > 0 and :
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(true_values))

    # this does not change anything about the std dev
    # values += 1.0
    # true_values += 1.0

    if values.std() != 0 and true_values.std() != 0:
        return np.corrcoef(true_values, values)[0][1]
    elif atol is not None and np.allclose(
        np.amin([values, true_values], axis=0),
        np.amax([values, true_values], axis=0),
        atol=atol,
    ):
        return 1.0
    else:
        assert len(values) == len(true_values)
        assert len(np.amin([values, true_values], axis=0)) == len(values)
        a = np.amin([values, true_values], axis=0)
        b = np.amax([values, true_values], axis=0)
        print(a, b)
        print(np.abs(a - b))
        return np.nan


class TargetDataframes(typing.NamedTuple):
    native_df: pd.DataFrame
    accelsim_df: pd.DataFrame
    serial_gpucachesim_df: pd.DataFrame
    serial_gpucachesim_mem_only_df: pd.DataFrame
    serial_gpucachesim_exec_driven_df: pd.DataFrame
    parallel_gpucachesim_df: pd.DataFrame


def aggregate_mean_input_config_stats(
    df: pd.DataFrame,
    per_kernel=True,
    mean=True,
    inspect=False,
) -> typing.Tuple[pd.DataFrame, typing.List[str]]:
    bench_input_cols = copy.deepcopy(list(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    input_cols = copy.deepcopy(benchmarks.SIMULATE_INPUT_COLS)
    input_config_group_cols = list(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + input_cols
        + bench_input_cols
        + ["input_id"]
    )
    input_config_group_cols = [col for col in input_config_group_cols if col in df]

    preview_cols = ["target", "benchmark", "input_id", "run", "kernel_launch_id", "kernel_name", "kernel_name_mangled"]
    preview_cols += ["exec_time_sec"]
    preview_cols += ["cycles"]
    # print(df.loc[:,preview_cols][:])

    if not per_kernel:
        # sum metrics for all kernels per input_id and run
        group_cols = input_config_group_cols + ["run"]

        # TODO:
        # how do we deal with NOT summing cycles for <nan> kernel, while
        # we want to sum the execution time
        aggregations = {
            **{c: "sum" for c in sorted(df.columns)},
            **{c: "mean" for c in benchmarks.RATE_COLUMNS},
            # **{c: "first" for c in bench_input_cols + input_cols},
            **benchmarks.NON_NUMERIC_COLS,
        }
        aggregations = {
            col: agg
            for col, agg in aggregations.items()
            if col in df and not col in group_cols
        }

        grouped = df.groupby(group_cols, dropna=False)

        def _inspect_per_config(df):
            print("\nINSPECT: metrics (per input config, PER RUN)")
            print(df.loc[:,preview_cols][:10])
            pass

        if inspect:
            grouped[df.columns].apply(_inspect_per_config)
        df = grouped.agg(aggregations).reset_index()

        # we no longer have kernels now
        df["kernel_launch_id"] = np.nan
        df["kernel_name"] = np.nan
        df["kernel_name_mangled"] = np.nan

    # compute mean per input_id and kernel launch id over all runs
    group_cols = input_config_group_cols + ["kernel_launch_id", "kernel_name"]

    if mean:
        aggregations = {
            **{c: "mean" for c in sorted(df.columns)},
            **{c: "first" for c in bench_input_cols + input_cols},
            **benchmarks.NON_NUMERIC_COLS,
        }
        aggregations = {
            col: agg
            for col, agg in aggregations.items()
            if col in df and not col in group_cols
        }
        grouped = df.groupby(group_cols, dropna=False)

        def _inspect_per_config_per_kernel(df):
            print("\nINSPECT: metrics (per input config, PER KERNEL)")
            print(df.loc[:,preview_cols][:10])
            pass

        if inspect:
            grouped[df.columns].apply(_inspect_per_config_per_kernel)
        df = grouped.agg(aggregations).reset_index()

    return df.copy(), group_cols


class FunctionalConfig(typing.TypedDict):
    num_clusters: int
    cores_per_cluster: int


def split_into_target_dfs(
    df,
    per_kernel=False,
    mean=False,
    functional_config: typing.Optional[FunctionalConfig] = None,
) -> TargetDataframes:
    df = df.reset_index()

    baseline_cores_per_cluster = common.BASELINE["cores_per_cluster"]
    baseline_num_clusters = common.BASELINE["num_clusters"]
    functional_config = FunctionalConfig(
        cores_per_cluster=baseline_cores_per_cluster,
        num_clusters=baseline_num_clusters,
    )

    def _label(label, shape):
        return "{:>50}\t{}".format(label, shape)

    # native
    native_mask = df["target"] == Target.Profile.value
    native_df = df[native_mask]
    native_df, _ = aggregate_mean_input_config_stats(native_df, per_kernel=per_kernel, mean=mean)
    print(_label("native", native_df.shape))

    # accelsim
    accelsim_mask = df["target"] == Target.AccelsimSimulate.value
    accelsim_df = df[accelsim_mask]
    accelsim_df, _ = aggregate_mean_input_config_stats(accelsim_df, per_kernel=per_kernel, mean=mean)
    print(_label("accelsim", accelsim_df.shape))

    # gpucachesim (serial)
    serial_gpucachesim_mask = df["target"] == Target.Simulate.value
    serial_gpucachesim_mask &= df["input_mode"].isin(["serial", np.nan])
    serial_gpucachesim_mask &= df["input_memory_only"] == False
    if functional_config is not None:
        serial_gpucachesim_mask &= (
            df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
        )
        serial_gpucachesim_mask &= (
            df["input_num_clusters"] == functional_config["num_clusters"]
        )
    serial_gpucachesim_df = df[serial_gpucachesim_mask]
    serial_gpucachesim_df, _ = aggregate_mean_input_config_stats(
        serial_gpucachesim_df, per_kernel=per_kernel, mean=mean
    )
    print(_label("serial gpucachesim", serial_gpucachesim_df.shape))

    # gpucachesim (serial, mem only)
    serial_gpucachesim_mem_only_mask = df["target"] == Target.Simulate.value
    serial_gpucachesim_mem_only_mask &= df["input_memory_only"] == True
    serial_gpucachesim_mem_only_mask &= df["input_mode"].isin(["serial", np.nan])
    if functional_config is not None:
        serial_gpucachesim_mem_only_mask &= (
            df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
        )
        serial_gpucachesim_mem_only_mask &= (
            df["input_num_clusters"] == functional_config["num_clusters"]
        )
    serial_gpucachesim_mem_only_df = df[serial_gpucachesim_mem_only_mask]
    serial_gpucachesim_mem_only_df, _ = aggregate_mean_input_config_stats(
        serial_gpucachesim_mem_only_df, per_kernel=per_kernel, mean=mean
    )
    print(_label("serial gpucachesim (mem only)", serial_gpucachesim_mem_only_df.shape))

    # gpucachesim (serial, exec-driven)
    serial_gpucachesim_exec_driven_mask = (
        df["target"] == Target.ExecDrivenSimulate.value
    )
    # print("mask num", sum(serial_gpucachesim_exec_driven_mask))
    # print(df.loc[serial_gpucachesim_exec_driven_mask, ["target", "input_memory_only", "input_mode"]])
    serial_gpucachesim_exec_driven_mask &= df["input_mode"].isin(["serial", "", np.nan])
    serial_gpucachesim_exec_driven_df = df[serial_gpucachesim_exec_driven_mask]
    serial_gpucachesim_exec_driven_df, _ = aggregate_mean_input_config_stats(
        serial_gpucachesim_exec_driven_df, per_kernel=per_kernel, mean=mean
    )
    print(_label("serial gpucachesim (exec driven)", serial_gpucachesim_exec_driven_df.shape))

    # gpucachesim (parallel)
    parallel_gpucachesim_mask = df["target"] == Target.Simulate.value
    parallel_gpucachesim_mask &= df["input_mode"] != "serial"
    parallel_gpucachesim_mask &= df["input_memory_only"] == False
    if functional_config is not None:
        parallel_gpucachesim_mask &= (
            df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
        )
        parallel_gpucachesim_mask &= (
            df["input_num_clusters"] == functional_config["num_clusters"]
        )
    parallel_gpucachesim_df = df[parallel_gpucachesim_mask]
    parallel_gpucachesim_df, _ = aggregate_mean_input_config_stats(
        parallel_gpucachesim_df, per_kernel=per_kernel, mean=mean
    )
    print(_label("parallel gpucachesim", parallel_gpucachesim_df.shape))

    return TargetDataframes(
        native_df=native_df,
        accelsim_df=accelsim_df,
        serial_gpucachesim_df=serial_gpucachesim_df,
        serial_gpucachesim_mem_only_df=serial_gpucachesim_mem_only_df,
        serial_gpucachesim_exec_driven_df=serial_gpucachesim_exec_driven_df,
        parallel_gpucachesim_df=parallel_gpucachesim_df,
    )


def choose_fastest_parallel_implementation(df) -> pd.DataFrame:
    bench_input_cols = copy.deepcopy(list(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    # note, we do NOT group by SIMULATE_EXECUTION_CONFIG_COLS or SIMULATE_INPUT_COLS.
    # this means we do NOT group on input_mode, input_run_ahead, or input_threads
    functinoal_input_cols = copy.deepcopy(benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS)
    input_config_group_cols = (
        ["target", "benchmark"] + functinoal_input_cols + bench_input_cols
    )
    input_config_group_cols = [col for col in input_config_group_cols if col in df]

    group_cols = input_config_group_cols + ["run"]
    min_exec_times = df.groupby(group_cols, dropna=False)["exec_time_sec"].transform(
        "min"
    )
    df = df[df["exec_time_sec"] == min_exec_times]
    return df


@main.command()
# @click.pass_context
@click.option("-p", "--path", help="Path to materialized benchmark config")
@click.option("-b", "--bench", "bench_name_arg", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option("--mean-time", "include_mean_time", type=bool, is_flag=True, help="include mean time")
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
def speed_table(bench_name_arg, path, nsight, verbose, include_mean_time):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)

    # remove non-kernel results
    no_kernel_mask = selected_df["kernel_name"].isna()
    selected_df = selected_df[~no_kernel_mask]

    # print(selected_df.loc[
    #     (selected_df["target"] == Target.Simulate.value)
    #         & (selected_df["input_id"] == 210),
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    # print(selected_df.loc[
    #     (selected_df["target"] == Target.AccelsimSimulate.value)
    #         & (selected_df["input_id"] == 3),
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    target_dfs = split_into_target_dfs(
            selected_df, per_kernel=False, mean=True)

    # print(target_dfs.serial_gpucachesim_df.loc[
    #     target_dfs.serial_gpucachesim_df["input_id"] == 210,
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    # print(target_dfs.accelsim_df.loc[
    #     target_dfs.accelsim_df["input_id"] == 3,
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    native_df = target_dfs.native_df
    accelsim_df = target_dfs.accelsim_df
    serial_gpucachesim_df = target_dfs.serial_gpucachesim_df
    serial_gpucachesim_mem_only_df = target_dfs.serial_gpucachesim_mem_only_df
    serial_gpucachesim_exec_driven_df = target_dfs.serial_gpucachesim_exec_driven_df
    parallel_gpucachesim_df = choose_fastest_parallel_implementation(
        target_dfs.parallel_gpucachesim_df
    )
    print("{:>50}\t{}".format("fastest parallel gpucachesim", parallel_gpucachesim_df.shape))

    benches = sorted(selected_df["benchmark"].unique().tolist())

    # dtypes = {
    #     **{col: "float64" for col in native_df.columns},
    #     **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    # }
    # dtypes = {col: dtype for col, dtype in dtypes.items() if col in native_df}
    # native_df = native_df.astype(dtypes)

    dtypes = dict()
    sim_targets = {
        "_accelsim": accelsim_df.astype(dtypes),
        "_gpucachesim": serial_gpucachesim_df.astype(dtypes),
        "_gpucachesim_mem_only": serial_gpucachesim_mem_only_df.astype(dtypes),
        "_gpucachesim_exec_driven": serial_gpucachesim_exec_driven_df.astype(dtypes),
        "_gpucachesim_parallel": parallel_gpucachesim_df.astype(dtypes),
    }

    print("\n")

    for suffix, sim_df in sim_targets.items():
        print("computing =>", suffix)
        # print(sim_df[benchmarks.PREVIEW_COLS][:4].T)
        join_cols = list(
            # we do NOT join based on target
            ["benchmark", "kernel_launch_id"]
            + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
            # we do NOT join based on input_memory_only
            + ["input_num_clusters", "input_cores_per_cluster"],
        )
        join_cols = [col for col in join_cols if col in selected_df]
        # pprint(join_cols)

        missing_df = (
            native_df[join_cols]
            .merge(
                sim_df[join_cols],
                how="left",
                indicator=True,
            )
            .loc[lambda x: x["_merge"] != "both"]
        )
        if len(missing_df) > 0:
            if suffix == "_gpucachesim_parallel":
                # temp: ignore for now
                pass
            elif suffix == "_gpucachesim_exec_driven":
                # we do not have an exec driven version of babelstream
                missing_exec_driven_benches = sorted(
                    missing_df["benchmark"].unique().tolist()
                )
                if missing_exec_driven_benches != ["babelstream"]:
                    print("MISSING {}".format(missing_df.shape))
                    print(missing_df)
                    raise ValueError(
                        "missing exec driven {} but should only miss babelstream".format(
                            missing_exec_driven_benches
                        )
                    )
            else:
                print("MISSING {}".format(missing_df.shape))
                print(missing_df)
                assert len(missing_df) == 0

        joined_df = native_df.merge(
            sim_df,
            on=join_cols,
            how="left",
            suffixes=(None, suffix),
        )
        assert joined_df.shape[0] == native_df.shape[0]
        if len(joined_df) == 0:
            raise ValueError("joined dataframe is empty")

        native_df = joined_df

    native_df["exec_time_nsec"] = native_df["exec_time_sec"] * 1e9
    # preview_metrics = ["cycles", "instructions", "exec_time_sec", "input_id"]
    preview_metrics = ["input_id", "kernel_name", "exec_time_sec"]
    preview_cols = ["benchmark", "exec_time_nsec"] + [
        col + suffix
        for col, suffix in itertools.product(
            preview_metrics, [""] + list(sim_targets.keys())
        )
    ]

    table = ""
    for bench in benches + [None]:
        print(bench)
        if bench is None:
            bench_name = "Combined"
        else:
            match bench.lower():
                case "vectoradd":
                    bench_name = "VectorAdd"
                case "matrixmul":
                    bench_name = "Matrixmul"
                case "simple_matrixmul":
                    bench_name = "Naive Matrixmul"
                case "transpose":
                    bench_name = "Transpose"
                case "babelstream":
                    bench_name = "BabelStream"
                case other:
                    bench_name = str(other)

        table += r"\rowcolor{gray!10}"
        table += r"\multicolumn{6}{c}{\textbf{" + bench_name + r"}} \\"
        if bench is None:
            table += r"\hline \hline"
        else:
            table += r"\hline"
        table += "\n"

        # for metric in metrics:
        if bench is not None:
            bench_df = native_df[native_df["benchmark"] == bench]
        else:
            bench_df = native_df

        bench_df = bench_df.copy()
        if verbose:
            print(bench_df[preview_cols + benchmarks.BENCHMARK_INPUT_COLS[bench]])
            print(bench_df.shape)

        table += r"Slowdown"
        slowdowns_over_native = np.array(
            [
                slowdown(
                    baseline=bench_df["exec_time_sec"],
                    values=bench_df["exec_time_sec_accelsim"],
                ),
                slowdown(
                    baseline=bench_df["exec_time_sec"],
                    values=bench_df["exec_time_sec_gpucachesim"],
                ),
                slowdown(
                    baseline=bench_df["exec_time_sec"],
                    values=bench_df["exec_time_sec_gpucachesim_mem_only"],
                ),
                slowdown(
                    baseline=bench_df["exec_time_sec"],
                    values=bench_df["exec_time_sec_gpucachesim_exec_driven"],
                ),
                slowdown(
                    baseline=bench_df["exec_time_sec"],
                    values=bench_df["exec_time_sec_gpucachesim_parallel"],
                ),
            ]
        )
        if bench is None:
            slowdowns_over_native = np.nanmean(slowdowns_over_native, axis=1)
        else:
            slowdowns_over_native = np.mean(slowdowns_over_native, axis=1)
        for slowdown_value in slowdowns_over_native:
            table += " & "
            if np.isnan(slowdown_value):
                continue
            bold = np.isfinite(slowdown_value) and slowdown_value == np.nanmin(
                slowdowns_over_native
            )
            if bold:
                table += r"\boldmath"
            table += "${}$".format(plot.human_format_thousands(slowdown_value))
        table += r"\\" + "\n"

        table += r"KIPS"
        native_kilo_instructions = bench_df["instructions"] / 1000.0
        kips = np.array(
            [
                native_kilo_instructions / bench_df["exec_time_sec_accelsim"],
                native_kilo_instructions / bench_df["exec_time_sec_gpucachesim"],
                (bench_df["instructions_gpucachesim_mem_only"] / 1000.0)
                / bench_df["exec_time_sec_gpucachesim_mem_only"],
                (bench_df["instructions_gpucachesim_exec_driven"] / 1000.0)
                / bench_df["exec_time_sec_gpucachesim_exec_driven"],
                native_kilo_instructions
                / bench_df["exec_time_sec_gpucachesim_parallel"],
            ]
        )

        # print("kips:")
        # print(kips)
        if bench is None:
            kips = np.nanmean(kips, axis=1)
        else:
            kips = np.mean(kips, axis=1)
        for kips_value in kips:
            table += " & "
            if np.isnan(kips_value):
                continue
            bold = np.isfinite(kips_value) and kips_value == np.nanmax(kips)
            if bold:
                table += r"\boldmath"
            table += "${}$".format(plot.human_format_thousands(kips_value))

        if include_mean_time:
            table += r"\\" + "\n"
            table += r"mean time"
            mean_time = np.array(
                [
                    bench_df["exec_time_sec_accelsim"],
                    bench_df["exec_time_sec_gpucachesim"],
                    bench_df["exec_time_sec_gpucachesim_mem_only"],
                    bench_df["exec_time_sec_gpucachesim_exec_driven"],
                    bench_df["exec_time_sec_gpucachesim_parallel"],
                ]
            )
            if bench is None:
                mean_time = np.nanmean(mean_time, axis=1)
            else:
                mean_time = np.mean(mean_time, axis=1)
            for mean_time_value in mean_time:
                table += " & "
                if np.isnan(mean_time_value):
                    continue
                bold = np.isfinite(mean_time_value) and mean_time_value == np.nanmin(
                    mean_time
                )
                if bold:
                    table += r"\boldmath"
                table += "${:5.1f}s$".format(mean_time_value)
            # table += r"\\" + "\n"

        table += r"\\"
        # if bench is not None:
        table += r" \hline"
        table += "\n"
        table += "% \n"

    table += "%\n%\n"

    print(table)
    utils.copy_to_clipboard(table)
    print("copied table to clipboard")


class ErrorMetric(enum.Enum):
    MAPE = "MAPE"
    SMAPE = "SMAPE"
    MAE = "MAE"
    Correlation = "Corr."
    EMALE = "EMALE"
    ERMSLE = "ERMSLE"
    # RelErr = "Rel err."

    # MAPE = ("mape", "MAPE")
    # Correlation = ("corr", "Corr.")
    # RelErr = ("rel_err", "Rel err.")


# from collections import namedtuple

# ErrorMetric = namedtuple('ErrorMetric', ['value', 'label', 'column'])

# class ErrorMetrics(enum.Enum):
#
#     @property
#     def column(self):
#         return self.value.column
#
#     yellow = ErrorMetric(1, 'Yellow')
#     green = Color(2, 'Green')


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name_arg", help="Benchmark name")
@click.option("--metric", "metric_arg", type=str, help="metric")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option(
    "-v", "--vebose", "verbose", type=bool, is_flag=True, help="enable verbose output"
)
def result_table(path, bench_name_arg, metric_arg, nsight, verbose):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)

    # remove non-kernel results
    selected_df = selected_df[~selected_df["kernel_name"].isna()]

    # target benchmark histogram 
    target_bench_input_count_hist = selected_df[["target", "benchmark", "input_id"]].drop_duplicates().value_counts(["target", "benchmark"], dropna=False).sort_index()
    print(target_bench_input_count_hist)

    target_dfs = split_into_target_dfs(
        selected_df, per_kernel=False, mean=True)
    native_df = target_dfs.native_df
    accelsim_df = target_dfs.accelsim_df
    serial_gpucachesim_df = target_dfs.serial_gpucachesim_df
    serial_gpucachesim_mem_only_df = target_dfs.serial_gpucachesim_mem_only_df
    serial_gpucachesim_exec_driven_df = target_dfs.serial_gpucachesim_exec_driven_df

    
    class Metric(typing.TypedDict):
        label: str
        is_percent: bool
        error_metrics: typing.Sequence[typing.Tuple[str, ErrorMetric]]

    benches = sorted(selected_df["benchmark"].unique().tolist())
    all_metrics = [
        Metric(
            label="Cycles",
            is_percent=False,
            error_metrics=[
                # ("cycles", ErrorMetric.RelErr),
                ("cycles", ErrorMetric.EMALE),
                ("cycles", ErrorMetric.ERMSLE),
                ("cycles", ErrorMetric.SMAPE),
                ("cycles", ErrorMetric.MAPE),
                ("cycles", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="DRAM reads",
            is_percent=False,
            error_metrics=[
                ("dram_reads", ErrorMetric.EMALE),
                ("dram_reads_percent", ErrorMetric.MAPE),
                ("dram_reads", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="DRAM writes",
            is_percent=False,
            error_metrics=[
                ("dram_writes", ErrorMetric.EMALE),
                ("dram_writes_percent", ErrorMetric.MAPE),
                ("dram_writes", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L1 Accesses",
            is_percent=False,
            error_metrics=[
                ("l1_accesses", ErrorMetric.EMALE),
                ("l1_accesses", ErrorMetric.MAPE),
                ("l1_accesses", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L2 Accesses",
            is_percent=False,
            error_metrics=[
                ("l2_accesses", ErrorMetric.EMALE),
                ("l2_accesses", ErrorMetric.MAPE),
                ("l2_accesses", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L2 reads",
            is_percent=False,
            error_metrics=[
                ("l2_reads", ErrorMetric.EMALE),
                ("l2_reads", ErrorMetric.MAPE),
                ("l2_reads", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L2 writes",
            is_percent=False,
            error_metrics=[
                ("l2_writes", ErrorMetric.EMALE),
                ("l2_writes", ErrorMetric.MAPE),
                ("l2_writes", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L1D hitrate",
            is_percent=True,
            error_metrics=[
                ("l1_global_hit_rate", ErrorMetric.EMALE),
                ("l1_global_hit_rate", ErrorMetric.MAE),
                ("l1_global_hit_rate", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L2D hitrate",
            is_percent=True,
            error_metrics=[
                ("l2_hit_rate", ErrorMetric.EMALE),
                ("l2_hit_rate", ErrorMetric.MAE),
                ("l2_hit_rate", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L2D read hitrate",
            is_percent=True,
            error_metrics=[
                ("l2_read_hit_rate", ErrorMetric.EMALE),
                ("l2_read_hit_rate", ErrorMetric.MAE),
                ("l2_read_hit_rate", ErrorMetric.Correlation),
            ],
        ),
        Metric(
            label="L2D write hitrate",
            is_percent=True,
            error_metrics=[
                ("l2_write_hit_rate", ErrorMetric.EMALE),
                ("l2_write_hit_rate", ErrorMetric.MAE),
                ("l2_write_hit_rate", ErrorMetric.Correlation),
            ],
        )
    ]

    # metrics = [dram_reads]
    # metrics = [dram_writes]
    # metrics = [l1_accesses]
    # metrics = [l2_accesses]
    # metrics = [
    #     l1_hit_rate,
    #     l2_hit_rate,
    #     l1_accesses,
    #     l2_accesses,
    #     cycles,
    #     dram_reads,
    #     dram_writes,
    # ]
    # metrics = [l1_hit_rate]
    # metrics = [l2_hit_rate]
    # metrics = [cycles]

    
    if metric_arg is None:
        metrics = all_metrics[:1]
    else:
        metrics = [m for m in all_metrics if metric_arg.replace(" ", "").lower() == m["label"].replace(" ", "").lower()]
        if len(metrics) == 0:
            raise ValueError("no metric named {}, have {}", metric_arg, [m["label"].replace(" ", "").lower() for m in all_metrics])

    print("\n")
    print("computing {} metrics: {} for {} benches: {}".format(len(metrics), [m["label"] for m in metrics], len(benches), benches))

    # dtypes = {
    #     **{col: "float64" for col in native_df.columns},
    #     **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    # }
    # dtypes = {col: dtype for col, dtype in dtypes.items() if col in native_df}
    # native_df = native_df.astype(dtypes)

    dtypes = dict()
    sim_targets = {
        "_accelsim": accelsim_df.astype(dtypes),
        "_gpucachesim": serial_gpucachesim_df.astype(dtypes),
        "_gpucachesim_mem_only": serial_gpucachesim_mem_only_df.astype(dtypes),
        "_gpucachesim_exec_driven": serial_gpucachesim_exec_driven_df.astype(dtypes),
    }

    for suffix, sim_df in sim_targets.items():
        print("computing =>", suffix)
        # print(sim_df[benchmarks.PREVIEW_COLS][:4].T)
        join_cols = list(
            # we do NOT join based on target
            ["benchmark", "kernel_launch_id"]
            + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
            # we do NOT join based on input_memory_only
            + ["input_num_clusters", "input_cores_per_cluster"],
        )
        join_cols = [col for col in join_cols if col in selected_df]
        # pprint(join_cols)

        missing_df = (
            native_df[join_cols]
            .merge(
                sim_df[join_cols],
                how="left",
                indicator=True,
            )
            .loc[lambda x: x["_merge"] != "both"]
        )
        if len(missing_df) > 0:
            if suffix == "_gpucachesim_parallel":
                # temp: ignore for now
                pass
            elif suffix == "_gpucachesim_exec_driven":
                # we do not have an exec driven version of babelstream
                missing_exec_driven_benches = sorted(
                    missing_df["benchmark"].unique().tolist()
                )
                if missing_exec_driven_benches != ["babelstream"]:
                    print("MISSING {}".format(missing_df.shape))
                    print(missing_df)
                    raise ValueError(
                        "missing exec driven {} but should only miss babelstream".format(
                            missing_exec_driven_benches
                        )
                    )
            else:
                print("MISSING {}".format(missing_df.shape))
                print(missing_df)
                assert len(missing_df) == 0

        joined_df = native_df.merge(
            sim_df,
            on=join_cols,
            how="left",
            suffixes=(None, suffix),
        )
        assert joined_df.shape[0] == native_df.shape[0]
        if len(joined_df) == 0:
            raise ValueError("joined dataframe is empty")

        native_df = joined_df
        # break

    for suffix in list(sim_targets.keys()) + [""]:
        native_df["dram_reads_percent" + suffix] = native_df[
            "dram_reads" + suffix
        ].fillna(0.0)
        scale = (
            native_df[["num_global_loads", "num_global_stores"]].max(axis=1) + 0.00001
        )
        native_df["dram_reads_percent" + suffix] /= scale
        native_df["dram_writes_percent" + suffix] = native_df[
            "dram_writes" + suffix
        ].fillna(0.0)
        native_df["dram_writes_percent" + suffix] /= scale
        assert (native_df["dram_writes_percent" + suffix] <= 1.0).all()
        assert (native_df["dram_reads_percent" + suffix] <= 1.0).all()

    assert all(
        [
            col in native_df
            for col, _ in utils.flatten([m["error_metrics"] for m in metrics])
        ]
    )

    # preview_cols = [
    #     "benchmark",
    #     "input_id",
    #     "num_global_loads",
    #     "num_global_stores",
    # ] + [
    #     col + suffix
    #     for col, suffix in itertools.product(
    #         # ["cycles"],
    #         # ["dram_writes", "dram_writes_percent"],
    #         # ["dram_reads", "dram_reads_percent"],
    #         ["l1_accesses"],
    #         # [""] + list(sim_targets.keys())
    #         ["", "_accelsim", "_gpucachesim"],
    #     )
    # ]
    # print(native_df[preview_cols])

    for metric in metrics:
        metric_cols = sorted(list(set([metric_col for metric_col, _ in metric["error_metrics"]])))
        print("==> PREVIEW: {}".format(metric_cols))
        preview_cols = [
            "benchmark",
            "input_id",
            # "num_global_loads",
            # "num_global_stores",
        ] + [
            col + suffix
            for col, suffix in itertools.product(
                metric_cols,
                [""] + list(sim_targets.keys()),
                # ["", "_accelsim", "_gpucachesim"],
            )
        ]
        print(native_df[preview_cols])

    table = ""
    for bench in benches + [None]:
        if bench is None:
            bench_name = "Combined"
        else:
            match bench.lower():
                case "vectoradd":
                    bench_name = "VectorAdd"
                case "matrixmul":
                    bench_name = "Matrixmul"
                case "simple_matrixmul":
                    bench_name = "Naive Matrixmul"
                case "transpose":
                    bench_name = "Transpose"
                case "babelstream":
                    bench_name = "BabelStream"
                case other:
                    bench_name = str(other)

        table += r"\rowcolor{gray!10}"
        table += r"\multicolumn{6}{c}{\textbf{" + bench_name + r"}} \\"
        if bench is None:
            table += r"\hline \hline"
        else:
            table += r"\hline"
        table += "\n"


        
        for metric in metrics:
            print(bench, metric["label"])

            if bench is not None:
                bench_df = native_df[native_df["benchmark"] == bench]
            else:
                bench_df = native_df
                # continue

            table += r"\multirow{" + str(len(metric["error_metrics"])) + "}{*}{"
            table += " ".join(str(metric["label"]).split("_"))
            table += "} \n"

            for metric_col, error_metric in metric["error_metrics"]:
                preview_cols = ["benchmark"] + [
                    col + suffix
                    for col, suffix in itertools.product(
                        [metric_col], [""] + list(sim_targets.keys())
                    )
                ]

                bench_df = bench_df.copy()
                if bench is not None and verbose:
                    print(
                        bench_df[
                            preview_cols + benchmarks.BENCHMARK_INPUT_COLS[bench]
                        ].fillna(0.0)
                    )
                    print(bench_df.shape)

                error_values: pd.DataFrame

                metric_is_percent = metric["is_percent"]
                value_scale = 100.0 if metric_is_percent else 1.0

                match error_metric:
                    case ErrorMetric.Correlation:
                        error_values = []
                        for suffix in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + suffix] * value_scale
                            atol = 1.0 if metric_is_percent else 0.1
                            error = correlation(
                                true_values=true_values, values=values, atol=atol
                            )
                            bench_df[
                                metric_col + "_" + error_metric.name.lower() + suffix
                            ] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    # case ErrorMetric.RelErr:
                    #     error_values = []
                    #     for suffix in sim_targets.keys():
                    #         true_values=bench_df[metric_col]
                    #         values=bench_df[metric_col + suffix]
                    #         error = rel_err(true_values=true_values, values=values)
                    #         bench_df[metric_col + "_" + error_metric.name.lower() + suffix] = error
                    #         error_values.append(error)
                    #     error_values = pd.DataFrame(error_values)
                    #     error_values = error_values.mean(axis=1)
                    #     # error_values *= 100.0

                    case ErrorMetric.EMALE:
                        error_values = []
                        for suffix in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + suffix] * value_scale
                            error = emale(true_values=true_values, values=values)
                            bench_df[
                                metric_col + "_" + error_metric.name.lower() + suffix
                            ] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.ERMSLE:
                        error_values = []
                        for suffix in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + suffix] * value_scale
                            error = ermsle(true_values=true_values, values=values)
                            bench_df[
                                metric_col + "_" + error_metric.name.lower() + suffix
                            ] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.MAE:
                        error_values = []
                        for suffix in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + suffix] * value_scale
                            error = abs_err(true_values=true_values, values=values)
                            bench_df[
                                metric_col + "_" + error_metric.name.lower() + suffix
                            ] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.SMAPE:
                        error_values = []
                        for suffix in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + suffix] * value_scale
                            error = smape(true_values=true_values, values=values)
                            bench_df[
                                metric_col + "_" + error_metric.name.lower() + suffix
                            ] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values *= 100.0
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.MAPE:
                        error_values = []
                        for suffix in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + suffix] * value_scale
                            error = mape(true_values=true_values, values=values)
                            bench_df[
                                metric_col + "_" + error_metric.name.lower() + suffix
                            ] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values *= 100.0
                        error_values = error_values.mean(axis=1)
                        # error_values = error_values.aggregate(scipy.stats.gmean, axis=1)
                        # .apply(np.exp)
                        # error_values = pd.DataFrame([
                        #     abs_err(
                        #         true_values=bench_df[metric_col],
                        #         values=bench_df[metric_col + suffix]
                        #     ) for suffix in sim_targets.keys()
                        # ])
                        # keys = [
                        #     metric_col + "_" + error_metric.name.lower() + suffix
                        #     for suffix in sim_targets.keys()
                        # ]
                        # # print(keys)
                        # print(error_values.shape)
                        # bench_df[keys] = error_values.to_numpy().ravel()
                        # error_values = error_values.mean(axis=1)
                    case _:
                        raise ValueError(
                            "unknown error metric {}".format(error_metric.name)
                        )

                # assert isinstance(error_values, (np.ndarray, pd.Series))
                for col, suffix in enumerate(sim_targets.keys()):
                    valid = not np.isnan(bench_df[metric_col + suffix]).all()
                    if not valid:
                        error_values[col] = np.nan

                table += r" & {} ".format(error_metric.value)
                print(error_metric.name)
                print(error_values)
                for value in error_values:
                    table += " & "
                    if np.isnan(value):
                        continue
                    match error_metric:
                        case ErrorMetric.Correlation:
                            if value == np.nanmax(error_values):
                                table += r"\boldmath"
                            table += "${:5.3f}$".format(value)
                        # case ErrorMetric.RelErr:
                        #     if value == np.nanmin(error_values):
                        #         table += r"\boldmath"
                        #     table += "${:5.2f}\\%$".format(value)
                        # case ErrorMetric.MALE:
                        #     if value == np.nanmin(error_values):
                        #         table += r"\boldmath"
                        #     table += "${}\\%$".format(
                        #         plot.human_format_thousands(value)
                        #     )
                        # case ErrorMetric.SMAPE:
                        #     if value == np.nanmin(error_values):
                        #         table += r"\boldmath"
                        #     table += "${}\\%$".format(
                        #         plot.human_format_thousands(value)
                        #     )
                        case ErrorMetric.SMAPE | ErrorMetric.MAPE:
                            if value == np.nanmin(error_values):
                                table += r"\boldmath"
                            table += "${}\\%$".format(
                                plot.human_format_thousands(value)
                            )
                        case ErrorMetric.EMALE | ErrorMetric.ERMSLE | ErrorMetric.MAE:
                            if value == np.nanmin(error_values):
                                table += r"\boldmath"
                            if metric_is_percent:
                                table += "${:5.2f}\\%$".format(value)
                            else:
                                table += "${}$".format(
                                    plot.human_format_thousands(value)
                                )

                table += r"\\" + "\n"

                # if not accelsim_valid:
                #     metric_row[0] = np.nan
                # if not gpucachesim_valid:
                #     metric_row[1] = np.nan
                # if not gpucachesim_mem_only_valid:
                #     metric_row[2] = np.nan
                # if not gpucachesim_exec_valid:
                #     metric_row[3] = np.nan

                if bench is not None and verbose:
                    print(
                        bench_df[
                            # + [sim + "_rel_err" for sim in ["accelsim", "gpucachesim"]]
                            # + [sim + "_rmse" for sim in ["accelsim", "gpucachesim"]]
                            preview_cols
                            + [
                                metric_col + "_" + error_metric.name.lower() + suffix
                                for suffix in ["_accelsim", "_gpucachesim"]
                            ]
                            # + [sim + "_rpd" for sim in ["accelsim", "gpucachesim"]]
                        ].fillna(0.0)
                    )

            # if bench is not None:
            table += r" \hline"
            table += "\n"

        table += "%\n%\n"

    print(table)
    utils.copy_to_clipboard(table)
    print("copied table to clipboard")


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name_arg", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
def parallel_table(bench_name_arg, path, nsight):
    profiler = "nsight" if nsight else "nvprof"
    all_benchmarks = bench_name_arg is None
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)

    print(selected_df[["target", "run"]].drop_duplicates())

    # only keep simulation and remove non kernel stats
    selected_df = selected_df[selected_df["target"] == Target.Simulate.value]
    selected_df = selected_df[~selected_df["kernel_name"].isna()]
    # selected_df = sum_per_config_kernel_metrics(selected_df)
    selected_df, _ = aggregate_mean_input_config_stats(selected_df, per_kernel=False, mean=False)

    num_benchmarks = len(selected_df["benchmark"].unique().tolist())

    all_input_cols = copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS)
    all_input_cols = sorted(list([col for col in all_input_cols if col in selected_df]))

    # bench_cols = copy.deepcopy(benchmarks.BENCH_TARGET_INDEX_COLS)
    bench_input_cols = (
        []
        if all_benchmarks
        else copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[bench_name_arg])
    )
    # bench_input_cols = (
    #     list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS) - set(["input_mode"]))
    #     if all_benchmarks else copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[bench_name_arg])
    # )

    # get serial
    serial = selected_df[selected_df["input_mode"] == "serial"].copy()

    metric_cols = set(serial.columns)
    metric_cols -= set([c for c in serial.columns if c.startswith("input_")])
    metric_cols -= set(benchmarks.NON_NUMERIC_COLS)
    metric_cols -= set(["exec_time_sec", "run"])
    metric_cols = list(metric_cols)
    # pprint(metric_cols)
    # print(serial.loc[
    #     serial["input_id"] == 0,
    #     # ["cycles", "kernel_launch_id", "stream_id", "run"],
    #     ["target", "benchmark", "input_id", "kernel_name_mangled", "kernel_name", "run"]
    #     + metric_cols,
    # ].T)

    deterministic_group_cols = [
        "target",
        "benchmark",
        "input_id",
        "kernel_launch_id",
        "kernel_name_mangled",
        "kernel_name",
    ]
    metric_cols = [col for col in metric_cols if col not in deterministic_group_cols]

    def _inspect_deterministic_metrics(df):
        print(df[metric_cols].nunique().T)
        pass

    # print(serial.groupby(deterministic_group_cols, dropna=False)[metric_cols].apply(lambda df: print(df.T)))
    # print(deterministic_group_cols)
    # print(metric_cols)
    serial_deterministic_grouped = serial.groupby(deterministic_group_cols, dropna=False)
    # serial_deterministic_grouped[serial.columns].apply(_inspect_deterministic_metrics)
    unique_simulation_metrics = serial_deterministic_grouped[metric_cols].nunique()
    assert (unique_simulation_metrics == 1).all(axis=1).all()

    # parallel
    parallel = selected_df[~selected_df["input_mode"].isin([np.nan, "serial"])]
    assert "total_cores" in serial
    assert "total_cores" in parallel

    print("serial size", serial.shape)
    print("parallel size", parallel.shape)

    # those are fully distinct
    serial_input_ids = sorted(serial["input_id"].unique().tolist())
    parallel_input_ids = sorted(parallel["input_id"].unique().tolist())
    print("num serial input ids:   ", len(serial_input_ids))
    print("num parallel input ids: ", len(parallel_input_ids))
    if len(serial_input_ids) == 0:
        raise ValueError("have zero serial benchmark configurations")
    if len(parallel_input_ids) == 0:
        raise ValueError("have zero parallel benchmark configurations")

    print("serial input ids", serial_input_ids)
    print("parallel input ids", parallel_input_ids)

    deterministic = parallel[parallel["input_mode"] == "deterministic"]
    unique_simulation_metrics = deterministic.groupby(
        deterministic_group_cols, dropna=False
    )[metric_cols].nunique()
    assert (unique_simulation_metrics == 1).all(axis=1).all()

    # non deterministic without interleaving is also deterministic actually
    nondeterministic = parallel[parallel["input_mode"] == "nondeterministic"]
    unique_simulation_metrics = nondeterministic.groupby(
        deterministic_group_cols, dropna=False
    )[metric_cols].nunique()
    assert len(nondeterministic) > 0

    input_id_partitoning = set(serial["input_id"].unique()).intersection(
        set(parallel["input_id"].unique())
    )
    if len(input_id_partitoning) > 0:
        print(color("serial and parallel input ids intersect ", fg="red"))
        for input_id in input_id_partitoning:
            print("serial input", input_id)
            print(
                serial.loc[
                    serial["input_id"] == input_id,
                    benchmarks.BENCH_TARGET_INDEX_COLS
                    + ["kernel_launch_id"]
                    + bench_input_cols
                    + benchmarks.SIMULATE_INPUT_COLS,
                ]
            )
            print("parallel input", input_id)
            print(
                parallel.loc[
                    parallel["input_id"] == input_id,
                    benchmarks.BENCH_TARGET_INDEX_COLS
                    + ["kernel_launch_id"]
                    + bench_input_cols
                    + benchmarks.SIMULATE_INPUT_COLS,
                ]
            )
            break
        assert len(input_id_partitoning) == 0

    # join based on input_cols, NOT based on mode
    join_cols = list(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + ["kernel_name", "kernel_launch_id", "run"]
        + (
            list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS) - set(["input_mode"]))
            if all_benchmarks else copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[bench_name_arg])
        )
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
    )
    pprint(join_cols)

    pre_join_preview_cols = ["benchmark", "kernel_name", "kernel_launch_id", "run"]
    serial_indices = serial[pre_join_preview_cols].drop_duplicates(ignore_index=True)
    parallel_indices = parallel[pre_join_preview_cols].drop_duplicates(ignore_index=True)
    # print(serial_indices)
    # print(parallel_indices)
    diff = parallel_indices.compare(serial_indices)
    if len(diff) != 0:
        print("DIFF START")
        print(diff)
        print("DIFF END")
    assert len(diff) == 0

    joined = parallel.merge(
        serial,
        on=join_cols,
        how="left",
        suffixes=("_parallel", "_serial"),
    )
    print(
        "joined={} parallel={} serial={}".format(
            joined.shape, parallel.shape, serial.shape
        )
    )

    # test = joined["target"] == Target.Simulate.value
    # test &= joined["benchmark"] == "vectorAdd"
    # test &= joined["kernel_name"] == "vecAdd"
    # test &= joined["kernel_launch_id"] == 0
    # test &= joined["run"] == 1
    # test &= joined["input_memory_only"] == False
    # test &= joined["input_num_clusters"] == 56
    # test &= joined["input_cores_per_cluster"] == 1
    # pprint(list(joined.columns.tolist()))
    # print(joined[test])

    assert joined.shape[0] == parallel.shape[0]
    assert "mean_blocks_per_sm_parallel" in joined
    assert "total_cores_parallel" in joined
    assert "cores_per_cluster_parallel" in joined
    assert set(joined["input_id_serial"].values) == set(serial["input_id"].values)

    if len(joined) == 0:
        raise ValueError("joined parallel and serial dataframe is empty")

    preview_metric_cols = ["cycles", "exec_time_sec", "l2_hit_rate", "l1_hit_rate"]
    preview_cols = list(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + ["kernel_name", "kernel_launch_id", "run"]
        + ["input_id_serial", "input_id_parallel"]
        + bench_input_cols
        + [c + "_serial" for c in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + [c + "_parallel" for c in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + sorted(
            [c + "_serial" for c in preview_metric_cols]
            + [c + "_parallel" for c in preview_metric_cols]
        )
    )
    # print(joined[preview_cols][:4].T)

    group_cols = sorted(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + bench_input_cols
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [col + "_parallel" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + [col + "_serial" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
    )
    assert "input_id" not in group_cols
    assert "input_id_serial" not in group_cols
    aggregations = {
        **{c: "mean" for c in sorted(joined.columns)},
        **{c: agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
        **{c + "_parallel": agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
        **{c + "_serial": agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
    }
    aggregations = {
        col: agg
        for col, agg in aggregations.items()
        if col in joined and not col in group_cols
    }
    # pprint(aggregations)
    # pprint(group_cols)

    if set(joined.columns.tolist()) - set(group_cols) != set(aggregations.keys()):
        pprint(
            (set(joined.columns.tolist()) - set(group_cols)).symmetric_difference(
                set(aggregations.keys())
            )
        )
        raise ValueError

    # def add_no_kernel_exec_time(df):
    #     # print(df[preview_cols].T)
    #     assert len(df) >= 2, "have no kernel row and at least one kernel for the config"
    #     valid_kernels = ~df["kernel_name"].isna()
    #     no_kernel = df[~valid_kernels]
    #     assert len(no_kernel) == 1
    #     num_valid_kernels = valid_kernels.sum()
    #     assert num_valid_kernels >= 1
    #     serial_delta = float(no_kernel["exec_time_sec_serial"].iloc[0]) / num_valid_kernels
    #     parallel_delta = float(no_kernel["exec_time_sec_parallel"].iloc[0]) / num_valid_kernels
    #     df.loc[valid_kernels, "exec_time_sec_serial"] += serial_delta
    #     df.loc[valid_kernels, "exec_time_sec_parallel"] += parallel_delta
    #     return df
    #
    # joined = joined.groupby(
    #     group_cols + ["run"], dropna=False).apply(
    #         add_no_kernel_exec_time).reset_index(drop=True)

    # # remove non kernel stats
    # grouped = joined[~joined["kernel_name"].isna()].groupby(group_cols, dropna=False)
    grouped = joined.groupby(group_cols, dropna=False)

    # this is just for checking things
    def _inspect(df):
        # print(df)
        # print(df.columns)
        # print(df.index)
        if not all_benchmarks:
            assert len(df["input_id_serial"].unique()) == 1
        # print("num runs", len(df["run"].unique()))
        pass

    grouped[joined.columns].apply(_inspect)

    aggregated = grouped.agg(aggregations, squeeze=False)

    # speedup
    def compute_speedup(df):
        # only count speedup for large enough inputs
        exec_time_sec_serial = df["exec_time_sec_serial"]
        exec_time_sec_parallel = df["exec_time_sec_parallel"]
        exec_time_sec_parallel = df[
            ["exec_time_sec_serial", "exec_time_sec_parallel"]
        ].min(axis=1)
        return speedup(
            baseline=exec_time_sec_serial, values=exec_time_sec_parallel
        ).mean()

    aggregated["exec_time_sec_speedup"] = grouped[joined.columns].apply(compute_speedup)

    # cycles error
    aggregated["cycles_mape"] = grouped[joined.columns].apply(
        lambda df: mape(
            true_values=df["cycles_serial"], values=df["cycles_parallel"]
        ).mean()
    )
    # l1 hit rate error
    aggregated["l1_hit_rate_mae"] = grouped[joined.columns].apply(
        lambda df: abs_err(
            true_values=df["l1_hit_rate_serial"], values=df["l1_hit_rate_parallel"]
        ).mean()
    )
    # # l2 hit rate error
    aggregated["l2_hit_rate_mae"] = grouped[joined.columns].apply(
        lambda df: abs_err(
            true_values=df["l2_hit_rate_serial"], values=df["l2_hit_rate_parallel"]
        ).mean()
    )
    # dram reads error
    aggregated["dram_reads_smape"] = grouped[joined.columns].apply(
        lambda df: smape(
            true_values=df["dram_reads_serial"], values=df["dram_reads_parallel"]
        )  # .mean()
    )
    # dram writes error
    aggregated["dram_writes_smape"] = grouped[joined.columns].apply(
        lambda df: smape(
            true_values=df["dram_writes_serial"], values=df["dram_writes_parallel"]
        )  # .mean()
    )

    # print(aggregated[[
    #     "target",
    #     "benchmark",
    #     "input_variant",
    #     "dram_reads_serial",
    #     "dram_reads_parallel",
    #     "dram_reads_rel_err",
    #     "dram_writes_serial",
    #     "dram_writes_parallel",
    #     "dram_writes_rel_err",
    # ]])

    aggregated = aggregated.reset_index()
    print(
        aggregated.loc[
            # 500_000 vectoradd
            aggregated["input_id_serial"] == 210.0,
            preview_cols
            + [
                "cycles_mape",
                "dram_reads_smape",
                "dram_writes_smape",
                "exec_time_sec_speedup",
            ],
        ][0:4].T.drop_duplicates()
    )

    # build the table data
    functional_configs: typing.Sequence[typing.Dict[str, typing.Any]] = [
        dict(
            input_memory_only=False,
            input_num_clusters=common.BASELINE["num_clusters"],
            input_cores_per_cluster=1,
        ),
        dict(
            input_memory_only=False,
            input_num_clusters=common.BASELINE["num_clusters"],
            input_cores_per_cluster=8,
        ),
    ]
    selected_benchmarks: typing.Sequence[typing.Dict[str, typing.Any]] = []
    for functional_config in functional_configs:
        selected_benchmarks += [
            dict(
                name="vectorAdd",
                inputs={
                    **{"input_dtype": 32, "input_length": 500_000},
                    **functional_config,
                },
            ),
            # dict(
            #     name="transpose",
            #     inputs={
            #         **{"input_variant": "naive", "input_dim": 512},
            #         **functional_config,
            #     },
            # ),
            dict(
                name="transpose",
                inputs={
                    **{"input_variant": "coalesced", "input_dim": 512},
                    **functional_config,
                },
            ),
            dict(
                name="matrixmul",
                inputs={
                    **{"input_dtype": 32, "input_rows": 512},
                    **functional_config,
                },
            ),
            dict(
                name="simple_matrixmul",
                inputs={
                    **{
                        "input_dtype": 32,
                        "input_m": 512,
                        "input_n": 32,
                        "input_p": 512,
                    },
                    **functional_config,
                },
            ),
        ]

    def compute_table_row_label(bench_config, df):
        benchmark = df["benchmark"]
        bench_input_cols = copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[benchmark])
        assert all([c in df for c in bench_input_cols])

        assert (
            df[["total_cores_parallel"]].values == df[["total_cores_serial"]].values
        ).all()

        assert len(df[["input_cores_per_cluster"]].value_counts()) == 1
        assert len(df[["input_num_clusters"]].value_counts()) == 1
        assert len(df[["total_cores_parallel"]].value_counts()) == 1

        cores_per_cluster = int(df[["input_cores_per_cluster"]].values[0])
        num_clusters = int(df[["input_num_clusters"]].values[0])
        total_cores = num_clusters * cores_per_cluster

        assert bench_config["inputs"]["input_cores_per_cluster"] == cores_per_cluster
        assert bench_config["inputs"]["input_num_clusters"] == num_clusters
        print(
            df[
                [
                    "benchmark",
                    "input_cores_per_cluster",
                    "input_num_clusters",
                    "total_cores_parallel",
                ]
            ]
        )
        assert total_cores == int(df[["total_cores_parallel"]].values[0])

        match benchmark.lower():
            case "vectoradd":
                label = "VectorAdd (f{:<2}, {})".format(
                    int(df["input_dtype"]),
                    int(df["input_length"]),
                )
            case "matrixmul":
                label = "MatrixMul (f{:<2}, {}x{}x{})".format(
                    int(df["input_dtype"]),
                    int(df["input_rows"]),
                    int(df["input_rows"]),
                    int(df["input_rows"]),
                )
            case "simple_matrixmul":
                label = "Naive MatrixMul (f{:<2}, {}x{}x{})".format(
                    int(df["input_dtype"]),
                    int(df["input_m"]),
                    int(df["input_n"]),
                    int(df["input_p"]),
                )
            case "transpose":
                label = "Transpose ({}, {}x{})".format(
                    df["input_variant"],
                    int(df["input_dim"]),
                    int(df["input_dim"]),
                )
            case "babelstream":
                label = "BabelStream ({})".format(int(df["input_size"]))
            case other:
                label = str(other)

        label += " @ {} SM's [{:.2f} CTA/SM]".format(
            int(df["total_cores_parallel"]),
            float(df["mean_blocks_per_sm_parallel"]),
        )
        return label

    def write_table_row(row, bold_values=None):
        if bold_values is None:
            bold_values = set()

        def bold(v, formatted_v):
            if v in bold_values:
                formatted_v = formatted_v.strip()
                is_math = formatted_v[0] == "$" and formatted_v[-1] == "$"
                if is_math:
                    return r"\boldmath" + str(formatted_v)
                else:
                    return r"\textbf{" + str(formatted_v) + "}"
            return str(formatted_v)

        is_first_metric_row = row.threads == 4
        is_last_metric_row = row.threads == 8

        table_row = ""

        # metric name
        if is_first_metric_row:
            table_row += r"\multirow{2}{*}{\shortstack[r]{" + str(row.metric) + r"}}"

        # threads
        table_row += r" & $t=" + str(row.threads) + r"$ "

        # serial value
        if row.serial_value is not None and is_first_metric_row:
            table_row += (
                r" & \multirow{2}{*}{\shortstack[r]{"
                + bold(row.serial_value[0], row.serial_value[1])
                + r"}} "
            )
        else:
            table_row += r" &  "

        # deterministic value
        if row.det_value is not None:
            table_row += r" & " + bold(row.det_value[0], row.det_value[1])
        else:
            table_row += r" & "

        # nondeterministic value
        for nondet_value, formatted_nondet_value in row.nondet_values:
            table_row += r" & " + bold(nondet_value, formatted_nondet_value)
        table_row += r" \\ "
        if is_last_metric_row:
            table_row += r" \hline "
        table_row += "\n"
        return table_row

    table = r"""
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{\textwidth}{zs|s|z|zz}
& & \multicolumn{1}{c|}{Serial} & \multicolumn{1}{c|}{Deterministic} & \multicolumn{2}{c}{Nondeterministic} \\
& & & & \multicolumn{1}{c}{$n=5$} & \multicolumn{1}{c}{$n=10$} \\ \hline
"""

    # absolute_exec_time = not all_benchmarks

    if all_benchmarks:
        for functional_config in functional_configs:
            mask_cols = list(functional_config.keys())
            mask_values = list(functional_config.values())
            mask = (aggregated[mask_cols] == mask_values).all(axis=1)

            label = "Average @ {} SM's".format(
                int(aggregated.loc[mask, "total_cores_parallel"].values[0]),
            )

            table += "%\n%\n"
            table += (
                r"\rowcolor{gray!10} \multicolumn{6}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            print("=> functional config: {}".format(functional_config))

            num_bench_configs = num_benchmarks  # todo
            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask],
                num_bench_configs=num_bench_configs,
                # all_benchmarks=True
            )

            table += "%\n%\n"

            table_rows = sorted(table_rows, key=lambda row: (row.metric, row.threads))
            for row in table_rows:
                bold_values = []
                if row.metric == r"exec\\time":
                    bold_values = [np.amax(row.values())]
                    # bold_values = [np.amin(row.values())]
                    # if absolute_exec_time:
                    #     # when exec time is absolute, take minimum
                    #     bold_values = [np.amin(row.values())]
                    # else:
                    #     # when exec time is speedup, take maximum
                    #     bold_values = [np.amax(row.values())]
                print(row.metric, bold_values, row.values())
                table += write_table_row(row, bold_values)

    else:
        for bench_config in selected_benchmarks:
            bench_inputs: typing.Dict[str, typing.Any] = bench_config["inputs"]
            if not all(aggregated["benchmark"] == bench_config["name"]):
                print(
                    color(
                        "SKIP: want {} (have {})".format(
                            bench_config["name"], aggregated["benchmark"][0]
                        ),
                        fg="red",
                    )
                )
                continue

            print("==> {}".format(bench_config["name"]))
            mask_cols = ["benchmark"] + list(bench_inputs.keys())
            mask_values = [bench_name_arg] + list(bench_inputs.values())

            mask = (aggregated[mask_cols] == mask_values).all(axis=1)
            # test_df = aggregated.loc[
            #     mask,
            #     benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
            #     + bench_input_cols
            #     + ["mean_blocks_per_sm_parallel"],
            # ]
            # test_df = test_df.drop_duplicates()
            # print(test_df)
            # assert len(test_df) == 1

            table += "%\n%\n"
            label = str(compute_table_row_label(
                bench_config, aggregated.loc[mask].iloc[0]))
            table += (
                r"\rowcolor{gray!10} \multicolumn{6}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask],
                num_bench_configs=1,  # all_benchmarks=False
            )

            table += "%\n%\n"

            table_rows = sorted(table_rows, key=lambda row: (row.metric, row.threads))
            for row in table_rows:
                bold_values = []
                if row.metric == r"exec\\time":
                    bold_values = [np.amin(row.values())]
                    # if absolute_exec_time:
                    #     bold_values = [np.amin(row.values())]
                    # else:
                    #     bold_values = [np.amax(row.values())]
                print(row.metric, bold_values, row.values())
                table += write_table_row(row, bold_values)

        # add averaged row
        for functional_config in functional_configs:
            mask_cols = list(functional_config.keys())
            mask_values = list(functional_config.values())
            mask = (aggregated[mask_cols] == mask_values).all(axis=1)

            label = "Average @ {} SM's".format(
                int(aggregated.loc[mask, "total_cores_parallel"].values[0]),
            )

            table += "%\n%\n"
            table += (
                r"\rowcolor{gray!10} \multicolumn{6}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            assert num_benchmarks == 1
            num_configs = len(aggregated.loc[mask, all_input_cols].drop_duplicates())
            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask],
                num_bench_configs=num_configs,  # all_benchmarks=True
            )
            table += "%\n%\n"

            table_rows = sorted(table_rows, key=lambda row: (row.metric, row.threads))
            for row in table_rows:
                bold_values = []
                if row.metric == r"exec\\time":
                    # if absolute_exec_time:
                    #     # when exec time is absolute, take minimum
                    #     bold_values = [np.amin(row.values())]
                    # else:
                    #     # when exec time is speedup, take maximum
                    bold_values = [np.amax(row.values())]

                print(row.metric, bold_values, row.values())
                table += write_table_row(row, bold_values)

    table += r"""
\end{tabularx}}
\end{table}
"""
    print(table)
    utils.copy_to_clipboard(table)
    print("copied table to clipboard")


def load_stats(bench_name, profiler="nvprof", path=None) -> pd.DataFrame:
    stats = []
    if bench_name is not None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
            profiler, bench_name
        )
        print("loading {}".format(stats_file))
        df = pd.read_csv(stats_file, header=0)
        if len(df) < 1:
            print(color("WARNING: {} is empty!".format(stats_file), fg="red"))
        else:
            stats.append(df)
    else:
        b = Benchmarks(path)
        benches = utils.flatten(list(b.benchmarks[Target.Profile.value].values()))
        bench_names = set([b["name"] for b in benches])
        for name in bench_names:
            stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
                profiler, name
            )
            print("loading {}".format(stats_file))
            df = pd.read_csv(stats_file, header=0)
            if len(df) < 1:
                print(color("WARNING: {} is empty!".format(stats_file), fg="red"))
            else:
                stats.append(df)

    stats_df = pd.concat(stats, ignore_index=False)
    stats_df = stats_df.sort_values(["benchmark", "target"])
    if bench_name is not None:
        if isinstance(bench_name, str):
            bench_names = [bench_name]
        elif isinstance(bench_name, list):
            bench_names = bench_name
        else:
            raise ValueError
        stats_df = stats_df[stats_df["benchmark"].isin(bench_names)]

    # special_dtypes = {
    #     # **{col: "float64" for col in stats_df.columns},
    #     # **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    #     "target": "str",
    #     "benchmark": "str",
    #     "Host Name": "str",
    #     "Process Name": "str",
    #     "device": "str",
    #     "context_id": "float",
    #     "is_release_build": "bool",
    #     "kernel_function_signature": "str",
    #     "kernel_name": "str",
    #     "kernel_name_mangled": "str",
    #     "input_id": "float",
    #     # "input_memory_only": "first",
    #     # "input_mode": "first",
    #     # makes no sense to aggregate
    #     "cores_per_cluster": "float",
    #     "num_clusters": "float",
    #     "total_cores": "float",
    #     "input_memory_only": "bool",
    #     "input_num_clusters": "float",
    #     "input_cores_per_cluster": "float",
    #     "input_mode": "str",
    #     "input_threads": "float",
    #     "input_run_ahead": "float",
    # }
    # missing_dtypes = set(benchmarks.NON_NUMERIC_COLS.keys()) - set(special_dtypes.keys())
    # assert len(missing_dtypes) == 0, "missing dtypes for {}".format(missing_dtypes)

    dtypes = {
        **{col: "float64" for col in stats_df.columns},
        **benchmarks.SPECIAL_DTYPES,
    }
    # raise ValueError("test")
    dtypes = {col: dtype for col, dtype in dtypes.items() if col in stats_df}
    stats_df = stats_df.astype(dtypes)

    simulation_targets = [
        Target.Simulate.value,
        Target.AccelsimSimulate.value,
        Target.PlaygroundSimulate.value,
    ]
    simulation_targets_df = stats_df[stats_df["target"].isin(simulation_targets)]
    if not (simulation_targets_df["is_release_build"] == True).all():
        print(color("WARNING: non release results:", fg="red"))
        non_release_results = simulation_targets_df[
            simulation_targets_df["is_release_build"] == True
        ]
        grouped = non_release_results.groupby(["benchmark", "target"])
        print(grouped["input_id"].count())
        print("====")

    non_float_cols = set([
        col for col, dtype in benchmarks.SPECIAL_DTYPES.items()
        if dtype not in ["float", "float64", "int", "int64"]
    ])
    nan_dtype = pd.NA
    fill = {
        **{col: 0.0 for col in stats_df.columns},
        **{col: nan_dtype for col in non_float_cols},
        **{
            "kernel_name_mangled": nan_dtype,
            "kernel_name": nan_dtype,
            "device": nan_dtype,
            # test this out
            "kernel_launch_id": nan_dtype,
            "run": nan_dtype,
        },
        **{c: nan_dtype for c in benchmarks.ALL_BENCHMARK_INPUT_COLS},
        **{c: nan_dtype for c in benchmarks.SIMULATE_INPUT_COLS},
        **{
            "input_memory_only": False,
            "input_num_clusters": 28,
            "input_cores_per_cluster": 1,
        },
    }
    assert pd.isnull(fill["kernel_launch_id"])
    assert pd.isnull(fill["kernel_name"])
    # fill = {
    #     col: dtype for col, dtype in fill.items()
    #     if col not in benchmarks.CATEGORICAL_COLS
    # }

    stats_df = stats_df.fillna(fill).infer_objects(copy=False)
    assert stats_df["run"].isna().sum() == 0

    def add_no_kernel_exec_time(df):
        # print(df[benchmarks.PREVIEW_COLS][:4].T)
        try:
            before = copy.deepcopy(df.dtypes)
            if df["target"].iloc[0] != Target.Simulate.value:
                return df

            assert (
                len(df) >= 2
            ), "expected at least two rows: a no kernel row and at least one kernel for the config"
            # print("df")
            # print(df)
            valid_kernels = ~df["kernel_name"].isna()
            # print("valid_kernels")
            # print(valid_kernels)
            no_kernel = df[~valid_kernels]
            # print("no kernel")
            # print(no_kernel)
            assert len(no_kernel) == 1
            num_valid_kernels = valid_kernels.sum()
            assert num_valid_kernels >= 1
            delta = float(no_kernel["exec_time_sec"].iloc[0]) / num_valid_kernels
            df.loc[valid_kernels, "exec_time_sec"] += delta
            assert (df.dtypes == before).all()
            return df
        except Exception as e:
            print(e)
            return str(e)

    group_cols = list(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
        + benchmarks.SIMULATE_INPUT_COLS
        + ["run"]
    )
    print(len(stats_df))
    group_cols = [col for col in group_cols if col in stats_df]
    # pprint(group_cols)
    # pprint(stats_df[group_cols].dtypes)
    # stats_df = stats_df.fillna({'target': "", "benchmark": "", "input_mode": ""})
    grouped = stats_df.groupby(group_cols, dropna=False)
    # grouped = grouped[stats_df.columns].fillna({'target': "", "benchmark": "", "input_mode": ""})
    # print(grouped.isna())
    # raise ValueError("grouped")
    stats_df = grouped[stats_df.columns].apply(add_no_kernel_exec_time)
    stats_df = stats_df.reset_index(drop=True)
    # raise ValueError("its over")

    assert stats_df["run"].isna().sum() == 0
    # assert stats_df["kernel_launch_id"].isna().sum() == 0
    assert stats_df["num_clusters"].isna().sum() == 0
    return stats_df


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
                    int(common.BASELINE["num_clusters"]),
                    int(common.BASELINE["cores_per_cluster"]),
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
@click.option(
    "--per-kernel", "per_kernel", type=bool, default=False, help="per kernel"
)
@click.option(
    "--inspect", "inspect", type=bool, default=False, help="inspet aggregations"
)
def view(path, bench_name_arg, should_plot, nsight, mem_only, verbose, strict, per_kernel, inspect):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)

    # gpucachesim stats include "no kernel" (e.g. memcopies) stats
    assert selected_df["kernel_name"].isna().sum() > 0

    target_bench_input_count_hist = selected_df[["target", "benchmark", "input_id"]].drop_duplicates().value_counts(["target", "benchmark"], dropna=False).sort_index()
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
        selected_df, memory_only=mem_only,
        per_kernel=per_kernel, inspect=inspect, mean=False)

    print(per_config[["target", "benchmark"] + (["input_id", "kernel_name"] if per_kernel else [])].drop_duplicates())


    all_input_cols = list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    if per_kernel:
        all_input_cols += ["kernel_launch_id", "kernel_name"]
    all_input_cols = [col for col in all_input_cols if col in selected_df]
    all_input_cols = sorted(all_input_cols)
    

    # # all_input_cols = sorted(list([col for col in all_input_cols if col in per_config]))
    #     all_input_cols = [col for col in all_input_cols if col in per_config]


    per_config_group_cols = utils.dedup([
        "target", "benchmark", "input_id", "kernel_name",
        "kernel_name_mangled", "run",
    ] + benchmarks.SIMULATE_INPUT_COLS + all_input_cols)

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

    preview_cols = ["target", "benchmark", "input_id", "run", "kernel_launch_id", "kernel_name", "kernel_name_mangled"]
    preview_cols += ["exec_time_sec"]
    preview_cols += ["cycles"]

    def _inspect(df):
        print("\nINSPECT")
        print(df.loc[:,preview_cols][:10])
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
    print(per_config_pivoted[stat_cols].T)

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
            [
                ("babelstream", 10240.0),
                ("babelstream", 102400.0),
            ] if per_kernel else [
                ("babelstream", 1024.0),
                ("babelstream", 10240.0),
                ("babelstream", 102400.0),
            ],
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
                ("simple_matrixmul", 32, 32, 32),
                ("simple_matrixmul", 128, 128, 128),
                ("simple_matrixmul", 32, 64, 128),
                ("simple_matrixmul", 128, 32, 32),
                ("simple_matrixmul", 128, 512, 128),
                ("simple_matrixmul", 512, 32, 512),
            ],
            columns=["benchmark", "input_m", "input_n", "input_p"],
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
    per_config.loc[:,"label"] = per_config.apply(partial(compute_label_for_benchmark_df, per_kernel=per_kernel), axis=1)
    per_config.loc[
        per_config["target"] == Target.Simulate.value, "target_name"
    ] = "gpucachesim"
    per_config.loc[
        per_config["target"] == Target.AccelsimSimulate.value, "target_name"
    ] = "AccelSim"
    per_config.loc[
        per_config["target"] == Target.Profile.value, "target_name"
    ] = per_config.loc[~per_config["device"].isna(), "device"].apply(
        gpucachesim.stats.native.normalize_nvprof_device_name
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
    plot_per_config = per_config.reset_index(drop=True).merge(selected_table_benchmarks, how="inner")
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
            for input_idx, (_, input_values_df) in enumerate(target_df.groupby([col for col in group_cols if col in target_df])):
            # for input_idx, (_input_id, input_values_df) in enumerate(target_df.groupby("input_id")):

                # key = (target, benchmark) + tuple(input_values.values)

                # print(input_idx, input_values)
                input_values = input_values_df[[col for col in all_input_cols if col in input_values_df]].drop_duplicates().dropna()
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
                idx = input_idx * group_width + (target_idx + 0.5) * (bar_width + spacing)

                # target = target_df["target"].first().values[0]
                # assert target == target_df["target"].first().values[0]
                assert input_values_df["target_name"].nunique() == 1
                target_name = input_values_df["target_name"].iloc[0]
                # target_name = target_df["target_name"].first().values[0]

                x = [idx]
                raw_y = input_values_df[stat_col] # .fillna(0.0)
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
                    y = raw_y.mean() #.fillna(0.0)

                ystd = raw_y.std() #.fillna(0.0)

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
        all_values_df = plot_per_config.loc[pd.IndexSlice[:,benchmark], :]
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
            plot.human_format_thousands(v, round_to=0).rjust(6, " ") for v in ytick_values
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
        fig.set_size_inches( new_width, height)

        filename = plot_dir / "{}.{}.{}_no_xticks_no_legend.pdf".format(
            profiler, benchmark, stat_col
        )
        fig.savefig(filename)

        # plot with xticks but without legend (bottom)
        ax.set_xticks(xtick_values, xtick_labels, rotation=0)
        fig.set_size_inches( new_width, height)

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

        fig.set_size_inches( new_width, height)

        filename = plot_dir / "{}.{}.{}.pdf".format(profiler, benchmark, stat_col)
        fig.savefig(filename)
        print(color("wrote {}".format(filename), fg="cyan"))

        # plot with legend but without xticks (top)
        ax.set_xticks(xtick_values, ["" for _ in range(len(xtick_values))], rotation=0)
        fig.set_size_inches( new_width, height)

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
                        int(common.BASELINE["cores_per_cluster"]),
                        None,
                    ]:
                        continue
                    if input_values.get("num_clusters") not in [
                        int(common.BASELINE["num_clusters"]),
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
                    if (target.lower(), name.lower()) == ("execdrivensimulate", "babelstream"):
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
def timings(path, bench_name_arg, baseline, strict):
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
                    in [int(common.BASELINE["num_clusters"]), None],
                    config["values"].get("cores_per_cluster")
                    in [int(common.BASELINE["cores_per_cluster"]), None],
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
            for r in range(repetitions):
                sim_df = pd.read_csv(
                    stats_dir / f"stats.sim.{r}.csv",
                    header=0,
                )
                sim_df["run"] = r
                grouped_sim = sim_df.groupby(
                    gpucachesim.stats.stats.INDEX_COLS, dropna=False
                )

                # timings_path = stats_dir / f"timings.{r}.csv"
                timings_path = stats_dir / f"timings.csv"
                # print(timings_path)
                if not strict and not timings_path.is_file():
                    continue

                assert timings_path.is_file()

                timing_df = pd.read_csv(timings_path, header=0)
                timing_df["benchmark"] = bench_config["name"]
                timing_df["input_id"] = bench_config["input_idx"]
                timing_df["target"] = bench_config["target"]
                timing_df["run"] = r
                timing_df["exec_time_sec"] = grouped_sim["elapsed_millis"].sum().sum()
                timing_df["exec_time_sec"] /= 1000.0

                timings_dfs.append(timing_df)

    timings_df = pd.concat(timings_dfs)
    timings_df["max_total"] = timings_df.groupby(
        ["target", "benchmark", "input_id", "run"]
    )["total"].transform("max")
    timings_df["exec_time_sec"] = timings_df["max_total"]
    # timings_df["exec_time_sec"] = timings_df[["max_total", "exec_time_sec"]].max(axis=1)
    timings_df["mean_sec"] = timings_df["total"] / timings_df["count"]
    timings_df["mean_millis"] = timings_df["mean_sec"] * 1000.0
    timings_df["mean_micros"] = timings_df["mean_millis"] * 1000.0
    timings_df["share"] = timings_df["total"] / timings_df["exec_time_sec"]

    print(timings_df.head(n=100))

    def stderr(df):
        return df.std() / np.sqrt(len(df))

    averaged = timings_df.groupby("name")[
        ["total", "share", "mean_micros", "mean_millis"]
    ].agg(["min", "max", "mean", "std", "sem", stderr])

    # make sure sem is correct
    all_sem = averaged.iloc[:, averaged.columns.get_level_values(1) == "sem"]
    all_sem.columns = all_sem.columns.droplevel(1)
    all_stderr = averaged.iloc[:, averaged.columns.get_level_values(1) == "stderr"]
    all_stderr.columns = all_stderr.columns.droplevel(1)
    assert ((all_sem - all_stderr).abs() > 0.001).sum().sum() == 0

    print("\n\n=== TOTAL")
    pd.options.display.float_format = "{:.2f}".format
    print(averaged["total"])
    print("\n\n=== MEAN MICROSECONS")
    pd.options.display.float_format = "{:.6f}".format
    print(averaged["mean_micros"])
    print("\n\n=== SHARE")
    pd.options.display.float_format = "{:.2f}".format
    print(averaged["share"] * 100.0)

    total = averaged["share"]["mean"].T["cycle::total"]
    summed = averaged["share"]["mean"].T[
        [
            "cycle::core",
            "cycle::dram",
            "cycle::interconn",
            "cycle::issue_block_to_core",
            "cycle::l2",
            "cycle::subpartitions",
        ]
    ]
    print(total, summed.sum())
    assert summed.sum() <= total

    # issue blocks = cycle::issue_block_to_core
    # cores = cycle::core
    # dram = cycle::dram
    # interconn = cycle::subpartitions, cycle::interconn
    # cache cycle = cycle::l2

    # timings_df["rel_err"] = timings_df["total"] / timings_df["exec_time_sec"]
    # timings_df["abs_err"] = (timings_df["total"] - timings_df["exec_time_sec"]).abs()

    valid_rel = (timings_df["total"] / timings_df["exec_time_sec"]) <= 1.2
    valid_abs = (timings_df["total"] - timings_df["exec_time_sec"]).abs() <= 0.1
    # print(timings_df[timings_df["total"] > timings_df["exec_time_sec"]])
    # print(timings_df[~(valid_rel | valid_abs)])
    assert (valid_rel | valid_abs).all()


if __name__ == "__main__":
    main()
    # main(ctx={})
