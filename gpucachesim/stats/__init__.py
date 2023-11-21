import click
import yaml
import typing
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from wasabi import color
import wasabi
import itertools
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

import gpucachesim.stats.stats
import gpucachesim.stats.native
import gpucachesim.stats.accelsim
import gpucachesim.stats.playground
import gpucachesim.stats.common as common
import gpucachesim.benchmarks as benchmarks
# from gpucachesim.stats.human import human_readable
import gpucachesim.plot as plot
import gpucachesim.utils as utils

from gpucachesim.benchmarks import (
    # SIMULATE_FUNCTIONAL_CONFIG_COLS,
    Target,
    Benchmarks,
    GPUConfig,
    REPO_ROOT_DIR,
)


# suppress scientific notation by setting float_format
# pd.options.display.float_format = "{:.3f}".format
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.set_printoptions(suppress=True, formatter={"float_kind": "{:f}".format})
np.seterr(all="raise")

DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.config.yml"


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
    cores_per_cluster=int(common.BASELINE["cores_per_cluster"]),
    num_clusters=int(common.BASELINE["num_clusters"]),
    # ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
) -> pd.DataFrame:
    """View results for a benchmark"""
    for col in benchmarks.SIMULATE_INPUT_COLS:
        if col not in selected_df:
            selected_df[col] = np.nan

    non_gpucachesim = selected_df["input_mode"].isnull()
    # print(selected_df[non_gpucachesim]["target"].unique().tolist())

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
    print(kernels)

    # only keep gold gpucachesim and other targets
    no_kernel = selected_df["kernel_name"].isna() ^ (selected_df["kernel_name"] == "")
    valid_kernel = selected_df["kernel_name"].isin(kernels)
    selected_df = selected_df[
        (gold_gpucachesim ^ non_gpucachesim) & (valid_kernel ^ no_kernel)
    ]

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

    per_config = sum_per_config_kernel_metrics(selected_df)

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
    return per_config


def sum_per_config_kernel_metrics(df):
    input_cols = benchmarks.ALL_BENCHMARK_INPUT_COLS
    input_cols = sorted(list([col for col in input_cols if col in df]))
    group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + ["input_id", "run"]
    # pprint(group_cols)
    # pprint(benchmarks.NON_NUMERIC_COLS)
    # pprint(sorted(list(set(df.columns) - set(benchmarks.NON_NUMERIC_COLS))))

    grouped = df.groupby(group_cols, dropna=False)
    aggregations = {
        **{c: "sum" for c in set(df.columns)},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {
        col: agg
        for col, agg in aggregations.items()
        if col in df and not col in group_cols
    }
    # def _inspect(_df):
    #     print(_df.head(n=100))
    #
    # grouped.apply(_inspect)
    return grouped.agg(aggregations).reset_index()


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
    num_benchmarks,
    all_benchmarks,
    thousands_round_to=1,
    variable_precision=True,
) -> typing.Sequence[ParallelTableRow]:
    interleave_n = list(itertools.product([False, True], [5, 10]))
    table_rows: typing.Sequence[ParallelTableRow] = []

    for threads in [4, 8]:
        threads_mask = df["input_threads_parallel"] == threads
        det_mask = df["input_mode_parallel"] == "deterministic"
        nondet_no_interleave_mask = df["input_mode_parallel"] == "nondeterministic"
        nondet_interleave_mask = (
            df["input_mode_parallel"] == "nondeterministic_interleave"
        )
        # print([m.sum() for m in [
        #     mask, threads_mask, det_mask, nondet_no_interleave_mask, nondet_interleave_mask
        # ]])

        det = df[threads_mask & det_mask]
        if False:
            print(
                det[
                    # bench_input_cols
                    +[
                        "input_threads_parallel",
                        "exec_time_sec_parallel",
                        "input_id_parallel",
                        "input_id_serial",
                        # "dram_reads_serial",
                        # "dram_reads_parallel",
                        # "dram_reads_rel_err",
                        "dram_writes_serial",
                        "dram_writes_parallel",
                        "dram_writes_rel_err",
                    ]
                    + different_cols(det)
                ]
            )
        print("===")
        nondet_no_interleave = df[threads_mask & nondet_no_interleave_mask]
        nondet_interleave = df[threads_mask & nondet_interleave_mask]

        assert len(det) == num_benchmarks
        assert len(nondet_no_interleave) == 2 * num_benchmarks
        assert len(nondet_interleave) == 2 * num_benchmarks
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
        serial_exec_time = df.loc[threads_mask, "exec_time_sec_serial"].values[0]
        det_exec_time = det["exec_time_sec_parallel"].values[0]
        det_speedup = det["exec_time_sec_speedup"].values[0]
        nondet_values = []
        for interleave, n in interleave_n:
            nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            assert len(nondet) == 1

            nondet_exec_time = nondet["exec_time_sec_parallel"].values[0]
            nondet_speedup = nondet["exec_time_sec_speedup"].values[0]
            if all_benchmarks:
                nondet_values.append(
                    (
                        nondet_speedup,
                        "${}x$".format(
                            plot.round_to_precision(
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
                            plot.round_to_precision(
                                nondet_speedup,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if all_benchmarks
            else (serial_exec_time, "${:>3.1f}s$".format(serial_exec_time))
        )
        if all_benchmarks:
            det_value = (
                det_speedup,
                "${}x$".format(
                    plot.round_to_precision(
                        det_speedup, round_to=1, variable_precision=variable_precision
                    )
                ),
            )
        else:
            det_value = (
                det_exec_time,
                "${:>3.1f}s~({}x)$".format(
                    det_exec_time,
                    plot.round_to_precision(
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
        serial_cycles = int(df.loc[threads_mask, "cycles_serial"].values[0])
        det_cycles = int(det["cycles_parallel"].values[0])
        det_rel_err = det["cycles_rel_err"].values[0]
        nondet_values = []
        for interleave, n in interleave_n:
            nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            assert len(nondet) == 1

            nondet_cycles = int(nondet["cycles_parallel"].values[0])
            nondet_rel_err = nondet["cycles_rel_err"].values[0]
            if all_benchmarks:
                nondet_values.append(
                    (
                        nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision(
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
                            plot.round_to_precision(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if all_benchmarks
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
        if all_benchmarks:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision(
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
                    plot.round_to_precision(
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
        serial_l1_hit_rate = df.loc[threads_mask, "l1_hit_rate_serial"].values[0]
        det_l1_hit_rate = det["l1_hit_rate_parallel"].values[0]
        det_rel_err = det["l1_hit_rate_rel_err"].values[0]
        nondet_values = []
        for interleave, n in interleave_n:
            nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            assert len(nondet) == 1

            nondet_l1_hit_rate = nondet["l1_hit_rate_parallel"].values[0]
            nondet_rel_err = nondet["l1_hit_rate_rel_err"].values[0]
            if all_benchmarks:
                nondet_values.append(
                    (
                        100.0 * nondet_rel_err,
                        "{}\\%$".format(
                            plot.round_to_precision(
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
                            plot.round_to_precision(
                                100.0 * nondet_l1_hit_rate,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if all_benchmarks
            else (
                100.0 * serial_l1_hit_rate,
                "${:>2.1f}\\%$".format(100.0 * serial_l1_hit_rate),
            )
        )
        if all_benchmarks:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision(
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
                    plot.round_to_precision(
                        100.0 * det_l1_hit_rate,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision(
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
        serial_l2_hit_rate = df.loc[threads_mask, "l2_hit_rate_serial"].values[0]
        det_l2_hit_rate = det["l2_hit_rate_parallel"].values[0]
        det_rel_err = det["l2_hit_rate_rel_err"].values[0]
        nondet_values = []
        for interleave, n in interleave_n:
            nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            assert len(nondet) == 1

            nondet_l2_hit_rate = nondet["l2_hit_rate_parallel"].values[0]
            nondet_rel_err = nondet["l2_hit_rate_rel_err"].values[0]
            if all_benchmarks:
                nondet_values.append(
                    (
                        100.0 * nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision(
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
                            plot.round_to_precision(
                                100.0 * nondet_l2_hit_rate,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                            plot.round_to_precision(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if all_benchmarks
            else (
                100.0 * serial_l2_hit_rate,
                "${}\\%$".format(
                    plot.round_to_precision(
                        100.0 * serial_l2_hit_rate,
                        round_to=1,
                        variable_precision=variable_precision,
                    )
                ),
            )
        )
        if all_benchmarks:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision(
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
                    plot.round_to_precision(
                        100.0 * det_l2_hit_rate,
                        round_to=1,
                        variable_precision=variable_precision,
                    ),
                    plot.round_to_precision(
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
        serial_dram_reads = int(df.loc[threads_mask, "dram_reads_serial"].values[0])
        det_dram_reads = int(det["dram_reads_parallel"].values[0])
        det_rel_err = det["dram_reads_rel_err"].values[0]
        nondet_values = []
        for interleave, n in interleave_n:
            nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            assert len(nondet) == 1

            nondet_dram_reads = int(nondet["dram_reads_parallel"].values[0])
            nondet_rel_err = nondet["dram_reads_rel_err"].values[0]
            if all_benchmarks:
                nondet_values.append(
                    (
                        nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision(
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
                            plot.round_to_precision(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if all_benchmarks
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
        if all_benchmarks:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision(
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
                    plot.round_to_precision(
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
        serial_dram_writes = int(df.loc[threads_mask, "dram_writes_serial"].values[0])
        det_dram_writes = int(det["dram_writes_parallel"].values[0])
        det_rel_err = det["dram_writes_rel_err"].values[0]
        nondet_values = []
        for interleave, n in interleave_n:
            nondet = nondet_interleave if interleave else nondet_no_interleave
            nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            assert len(nondet) == 1

            nondet_dram_writes = int(nondet["dram_writes_parallel"].values[0])
            nondet_rel_err = nondet["dram_writes_rel_err"].values[0]
            if all_benchmarks:
                nondet_values.append(
                    (
                        100.0 * nondet_rel_err,
                        "${}\\%$".format(
                            plot.round_to_precision(
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
                            plot.round_to_precision(
                                100.0 * nondet_rel_err,
                                round_to=1,
                                variable_precision=variable_precision,
                            ),
                        ),
                    )
                )

        serial_value = (
            None
            if all_benchmarks
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
        if all_benchmarks:
            det_value = (
                100.0 * det_rel_err,
                "${}\\%$".format(
                    plot.round_to_precision(
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
                    plot.round_to_precision(
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


def speedup(baseline, values):
    return baseline / values

def rel_err(true_values: np.ndarray, values: np.ndarray):
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    rel_err = (values - true_values).abs() / true_values
    rel_err = rel_err.fillna(0.0)
    rel_err[rel_err == 0.0] = 0.0
    return rel_err

def rmse(true_values, values) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return ((values - true_values) ** 2).mean() ** 0.5


def mae(true_values: np.ndarray, values: np.ndarray) -> pd.Series:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    return (true_values - values).abs()


def correlation(true_values: np.ndarray, values: np.ndarray) -> float:
    values = values.fillna(0.0)
    true_values = true_values.fillna(0.0)
    if values.sum() > 0:
        return np.corrcoef(true_values, values)[0][1]
    else:
        return np.nan

@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name_arg", help="Benchmark name")
@click.option("--metric", "metric", type=str, help="metric")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
def result_table(bench_name_arg, metric, path, nsight):
    profiler = "nsight" if nsight else "nvprof"
    all_benchmarks = bench_name_arg is None
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)

    bench_cols = ["target", "benchmark"]
    bench_input_cols = (
        [] if all_benchmarks else benchmarks.BENCHMARK_INPUT_COLS[bench_name_arg]
    )
    input_cols = benchmarks.SIMULATE_INPUT_COLS

    # metric_cols = ["cycles", "exec_time_sec", "l2_hit_rate", "l1_hit_rate"]
    group_cols = bench_cols + input_cols + bench_input_cols + ["kernel_launch_id", "input_id"]
    aggregations = {
        **{c: "mean" for c in sorted(selected_df.columns)},
        **{c: "first" for c in selected_df.columns if c.startswith("input_")},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {col: agg for col, agg in aggregations.items() if col in selected_df}
    aggregations = {
        col: agg for col, agg in aggregations.items() if col not in group_cols
    }

    # native
    native_mask = selected_df["target"] == Target.Profile.value
    native_df = selected_df[native_mask]
    native_df = native_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    print("native", native_df.shape)

    # gpucachesim
    accelsim_mask = selected_df["target"] == Target.AccelsimSimulate.value
    accelsim_df = selected_df[accelsim_mask]
    accelsim_df = accelsim_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    print("accelsim", accelsim_df.shape)

    # gpucachesim
    serial_gpucachesim_mask = selected_df["target"] == Target.Simulate.value
    serial_gpucachesim_mask &= selected_df["input_mode"] == "serial"
    serial_gpucachesim_mask &= selected_df["input_memory_only"] == False
    serial_gpucachesim_mask &= selected_df["input_cores_per_cluster"] == common.BASELINE["cores_per_cluster"]
    serial_gpucachesim_mask &= selected_df["input_num_clusters"] == common.BASELINE["num_clusters"]
    serial_gpucachesim_df = selected_df[serial_gpucachesim_mask]
    serial_gpucachesim_df = serial_gpucachesim_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    print("gpucachesim", serial_gpucachesim_df.shape)

    # gpucachesim (mem only)
    serial_gpucachesim_mem_only_mask = selected_df["target"] == Target.Simulate.value
    serial_gpucachesim_mem_only_mask &= selected_df["input_mode"] == "serial"
    serial_gpucachesim_mem_only_mask &= selected_df["input_memory_only"] == True
    serial_gpucachesim_mem_only_mask &= selected_df["input_cores_per_cluster"] == common.BASELINE["cores_per_cluster"]
    serial_gpucachesim_mem_only_mask &= selected_df["input_num_clusters"] == common.BASELINE["num_clusters"]
    serial_gpucachesim_mem_only_df = selected_df[serial_gpucachesim_mem_only_mask]
    serial_gpucachesim_mem_only_df = serial_gpucachesim_mem_only_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    print("gpucachesim mem only", serial_gpucachesim_mem_only_df.shape)

    # gpucachesim (exec-driven)
    serial_gpucachesim_exec_driven_mask = selected_df["target"] == Target.ExecDrivenSimulate.value
    serial_gpucachesim_exec_driven_mask &= selected_df["input_mode"].isin(["serial", np.nan])
    serial_gpucachesim_exec_driven_df = selected_df[serial_gpucachesim_exec_driven_mask]
    serial_gpucachesim_exec_driven_df = serial_gpucachesim_exec_driven_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    print("gpucachesim exec driven", serial_gpucachesim_exec_driven_df.shape)

    benches = sorted(selected_df["benchmark"].unique().tolist())
    if metric is None:
        metrics = ["cycles"]
    elif isinstance(metric, str):
        metrics = [metric]
    elif isinstance(metric, list):
        metrics = metric
    else:
        raise ValueError

    print(benches)
    print(metrics)

    assert all([m in selected_df for m in metrics])

    sim_targets = {
        "_accelsim": accelsim_df,
        "_gpucachesim": serial_gpucachesim_df,
        "_gpucachesim_mem_only": serial_gpucachesim_mem_only_df,
        "_gpucachesim_exec_driven": serial_gpucachesim_exec_driven_df,
    }

    dtypes = {
        **{col: "float64" for col in native_df.columns},
        **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()}
    }
    dtypes = {col: dtype for col, dtype in dtypes.items() if col in native_df}
    native_df = native_df.astype(dtypes)

    for suffix, sim_df in sim_targets.items():
        print("computing =>", suffix)
        join_cols = ["benchmark", "kernel_launch_id"] + sorted(list([
            col for col in benchmarks.ALL_BENCHMARK_INPUT_COLS if col in selected_df
        ]))
        missing_df = native_df[join_cols].merge(
            sim_df[join_cols],
            how="left",
            indicator=True,
        ).loc[lambda x: x["_merge"] != "both"]
        # print(missing_df)
        # print(missing_df.shape)
        if suffix == "_gpucachesim_exec_driven":
            assert sorted(missing_df["benchmark"].unique().tolist()) == ["babelstream"]
        else:
            assert len(missing_df) == 0

        sim_df = sim_df.astype(dtypes)

        print(native_df[["input_mode"]].dtypes)
        print(sim_df[["input_mode"]].dtypes)

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

        # new_cols = [col + suffix for col in native_df.columns if col + suffix in joined_df]
        # native_df[new_cols] = joined_df[new_cols]

        # if False:
        #     group_cols = sorted(
        #         bench_cols
        #         # + bench_input_cols
        #         # + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
        #         # + [col + "_parallel" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        #         # + [col + "_serial" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        #     )
        #     aggregations = {
        #         **{c: "mean" for c in sorted(joined_df.columns)},
        #         **{c + "_native": agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
        #         **{c + "_sim": agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
        #     }
        #     aggregations = {col: agg for col, agg in aggregations.items() if col in joined_df}
        #     aggregations = {
        #         col: agg for col, agg in aggregations.items() if col not in group_cols
        #     }
        #     print(group_cols)
        #
        #     # if set(joined.columns.tolist()) - set(group_cols) != set(aggregations.keys()):
        #     #     pprint(
        #     #         (set(joined.columns.tolist()) - set(group_cols)).symmetric_difference(
        #     #             set(aggregations.keys())
        #     #         )
        #     #     )
        #     #     raise ValueError
        #
        #     grouped = joined.groupby(group_cols, dropna=False)
        #     # aggregated = grouped.agg(aggregations, squeeze=False)

        # # speedup
        # joined_df["exec_time_sec_speedup"] = grouped.apply(
        #     lambda df: speedup(
        #         baseline=df["exec_time_sec_serial"], values=df["exec_time_sec_parallel"]
        #     ).mean()
        # )
        # # cycles error
        # aggregated["cycles_rel_err"] = grouped.apply(
        #     lambda df: rel_err(
        #         true_values=df["cycles_serial"], values=df["cycles_parallel"]
        #     ).mean()
        # )
        # aggregated["cycles_rmse"] = grouped.apply(
        #     lambda df: rmse(
        #         true_values=df["cycles_serial"], values=df["cycles_parallel"]
        #     ).mean()
        # )
        # aggregated["cycles_mae"] = grouped.apply(
        #     lambda df: mae(
        #         true_values=df["cycles_serial"], values=df["cycles_parallel"]
        #     ).mean()
        # )
        # # l1 hit rate error
        # aggregated["l1_hit_rate_rel_err"] = grouped.apply(
        #     lambda df: rel_err(
        #         true_values=df["l1_hit_rate_serial"], values=df["l1_hit_rate_parallel"]
        #     ).mean()
        # )
        # # l2 hit rate error
        # aggregated["l2_hit_rate_rel_err"] = grouped.apply(
        #     lambda df: rel_err(
        #         true_values=df["l2_hit_rate_serial"], values=df["l2_hit_rate_parallel"]
        #     ).mean()
        # )
        # # dram reads error
        # aggregated["dram_reads_rel_err"] = grouped.apply(
        #     lambda df: rel_err(
        #         true_values=df["dram_reads_serial"], values=df["dram_reads_parallel"]
        #     ).mean()
        # )
        # # dram writes error
        # aggregated["dram_writes_rel_err"] = grouped.apply(
        #     lambda df: rel_err(
        #         true_values=df["dram_writes_serial"], values=df["dram_writes_parallel"]
        #     ).mean()
        # )
        # break

    preview_cols = ["benchmark"] + [
        col + suffix for col, suffix in itertools.product(
            ["cycles"],
            [""] + list(sim_targets.keys())
        )]
    # print(native_df[preview_cols])

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
            print(bench, metric)
            if bench is not None:
                bench_df = native_df[native_df["benchmark"] == bench]
            else:
                bench_df = native_df
                # continue

            bench_df = bench_df.copy()

            bench_df["accelsim_rel_err"] = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_accelsim"])
            accelsim_valid = not np.isnan(bench_df[metric + "_accelsim"]).all()
            accelsim_rel_err = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_accelsim"]).mean()
            accelsim_mae = mae(
                true_values=bench_df[metric], values=bench_df[metric + "_accelsim"]).mean()
            accelsim_rmse = rmse(
                true_values=bench_df[metric], values=bench_df[metric + "_accelsim"])
            accelsim_correl = correlation(
                true_values=bench_df[metric], values=bench_df[metric + "_accelsim"])

            bench_df["gpucachesim_rel_err"] = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim"])
            gpucachesim_valid = not np.isnan(bench_df[metric + "_gpucachesim"]).all()
            gpucachesim_rel_err = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim"]).mean()
            gpucachesim_mae = mae(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim"]).mean()
            gpucachesim_rmse = rmse(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim"])
            gpucachesim_correl = correlation(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim"])


            bench_df["gpucachesim_mem_only_rel_err"] = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_mem_only"])
            gpucachesim_mem_only_valid = not np.isnan(bench_df[metric + "_gpucachesim_mem_only"]).all()
            gpucachesim_mem_only_rel_err = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_mem_only"]).mean()
            gpucachesim_mem_only_mae = mae(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_mem_only"]).mean()
            gpucachesim_mem_only_rmse = rmse(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_mem_only"])
            gpucachesim_mem_only_correl = correlation(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_mem_only"])



            bench_df["gpucachesim_exec_driven_rel_err"] = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_exec_driven"])
            gpucachesim_exec_valid = not np.isnan(bench_df[metric + "_gpucachesim_exec_driven"]).all()
            gpucachesim_exec_rel_err = rel_err(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_exec_driven"]).mean()
            gpucachesim_exec_mae = mae(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_exec_driven"]).mean()
            gpucachesim_exec_rmse = rmse(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_exec_driven"])
            gpucachesim_exec_correl = correlation(
                true_values=bench_df[metric], values=bench_df[metric + "_gpucachesim_exec_driven"])

            # print(bench_df[preview_cols + ["accelsim_rel_err", "gpucachesim_rel_err"]])
            #
            # print(accelsim_rel_err, gpucachesim_rel_err, gpucachesim_mem_only_rel_err, gpucachesim_exec_rel_err)
            # print(accelsim_mae, gpucachesim_mae, gpucachesim_mem_only_mae, gpucachesim_exec_mae)
            # print(accelsim_rmse, gpucachesim_rmse, gpucachesim_mem_only_rmse, gpucachesim_exec_rmse)
            # print(accelsim_correl, gpucachesim_correl, gpucachesim_mem_only_correl, gpucachesim_exec_correl)

            table += r"\multirow{4}{*}{" + str(metric) + "}"

            metric_values = np.array([
                [accelsim_rel_err,gpucachesim_rel_err, gpucachesim_mem_only_rel_err, gpucachesim_exec_rel_err],
                [accelsim_mae, gpucachesim_mae, gpucachesim_mem_only_mae, gpucachesim_exec_mae],
                [accelsim_rmse, gpucachesim_rmse, gpucachesim_mem_only_rmse, gpucachesim_exec_rmse],
                [accelsim_correl, gpucachesim_correl, gpucachesim_mem_only_correl, gpucachesim_exec_correl],
            ])
            if not accelsim_valid:
                metric_values[:,0] = np.nan
            if not gpucachesim_valid:
                metric_values[:,1] = np.nan
            if not gpucachesim_mem_only_valid:
                metric_values[:,2] = np.nan
            if not gpucachesim_exec_valid:
                metric_values[:,3] = np.nan

            table += r" & Rel err"
            rel_errs = metric_values[0,:] * 100.0
            for rel_err_value in rel_errs:
                table += " & "
                if np.isnan(rel_err_value):
                    continue
                bold = rel_err_value == np.nanmin(rel_errs)
                if bold:
                    table += r"\boldmath"
                    # table += r"\textbf{"
                table += "${:5.2f}\\%$".format(rel_err_value)
                # if bold:
                #     table += r"}"
            table += r"\\" + "\n"

            table += r" & MAE "
            maes = metric_values[1,:]
            for mae_value in maes:
                table += " & "
                if np.isnan(mae_value):
                    continue
                bold = mae_value == np.nanmin(maes)
                if bold:
                    # table += r"\textbf{"
                    table += r"\boldmath"
                table += "${}$".format(plot.human_format_thousands(mae_value))
                # if bold:
                #     table += r"}"
            table += r"\\" + "\n"

            table += r" & RMSE "
            rmses = metric_values[2,:]
            for rmse_value in rmses:
                table += " & "
                if np.isnan(rmse_value):
                    continue
                bold = rmse_value == np.nanmin(rmses)
                if bold:
                    table += r"\boldmath"
                table += "${}$".format(plot.human_format_thousands(rmse_value))
                # if bold:
                #     table += r"}"
            table += r"\\" + "\n"

            table += r" & Corr. "
            correls = metric_values[3,:]
            for correl_value in correls:
                table += " & "
                if np.isnan(correl_value):
                    continue
                bold = correl_value == np.nanmax(correls)
                if bold:
                    # table += r"\textbf{"
                    table += r"\boldmath"
                table += "${:5.3f}$".format(correl_value)
                # if bold:
                #     table += r"}"
            table += r"\\"
            # if bench is not None:
            table += r" \hline"
            table += "\n"

            # print(bench_df[preview_cols])

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
    num_benchmarks = len(selected_df["benchmark"].unique().tolist())

    bench_cols = ["target", "benchmark"]
    bench_input_cols = (
        [] if all_benchmarks else benchmarks.BENCHMARK_INPUT_COLS[bench_name_arg]
    )
    input_cols = benchmarks.SIMULATE_INPUT_COLS

    selected_df = sum_per_config_kernel_metrics(selected_df)

    # get serial
    serial = selected_df[selected_df["input_mode"] == "serial"].copy()

    # we are joining all parallel configs with their serial variant
    # however, we do not assume equal number of repetitions necessarily,
    # hence we compute the mean.
    # Note that repetitions also only have a very minimal influence on serial execution time,
    # since serial execution is deterministic
    group_cols = bench_cols + ["kernel_launch_id"] + input_cols + bench_input_cols

    aggregations = {
        **{c: "mean" for c in sorted(serial.columns)},
        **{c: "first" for c in serial.columns if c.startswith("input_")},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {col: agg for col, agg in aggregations.items() if col in serial}
    aggregations = {
        col: agg for col, agg in aggregations.items() if col not in group_cols
    }
    mean_serial = serial.groupby(group_cols).agg(aggregations).reset_index()

    metric_cols = ["cycles", "exec_time_sec", "l2_hit_rate", "l1_hit_rate"]

    serial = mean_serial
    parallel = selected_df[~selected_df["input_mode"].isin([np.nan, "serial"])]
    assert "total_cores" in serial
    assert "total_cores" in parallel

    # cols = ["dram_writes", "l1_accesses"]
    # print(serial[(serial["benchmark"] == "transpose") & (serial["input_variant"] == "naive") & (serial["input_dim"] == 512)][benchmarks.INDEX_COLS + benchmarks.SIMULATE_INPUT_COLS + cols])
    # print(parallel[(parallel["benchmark"] == "transpose") & (parallel["input_variant"] == "naive") & (parallel["input_dim"] == 512) & (parallel["input_mode"] == "deterministic")][benchmarks.INDEX_COLS + benchmarks.SIMULATE_INPUT_COLS + cols])

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

    input_id_partitoning = set(serial["input_id"].unique()).intersection(
        set(parallel["input_id"].unique())
    )
    if len(input_id_partitoning) > 0:
        for input_id in input_id_partitoning:
            print("serial input", input_id)
            print(
                serial.loc[
                    serial["input_id"] == input_id,
                    bench_cols
                    + ["kernel_launch_id"]
                    + bench_input_cols
                    + benchmarks.SIMULATE_INPUT_COLS,
                ]
            )
            print("parallel input", input_id)
            print(
                parallel.loc[
                    parallel["input_id"] == input_id,
                    bench_cols
                    + ["kernel_launch_id"]
                    + bench_input_cols
                    + benchmarks.SIMULATE_INPUT_COLS,
                ]
            )
            break
        assert len(input_id_partitoning) == 0

    # join based on input_cols, NOT based on mode
    joined = parallel.merge(
        serial,
        on=bench_cols
        + ["kernel_launch_id"]
        + bench_input_cols
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS,
        how="left",
        suffixes=("_parallel", "_serial"),
    )
    print(
        "joined={} parallel={} serial={}".format(
            joined.shape, parallel.shape, serial.shape
        )
    )
    assert joined.shape[0] == parallel.shape[0]
    assert "mean_blocks_per_sm_parallel" in joined
    assert "total_cores_parallel" in joined
    assert "cores_per_cluster_parallel" in joined

    if len(joined) == 0:
        raise ValueError("joined parallel and serial dataframe is empty")

    PREVIEW_COLS = sorted(
        list(
            bench_cols
            + ["kernel_launch_id"]
            + bench_input_cols
            + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
            + [c + "_parallel" for c in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
            + [c + "_parallel" for c in metric_cols]
            + [c + "_serial" for c in metric_cols]
            + ["input_id_serial", "input_id_parallel"]
        )
    )

    if True:
        print(joined.loc[0:3, PREVIEW_COLS].T)

    # cols = ["dram_writes", "l1_accesses"]
    # joined_tmp = joined.reset_index()
    # print(joined_tmp[(joined_tmp["benchmark"] == "transpose") & (joined_tmp["input_variant"] == "naive") & (joined_tmp["input_dim"] == 512)][bench_cols + ["kernel_launch_id"] + [c + "_serial" for c in cols] + [c + "_parallel" for c in cols]])

    print(
        joined[
            [
                "benchmark",
                "input_mode_parallel",
                "input_threads_parallel",
                "input_run_ahead_parallel",
            ]
            # ["benchmark"] + bench_input_cols + [c + "_parallel" for c in SIMULATE_EXECUTION_CONFIG_COLS]
        ].drop_duplicates()
    )

    group_cols = sorted(
        bench_cols
        + bench_input_cols
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [col + "_parallel" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + [col + "_serial" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
    )
    aggregations = {
        **{c: "sum" for c in sorted(joined.columns)},
        **{c + "_parallel": agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
        **{c + "_serial": agg for c, agg in benchmarks.NON_NUMERIC_COLS.items()},
    }
    aggregations = {col: agg for col, agg in aggregations.items() if col in joined}
    aggregations = {
        col: agg for col, agg in aggregations.items() if col not in group_cols
    }

    if set(joined.columns.tolist()) - set(group_cols) != set(aggregations.keys()):
        pprint(
            (set(joined.columns.tolist()) - set(group_cols)).symmetric_difference(
                set(aggregations.keys())
            )
        )
        raise ValueError

    
    print(joined.shape)
    grouped = joined.groupby(group_cols, dropna=False)
    aggregated = grouped.agg(aggregations, squeeze=False)

    # speedup
    aggregated["exec_time_sec_speedup"] = grouped.apply(
        lambda df: speedup(
            baseline=df["exec_time_sec_serial"], values=df["exec_time_sec_parallel"]
        ).mean()
    )
    # cycles error
    aggregated["cycles_rel_err"] = grouped.apply(
        lambda df: rel_err(
            true_values=df["cycles_serial"], values=df["cycles_parallel"]
        ).mean()
    )
    aggregated["cycles_rmse"] = grouped.apply(
        lambda df: rmse(
            true_values=df["cycles_serial"], values=df["cycles_parallel"]
        ).mean()
    )
    aggregated["cycles_mae"] = grouped.apply(
        lambda df: mae(
            true_values=df["cycles_serial"], values=df["cycles_parallel"]
        ).mean()
    )
    # l1 hit rate error
    aggregated["l1_hit_rate_rel_err"] = grouped.apply(
        lambda df: rel_err(
            true_values=df["l1_hit_rate_serial"], values=df["l1_hit_rate_parallel"]
        ).mean()
    )
    # l2 hit rate error
    aggregated["l2_hit_rate_rel_err"] = grouped.apply(
        lambda df: rel_err(
            true_values=df["l2_hit_rate_serial"], values=df["l2_hit_rate_parallel"]
        ).mean()
    )
    # dram reads error
    aggregated["dram_reads_rel_err"] = grouped.apply(
        lambda df: rel_err(
            true_values=df["dram_reads_serial"], values=df["dram_reads_parallel"]
        ).mean()
    )
    # dram writes error
    aggregated["dram_writes_rel_err"] = grouped.apply(
        lambda df: rel_err(
            true_values=df["dram_writes_serial"], values=df["dram_writes_parallel"]
        ).mean()
    )

    def _inspect(df):
        # if df["input_rows"].values[0] != 512:
        #     return

        print(
            df.reset_index()[
                bench_cols
                + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
                + ["total_cores_parallel"]
                # + ["input_variant"]
                + ["input_rows"]
                + ["l1_hit_rate_serial", "l1_hit_rate_parallel"]
            ]
        )
        print(
            rel_err(
                true_values=df["l1_hit_rate_serial"], values=df["l1_hit_rate_parallel"]
            )
        )

    # grouped.apply(_inspect)

    # print(aggregated.reset_index()[[
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

    # return

    aggregated = aggregated.reset_index()
    print(aggregated.shape)
    print(
        aggregated.reset_index()
        .loc[
            0:4,
            PREVIEW_COLS
            + bench_input_cols
            + ["cycles_rel_err", "exec_time_sec_speedup"],
        ]
        .T.drop_duplicates()
    )

    print(
        aggregated.loc[
            (aggregated["input_mode_parallel"] == "deterministic")
            & (aggregated["input_threads_parallel"] == 4),
            PREVIEW_COLS + ["cycles_rel_err", "exec_time_sec_speedup"],
        ].T.drop_duplicates()
    )
    # print(aggregated.reset_index().loc[0:4,
    #     ["cycles_rel_err", "cycles_serial", "cycles_parallel"]].T) #.drop_duplicates())

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

    # print(aggregated[bench_input_cols].drop_duplicates())

    def compute_label(bench_config, df):
        benchmark = df["benchmark"]
        bench_input_cols = benchmarks.BENCHMARK_INPUT_COLS[benchmark]
        assert all([c in df for c in bench_input_cols])

        assert (df["total_cores_parallel"] == df["total_cores_serial"]).all()

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
            table_row += r"\multirow{2}{*}{\shortstack[l]{" + str(row.metric) + r"}}"

        # threads
        table_row += r" & $t=" + str(row.threads) + r"$ "

        # serial value
        if row.serial_value is not None and is_first_metric_row:
            table_row += (
                r" & \multirow{2}{*}{\shortstack[l]{"
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

    table = ""
    # thousands_round_to = 1
    # variable_precision = True

    if all_benchmarks:
        # mask_cols = ["benchmark"] + list(bench_inputs.keys())
        # mask_values = [bench_name] + list(bench_inputs.values())
        # mask = aggregated["benchmark"] == bench_name
        # for col, value in zip(mask_cols, mask_values):
        #     mask &= aggregated[col] == value
        # print((aggregated[mask_cols] == mask_values).sum(axis=0))
        for functional_config in functional_configs:
            mask_cols = list(functional_config.keys())
            mask_values = list(functional_config.values())
            mask = (aggregated[mask_cols] == mask_values).all(axis=1)

            # df = aggregated[mask]
            # test_df = aggregated.loc[
            #     mask,
            #     benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
            #     + bench_input_cols
            #     + ["mean_blocks_per_sm_parallel"]]
            # test_df = test_df.drop_duplicates()
            # assert len(test_df) == 1

            label = "Average @ {} SM's".format(  # [{:.2f} CTA/SM]".format(
                int(aggregated.loc[mask, "total_cores_parallel"].values[0]),
                # float(aggregated.loc[mask, "mean_blocks_per_sm_parallel"].values[0]),
            )

            table += "%\n%\n"
            table += (
                r"\rowcolor{gray!10} \multicolumn{8}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask], num_benchmarks=num_benchmarks, all_benchmarks=True
            )

            # for threads in [4, 8]:
            #     threads_mask = aggregated["input_threads_parallel"] == threads
            #     det_mask = aggregated["input_mode_parallel"] == "deterministic"
            #     nondet_no_interleave_mask = (
            #         aggregated["input_mode_parallel"] == "nondeterministic"
            #     )
            #     nondet_interleave_mask = (
            #         aggregated["input_mode_parallel"] == "nondeterministic_interleave"
            #     )
            #
            #     det = aggregated[mask & threads_mask & det_mask]
            #
            #     # det_preview = det[
            #     #     PREVIEW_COLS
            #     #     + ["input_threads_parallel", "cycles_rel_err", "exec_time_sec_speedup"]
            #     # ]
            #
            #     print("===")
            #     nondet_no_interleave = aggregated[
            #         mask & threads_mask & nondet_no_interleave_mask
            #     ]
            #     nondet_interleave = aggregated[
            #         mask & threads_mask & nondet_interleave_mask
            #     ]
            #
            #     assert len(det) == num_benchmarks
            #     assert len(nondet_interleave) == 2 * num_benchmarks
            #     assert len(nondet_no_interleave) == 2 * num_benchmarks
            #
            #     # exec time (speedup)
            #     det_speedup = det["exec_time_sec_speedup"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #         nondet_speedup = nondet["exec_time_sec_speedup"].values[0]
            #         nondet_values.append(
            #             (nondet_speedup, "${}x$".format(
            #                 plot.round_to_precision(nondet_speedup, round_to=1, variable_precision=variable_precision)
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric=r"exec\\time",
            #             threads=threads,
            #             serial_value=None,
            #             det_value=(det_speedup, "${}x$".format(
            #                 plot.round_to_precision(det_speedup, round_to=1,
            #                                         variable_precision=variable_precision)
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )
            #
            #     # cycles (rel err)
            #     serial_cycles = int(aggregated.loc[mask & threads_mask, "cycles_serial"].values[0])
            #     det_cycles = int(det["cycles_parallel"].values[0])
            #     det_rel_err = det["cycles_rel_err"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #
            #         nondet_cycles = int(nondet["cycles_parallel"].values[0])
            #         nondet_rel_err = nondet["cycles_rel_err"].values[0]
            #         nondet_values.append(
            #             (nondet_cycles, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(nondet_cycles, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * nondet_rel_err, round_to=1,
            #                                         variable_precision=variable_precision)
            #
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric="cycles",
            #             threads=threads,
            #             serial_value=(serial_cycles, "${}$".format(plot.human_format_thousands(serial_cycles, round_to=thousands_round_to, variable_precision=variable_precision))),
            #             det_value=(det_cycles, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(det_cycles, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * det_rel_err, round_to=1,
            #                                         variable_precision=variable_precision)
            #
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )

            table += "%\n%\n"

            table_rows = sorted(table_rows, key=lambda row: (row.metric, row.threads))
            for row in table_rows:
                bold_values = []
                if row.metric == r"exec\\time":
                    bold_values = [np.amin(row.values())]
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
            mask_values = [bench_name] + list(bench_inputs.values())
            # mask = aggregated["benchmark"] == bench_name
            # for col, value in zip(mask_cols, mask_values):
            #     mask &= aggregated[col] == value
            # print((aggregated[mask_cols] == mask_values).sum(axis=0))

            mask = (aggregated[mask_cols] == mask_values).all(axis=1)
            test_df = aggregated.loc[
                mask,
                benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
                + bench_input_cols
                + ["mean_blocks_per_sm_parallel"],
            ]
            test_df = test_df.drop_duplicates()
            print(test_df)
            assert len(test_df) == 1

            table += "%\n%\n"
            table += (
                r"\rowcolor{gray!10} \multicolumn{8}{c}{\textbf{"
                + str(compute_label(bench_config, aggregated.loc[mask].iloc[0]))
                + r"}} \\ \hline"
                + "\n"
            )

            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask], num_benchmarks=num_benchmarks, all_benchmarks=False
            )
            # table_rows: typing.Sequence[TableRow] = []

            # for threads in [4, 8]:
            #     threads_mask = aggregated["input_threads_parallel"] == threads
            #     det_mask = aggregated["input_mode_parallel"] == "deterministic"
            #     nondet_no_interleave_mask = (
            #         aggregated["input_mode_parallel"] == "nondeterministic"
            #     )
            #     nondet_interleave_mask = (
            #         aggregated["input_mode_parallel"] == "nondeterministic_interleave"
            #     )
            #     # print([m.sum() for m in [
            #     #     mask, threads_mask, det_mask, nondet_no_interleave_mask, nondet_interleave_mask
            #     # ]])
            #
            #     det = aggregated[mask & threads_mask & det_mask]
            #     print(
            #         det[
            #             bench_input_cols
            #             + [
            #                 "input_threads_parallel",
            #                 "exec_time_sec_parallel",
            #                 "input_id_parallel",
            #                 "input_id_serial",
            #                 # "dram_reads_serial",
            #                 # "dram_reads_parallel",
            #                 # "dram_reads_rel_err",
            #                 "dram_writes_serial",
            #                 "dram_writes_parallel",
            #                 "dram_writes_rel_err",
            #             ] + different_cols(det)
            #         ]
            #     )
            #     print("===")
            #     nondet_no_interleave = aggregated[
            #         mask & threads_mask & nondet_no_interleave_mask
            #     ]
            #     nondet_interleave = aggregated[
            #         mask & threads_mask & nondet_interleave_mask
            #     ]
            #
            #     assert len(det) == num_benchmarks
            #     assert len(nondet_no_interleave) == 2 * num_benchmarks
            #     assert len(nondet_interleave) == 2 * num_benchmarks
            #     assert (
            #         len(
            #             aggregated.loc[
            #                 mask,
            #                 [
            #                     "exec_time_sec_serial",
            #                     "cycles_serial",
            #                     "input_id_serial",
            #                 ],
            #             ].drop_duplicates()
            #         )
            #         == 1
            #     )
            #
            #     # exec time (speedup)
            #     serial_exec_time = aggregated.loc[
            #         mask & threads_mask, "exec_time_sec_serial"
            #     ].values[0]
            #     det_exec_time = det["exec_time_sec_parallel"].values[0]
            #     det_speedup = det["exec_time_sec_speedup"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #         nondet_exec_time = nondet["exec_time_sec_parallel"].values[0]
            #         nondet_speedup = nondet["exec_time_sec_speedup"].values[0]
            #         nondet_values.append(
            #             (nondet_exec_time, "${:>3.1f}s~({}x)$".format(
            #                 nondet_exec_time,
            #                 plot.round_to_precision(nondet_speedup, round_to=1, variable_precision=variable_precision)
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric=r"exec\\time",
            #             threads=threads,
            #             serial_value=(serial_exec_time, "${:>3.1f}s$".format(serial_exec_time)),
            #             det_value=(det_exec_time, "${:>3.1f}s~({}x)$".format(
            #                 det_exec_time,
            #                 plot.round_to_precision(det_speedup, round_to=1,
            #                                         variable_precision=variable_precision)
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )
            #
            #     # cycles (rel err)
            #     serial_cycles = int(aggregated.loc[mask & threads_mask, "cycles_serial"].values[0])
            #     det_cycles = int(det["cycles_parallel"].values[0])
            #     det_rel_err = det["cycles_rel_err"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #
            #         nondet_cycles = int(nondet["cycles_parallel"].values[0])
            #         nondet_rel_err = nondet["cycles_rel_err"].values[0]
            #         nondet_values.append(
            #             (nondet_cycles, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(nondet_cycles, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * nondet_rel_err, round_to=1,
            #                                         variable_precision=variable_precision)
            #
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric="cycles",
            #             threads=threads,
            #             serial_value=(serial_cycles, "${}$".format(plot.human_format_thousands(serial_cycles, round_to=thousands_round_to, variable_precision=variable_precision))),
            #             det_value=(det_cycles, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(det_cycles, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * det_rel_err, round_to=1,
            #                                         variable_precision=variable_precision)
            #
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )
            #
            #     # l1 data hit rate (rel err)
            #     serial_l1_hit_rate = aggregated.loc[mask & threads_mask, "l1_hit_rate_serial"].values[0]
            #     det_l1_hit_rate = det["l1_hit_rate_parallel"].values[0]
            #     det_rel_err = det["l1_hit_rate_rel_err"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #
            #         nondet_l1_hit_rate = nondet["l1_hit_rate_parallel"].values[0]
            #         nondet_rel_err = nondet["l1_hit_rate_rel_err"].values[0]
            #         nondet_values.append(
            #             (100.0 * nondet_l1_hit_rate, "${}\\%~({}\\%)$".format(
            #                 plot.round_to_precision(100.0 * nondet_l1_hit_rate, round_to=1,
            #                                         variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * nondet_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric=r"L1D\\hit rate",
            #             threads=threads,
            #             serial_value=(100.0 * serial_l1_hit_rate, "${:>2.1f}\\%$".format(100.0 * serial_l1_hit_rate)),
            #             det_value=(100.0 * det_l1_hit_rate, "${}\\%~({}\\%)$".format(
            #                 plot.round_to_precision(100.0 * det_l1_hit_rate, round_to=1,
            #                                         variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * det_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )
            #
            #     # l2 data hit rate (rel err)
            #     serial_l2_hit_rate = aggregated.loc[mask & threads_mask, "l2_hit_rate_serial"].values[0]
            #     det_l2_hit_rate = det["l2_hit_rate_parallel"].values[0]
            #     det_rel_err = det["l2_hit_rate_rel_err"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #
            #         nondet_l2_hit_rate = nondet["l2_hit_rate_parallel"].values[0]
            #         nondet_rel_err = nondet["l2_hit_rate_rel_err"].values[0]
            #         nondet_values.append(
            #             (100.0 * nondet_l2_hit_rate, "${}\\%~({}\\%)$".format(
            #                 plot.round_to_precision(100.0 * nondet_l2_hit_rate, round_to=1,
            #                                         variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * nondet_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric=r"L2D\\hit rate",
            #             threads=threads,
            #             serial_value=(
            #                 100.0 * serial_l2_hit_rate,
            #                 "${}\\%$".format(
            #                     plot.round_to_precision(
            #                         100.0 * serial_l2_hit_rate,
            #                         round_to=1, variable_precision=variable_precision)
            #             )),
            #             det_value=(100.0 * det_l2_hit_rate, "${}\\%~({}\\%)$".format(
            #                 plot.round_to_precision(100.0 * det_l2_hit_rate, round_to=1,
            #                                         variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * det_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )
            #
            #     # dram reads (rel err)
            #     serial_dram_reads = int(aggregated.loc[mask & threads_mask, "dram_reads_serial"].values[0])
            #     det_dram_reads = int(det["dram_reads_parallel"].values[0])
            #     det_rel_err = det["dram_reads_rel_err"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #
            #         nondet_dram_reads = int(nondet["dram_reads_parallel"].values[0])
            #         nondet_rel_err = nondet["dram_reads_rel_err"].values[0]
            #         nondet_values.append(
            #             (nondet_dram_reads, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(nondet_dram_reads, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * nondet_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric=r"DRAM\\reads",
            #             threads=threads,
            #             serial_value=(serial_dram_reads, "${}$".format(plot.human_format_thousands(serial_dram_reads, round_to=thousands_round_to, variable_precision=variable_precision))),
            #             det_value=(det_dram_reads, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(det_dram_reads, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * det_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )
            #
            #     # dram writes (rel err)
            #     serial_dram_writes = int(
            #         aggregated.loc[mask & threads_mask, "dram_writes_serial"].values[0]
            #     )
            #     det_dram_writes = int(det["dram_writes_parallel"].values[0])
            #     det_rel_err = det["dram_writes_rel_err"].values[0]
            #     nondet_values = []
            #     for interleave, n in interleave_n:
            #         nondet = nondet_interleave if interleave else nondet_no_interleave
            #         nondet = nondet[nondet["input_run_ahead_parallel"] == n]
            #
            #         nondet_dram_writes = int(nondet["dram_writes_parallel"].values[0])
            #         nondet_rel_err = nondet["dram_writes_rel_err"].values[0]
            #         nondet_values.append(
            #             (nondet_dram_writes, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(nondet_dram_writes, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * nondet_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #
            #             ))
            #         )
            #
            #     table_rows.append(
            #         TableRow(
            #             metric=r"DRAM\\writes",
            #             threads=threads,
            #             serial_value=(serial_dram_writes, "${}$".format(plot.human_format_thousands(serial_dram_writes, round_to=thousands_round_to, variable_precision=variable_precision))),
            #             # serial_value="${:>4}$".format(),
            #             det_value=(det_dram_writes, "${} ({}\\%)$".format(
            #                 plot.human_format_thousands(det_dram_writes, round_to=thousands_round_to, variable_precision=variable_precision),
            #                 plot.round_to_precision(100.0 * det_rel_err, round_to=1,
            #                                         variable_precision=variable_precision),
            #
            #             )),
            #             nondet_values=nondet_values,
            #         )
            #     )

            table += "%\n%\n"

            table_rows = sorted(table_rows, key=lambda row: (row.metric, row.threads))
            for row in table_rows:
                bold_values = []
                if row.metric == r"exec\\time":
                    bold_values = [np.amin(row.values())]
                print(row.metric, bold_values, row.values())
                table += write_table_row(row, bold_values)

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

    dtypes = {
        **{col: "float64" for col in stats_df.columns},
        **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()}
    }
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
        non_release_results = simulation_targets_df[simulation_targets_df["is_release_build"] == True]
        grouped = non_release_results.groupby(["benchmark", "target"])
        print(grouped["input_id"].count())
        print("====")
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

            bench_input_cols = benchmarks.BENCHMARK_INPUT_COLS[bench_name]
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
                **{c: "mean" for c in set(bench_df.columns) - set(group_cols)},
                **benchmarks.NON_NUMERIC_COLS,
            }
            aggregations = {
                col: agg for col, agg in aggregations.items() if col in bench_df
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
            gpucachesim_trace_reconstruction_df = gpucachesim_trace_reconstruction_df.groupby(
                bench_input_cols
            ).agg(aggregations)

            print("native                    ", native_df.shape)
            print("accelsim                  ", accelsim_df.shape)
            print("gpucachesim               ", gpucachesim_df.shape)
            print("gpucachesim (mem only)    ", gpucachesim_memory_only_df.shape)
            print("gpucachesim (exec driven) ", gpucachesim_trace_reconstruction_df.shape)

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
}

# @main.command()
# # @click.pass_context
# # @click.option("--path", help="Path to materialized benchmark config")
# # @click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
# @click.option("--bench", "bench_name", help="Benchmark name")
# # @click.option("--plot", "should_plot", type=bool, default=True, help="generate plots")
# @click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
# # @click.option("--memory-only", "mem_only", type=bool, is_flag=True, help="memory only")
# # @click.option("--verbose", "verbose", type=bool, is_flag=True, help="verbose output")
# # @click.option("--input", "input_idx", type=int, help="Input index")
# def test(bench_name, nsight):
#     # 0.95 0.95 0.78 0.59 0.45 0.95 0.95
#     profiler = "nsight" if nsight else "nvprof"
#     if bench_name is None:
#         stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.csv".format(profiler)
#     else:
#         stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
#             profiler, bench_name
#         )
#
#     per_kernel, per_target_pivoted = aggregate_benchmark_results(
#         sim_df, bench_name, memory_only=mem_only
#     )


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
def view(path, bench_name_arg, should_plot, nsight, mem_only, verbose, strict):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = load_stats(bench_name=bench_name_arg, profiler=profiler, path=path)
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

    per_config = aggregate_benchmark_results(selected_df, memory_only=mem_only)

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

    all_input_cols = benchmarks.ALL_BENCHMARK_INPUT_COLS
    all_input_cols = sorted(list([col for col in all_input_cols if col in per_config]))

    # average pivot preview table over runs
    group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + all_input_cols
    grouped = per_config.groupby(group_cols, dropna=False)
    aggregations = {
        **{c: "mean" for c in set(per_config.columns)},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {
        col: agg
        for col, agg in aggregations.items()
        if col in per_config and not col in group_cols
    }
    per_config_pivoted = grouped.agg(aggregations).reset_index()

    per_config_pivoted = per_config_pivoted.pivot(
        index=["benchmark"] + all_input_cols,
        columns="target",
    )

    print(" === {} === ".format(profiler))
    print(per_config_pivoted[stat_cols].T)

    if not should_plot:
        return

    def compute_label(df):
        assert isinstance(df, pd.Series)

        benchmark = df["benchmark"]
        bench_input_cols = benchmarks.BENCHMARK_INPUT_COLS[benchmark]
        assert all([c in df for c in bench_input_cols])

        match benchmark.lower():
            case "vectoradd":
                label = "VectorAdd\n"
                label += "f{:<2} {}".format(
                    int(df["input_dtype"]),
                    int(df["input_length"]),
                )
            case "matrixmul":
                label = "MatrixMul\n"
                label += "f{:<2} {}x{}x{}".format(
                    int(df["input_dtype"]),
                    int(df["input_rows"]),
                    int(df["input_rows"]),
                    int(df["input_rows"]),
                )
            case "simple_matrixmul":
                label = "Naive MatrixMul\n"
                label += "f{:<2} {}x{}x{}".format(
                    int(df["input_dtype"]),
                    int(df["input_m"]),
                    int(df["input_n"]),
                    int(df["input_p"]),
                )
            case "transpose":
                label = "Transpose\n"
                label += "{}\n".format(df["input_variant"])
                label += "{}x{}".format(
                    int(df["input_dim"]),
                    int(df["input_dim"]),
                )
            case "babelstream":
                label = "BabelStream\n"
                label += "{}".format(int(df["input_size"]))
            case other:
                label = str(other)

        return label

    per_config["label"] = per_config.apply(compute_label, axis=1)
    per_config.loc[
        per_config["target"] == Target.Simulate.value, "target_name"
    ] = "gpucachesim"
    per_config.loc[
        per_config["target"] == Target.AccelsimSimulate.value, "target_name"
    ] = "AccelSim"
    per_config.loc[
        per_config["target"] == Target.Profile.value, "target_name"
    ] = per_config.loc[~per_config["device"].isna(), "device"].apply(
        native.normalize_nvprof_device_name
    )

    targets = sorted(per_config["target"].unique().tolist())
    benches = sorted(per_config["benchmark"].unique().tolist())

    targets = [
        target
        for target in targets
        if target in ["Profile", "Simulate", "AccelsimSimulate"]
    ]

    for stat_col, benchmark in itertools.product(stat_cols, benches):
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

        group_width = len(targets) * (bar_width + spacing) + group_spacing

        plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
        fig = plt.figure(
            figsize=(1.0 * plot.DINA4_WIDTH_INCHES, 0.16 * plot.DINA4_HEIGHT_INCHES),
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

        bench_input_cols = benchmarks.BENCHMARK_INPUT_COLS[benchmark]
        group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + bench_input_cols

        bench_input_values = per_config.loc[
            per_config["benchmark"] == benchmark, bench_input_cols
        ]

        match benchmark:
            case "simple_matrixmul":
                subset = pd.DataFrame.from_records(
                    [
                        (32, 32, 32),
                        (128, 128, 128),
                        (32, 64, 128),
                        (128, 32, 32),
                        (128, 512, 128),
                        (512, 32, 512),
                    ],
                    columns=["input_m", "input_n", "input_p"],
                )
                bench_input_values = bench_input_values.merge(subset, how="inner")

        bench_input_values = bench_input_values.drop_duplicates().reset_index()

        target_configs = list(
            itertools.product(targets, list(bench_input_values.iterrows()))
        )

        for target, (input_idx, input_values) in target_configs:
            print(target, input_idx, dict(input_values))
            target_df_mask = per_config["target"] == target
            target_df_mask &= per_config["benchmark"] == benchmark
            for col in bench_input_cols:
                target_df_mask &= per_config[col] == input_values[col]
            target_df = per_config.loc[target_df_mask, :]

            if len(target_df) < 1:
                print(
                    color(
                        "missing {} {} [{}]".format(
                            target, benchmark, input_values.values.tolist()
                        ),
                        fg="red",
                    )
                )
                if strict:
                    return
                continue

            target_df = target_df.groupby(group_cols, dropna=False)

            target_idx = targets.index(target)
            idx = input_idx * group_width + (target_idx + 0.5) * (bar_width + spacing)

            target = target_df["target"].first().values[0]
            target_name = target_df["target_name"].first().values[0]

            if verbose:
                print(
                    "{:>15} {:<10} {:>15} [{:<3}]  {:<35}  {:<3} {:<4} = {:<8.2f} {:<8.2f}".format(
                        benchmark,
                        stat_col,
                        target_name,
                        target_idx,
                        str(input_values[bench_input_cols].tolist()),
                        input_idx,
                        idx,
                        target_df[stat_col].fillna(0.0).mean(),
                        target_df[stat_col].fillna(0.0).std(),
                    )
                )

            x = [idx]
            y = target_df[stat_col].fillna(0.0).mean()
            if stat_config.percent:
                y *= 100.0
            ystd = target_df[stat_col].fillna(0.0).std()

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

        simulate_df_mask = per_config["target"] == Target.Simulate.value
        simulate_df_mask &= per_config["benchmark"] == benchmark
        simulate_df = per_config.loc[simulate_df_mask, :]
        simulate_df = simulate_df.merge(bench_input_values, how="inner")
        # print(simulate_df.head(n=100))
        # simulate_df = simulate_df.drop_duplicates().reset_index()
        assert len(simulate_df) > 0

        labels = simulate_df["label"].values
        num_blocks = simulate_df["num_blocks"].values
        # print(labels.tolist())
        assert len(labels) == len(num_blocks)
        

        all_values_mask = per_config["benchmark"] == benchmark
        all_values_df = per_config.loc[all_values_mask, :]
        all_values_df = all_values_df.merge(bench_input_values, how="inner")
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
                ymax = utils.round_to_multiple_of(1.5 * ymax, multiple_of=25.0)
                ymax = np.clip(ymax, 25.0, 100.0)
                ax.set_ylim(0, ymax + 10.0)
            else:
                ymax = max(2 * ymax, 1)
                ax.set_ylim(0, ymax)
            ytick_values = np.linspace(0, ymax, 6)

        ytick_labels = [
            plot.human_format_thousands(v, round_to=0) for v in ytick_values
        ]
        ax.set_yticks(ytick_values, ytick_labels)

        plot_dir = plot.PLOT_DIR / "validation"
        plot_dir.parent.mkdir(parents=True, exist_ok=True)

        

        xtick_labels = [
            "{}\n{} {}".format(label, int(blocks), "blocks" if blocks > 1 else "block")
            for label, blocks in zip(labels, num_blocks)
        ]
        xtick_values = np.arange(0, len(labels), dtype=np.float64)
        xtick_values *= group_width
        xtick_values += 0.5 * float((group_width - group_spacing))
        ax.set_xlim(0, len(xtick_labels) * group_width)

        # plot without xticks
        ax.set_xticks(xtick_values, ["" for _ in range(len(xtick_values))], rotation=0)
        filename = plot_dir / "{}.{}.{}_no_xticks.pdf".format(profiler, benchmark, stat_col)
        fig.savefig(filename)

        ax.set_xticks(xtick_values, xtick_labels, rotation=0)

        # plot without legend
        filename = plot_dir / "{}.{}.{}_no_legend.pdf".format(profiler, benchmark, stat_col)
        fig.savefig(filename)

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

        filename = plot_dir / "{}.{}.{}.pdf".format(profiler, benchmark, stat_col)
        fig.savefig(filename)
        print(color("wrote {}".format(filename), fg="cyan"))


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
                            bench_stats = gpucachesim.stats.native.NvprofStats(config, bench_config)
                        case ("profile", "nsight"):
                            target_name += "[nsight]"
                            bench_stats = gpucachesim.stats.native.NsightStats(config, bench_config)
                        case ("simulate", _):
                            bench_stats = gpucachesim.stats.stats.Stats(config, bench_config)
                        case ("execdrivensimulate", _):
                            bench_stats = gpucachesim.stats.stats.ExecDrivenStats(config, bench_config)
                        case ("accelsimsimulate", _):
                            bench_stats = gpucachesim.stats.accelsim.Stats(config, bench_config)
                        case ("playgroundsimulate", _):
                            bench_stats = gpucachesim.stats.playground.Stats(config, bench_config)
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


if __name__ == "__main__":
    main()
    # main(ctx={})
