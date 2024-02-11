import typing
import copy

import numpy as np
import pandas as pd
from wasabi import color
from pprint import pprint

import gpucachesim.benchmarks as benchmarks
import gpucachesim.utils as utils
import gpucachesim.plot as plot
import gpucachesim.stats.metrics as metrics
import gpucachesim.stats.agg

from gpucachesim.benchmarks import (
    Target,
)

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
    # num_bench_configs: int,
    thousands_round_to=1,
    variable_precision=True,
    verbose=True,
) -> typing.Sequence[ParallelTableRow]:
    # interleave_n = list(itertools.product([False, True], [5, 10]))
    run_ahead_values = [5, 10]
    for run_ahead in run_ahead_values:
        # print(df["input_run_ahead_parallel"].unique())
        assert run_ahead in df["input_run_ahead_parallel"].unique()

    table_rows: typing.Sequence[ParallelTableRow] = []


    multiple_bench_configs = len(df[["target", "benchmark", "input_id_serial"]].drop_duplicates()) > 1

    # assert num_bench_configs > 0
    # multiple_bench_configs = num_bench_configs > 1

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

        # benchmarks.BENCH_TARGET_INDEX_COLS
        #     + ["kernel_name", "kernel_launch_id", "run"]
        #     + list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS))
        #     + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS

        if verbose:
            print(
                color(
                    "==> max speedup for {} threads is {}".format(
                        threads, all_parallel["exec_time_sec_speedup"].max()
                    ),
                    fg="green",
                )
            )

        weird_mask = all_parallel["exec_time_sec_speedup"] > threads
        weird = all_parallel.loc[weird_mask, preview_cols]
        if len(weird) > 0:
            print(
                color(
                    "WARNING: weird results for {} threads:".format(threads), fg="red"
                )
            )
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
        # if False:
        #     if num_bench_configs > 1:
        #         print(det.loc[det["benchmark"] == "vectorAdd", preview_cols].T)
        #     else:
        #         print(det.loc[:, preview_cols].T)

        all_nondet = df[threads_mask & nondet_mask]
        # nondet_no_interleave = df[threads_mask & nondet_no_interleave_mask]
        # nondet_interleave = df[threads_mask & nondet_interleave_mask]

        if verbose:
            print(
                "num deterministic={} num nondeterministic={}".format(
                    #  num benchmark configs={}".format(
                    len(det), len(all_nondet), # num_bench_configs
                )
            )

        # print(det)
        # if not large:

        # assert len(det) == num_bench_configs
        # assert len(all_nondet) == len(run_ahead_values) * num_bench_configs

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

        parallel_preview_cols = list(
            benchmarks.BENCH_TARGET_INDEX_COLS
            + ["input_id_serial", "input_id_parallel"]
            + benchmarks.INDEX_COLS
            + [c for c in benchmarks.SIMULATE_INPUT_COLS]
            + [c + "_parallel" for c in benchmarks.SIMULATE_INPUT_COLS]
            + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
        )
        parallel_preview_cols += [
            "total_cores_parallel",
            "num_blocks_parallel",
            "mean_blocks_per_sm_parallel",
            "exec_time_sec_serial",
            "exec_time_sec_parallel",
            "exec_time_sec_speedup",
            "cycles_serial",
            "cycles_parallel",
            "cycles_mape",
        ]
        parallel_preview_cols = [col for col in parallel_preview_cols if col in df]

        spacer = " " + ("=" * 20) + " "

        # exec time (speedup)
        serial_exec_time = df.loc[threads_mask, "exec_time_sec_serial"].mean()
        det_exec_time = det["exec_time_sec_parallel"].mean()
        det_speedup = det["exec_time_sec_speedup"].mean()
        # if multiple_bench_configs:

        if verbose:
            print("")
            print(
                spacer
                + "DETERMINISTIC {} threads={}".format(det.shape, threads)
                + spacer
            )
            print(det[parallel_preview_cols][:8].T)

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

            if verbose:
                print("")
                print(
                    spacer
                    + "NONDETERMINISTIC {} threads={} run ahead={}".format(
                        nondet.shape, threads, run_ahead
                    )
                    + spacer
                )
                print(nondet[parallel_preview_cols][:8].T)

            # print(nondet.T)
            # assert len(nondet) == 1
            # if not large:
            # assert len(nondet) == num_bench_configs

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
    # print(
    #     df[
    #         [
    #             "benchmark",
    #             "input_cores_per_cluster",
    #             "input_num_clusters",
    #             "total_cores_parallel",
    #         ]
    #     ]
    # )
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

    label += " @ {} SM's [{:.2f} blocks/SM]".format(
        int(df["total_cores_parallel"]),
        float(df["mean_blocks_per_sm_parallel"]),
    )
    return label

def write_table_row(row: ParallelTableRow, _bold_values: typing.Optional[typing.Sequence[float]]=None):
    if _bold_values is None:
        bold_values = set()
    else:
        bold_values = set(_bold_values)

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




def parallel_table(selected_df, bench_name, scale_clusters=True, large=False, verbose=True, batch=False, png=False):
    all_benchmarks = bench_name is None

    if verbose:
        print(selected_df[["target", "run"]].drop_duplicates())

    # only keep simulation and remove non kernel stats
    selected_df = selected_df[selected_df["target"] == Target.Simulate.value]
    selected_df = selected_df[~selected_df["kernel_name"].isna()]
    # selected_df = sum_per_config_kernel_metrics(selected_df)
    selected_df, _ = gpucachesim.stats.agg.aggregate_mean_input_config_stats(
        selected_df, per_kernel=False, mean=False
    )

    # num_benchmarks = len(selected_df["benchmark"].unique().tolist())

    all_input_cols = copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS)
    all_input_cols = sorted(list([col for col in all_input_cols if col in selected_df]))

    # bench_cols = copy.deepcopy(benchmarks.BENCH_TARGET_INDEX_COLS)
    bench_input_cols = (
        []
        if all_benchmarks
        else copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[bench_name])
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
    metric_cols = sorted(metric_cols)
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
    serial_deterministic_grouped = serial.groupby(
        deterministic_group_cols, dropna=False
    )
    # serial_deterministic_grouped[serial.columns].apply(_inspect_deterministic_metrics)
    unique_simulation_metrics = serial_deterministic_grouped[metric_cols].nunique()
    assert (unique_simulation_metrics == 1).all(axis=1).all()

    # parallel
    parallel = selected_df[~selected_df["input_mode"].isin([np.nan, "serial"])]
    assert "total_cores" in serial
    assert "total_cores" in parallel

    if verbose:
        print("serial size", serial.shape)
        print("parallel size", parallel.shape)

    # those are fully distinct
    serial_input_ids = sorted(serial["input_id"].unique().tolist())
    parallel_input_ids = sorted(parallel["input_id"].unique().tolist())
    
    if verbose:
        print("{:>3} serial input ids".format(len(serial_input_ids), serial_input_ids))
        print("{:>3} parallel input ids".format(len(parallel_input_ids), parallel_input_ids))

    if len(serial_input_ids) == 0:
        raise ValueError("have zero serial benchmark configurations")
    if len(parallel_input_ids) == 0:
        raise ValueError("have zero parallel benchmark configurations")


    deterministic = parallel[parallel["input_mode"] == "deterministic"]
    assert len(deterministic) > 0
    unique_simulation_metrics = deterministic.groupby(
        deterministic_group_cols,
        dropna=False,
    )[metric_cols].nunique()

    config_with_identical_results = (unique_simulation_metrics == 1).all(axis=1)
    if not config_with_identical_results.all():
        bad_configs = unique_simulation_metrics[
            ~config_with_identical_results
        ].reset_index()
        # print(bad_configs.T)
        bad = deterministic.merge(
            bad_configs,
            on=deterministic_group_cols,
            how="inner",
            suffixes=("", "_nunique"),
        )
        # print(bad.T)
        print(bad[deterministic_group_cols + ["run"] + metric_cols].T)

    assert (
        config_with_identical_results.all()
    ), "deterministic configuration results differ for different runs, which makes them rather nondeterministic"

    # non deterministic without interleaving is also deterministic actually
    nondeterministic = parallel[parallel["input_mode"] == "nondeterministic"]
    # unique_simulation_metrics = nondeterministic.groupby(
    #     deterministic_group_cols, dropna=False
    # )[metric_cols].nunique()
    assert len(nondeterministic) > 0

    input_id_partitoning = set(serial["input_id"].unique()).intersection(
        set(parallel["input_id"].unique())
    )
    if len(input_id_partitoning) > 0:
        print(color("serial and parallel input ids intersect ", fg="red"))
        for input_id in input_id_partitoning:
            input_preview_cols = list(
                ["input_id"]
                + benchmarks.BENCH_TARGET_INDEX_COLS
                + ["kernel_launch_id"]
                + bench_input_cols
                + benchmarks.SIMULATE_INPUT_COLS
            )

            print("serial with input id", input_id)
            print(serial.loc[serial["input_id"] == input_id, input_preview_cols])
            print("parallel input", input_id)
            print(parallel.loc[parallel["input_id"] == input_id, input_preview_cols])
            break
        assert (
            len(input_id_partitoning) == 0
        ), "serial and parallel inputs intersect, this is generally solved by regenerating the aggregated csv stats"

    # join based on input_cols, NOT based on mode
    join_cols = list(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + ["kernel_name", "kernel_launch_id", "run"]
        + (
            list(
                copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS) - set(["input_mode"])
            )
            if all_benchmarks
            else copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[bench_name])
        )
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
    )
    if verbose:
        print("JOIN COLS:")
        pprint(join_cols)

    pre_join_preview_cols = ["benchmark", "kernel_name", "kernel_launch_id", "run"]
    serial_indices = serial[pre_join_preview_cols].drop_duplicates(ignore_index=True)
    parallel_indices = parallel[pre_join_preview_cols].drop_duplicates(
        ignore_index=True
    )
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
    if verbose:
        print(
            "joined={} parallel={} serial={}".format(
                joined.shape, parallel.shape, serial.shape
            )
        )

    # test_df = joined
    # test_df = serial
    # test = test_df["target"] == Target.Simulate.value
    # test &= test_df["benchmark"] == "vectorAdd"
    # test &= test_df["input_id"] == 1
    # # test &= joined["kernel_name"] == "vecAdd"
    # # test &= joined["kernel_launch_id"] == 0
    # # test &= joined["run"] == 1
    # # test &= joined["input_memory_only"] == False
    # # test &= joined["input_num_clusters"] == 56
    # # test &= joined["input_cores_per_cluster"] == 1
    # pprint(list(test_df.columns.tolist()))
    # print(test_df.loc[test, join_cols])

    if verbose:
        print(
            "post join serial input ids",
            sorted(joined["input_id_serial"].unique().tolist()),
        )

    assert joined.shape[0] == parallel.shape[0]
    assert "mean_blocks_per_sm_parallel" in joined
    assert "total_cores_parallel" in joined
    assert "cores_per_cluster_parallel" in joined

    # this does no longer hold, since for parallel we currently do not run
    # memory only, so there are some serial input ids that cannot be compared
    # to parallel input ids.
    # assert set(joined["input_id_serial"].values) == set(serial["input_id"].values)

    if len(joined) == 0:
        raise ValueError("joined parallel and serial dataframe is empty")

    if large:
        joined = joined[joined["mean_blocks_per_sm_parallel"] > 1.0]

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
        # + ["input_id_serial"]
        + ["input_id_serial", "input_id_parallel"]
        + bench_input_cols
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [col + "_parallel" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + [col + "_serial" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
    )
    if verbose:
        print("GROUP COLS:")
        pprint(group_cols)
    # assert "input_id" not in group_cols
    # assert "input_id_serial" not in group_cols

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
    # print("AGGREGATIONS:")
    # pprint(aggregations)

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
        configs = df[["input_id_parallel", "input_id_serial"]].drop_duplicates()
        if not len(configs) == 1:
            print("WARN", configs)
        # assert len(configs) == 1

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
        # print(df[["benchmark", "target", "input_id_serial", "input_id_parallel", "run", "mean_blocks_per_sm_parallel", "exec_time_sec_serial", "exec_time_sec_parallel"]])
        return metrics.speedup(
            baseline=exec_time_sec_serial, values=exec_time_sec_parallel
        ).mean()

    if True:
        # exec time speedup
        aggregated["exec_time_sec_speedup"] = grouped[joined.columns].apply(compute_speedup)

        # cycles error
        aggregated["cycles_mape"] = grouped[joined.columns].apply(
            lambda df: metrics.mape(
                true_values=df["cycles_serial"], values=df["cycles_parallel"]
            )
        )

        # l1 hit rate error
        aggregated["l1_hit_rate_mae"] = grouped[joined.columns].apply(
            lambda df: metrics.abs_err(
                true_values=df["l1_hit_rate_serial"], values=df["l1_hit_rate_parallel"]
            ).mean()
        )

        # l2 hit rate error
        aggregated["l2_hit_rate_mae"] = grouped[joined.columns].apply(
            lambda df: metrics.abs_err(
                true_values=df["l2_hit_rate_serial"], values=df["l2_hit_rate_parallel"]
            ).mean()
        )

        # dram reads error
        aggregated["dram_reads_smape"] = grouped[joined.columns].apply(
            lambda df: metrics.smape(
                true_values=df["dram_reads_serial"], values=df["dram_reads_parallel"]
            )  # .mean()
        )

        # dram writes error
        aggregated["dram_writes_smape"] = grouped[joined.columns].apply(
            lambda df: metrics.smape(
                true_values=df["dram_writes_serial"], values=df["dram_writes_parallel"]
            )  # .mean()
        )

    else:
        # exec time speedup
        aggregated["exec_time_sec_speedup"] = metrics.speedup(
            baseline=aggregated["exec_time_sec_serial"],
            values=aggregated[
                ["exec_time_sec_serial", "exec_time_sec_parallel"]
            ].min(axis=1))

        # cycles error
        aggregated["cycles_mape"] = metrics.mape(
            true_values=aggregated["cycles_serial"], values=aggregated["cycles_parallel"]
        )

        # l1 hit rate error
        aggregated["l1_hit_rate_mae"] = metrics.abs_err(
            true_values=aggregated["l1_hit_rate_serial"],
            values=aggregated["l1_hit_rate_parallel"]
        )

    
        # l2 hit rate error
        aggregated["l2_hit_rate_mae"] = metrics.abs_err(
            true_values=aggregated["l2_hit_rate_serial"], values=aggregated["l2_hit_rate_parallel"]
        )

    
        # dram reads error
        aggregated["dram_reads_smape"] = metrics.smape(
            true_values=aggregated["dram_reads_serial"], values=aggregated["dram_reads_parallel"]
        )

    
        # dram writes error
        aggregated["dram_writes_smape"] = metrics.smape(
            true_values=aggregated["dram_writes_serial"], values=aggregated["dram_writes_parallel"]
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
    # print(
    #     aggregated.loc[
    #         # 500_000 vectoradd
    #         aggregated["input_id_serial"] == 210.0,
    #         preview_cols
    #         + [
    #             "cycles_mape",
    #             "dram_reads_smape",
    #             "dram_writes_smape",
    #             "exec_time_sec_speedup",
    #         ],
    #     ][0:4].T.drop_duplicates()
    # )

    # build the table data
    assert 8 * benchmarks.BASELINE["num_clusters"] == 224

    functional_configs: typing.Sequence[typing.Dict[str, typing.Any]] = [
        dict(
            input_memory_only=False,
            input_num_clusters=benchmarks.BASELINE["num_clusters"],
            input_cores_per_cluster=1,
        ),
    ]
    if scale_clusters:
        functional_configs += [
            dict(
                input_memory_only=False,
                input_num_clusters=4 * benchmarks.BASELINE["num_clusters"],
                input_cores_per_cluster=1,
            )
        ]
    else:
        functional_configs += [
            dict(
                input_memory_only=False,
                input_num_clusters=benchmarks.BASELINE["num_clusters"],
                input_cores_per_cluster=4,
            )
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
            dict(
                name="babelstream",
                inputs={
                    **{"input_size": 102400},
                    **functional_config,
                },
            ),
            dict(
                name="transpose",
                inputs={
                    # **{"input_variant": "naive", "input_dim": 512},
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

    
    table = ""

    # absolute_exec_time = not all_benchmarks

    if all_benchmarks:
        for functional_config in functional_configs:
            mask_cols = list(functional_config.keys())
            mask_values = list(functional_config.values())
            mask = (aggregated[mask_cols] == mask_values).all(axis=1)

            # print(aggregated.loc[mask, list(
            #     ["benchmark", "input_id_serial", "input_id_parallel"]
            #     + ["mean_blocks_per_sm_serial", "mean_blocks_per_sm_parallel"]
            #     + ["exec_time_sec_serial", "exec_time_sec_parallel", "exec_time_sec_speedup"]
            #     )])

            # return

            total_cores = int(aggregated.loc[mask, "total_cores_parallel"].values[0])

            num_unique_bench_configs = len(aggregated.loc[mask, ["benchmark", "input_id_serial"]].drop_duplicates())
            label = "Average ({} benchmark configurations) @ {} SM's".format(
                num_unique_bench_configs, total_cores
            )
            if large:
                label += " [blocks/SM > 1]"
                # label += " [blocks/SM > 1, {} benchmarks]".format(num_unique_bench_configs)

            table += "%\n%\n"
            table += (
                r"\rowcolor{gray!10} \multicolumn{6}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            print("=> functional config: {}".format(functional_config))

            # num_bench_configs = num_benchmarks  # todo
            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask],
                # num_bench_configs=num_bench_configs,
                verbose=verbose,
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
                if verbose:
                    print(row.metric, bold_values, row.values())
                table += write_table_row(row, bold_values)

    else:
        for bench_config in selected_benchmarks:
            bench_inputs: typing.Dict[str, typing.Any] = bench_config["inputs"]
            if not all(aggregated["benchmark"] == bench_config["name"]):
                # print(
                #     "SKIP: want {} (have {})".format(
                #         aggregated["benchmark"][0], bench_config["name"]
                #     )
                # )
                continue

            print("")
            print(
                color("==> {} {}".format(bench_config["name"], bench_inputs), fg="cyan")
            )

            mask_cols = ["benchmark"] + list(bench_inputs.keys())
            mask_values = [bench_name] + list(bench_inputs.values())

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
            label = str(
                compute_table_row_label(bench_config, aggregated.loc[mask].iloc[0])
            )
            table += (
                r"\rowcolor{gray!10} \multicolumn{6}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            # print(aggregated.loc[mask, list(
            #     ["benchmark", "input_id_serial", "input_id_parallel"]
            #     + ["mean_blocks_per_sm_serial", "mean_blocks_per_sm_parallel"]
            #     + ["exec_time_sec_serial", "exec_time_sec_parallel", "exec_time_sec_speedup"]
            #     )])


            # assert len(aggregated.loc[mask, ["target", "benchmark", "input_id_serial"]].drop_duplicates()) == 1
            num_unique_bench_configs = len(aggregated.loc[mask, ["benchmark", "input_id_serial"]].drop_duplicates())
            assert num_unique_bench_configs == 1
            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask],
                # num_bench_configs=1,  # all_benchmarks=False
                verbose=verbose,
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
                if verbose:
                    print(
                        "writing table row {:<30} values={} bold={}".format(
                            row.metric, row.values(), bold_values
                        )
                    )
                table += write_table_row(row, bold_values)

        # add averaged row
        for functional_config in functional_configs:
            mask_cols = list(functional_config.keys())
            mask_values = list(functional_config.values())
            mask = (aggregated[mask_cols] == mask_values).all(axis=1)

            total_cores = int(aggregated.loc[mask, "total_cores_parallel"].values[0])

            if verbose:
                print(
                    color(
                        "==> AVERAGE for {:<4} SM's {}".format(
                            total_cores, functional_config
                        ),
                        fg="cyan",
                    )
                )

            num_unique_bench_configs = len(aggregated.loc[mask, ["benchmark", "input_id_serial"]].drop_duplicates())
            if num_unique_bench_configs == 1:
                # does not make sense to average this, we have this in the
                # previous section
                continue
            # label = "Average @ {} SM's".format(total_cores)
            # unique_bench_names = [bench_config["name"] for bench_config in selected_benchmarks]
            unique_bench_names = sorted(
                [
                    benchmarks.benchmark_name_human_readable(name)
                    for name in aggregated.loc[mask, "benchmark"].unique()
                ]
            )
            label = "Average {} ({} configurations) @ {} SM's".format(
                ", ".join(unique_bench_names),
                num_unique_bench_configs,
                total_cores,
            )
            if large:
                label += " [blocks/SM > 1]"
                # label += " [blocks/SM > 1, {} benchmarks]".format(num_unique_bench_configs)

            assert "_" not in label

            table += "%\n%\n"
            table += (
                r"\rowcolor{gray!10} \multicolumn{6}{c}{\textbf{"
                + label
                + r"}} \\ \hline"
                + "\n"
            )

            # assert num_benchmarks == 1
            # num_configs = len(aggregated.loc[mask, all_input_cols].drop_duplicates())
            table_rows: typing.Sequence[ParallelTableRow] = build_parallel_table_rows(
                aggregated[mask],
                # num_bench_configs=num_configs,  # all_benchmarks=True
                verbose=verbose,
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

                # print(row.metric, bold_values, row.values())
                if verbose:
                    print(
                        "writing table row {:<30} values={} bold={}".format(
                            row.metric, row.values(), bold_values
                        )
                    )
                table += write_table_row(row, bold_values)

    clipboard_table = r"""
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{\textwidth}{zs|s|z|zz}
& & \multicolumn{1}{c|}{Serial} & \multicolumn{1}{c|}{Deterministic} & \multicolumn{2}{c}{Nondeterministic} \\
& & & & \multicolumn{1}{c}{$n=5$} & \multicolumn{1}{c}{$n=10$} \\ \hline
"""

    clipboard_table += table
    clipboard_table += r"""
\end{tabularx}}
\end{table}
    """

    if not batch:
        print(clipboard_table)
        utils.copy_to_clipboard(clipboard_table)
        print("copied table to clipboard")


    caption = r"Average relative speedup and percentage error for serial and parallel simulation using \textsc{gpucachesim} on selected simulation output metrics using $t$ threads."

    tex_code = r"""
\documentclass[preview]{standalone}
"""
    tex_code += utils.TEX_PACKAGES
    tex_code += r"""
\begin{document}
"""

    tex_code += r"""
\begin{table}[tbh]
\fontsize{8}{10}\selectfont
\footnotesize"""
    tex_code += r"\caption{\small " + caption + r"}"
    tex_code += r"""
\centering
% \setlength\extrarowheight{2pt}
% \rowcolors{2}{white}{gray!20}
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{\textwidth}{zs|s|z|zz}
& 
& \multicolumn{1}{c|}{Serial}
& \multicolumn{1}{c|}{Deterministic} 
& \multicolumn{2}{c}{Nondeterministic} \\
& & & & \multicolumn{1}{c}{$n=5$} & \multicolumn{1}{c}{$n=10$} \\ \hline
"""
    tex_code += table
    tex_code += r"""
\end{tabularx}}
\end{table}
"""
    tex_code += r"""
\end{document}
"""

    filename = "parallel_table"
    if all_benchmarks:
        filename += "_all"
    else:
        filename += "_{}".format(bench_name)
    if scale_clusters:
        filename += "_scaled_clusters"
    if large:
        filename += "_large"
    pdf_output_path = (plot.TABLE_DIR / filename).with_suffix(".pdf")
    try:
        utils.render_latex(tex_code, output_path=pdf_output_path)
    except Exception as e:
        print(tex_code)
        raise e
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))

    if png:
        png_output_path = (plot.TABLE_DIR / "png" / filename).with_suffix(".png")
        utils.convert_to_png(input_path=pdf_output_path, output_path=png_output_path)
        print(color("wrote {}".format(png_output_path), fg="cyan"))
