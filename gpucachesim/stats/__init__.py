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

import gpucachesim.stats.stats as stats
import gpucachesim.stats.native as native
import gpucachesim.stats.accelsim as accelsim
import gpucachesim.stats.playground as playground
import gpucachesim.benchmarks as benchmarks
from gpucachesim.stats.human import human_readable
import gpucachesim.plot as plot
import gpucachesim.utils as utils

from gpucachesim.benchmarks import Target, Benchmarks, GPUConfig, REPO_ROOT_DIR


# suppress scientific notation by setting float_format
# pd.options.display.float_format = "{:.3f}".format
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.seterr(all="raise")

DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.config.yml"


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass

def aggregate_benchmark_results(
    sim_df: pd.DataFrame,
    bench_name: str,
    targets=None,
    mode="serial",
    memory_only=False,
    cores_per_cluster=1,
    num_clusters=20,
) -> pd.DataFrame:
    """View results for a benchmark"""

    selected_df = sim_df.copy()
    selected_df = selected_df[selected_df["benchmark"] == bench_name]
    # print(selected_df)
    # only compare serial gpucachesim
    # selected_df = selected_df[selected_df["input_mode"] != "nondeterministic"]

    for col in benchmarks.SIMULATE_INPUT_COLS:
        if col not in selected_df:
            selected_df[col] = np.nan

    non_gpucachesim = selected_df["input_mode"].isnull()
    print(selected_df[non_gpucachesim]["target"].unique().tolist())

    serial_gpucachesim = selected_df["input_mode"] == mode
    compute_gpucachesim = selected_df["input_memory_only"] == memory_only
    gtx1080_gpucachesim = selected_df["input_cores_per_cluster"] == cores_per_cluster
    gtx1080_gpucachesim &= selected_df["input_num_clusters"] == num_clusters
    gold_gpucachesim = serial_gpucachesim & compute_gpucachesim & gtx1080_gpucachesim
    print(
        "gpucachesim gold input ids:",
        sorted(selected_df.loc[gold_gpucachesim, "input_id"].unique().tolist()),
    )

    # only keep gold gpucachesim and other targets
    # selected_df = selected_df[gold_gpucachesim ^ non_gpucachesim]
    # kernels = selected_df[non_gpucachesim][["kernel_name_mangled", "kernel_name"]].drop_duplicates()
    # print(kernels)
    #
    print(
        selected_df[gold_gpucachesim][
            ["kernel_name_mangled", "kernel_name"]
        ].drop_duplicates()
    )
    # kernels = selected_df[gold_gpucachesim][["kernel_name_mangled", "kernel_name"]].drop_duplicates()
    kernels = selected_df[gold_gpucachesim]["kernel_name"].unique().tolist()
    print(kernels)
    # print(selected_df[gold_gpucachesim][["target", "benchmark", "input_id", "kernel_name_mangled", "cycles"]])
    # print(
    #     selected_df[non_gpucachesim][
    #         ["target", "kernel_name_mangled", "kernel_name", "kernel_launch_id"]
    #     ].drop_duplicates()
    # )

    no_kernel = selected_df["kernel_name"].isna() ^ (selected_df["kernel_name"] == "")
    valid_kernel = selected_df["kernel_name"].isin(kernels)
    selected_df = selected_df[
        (gold_gpucachesim ^ non_gpucachesim) & (valid_kernel ^ no_kernel)
    ]

    if isinstance(targets, list):
        selected_df = selected_df[selected_df["target"].isin(targets)]

    # restrict inputs to fit screen
    assert (selected_df["benchmark"] == bench_name).all()
    if bench_name == "simple_matrixmul":
        # m: [32, 64, 128]
        # n: [32, 64, 128]
        # p: [32, 64, 128]

        subset = pd.DataFrame.from_records(
            [
                (32, 32, 32),
                (128, 128, 128),
                (32, 64, 128),
                (128, 32, 32),
                (32, 1024, 32),
                (32, 2048, 32),
                (32, 4096, 32),
            ],
            columns=["input_m", "input_n", "input_p"]
        )
        # print(subset.index)
        # print(selected_df.index)
        selected_df = selected_df.merge(subset, how="inner")
        # selected_df = selected_df.merge(subset, on=["input_m", "input_n", "input_p"], how="inner")
        # selected_df = selected_df.join(subset, on=["input_m", "input_n", "input_p"], how="left")
        # subset = [(32, 32, 32), (128, 128, 128)]
        # pprint([c for c in selected_df.columns if "input_" in c])
        # print(selected_df[["input_m", "input_n", "input_p"]])
        # selected_df = selected_df[selected_df[["input_m", "input_n", "input_p"]].isin(subset)]

    input_cols = benchmarks.BENCHMARK_INPUT_COLS[bench_name]
    # print(selected_df[input_cols].drop_duplicates())

    # INDEX_COLS = [
    #     "kernel_name",
    #     "kernel_name_mangled",
    #     "kernel_launch_id",
    #     "run",
    # ]

    # aggregate over all kernel launch ids in a single run
    # grouped = selected_df.groupby(["kernel_name", "run"], dropna=False)

    profile_df = selected_df[selected_df["target"] == "Profile"]
    if False:
        print(
            profile_df[
                benchmarks.BENCH_TARGET_INDEX_COLS
                + input_cols
                + ["l2_hits", "l2_misses", "l2_hit_rate"]
                + ["l1_hits", "l1_misses", "l1_hit_rate"]
            ]
        )

    group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + ["kernel_name", "run"] + input_cols

    # print(selected_df.index)
    # non_numeric_cols = sorted(selected_df.select_dtypes(include=["object"]).columns.tolist())
    # print(sorted(set(non_numeric_cols) - set(group_cols)))

    pprint(group_cols)
    aggregations = {
        **{c: "mean" for c in set(selected_df.columns)},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {
        col: agg for col, agg in aggregations.items()
        if col in selected_df and col not in group_cols
    }
    pprint(aggregations)

    # print(sorted(selected_df.columns.tolist()))
    per_kernel = selected_df.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    # print(sorted(per_kernel.columns.tolist()))
    # selected_df.groupby(group_cols, dropna=False)[STAT_COLS].mean().reset_index()

    # per_kernel["label"] = per_kernel.apply(compute_label, axis=1)
    # per_kernel["target_name"] = per_kernel["target"].apply(compute_target_name)
    # print(per_kernel)

    group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + input_cols
    grouped = per_kernel.groupby(group_cols, dropna=False)
    aggregations = {
        **{c: "mean" for c in set(per_kernel.columns)},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {
        col: agg for col, agg in aggregations.items()
        if col in per_kernel and not col in group_cols
    }
    pprint(aggregations)

    per_target = grouped.agg(aggregations).reset_index()
    # print(selected_df[INDEX_COLS + input_cols + ["dram_writes", "l2_accesses"]].head(n=200))
    # print(grouped["dram_writes"].sum())
    # averaged = grouped.mean().reset_index()
    # averaged = grouped[STAT_COLS].mean().reset_index()
    # print(per_kernel)
    # averaged = grouped[STAT_COLS + input_cols].mean().reset_index()
    # print(averaged)
    # print(averaged.drop_duplicates())

    # stat_cols = set(averaged.columns) - set(["benchmark"]) - set(input_cols)
    per_target_pivoted = per_target.pivot(
        index=["benchmark"] + input_cols, columns="target", # values=STAT_COLS
    )
    # per_target = averaged.set_index(["target", "benchmark"] + input_cols)
    return per_kernel, per_target_pivoted


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
def parallel_plot(path, bench_name, nvprof):
    # load the materialized benchmark config
    profiler = "nvprof" if nvprof else "nsight"
    if bench_name is None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.csv".format(profiler)
    else:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
            profiler, bench_name
        )

    selected_df = pd.read_csv(stats_file, header=0)
    selected_df = selected_df[selected_df["benchmark"] == bench_name]
    # print(selected_df)

    # selected_df["parallelization_method"] = "serial" # to be overridden

    if not (selected_df["is_release_build"] == True).all():
        print(color("WARNING: non release results:", fg="red"))
        non_release_results = selected_df[selected_df["is_release_build"] == True]
        grouped = non_release_results.groupby(["benchmark", "target"])
        print(grouped["input_id"].count())

    bench_cols = ["target", "benchmark"]
    # 'input_mode', 'input_threads', 'input_run_ahead', 'input_memory_only', 'input_num_clusters', 'input_cores_per_cluster'
    # input_cols = ["input_dtype", "input_length", "input_memory_only", "input_cores_per_cluster"]
    bench_input_cols = benchmarks.BENCHMARK_INPUT_COLS[bench_name]
    input_cols = benchmarks.SIMULATE_INPUT_COLS
    print(bench_input_cols)
    print(input_cols)

    # get serial
    serial = selected_df[selected_df["input_mode"] == "serial"].copy()

    # we are joining all parallel configs with their serial variant
    # however, we do not assume equal number of repetitions necessarily,
    # hence we compute the mean.
    # Note that repetitions also only have a very minimal influence on serial execution time,
    # since serial execution is deterministic
    group_cols = bench_cols + input_cols + bench_input_cols
    # print(group_cols)

    # non_numeric_cols = sorted(serial.select_dtypes(include=["object"]).columns.tolist())
    # print(sorted(set(non_numeric_cols) - set(group_cols)))

    aggregations = {
        **{c: "mean" for c in set(serial.columns) - set(group_cols)},
        **benchmarks.NON_NUMERIC_COLS,
    }
    aggregations = {col: agg for col, agg in aggregations.items() if col in serial}
    mean_serial = serial.groupby(group_cols).agg(aggregations).reset_index()

    metric_cols = ["cycles", "exec_time_sec", "l2_hit_rate", "l1_hit_rate"]

    if False:
        print("before", serial.shape)
        print("averaged", mean_serial.shape)

        print("before:")
        print(
            serial.sort_values(input_cols + bench_input_cols)[
                input_cols + bench_input_cols + metric_cols
            ]
        )
        print("averaged:")
        print(
            mean_serial.sort_values(input_cols + bench_input_cols)[
                input_cols + bench_input_cols + metric_cols
            ]
        )

    # assert mean_serial.shape == serial.shape, "this does not hold when we have repetitions"
    serial = mean_serial

    # print(mean_serial[input_cols + bench_input_cols + metric_cols])

    # get parallel
    # print(selected_df["input_mode"].unique())
    # parallel = selected_df[selected_df["input_mode"] == "nondeterministic"]
    parallel = selected_df[~selected_df["input_mode"].isin([np.nan, "serial"])]
    # if False:
    #     # do NOT average parallel here, because then we cannot compute stddev etc.
    #     mean_parallel = parallel.groupby(group_cols).agg({
    #         **{c: "mean" for c in set(parallel.columns) - set(group_cols)},
    #         **{
    #            # 'Host Name', 'Process Name', 'device', 'is_release_build', 'kernel_function_signature', 'kernel_name', 'kernel_name_mangled', 'parallelization_method', 'target'
    #            'Host Name': "first",
    #            'Process Name': "first",
    #            'device': "first",
    #            'is_release_build': "first",
    #            'kernel_function_signature': "first",
    #            'kernel_name': "first",
    #            'kernel_name_mangled': "first",
    #            # 'parallelization_method': "first",
    #            # 'target': "first",
    #            # 'input_mode': "first",
    #     }}).reset_index()
    #
    #     parallel = mean_parallel

    # print(parallel)

    # parallelism methods
    # - deterministic
    # - nondeterministic (n=5)
    # - nondeterministic (n=10)
    # - nondeterministic interleaved (n=5)
    # - nondeterministic interleaved (n=10)

    # parallel.loc[parallel["input_mode"] == "deterministic", "parallelization_method"] = "deterministic"
    # parallel.loc[parallel["input_mode"] == "deterministic", "parallelization_method"] != "serial"

    # print((parallel["input_id"] == 3).sum())
    # print(parallel[parallel["input_id"] == 41][["cycles", "l2_reads", "dram_accesses"]])
    print("serial size", serial.shape)
    print("parallel size", parallel.shape)

    # those are fully distinct
    print("serial input ids", len(serial["input_id"].unique()))
    print("parallel input ids", len(parallel["input_id"].unique()))
    input_id_partitoning = set(serial["input_id"].unique()).intersection(
        set(parallel["input_id"].unique())
    )
    if len(input_id_partitoning) > 0:
        for input_id in input_id_partitoning:
            print("serial input", input_id)
            print(
                serial.loc[
                    serial["input_id"] == input_id,
                    bench_cols + bench_input_cols + benchmarks.SIMULATE_INPUT_COLS,
                ]
            )
            print("parallel input", input_id)
            print(
                parallel.loc[
                    parallel["input_id"] == input_id,
                    bench_cols + bench_input_cols + benchmarks.SIMULATE_INPUT_COLS,
                ]
            )
            break
        assert len(input_id_partitoning) == 0

    # join based on input_cols, NOT based on mode
    joined = parallel.merge(
        serial,
        on=bench_cols + bench_input_cols + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS,
        how="left",
        suffixes=("_parallel", "_serial"),
    )
    print(joined.shape)
    assert joined.shape[0] == parallel.shape[0]
    # assert joined.shape[0] == len(parallel["input_id"].unique())

    PREVIEW_COLS = sorted(
        bench_cols
        + bench_input_cols
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [c + "_parallel" for c in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + [c + "_parallel" for c in metric_cols]
        + [c + "_serial" for c in metric_cols]
        + ["input_id_serial", "input_id_parallel"]
    )

    if True:
        print(joined.loc[0:3, PREVIEW_COLS].T)

    # if False:
    #     # compute speedup
    #     joined["exec_time_sec_speedup"] = speedup(
    #         baseline=joined["exec_time_sec_serial"],
    #         values=joined["exec_time_sec_parallel"],
    #     )
    #
    #     # compute cycle error
    #     joined["cycles_rel_err"] = (joined["cycles_parallel"] - joined["cycles_serial"]).abs() / joined["cycles_serial"]
    #     joined["cycles_rmse"] = rmse(true_values=joined["cycles_serial"], values=joined["cycles_parallel"])
    #     joined["cycles_mae"] = mae(true_values=joined["cycles_serial"], values=joined["cycles_parallel"])

    # benchmark_values = sorted(joined["benchmark"].unique().tolist())
    # mode_values = sorted(joined["input_mode_parallel"].unique().tolist())
    # thread_values = sorted(joined["input_threads_parallel"].astype(int).unique().tolist())
    # run_ahead_values = sorted(joined["input_run_ahead_parallel"].astype(int).unique().tolist())
    # print(benchmark_values)
    # print(mode_values)
    # print(thread_values)
    # print(run_ahead_values)

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
    # print(joined[[
    #     "benchmark", "input_mode_parallel", "input_threads_parallel", "input_run_ahead_parallel", "cycles_rmse",
    # ]].drop_duplicates())

    # def dummy_first(_df):
    #     print("first of", _df, type(_df))
    #     return _df

    # def first(_df):
    #     print("=====")
    #     if isinstance(_df, pd.Series):
    #         return _df.first()
    #     return _df
    #     # print(type(_df))
    #     # print(_df)
    #     # # print(_df.columns)
    #     # print(_df.first())
    #     # return _df

    group_cols = sorted(
        bench_cols
        + bench_input_cols
        + benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [col + "_parallel" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
        + [col + "_serial" for col in benchmarks.SIMULATE_EXECUTION_CONFIG_COLS]
    )
    # table_df = joined.groupby(group_cols).apply(test_func)
    # .set_index(group_cols)
    # print(table_df.index)

    # joined.loc[:,list(set(joined.columns) - set(non_numeric.keys()))] = joined[:,list(set(joined.columns) - set(non_numeric.keys()))].astype(float)
    # pprint(sorted(table_df.columns.tolist()))
    # print(len(table_df))
    aggregations = {
        **{c: "mean" for c in sorted(set(joined.columns) - set(group_cols))},
        **{c + "_parallel": agg for c, agg in non_numeric_cols.items()},
        **{c + "_serial": agg for c, agg in non_numeric_cols.items()},
    }
    aggregations = {col: agg for col, agg in aggregations.items() if col in joined}
    # pprint(sorted(aggregations.items(), key=lambda x: (
    #     x[1] if isinstance(x[1], str) else x[1].__name__, x[0]
    # )))

    # non_numeric_cols = sorted(joined.select_dtypes(include=["object"]).columns.tolist())
    # print(joined[non_numeric_cols])
    # return

    if set(joined.columns.tolist()) - set(group_cols) != set(aggregations.keys()):
        pprint(
            (set(joined.columns.tolist()) - set(group_cols)).symmetric_difference(
                set(aggregations.keys())
            )
        )
        raise ValueError

    # table_df = []
    # for _, df in joined.groupby(group_cols, dropna=False):
    #     # assert all([c in df for c in aggregations.keys()])
    #     # df.mean()
    #     out = df.agg(aggregations)
    #     table_df.append(out)

    # pprint(sorted(set(non_numeric_cols) - (set(group_cols).union(set(aggregations.keys())))))
    # for col in non_numeric:
    #     test = joined.groupby(group_cols, dropna=False).agg({col + "_parallel": "first", col + "_serial": "first"})
    #     # print(test)
    #     # test = joined.groupby(group_cols, dropna=False).agg({)

    # def compute_per_config_metrics(_df):
    #     print("=====")
    #     # print(_df.T)
    #     # compute speedup
    #     per_config_metrics = _df.agg(aggregations, squeeze=False)
    #
    #     # print(list(_df["exec_time_sec_parallel"]))
    #     # print(_df["exec_time_sec_parallel"].groups)
    #
    #
    #
    #     if False:
    #         per_config_metrics["exec_time_sec_speedup"] = _df[["exec_time_sec_serial", "exec_time_sec_parallel"]].apply(compute_speedup)
    #
    #     speedups = _df[
    #         ["exec_time_sec_serial", "exec_time_sec_parallel"]].apply(compute_speedup)
    #     per_config_metrics = per_config_metrics.assign(exec_time_sec_speedup=speedups)
    #
    #     # speedup(
    #     #     baseline=_df["exec_time_sec_serial"].groups,
    #     #     values=_df["exec_time_sec_parallel"],
    #     # )
    #
    #     # _df["exec_time_sec_speedup"] = speedup(
    #     #     baseline=_df["exec_time_sec_serial"],
    #     #     values=_df["exec_time_sec_parallel"],
    #     # )
    #     #
    #     # # compute cycle error
    #     # _df["cycles_rel_err"] = (_df["cycles_parallel"] - _df["cycles_serial"]).abs() / _df["cycles_serial"]
    #     # _df["cycles_rmse"] = rmse(true_values=_df["cycles_serial"], values=_df["cycles_parallel"])
    #     # _df["cycles_mae"] = mae(true_values=_df["cycles_serial"], values=_df["cycles_parallel"])
    #     # return _df
    #     return per_config_metrics

    def speedup(baseline, values):
        return baseline / values

    def rel_err(true_values, values):
        rel_err = (values - true_values).abs() / true_values
        rel_err = rel_err.fillna(0.0)
        return rel_err

    def rmse(true_values, values):
        return ((values - true_values) ** 2).mean() ** 0.5

    def mae(true_values, values):
        return (true_values - values).abs().mean()

    # def compute_mean_speedup(_df):
    #     return speedup(baseline=_df["exec_time_sec_serial"], values=_df["exec_time_sec_parallel"]).mean()
    #
    # def compute_mean_cycles_rel_err(_df):
    #     return rel_err(true_values=_df["cycles_serial"], values=_df["cycles_parallel"])
    #     # return (_df["cycles_parallel"] - _df["cycles_serial"]).abs() / _df["cycles_serial"]
    #
    # def compute_mean_cycles_rmse(_df):
    #     return rmse(true_values=_df["cycles_serial"], values=_df["cycles_parallel"])
    #
    # def compute_mean_cycles_rmse(_df):
    #     return rmse(true_values=_df["cycles_serial"], values=_df["cycles_parallel"])

    print(joined.shape)
    grouped = joined.groupby(group_cols, dropna=False)
    aggregated = grouped.agg(aggregations, squeeze=False)

    if False:

        def inspect_func(df):
            if (df["input_mode_parallel"] == "deterministic").all():
                print(df[["input_mode_parallel", "cycles_serial", "cycles_parallel"]])
            return df

        grouped.apply(inspect_func)

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

    # aggregated["exec_time_sec_speedup"] = joined.groupby(group_cols, dropna=False)[
    #     ["exec_time_sec_serial", "exec_time_sec_parallel"]].apply(compute_speedup).mean()
    # aggregated["cycles_rel_err"] = joined.groupby(group_cols, dropna=False).apply(compute_cycles_rel_err).mean()
    # [["cycles_parallel", "cycles_serial"]].apply(compute_speedup).mean()
    # joined = joined.groupby(group_cols, dropna=False).apply(compute_per_config_metrics).reset_index()

    aggregated = aggregated.reset_index()
    print(aggregated.shape)
    # print(joined.T)
    # print(aggregated.loc[0:4, PREVIEW_COLS].T.drop_duplicates())
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
    functional_config = {
        "input_memory_only": False,
        "input_num_clusters": 20,
        "input_cores_per_cluster": 1,
        # "input_cores_per_cluster": 8,
    }
    selected_benchmarks: typing.Dict[str, typing.Any] = {
        "vectorAdd": dict(
            label="VectorAdd", inputs={"input_dtype": 32, "input_length": 500_000}
        ),
    }

    interleave_n = list(itertools.product([False, True], [5, 10]))

    tables = [(functional_config, selected_benchmarks)]
    for functional_config, selected_benchmarks in tables:
        print(functional_config, selected_benchmarks)

        table = ""

        assert set(functional_config.keys()) == set(benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS)

        assert all(
            [
                bench_name in benchmarks.BENCHMARK_INPUT_COLS.keys()
                for bench_name in selected_benchmarks.keys()
            ]
        )
        assert all(
            [
                set(benchmarks.BENCHMARK_INPUT_COLS[bench_name])
                == set(bench_config["inputs"].keys())
                for bench_name, bench_config in selected_benchmarks.items()
            ]
        )

        for bench_name, bench_config in selected_benchmarks.items():
            bench_inputs: typing.Dict[str, typing.Any] = bench_config["inputs"]
            mask_cols = (
                list(functional_config.keys())
                + ["benchmark"]
                + list(bench_inputs.keys())
            )
            mask_values = (
                list(functional_config.values())
                + [bench_name]
                + list(bench_inputs.values())
            )
            # mask = aggregated["benchmark"] == bench_name
            # for col, value in zip(mask_cols, mask_values):
            #     mask &= aggregated[col] == value
            # print((aggregated[mask_cols] == mask_values).sum(axis=0))
            mask = (aggregated[mask_cols] == mask_values).all(axis=1)
            table += (
                r"\rowcolor{gray!10} \multicolumn{8}{c}{"
                + str(bench_config["label"])
                + r"} \\ \hline"
                + "\n"
            )
            # print(mask)
            # print(mask.shape)
            # print(mask.sum())
            # print(aggregated.loc[
            #     mask, PREVIEW_COLS + ["cycles_rel_err", "exec_time_sec_speedup"]].T.drop_duplicates())

            # def write_table_row(threads, det_value, nondet_values, serial_value=None):
            def write_table_row(row):
                is_first_metric_row = row.threads == 4
                is_last_metric_row = row.threads == 8

                table_row = ""
                # metric name
                if is_first_metric_row:
                    table_row += (
                        r"\multirow{2}{*}{\shortstack[l]{" + str(row.metric) + r"}}"
                    )
                # threads
                table_row += r" & $t=" + str(row.threads) + r"$ "
                # serial value
                # if row.serial_value is not None:
                if is_first_metric_row:
                    table_row += (
                        r" & \multirow{2}{*}{\shortstack[l]{"
                        + str(row.serial_value)
                        + r"}} "
                    )
                else:
                    table_row += r" &  "
                # deterministic value
                table_row += r" & " + str(row.det_value)
                # nondeterministic value
                for nondet_value in row.nondet_values:
                    table_row += r" & " + str(nondet_value)
                table_row += r" \\ "
                if is_last_metric_row:
                    table_row += r" \hline "
                table_row += "\n"
                return table_row

            class TableRow(typing.NamedTuple):
                metric: str
                threads: int
                serial_value: typing.Union[float, int, str]
                det_value: typing.Union[float, int, str]
                nondet_values: typing.Sequence[typing.Union[float, int, str]]

            table_rows: typing.Sequence[TableRow] = []

            for threads in [4, 8]:
                threads_mask = aggregated["input_threads_parallel"] == threads
                det_mask = aggregated["input_mode_parallel"] == "deterministic"
                nondet_no_interleave_mask = (
                    aggregated["input_mode_parallel"] == "nondeterministic"
                )
                nondet_interleave_mask = (
                    aggregated["input_mode_parallel"] == "nondeterministic_interleave"
                )
                # print([m.sum() for m in [
                #     mask, threads_mask, det_mask, nondet_no_interleave_mask, nondet_interleave_mask
                # ]])

                det = aggregated[mask & threads_mask & det_mask]
                print(
                    det[
                        bench_input_cols
                        + [
                            "input_threads_parallel",
                            "exec_time_sec_parallel",
                            "input_id_parallel",
                            "input_id_serial",
                        ]
                    ]
                )
                print("===")
                assert len(det) == 1
                nondet_no_interleave = aggregated[
                    mask & threads_mask & nondet_no_interleave_mask
                ]
                assert len(nondet_no_interleave) == 2
                nondet_interleave = aggregated[
                    mask & threads_mask & nondet_interleave_mask
                ]
                assert len(nondet_interleave) == 2

                # print([len(df) for df in [det, nondet_no_interleave, nondet_interleave]])
                # assert all([len(df) > 0 for df in [det, nondet_no_interleave, nondet_interleave]])
                assert (
                    len(
                        aggregated.loc[
                            mask,
                            [
                                "exec_time_sec_serial",
                                "cycles_serial",
                                "input_id_serial",
                            ],
                        ].drop_duplicates()
                    )
                    == 1
                )

                # exec time (speedup)
                serial_exec_time = aggregated.loc[
                    mask & threads_mask, "exec_time_sec_serial"
                ].values[0]
                det_exec_time = det["exec_time_sec_parallel"].values[0]
                det_speedup = det["exec_time_sec_speedup"].values[0]
                nondet_values = []
                for interleave, n in interleave_n:
                    # print(interleave, n)
                    nondet = nondet_interleave if interleave else nondet_no_interleave
                    # print(nondet["input_run_ahead_parallel"])
                    # print(nondet["input_run_ahead_parallel"] == n)
                    # print(nondet.shape)
                    nondet = nondet[nondet["input_run_ahead_parallel"] == n]
                    nondet_exec_time = nondet["exec_time_sec_parallel"].values[0]
                    nondet_speedup = nondet["exec_time_sec_speedup"].values[0]
                    nondet_values.append(
                        "${:>3.1f}s~({:>1.1f}x)$".format(
                            nondet_exec_time, nondet_speedup
                        )
                    )

                table_rows.append(
                    TableRow(
                        metric="exec time",
                        threads=threads,
                        serial_value="${:>3.1f}s$".format(serial_exec_time),
                        det_value="${:>3.1f}s~({:1.1f}x)$".format(
                            det_exec_time, det_speedup
                        ),
                        nondet_values=nondet_values,
                    )
                )

                # cycles (rel err)
                serial_cycles = int(
                    aggregated.loc[mask & threads_mask, "cycles_serial"].values[0]
                )
                det_cycles = int(det["cycles_parallel"].values[0])
                det_rel_err = det["cycles_rel_err"].values[0]
                nondet_values = []
                for interleave, n in interleave_n:
                    nondet = nondet_interleave if interleave else nondet_no_interleave
                    nondet = nondet[nondet["input_run_ahead_parallel"] == n]

                    nondet_cycles = int(nondet["cycles_parallel"].values[0])
                    nondet_rel_err = nondet["cycles_rel_err"].values[0]
                    nondet_values.append(
                        "${:>5}~({:>2.1f}\\%)$".format(
                            nondet_cycles, 100.0 * nondet_rel_err
                        )
                    )

                table_rows.append(
                    TableRow(
                        metric="cycles",
                        threads=threads,
                        serial_value="${:>5}$".format(serial_cycles),
                        det_value="${:>5}~({:>2.1f}\\%)$".format(
                            det_cycles, 100.0 * det_rel_err
                        ),
                        nondet_values=nondet_values,
                    )
                )

                # l1 data hit rate (rel err)
                serial_l1_hit_rate = int(
                    aggregated.loc[mask & threads_mask, "l1_hit_rate_serial"].values[0]
                )
                det_l1_hit_rate = int(det["l1_hit_rate_parallel"].values[0])
                det_rel_err = det["l1_hit_rate_rel_err"].values[0]
                nondet_values = []
                for interleave, n in interleave_n:
                    nondet = nondet_interleave if interleave else nondet_no_interleave
                    nondet = nondet[nondet["input_run_ahead_parallel"] == n]

                    nondet_l1_hit_rate = int(nondet["l1_hit_rate_parallel"].values[0])
                    nondet_rel_err = nondet["l1_hit_rate_rel_err"].values[0]
                    nondet_values.append(
                        "${:>2.1f}\\%~({:>2.1f}\\%)$".format(
                            nondet_l1_hit_rate, 100.0 * nondet_rel_err
                        )
                    )

                table_rows.append(
                    TableRow(
                        metric=r"L1D\\hit rate",
                        threads=threads,
                        serial_value="${:>2.1f}\\%$".format(100.0 * serial_l1_hit_rate),
                        det_value="${:>2.1f}\\%~({:>2.1f}\\%)$".format(
                            det_l1_hit_rate, 100.0 * det_rel_err
                        ),
                        nondet_values=nondet_values,
                    )
                )

                # l2 data hit rate (rel err)
                serial_l2_hit_rate = int(
                    aggregated.loc[mask & threads_mask, "l2_hit_rate_serial"].values[0]
                )
                det_l2_hit_rate = int(det["l2_hit_rate_parallel"].values[0])
                det_rel_err = det["l2_hit_rate_rel_err"].values[0]
                nondet_values = []
                for interleave, n in interleave_n:
                    nondet = nondet_interleave if interleave else nondet_no_interleave
                    nondet = nondet[nondet["input_run_ahead_parallel"] == n]

                    nondet_l2_hit_rate = int(nondet["l2_hit_rate_parallel"].values[0])
                    nondet_rel_err = nondet["l2_hit_rate_rel_err"].values[0]
                    nondet_values.append(
                        "${:>2.1f}\\%~({:>2.1f}\\%)$".format(
                            nondet_l2_hit_rate, 100.0 * nondet_rel_err
                        )
                    )

                table_rows.append(
                    TableRow(
                        metric=r"L2D\\hit rate",
                        threads=threads,
                        serial_value="${:>2.1f}\\%$".format(100.0 * serial_l2_hit_rate),
                        det_value="${:>2.1f}\\%~({:>2.1f}\\%)$".format(
                            det_l2_hit_rate, 100.0 * det_rel_err
                        ),
                        nondet_values=nondet_values,
                    )
                )

                # dram reads (rel err)
                serial_dram_reads = int(
                    aggregated.loc[mask & threads_mask, "dram_reads_serial"].values[0]
                )
                det_dram_reads = int(det["dram_reads_parallel"].values[0])
                det_rel_err = det["dram_reads_rel_err"].values[0]
                nondet_values = []
                for interleave, n in interleave_n:
                    nondet = nondet_interleave if interleave else nondet_no_interleave
                    nondet = nondet[nondet["input_run_ahead_parallel"] == n]

                    nondet_dram_reads = int(nondet["dram_reads_parallel"].values[0])
                    nondet_rel_err = nondet["dram_reads_rel_err"].values[0]
                    nondet_values.append(
                        "${:>4}~({:>2.1f}\\%)$".format(
                            nondet_dram_reads, 100.0 * nondet_rel_err
                        )
                    )

                table_rows.append(
                    TableRow(
                        metric=r"DRAM\\reads",
                        threads=threads,
                        serial_value="${:>4}$".format(serial_dram_reads),
                        det_value="${:>4}~({:>2.1f}\\%)$".format(
                            det_dram_reads, 100.0 * det_rel_err
                        ),
                        nondet_values=nondet_values,
                    )
                )

                # dram writes (rel err)
                serial_dram_writes = int(
                    aggregated.loc[mask & threads_mask, "dram_writes_serial"].values[0]
                )
                det_dram_writes = int(det["dram_writes_parallel"].values[0])
                det_rel_err = det["dram_writes_rel_err"].values[0]
                nondet_values = []
                for interleave, n in interleave_n:
                    nondet = nondet_interleave if interleave else nondet_no_interleave
                    nondet = nondet[nondet["input_run_ahead_parallel"] == n]

                    nondet_dram_writes = int(nondet["dram_writes_parallel"].values[0])
                    nondet_rel_err = nondet["dram_writes_rel_err"].values[0]
                    nondet_values.append(
                        "${:>4}~({:>2.1f}\\%)$".format(
                            nondet_dram_writes, 100.0 * nondet_rel_err
                        )
                    )

                table_rows.append(
                    TableRow(
                        metric=r"DRAM\\writes",
                        threads=threads,
                        serial_value="${:>4}$".format(serial_dram_writes),
                        det_value="${:>4}~({:>2.1f}\\%)$".format(
                            det_dram_writes, 100.0 * det_rel_err
                        ),
                        nondet_values=nondet_values,
                    )
                )

                # serial_exec_time = aggregated.loc[mask & threads_mask, "exec_time_sec_serial"].values[0]
                # serial_cycles = aggregated.loc[mask & threads_mask, "cycles_serial"].values[0]
                #
                # # get deterministic value
                # assert len(deterministic) == 1
                # # print(deterministic_4["exec_time_sec_speedup"].values[0])
                # determ_speedup = deterministic["exec_time_sec_speedup"].values[0]
                # determ_exec_time = deterministic["exec_time_sec_parallel"].values[0]
                # determ_cycles = deterministic["cycles_parallel"].values[0]

                # exec time

            # for row_num, threads in enumerate([4, 8]):
            #     threads_mask = aggregated["input_threads_parallel"] == threads
            #     det_mask = aggregated["input_mode_parallel"] == "deterministic"
            #     nondet_mask = aggregated["input_mode_parallel"] == "nondeterministic"
            #     nondet_interleave_mask = aggregated["input_mode_parallel"] == "nondeterministic_interleaved"
            #
            #     # print([m.sum() for m in [mask, threads_4_mask, deterministic_mask]])
            #
            #     assert len(aggregated.loc[mask,
            #         ["exec_time_sec_serial", "cycles_serial", "input_id_serial"]].drop_duplicates()) == 1
            #
            #     serial_exec_time = aggregated.loc[mask & threads_mask, "exec_time_sec_serial"].values[0]
            #     serial_cycles = aggregated.loc[mask & threads_mask, "cycles_serial"].values[0]
            #
            #     # get deterministic value
            #     deterministic = aggregated[mask & threads_mask & determ_mask]
            #     assert len(deterministic) == 1
            #     # print(deterministic_4["exec_time_sec_speedup"].values[0])
            #     determ_speedup = deterministic["exec_time_sec_speedup"].values[0]
            #     determ_exec_time = deterministic["exec_time_sec_parallel"].values[0]
            #     determ_cycles = deterministic["cycles_parallel"].values[0]
            #     # deterministic_speedup = "{:1.1f}x".format(deterministic_4["exec_time_sec_speedup"].values[0])
            #     # print(deterministic_speedup)
            #
            #     # table += r" & $t=" + str(threads) + r"$ "
            #     # # serial value
            #     # if row_num == 0:
            #     #     table += r" & \multirow{2}{*}{" + "${:3.2f}s".format(serial_exec_time) + r"$} "
            #     # else:
            #     #     table += r" &  "
            #     # # deterministic value
            #     # table += r" & " + "${:3.2f}s~({:1.1f}x)".format(determ_exec_time, determ_speedup) + r"$ "
            #     # # nondeterministic value
            #     # table += r" & " + "${:3.2f}s".format(0) + r"$ "
            #     # table += r" & " + "${:3.2f}s".format(0) + r"$ "
            #     # table += r" & " + "${:3.2f}s".format(0) + r"$ "
            #     # table += r" & " + "${:3.2f}s".format(0) + r"$ "
            #     # table += r"\\ " + "\n"

            # 8 threads
            # table += r" & $t=" + str(8) + r"$ "
            # r" & & $10\%$ & $10\%$ & $10\%$ & $10\%$ & $10\%$ \\ \hline"

            # last_metric = None
            table_rows = sorted(table_rows, key=lambda row: (row.metric, row.threads))
            for row in table_rows:
                # if row.metric != last_metric:
                #     table += r"\multirow{2}{*}{" + row.metric + r"}"
                table += write_table_row(row)
                # last_metric = row.metric

        print(table)

def flatten(l):
    return [item for ll in l for item in ll]

@main.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name_arg", help="Benchmark name")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
def correlation_plots(path, bench_name_arg, nsight):
    profiler = "nsight" if nsight else "nvprof"
    stats = []
    if bench_name_arg is not None:
        stats.append(pd.read_csv(REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
            profiler, bench_name_arg
        )))
    else:
        b = Benchmarks(path)
        benches = flatten(list(b.benchmarks[Target.Profile.value].values()))
        bench_names = set([b["name"] for b in benches])
        for bench_name in bench_names:
            stats.append(pd.read_csv(REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
                profiler, bench_name
            )))
    stats = pd.concat(stats, ignore_index=False)
    stats = stats.sort_values(["benchmark", "target"])
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
            bench_df = bench_df.set_index(["target"] + benchmarks.SIMULATE_INPUT_COLS).sort_index()

            def gpucachesim_baseline(target, memory_only=False):
                # "input_mode", "input_threads", "input_run_ahead",
                # "input_memory_only", "input_num_clusters", "input_cores_per_cluster",
                return (target, "serial", 4, 5, memory_only, 20, 1)

            group_cols = bench_input_cols

            aggregations = {
                **{c: "mean" for c in set(bench_df.columns) - set(group_cols)},
                **benchmarks.NON_NUMERIC_COLS,
            }
            aggregations = {col: agg for col, agg in aggregations.items() if col in bench_df}

            native = bench_df.loc[Target.Profile.value]
            native = native.groupby(bench_input_cols).agg(aggregations)

            accelsim = bench_df.loc[Target.AccelsimSimulate.value]
            accelsim = accelsim.groupby(bench_input_cols).agg(aggregations)

            gpucachesim = bench_df.loc[gpucachesim_baseline(target=Target.Simulate.value, memory_only=False)]
            gpucachesim = gpucachesim.groupby(bench_input_cols).agg(aggregations)

            gpucachesim_memory_only = bench_df.loc[gpucachesim_baseline(Target.Simulate.value, memory_only=True)]
            gpucachesim_memory_only = gpucachesim_memory_only.groupby(bench_input_cols).agg(aggregations)

            gpucachesim_trace_reconstruction = bench_df.loc[Target.ExecDrivenSimulate.value]
            gpucachesim_trace_reconstruction = gpucachesim_trace_reconstruction.groupby(bench_input_cols).agg(aggregations)

            print("native                    ", native.shape)
            print("accelsim                  ", accelsim.shape)
            print("gpucachesim               ", gpucachesim.shape)
            print("gpucachesim (mem only)    ", gpucachesim_memory_only.shape)
            print("gpucachesim (exec driven) ", gpucachesim_trace_reconstruction.shape)

            
            targets = [
                (("native", "native", "o"), native),
                (("AccelSim", "accelsim", "o"), accelsim),
                (("gpucachesim", "gpucachesim", "o"), gpucachesim),
                (("gpucachesim (memory only)", "gpucachesim", "x"), gpucachesim_memory_only),
                (("gpucachesim (exec driven)", "gpucachesim", "D"), gpucachesim_trace_reconstruction),
            ]
            assert all([len(target_df) == len(targets[0][1]) for _, target_df in targets])
            
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
                    native[stat_col],
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
                stat_col_max = 10 ** log_stat_col_max
                log_stat_col_min = np.floor(np.log10(stat_col_min))
                stat_col_min = 10 ** log_stat_col_min
                tick_values = np.arange(log_stat_col_min, log_stat_col_max, 
                                        step=int(np.ceil(log_stat_col_max / 6)))
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

            tick_labels = [plot.human_format_thousands(v, round_to=0) for v in tick_values]
            
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
            bench_df = stats.set_index(["target"] + benchmarks.SIMULATE_INPUT_COLS).sort_index()

def stat_cols_for_profiler(profiler: str) -> typing.Sequence[str]:
    stat_cols = [
        # "num_blocks",
        "input_id",
        "exec_time_sec",
        "cycles",
        "instructions",
        # "dram_reads",
        # "dram_writes",
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
            # "l2_read_hits",
            # "l2_write_hits",
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
            # "l2_miss_rate",
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
    "l2_reads": StatConfig(
        **{
            **DEFAULT_STAT_CONFIG._asdict(),
            **dict(label=r"L2 reads", log_y_axis=True, percent=False),
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
# @click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--plot", "should_plot", type=bool, default=True, help="generate plots")
@click.option("--nsight", "nsight", type=bool, is_flag=True, help="use nsight")
@click.option("--memory-only", "mem_only", type=bool, is_flag=True, help="memory only")
@click.option("--verbose", "verbose", type=bool, is_flag=True, help="verbose output")
# @click.option("--input", "input_idx", type=int, help="Input index")
def view(path, bench_name, should_plot, nsight, mem_only, verbose):
    # load the materialized benchmark config
    profiler = "nsight" if nsight else "nvprof"
    if bench_name is None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.csv".format(profiler)
    else:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
            profiler, bench_name
        )

    sim_df = pd.read_csv(stats_file, header=0)
    stat_cols = stat_cols_for_profiler(profiler)

    per_kernel, per_target_pivoted = aggregate_benchmark_results(
        sim_df, bench_name, memory_only=mem_only
    )
    
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
                    int(df["input_dim"]), int(df["input_dim"]),
                )
            case "babelstream":
                label = "BabelStream\n"
                label += "{}".format(int(df["input_size"]))
            case other:
                label = str(other)

        return label

    def compute_target_name(name):
        assert isinstance(name, str)

        match name.lower():
            case "simulate":
                return "gpucachesim"
            case "accelsimsimulate":
                return "AccelSim"
            case "profile":
                return "Native"

    per_kernel["label"] = per_kernel.apply(compute_label, axis=1)
    per_kernel["target_name"] = per_kernel["target"].apply(compute_target_name)

    targets = sorted(per_kernel["target"].unique().tolist())
    pprint(targets)
    benchmarks = sorted(per_kernel["benchmark"].unique().tolist())
    pprint(benchmarks)
    benchmark_inputs = {
        benchmark: per_kernel[
            ["label"] + benchmarks.BENCHMARK_INPUT_COLS[benchmark]
        ].drop_duplicates()
        for benchmark in benchmarks
    }
    pprint(benchmark_inputs)

    targets = [
        target
        for target in targets
        if target in ["Profile", "Simulate", "AccelsimSimulate"]
    ]

    input_cols = benchmarks.BENCHMARK_INPUT_COLS[bench_name]
    group_cols = benchmarks.BENCH_TARGET_INDEX_COLS + input_cols
    per_target = per_kernel.set_index(group_cols).sort_index()

    print(" === {} === ".format(profiler))
    print(per_target_pivoted[stat_cols].T)

    if not should_plot:
        return

    
    if False:
        stat_cols = ["cycles", "dram_reads", "dram_writes"]

    for stat_col, benchmark in itertools.product(stat_cols, benchmarks):
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
            figsize=(1.0 * plot.DINA4_WIDTH_INCHES, 0.21 * plot.DINA4_HEIGHT_INCHES),
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

        target_configs = itertools.product(
            targets, benchmark_inputs[benchmark].iterrows()
        )

        for target, (inputs_idx, inputs) in target_configs:
            key = [target, benchmark] + [
                inputs[col] for col in benchmarks.BENCHMARK_INPUT_COLS[benchmark]
            ]
            target_df = per_target.loc[tuple(key), :]
            target_df = target_df.groupby(group_cols, dropna=False)

            target_idx = targets.index(target)
            idx = inputs_idx * group_width + (target_idx + 0.5) * (bar_width + spacing)

            target_name = target_df["target_name"].first().values[0]

            if verbose:
                print(
                    "{:>15} {:<10} {:>15} [{:<3}]  {:<35}  {:<3} {:<4} = {:<8.2f} {:<8.2f}".format(
                        benchmark,
                        stat_col,
                        target_name,
                        target_idx,
                        str(inputs[benchmarks.BENCHMARK_INPUT_COLS[benchmark]].tolist()),
                        inputs_idx,
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

            color = plot.plt_rgba(*plot.SIM_RGB_COLOR[target_name.lower()], 1.0)
            ax.bar(
                x,
                y,
                color=color,
                hatch=plot.SIM_HATCH[target_name.lower()] * 1,
                width=bar_width,
                linewidth=1,
                edgecolor="black",
                zorder=2,
                label=target_name if inputs_idx == 0 else None,
            )

            ax.errorbar(
                x,
                y,
                yerr=ystd,
                linewidth=1,
                # ecolor=plot.plt_lighten_color(color, amount=1.2),
                # ecolor="#fcc45f",
                # ecolor="#f84f36",
                ecolor="black",
                capsize=0.5 * bar_width,
                # marker='o', markersize=4,
                linestyle="-",
            )
            # hatch.color and hatch.linewidth

        ax.set_ylabel(ylabel)
        ax.axes.set_zorder(10)

        inputs = benchmark_inputs[benchmark]
        labels = inputs["label"].values
        key = [Target.Simulate.value, benchmark] + [
            inputs[col] for col in benchmarks.BENCHMARK_INPUT_COLS[benchmark]
        ]
        simulate_df = per_target.loc[tuple(key), :]
        num_blocks = simulate_df["num_blocks"].values
        assert len(labels) == len(num_blocks)
        xtick_labels = [
            "{}\n{} blocks".format(label, int(blocks))
            for label, blocks in zip(labels, num_blocks)
        ]
        xtick_values = np.arange(0, len(labels), dtype=np.float64)
        xtick_values *= group_width
        xtick_values += 0.5 * float((group_width - group_spacing))
        ax.set_xticks(xtick_values, xtick_labels, rotation=0)
        ax.set_xlim(0, len(xtick_labels) * group_width)

        all_values = per_target[per_target.index.get_level_values(1) == benchmark]
        ymax = all_values[stat_col].max()

        if stat_config.log_y_axis:
            assert not stat_config.percent
            ymax_log = np.ceil(np.log10(ymax))
            ytick_values = np.arange(0, ymax_log+1, step=int(np.ceil(ymax_log / 6)))
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

        ytick_labels = [plot.human_format_thousands(v, round_to=0) for v in ytick_values]
        ax.set_yticks(ytick_values, ytick_labels)

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1, 1),
            edgecolor="none", fancybox=False, shadow=False,
        )
        filename = plot.PLOT_DIR / "validation/{}.{}.{}.pdf".format(
            profiler, bench_name, stat_col
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename)


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option(
    "--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config"
)
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--input", "input_idx", type=int, help="Input index")
@click.option(
    "--limit", "limit", type=int, help="Limit number of benchmark configs generated"
)
@click.option(
    "--baseline",
    "--quick",
    "quick",
    type=bool,
    is_flag=True,
    help="Fast mode: only collect baseline benchmark configurations",
)
@click.option(
    "--target",
    "target",
    type=str,
    help="target",
)
@click.option("--verbose", "verbose", type=bool, is_flag=True, help="verbose output")
@click.option("--strict", "strict", type=bool, default=True, help="fail on missing results")
@click.option("--nvprof", "nvprof", type=bool, default=True, help="use nvprof")
@click.option("--nsight", "nsight", type=bool, default=False, help="use nsight")
@click.option("--out", "output_path", help="Output path for combined stats")
def generate(
    path, config_path, bench_name, input_idx, limit, quick,
    target, verbose, strict, nvprof, nsight, output_path
):
    benches = []

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
    print("targets: {}".format(targets))
    for target in targets:
        if bench_name is None:
            for bench_configs in b.benchmarks[target.value].values():
                benches.extend(bench_configs)
        else:
            benches.extend(b.benchmarks[target.value][bench_name])

    if limit is not None:
        benches = benches[:limit]

    print(f"processing {len(benches)} benchmark configurations ({len(targets)} targets)")

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    profilers = []
    if nvprof:
        profilers += ["nvprof"]
    if nsight:
        profilers += ["nsight"]

    for profiler in profilers:
        all_stats = []
        assert all([b["name"] == benches[0]["name"] for b in benches])
        bench_name = benches[0]["name"]
        for bench_config in benches:
            # pprint(bench_config)
            name = bench_config["name"]
            target = bench_config["target"]
            input_idx = bench_config["input_idx"]
            input_values = bench_config["values"]
            target_name = f"[{target}]"

            # pprint(input_values)
            if quick:
                if input_values.get("mode") not in ["serial", None]:
                    continue
                # if input_values.get("memory_only") not in [False, None]:
                #     continue
                if input_values.get("cores_per_cluster") not in [1, None]:
                    continue
                if input_values.get("num_clusters") not in [20, None]:
                    continue

            current_bench_log_line = " ===> {:>20} {:>15}@{:<4} {}".format(
                target_name, name, input_idx, input_values
            )

            try:
                match (target.lower(), profiler):
                    case ("profile", "nvprof"):
                        target_name += "[nvprof]"
                        bench_stats = native.NvprofStats(config, bench_config)
                    case ("profile", "nsight"):
                        target_name += "[nsight]"
                        bench_stats = native.NsightStats(config, bench_config)
                    case ("simulate", _):
                        bench_stats = stats.Stats(config, bench_config)
                    case ("execdrivensimulate", _):
                        bench_stats = stats.ExecDrivenStats(config, bench_config)
                    case ("accelsimsimulate", _):
                        bench_stats = accelsim.Stats(config, bench_config)
                    case ("playgroundsimulate", _):
                        bench_stats = playground.Stats(config, bench_config)
                    case other:
                        print(
                            color(
                                f"WARNING: {name} has unknown target {other}", fg="red"
                            )
                        )
                        continue
                print(current_bench_log_line)
            except Exception as e:
                print(color(current_bench_log_line, fg="red"))
                if strict:
                    raise e

            values = pd.DataFrame.from_records([bench_config["values"]])
            values.columns = ["input_" + c for c in values.columns]
            # values.columns = [name + "input_" + c for c in values.columns]

            # this will be the new index
            values["target"] = target
            values["benchmark"] = name
            values["input_id"] = input_idx

            # print(bench_stats.result_df)
            # assert "run" in bench_stats.result_df.columns
            values = bench_stats.result_df.merge(values, how="cross")
            assert "run" in values.columns

            if verbose:
                print(values.T)
            all_stats.append(values)
            # print(bench_stats.result_df.T)

            # print("======")
            # print(bench_stats.print_all_stats())

        all_stats = pd.concat(all_stats)
        if verbose:
            print(all_stats)

        stats_output_path = (
            results_dir / f"combined.stats.{profiler}.{bench_name}.csv"
        )

        if output_path is not None:
            stats_output_path = Path(output_path)

        print(f"saving to {stats_output_path}")
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        all_stats.to_csv(stats_output_path, index=False)

    return

    pprint(config)

    for bench_config in benches:
        name = bench_config["name"]
        input_idx = bench_config["input_idx"]
        print(f"\n\n=== {name}@{input_idx} ===")

        our_stats = stats.Stats(config, bench_config)
        playground_stats = playground.Stats(config, bench_config)
        accelsim_stats = accelsim.Stats(config, bench_config)
        native_stats = native.Stats(config, bench_config)

        # data = [
        #     ("native", native_stats.instructions(), accelsim_stats.instructions()),
        #     ("cycles", native_stats.cycles(), accelsim_stats.cycles()),
        # ]
        # print(
        #     wasabi.table(
        #         data,
        #         header=("", "instructions", "cycles"),
        #         divider=True,
        #         aligns=("r", "r", "r"),
        #     )
        # )

        data = [
            (
                "instructions",
                native_stats.instructions(),
                our_stats.instructions(),
                accelsim_stats.instructions(),
                playground_stats.instructions(),
            ),
            (
                "num blocks",
                native_stats.num_blocks(),
                our_stats.num_blocks(),
                accelsim_stats.num_blocks(),
                playground_stats.num_blocks(),
            ),
            (
                "warp instructions",
                native_stats.warp_instructions(),
                our_stats.warp_instructions(),
                accelsim_stats.warp_instructions(),
                playground_stats.warp_instructions(),
            ),
            (
                "cycles",
                native_stats.cycles(),
                our_stats.cycles(),
                accelsim_stats.cycles(),
                playground_stats.cycles(),
            ),
            (
                "exec time sec",
                native_stats.exec_time_sec(),
                our_stats.exec_time_sec(),
                accelsim_stats.exec_time_sec(),
                playground_stats.exec_time_sec(),
            ),
            (
                "dram reads",
                native_stats.dram_reads(),
                our_stats.dram_reads(),
                accelsim_stats.dram_reads(),
                playground_stats.dram_reads(),
            ),
            (
                "dram writes",
                native_stats.dram_writes(),
                our_stats.dram_writes(),
                accelsim_stats.dram_writes(),
                playground_stats.dram_writes(),
            ),
            (
                "dram accesses",
                native_stats.dram_accesses(),
                our_stats.dram_accesses(),
                accelsim_stats.dram_accesses(),
                playground_stats.dram_accesses(),
            ),
            (
                "L2 reads",
                native_stats.l2_reads(),
                our_stats.l2_reads() * 4,
                accelsim_stats.l2_reads(),
                playground_stats.l2_reads(),
            ),
            (
                "L2 writes",
                native_stats.l2_writes(),
                our_stats.l2_writes() * 4,
                accelsim_stats.l2_writes(),
                playground_stats.l2_writes(),
            ),
            (
                "L2 accesses",
                native_stats.l2_accesses(),
                our_stats.l2_accesses() * 4,
                accelsim_stats.l2_accesses(),
                playground_stats.l2_accesses(),
            ),
            (
                "L2 read hits",
                native_stats.l2_read_hits(),
                our_stats.l2_read_hits() * 4,
                accelsim_stats.l2_read_hits(),
                playground_stats.l2_read_hits(),
            ),
            (
                "L2 write hits",
                native_stats.l2_write_hits(),
                our_stats.l2_write_hits() * 4,
                accelsim_stats.l2_write_hits(),
                playground_stats.l2_write_hits(),
            ),
            (
                "L2 read misses",
                native_stats.l2_read_misses(),
                our_stats.l2_read_misses() * 4,
                accelsim_stats.l2_read_misses(),
                playground_stats.l2_read_misses(),
            ),
            (
                "L2 write misses",
                native_stats.l2_write_misses(),
                our_stats.l2_write_misses() * 4,
                accelsim_stats.l2_write_misses(),
                playground_stats.l2_write_misses(),
            ),
        ]
        data = [
            (
                k,
                human_readable(native),
                human_readable(ours),
                human_readable(accel),
                human_readable(play),
            )
            for (k, native, ours, accel, play) in data
        ]
        # print(native_stats.df)
        print(
            wasabi.table(
                data,
                header=("", "native", "ours", "accelsim", "playground"),
                divider=True,
                aligns=("r", "r", "r", "r", "r"),
            )
        )
        # , widths=widths, ))


if __name__ == "__main__":
    main()
    # main(ctx={})
