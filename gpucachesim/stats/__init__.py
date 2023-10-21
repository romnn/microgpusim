import click
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from wasabi import color
import wasabi
import itertools

import gpucachesim.stats.stats as stats
import gpucachesim.stats.native as native
import gpucachesim.stats.accelsim as accelsim
import gpucachesim.stats.playground as playground
import gpucachesim.benchmarks as benchmarks
from gpucachesim.stats.human import human_readable
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


SIMULATE_FUNCTIONAL_CONFIG_COLS = [
    "input_memory_only",
    "input_num_clusters",
    "input_cores_per_cluster",
]
SIMULATE_EXECUTION_CONFIG_COLS = [
    "input_mode",
    "input_threads",
    "input_run_ahead",
]
SIMULATE_INPUT_COLS = SIMULATE_EXECUTION_CONFIG_COLS + SIMULATE_FUNCTIONAL_CONFIG_COLS 

BENCHMARK_INPUT_COLS = {
    "vectorAdd": ["input_dtype", "input_length"],
    "matrixmul": ["input_dtype", "input_rows"],
    "simple_matrixmul": ["input_dtype", "input_m", "input_n", "input_p"],
    "transpose": ["input_dim", "input_variant"],
    "babelstream": ["input_size"],
}

STAT_COLS = [
    "exec_time_sec",
    "cycles",
    "num_blocks",
    "instructions",
    "warp_inst",
    # dram stats
    "dram_reads",
    "dram_writes",
    # l2 stats
    "l2_accesses",
    "l2_reads",
    "l2_writes",
    "l2_read_hit_rate",
    "l2_write_hit_rate",
    "l2_read_miss_rate",
    "l2_write_miss_rate",
    "l2_hit_rate",
    "l2_miss_rate",
    "l2_read_hits",
    "l2_write_hits",
    "l2_read_misses",
    "l2_write_misses",
    "l2_hits",
    "l2_misses",
    # l1 rates
    "l1_hit_rate",
    "l1_miss_rate",
    # l1 accesses
    "l1_reads",
    "l1_writes",
    "l1_hits",
    "l1_misses",
    "l1_accesses",
]

INDEX_COLS = ["target", "benchmark", "input_id"]


def benchmark_results(sim_df: pd.DataFrame, bench_name: str, targets=None) -> pd.DataFrame:
    """View results for a benchmark"""

    selected_df = sim_df.copy()
    selected_df = selected_df[selected_df["benchmark"] == bench_name]
    # print(selected_df)
    # only compare serial gpucachesim
    # selected_df = selected_df[selected_df["input_mode"] != "nondeterministic"]

    for col in SIMULATE_INPUT_COLS:
        if col not in selected_df:
            selected_df[col] = np.nan

    non_gpucachesim = selected_df["input_mode"].isnull()
    print(selected_df[non_gpucachesim]["target"].unique().tolist())

    serial_gpucachesim = selected_df["input_mode"] == "serial"
    compute_gpucachesim = selected_df["input_memory_only"] == False
    gtx1080_gpucachesim = selected_df["input_cores_per_cluster"] == 1
    gtx1080_gpucachesim &= selected_df["input_num_clusters"] == 20
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
    print(selected_df[gold_gpucachesim][["kernel_name_mangled", "kernel_name"]].drop_duplicates())
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
    selected_df = selected_df[(gold_gpucachesim ^ non_gpucachesim) & (valid_kernel ^ no_kernel)]

    if isinstance(targets, list):
        selected_df = selected_df[selected_df["target"].isin(targets)]

    # restrict inputs to fit screen
    assert (selected_df["benchmark"] == bench_name).all()
    if bench_name == "simple_matrixmul":
        # m: [32, 64, 128]
        # n: [32, 64, 128]
        # p: [32, 64, 128]

        subset = pd.DataFrame.from_records([(32, 32, 32), (128, 128, 128)], columns=["input_m", "input_n", "input_p"])
        # print(subset.index)
        # print(selected_df.index)
        selected_df = selected_df.merge(subset, how="inner")
        # selected_df = selected_df.merge(subset, on=["input_m", "input_n", "input_p"], how="inner")
        # selected_df = selected_df.join(subset, on=["input_m", "input_n", "input_p"], how="left")
        # subset = [(32, 32, 32), (128, 128, 128)]
        # pprint([c for c in selected_df.columns if "input_" in c])
        # print(selected_df[["input_m", "input_n", "input_p"]])
        # selected_df = selected_df[selected_df[["input_m", "input_n", "input_p"]].isin(subset)]

    # assert False

    # assert (selected_df["is_release_build"] == True).all()

    input_cols = BENCHMARK_INPUT_COLS[bench_name]
    # print(selected_df[input_cols].drop_duplicates())

    # INDEX_COLS = [
    #     "kernel_name",
    #     "kernel_name_mangled",
    #     "kernel_launch_id",
    #     "run",
    # ]

    # aggregate over all kernel launch ids in a single run
    BENCH_TARGET_INDEX_COLS = ["target", "benchmark", "input_id"]
    # grouped = selected_df.groupby(["kernel_name", "run"], dropna=False)

    profile_df = selected_df[selected_df["target"] == "Profile"]
    print(profile_df[BENCH_TARGET_INDEX_COLS + input_cols + ["l2_hits", "l2_misses"]])

    per_kernel = (
        selected_df.groupby(BENCH_TARGET_INDEX_COLS + ["kernel_name", "run"] + input_cols, dropna=False)[STAT_COLS]
        .mean()
        .reset_index()
    )
    # print(per_kernel)
    # return None

    grouped = per_kernel.groupby(BENCH_TARGET_INDEX_COLS + input_cols, dropna=False)
    # print(selected_df[INDEX_COLS + input_cols + ["dram_writes", "l2_accesses"]].head(n=200))
    # print(grouped["dram_writes"].sum())
    averaged = grouped[STAT_COLS].mean().reset_index()
    # print(per_kernel)
    # averaged = grouped[STAT_COLS + input_cols].mean().reset_index()
    # print(averaged)
    # print(averaged.drop_duplicates())

    per_target = averaged.pivot(index=["benchmark"] + input_cols, columns="target", values=STAT_COLS)
    return per_target


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
def parallel_plot(path, bench_name, nvprof):
    # load the materialized benchmark config
    if bench_name is None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.csv"
    else:
        stats_file = REPO_ROOT_DIR / f"results/combined.stats.{bench_name}.csv"

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
    bench_input_cols = BENCHMARK_INPUT_COLS[bench_name]
    input_cols = SIMULATE_INPUT_COLS 
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

    mean_serial = serial.groupby(group_cols).agg({
        **{c: "mean" for c in set(serial.columns) - set(group_cols)},
        **{
           # 'Host Name', 'Process Name', 'device', 'is_release_build', 'kernel_function_signature', 'kernel_name', 'kernel_name_mangled', 'parallelization_method', 'target'
           'Host Name': "first",
           'Process Name': "first",
           'device': "first",
           'is_release_build': "first",
           'kernel_function_signature': "first",
           'kernel_name': "first",
           'kernel_name_mangled': "first",
           # 'parallelization_method': "first",
           # 'target': "first",
           # 'input_mode': "first",
    }}).reset_index()

    metric_cols = ["cycles", "exec_time_sec", "l2_hit_rate", "l1_hit_rate"]

    if False:
        print("before", serial.shape)
        print("averaged", mean_serial.shape)

        print("before:")
        print(serial.sort_values(input_cols + bench_input_cols)[input_cols + bench_input_cols + metric_cols])
        print("averaged:")
        print(mean_serial.sort_values(input_cols + bench_input_cols)[input_cols + bench_input_cols + metric_cols])

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
    input_id_partitoning = set(serial["input_id"].unique()).intersection(set(parallel["input_id"].unique()))
    if len(input_id_partitoning) > 0:
        for input_id in input_id_partitoning:
            print("serial input", input_id)
            print(serial.loc[serial["input_id"] == input_id, bench_cols + bench_input_cols + SIMULATE_INPUT_COLS])
            print("parallel input", input_id)
            print(parallel.loc[parallel["input_id"] == input_id, bench_cols + bench_input_cols + SIMULATE_INPUT_COLS])
            break
        assert len(input_id_partitoning) == 0

    # join based on input_cols, NOT based on mode
    joined = parallel.merge(
        serial,
        on=bench_cols + bench_input_cols + SIMULATE_FUNCTIONAL_CONFIG_COLS,
        how="left",
        suffixes=('_parallel', '_serial'),
    )
    print(joined.shape)
    assert joined.shape[0] == parallel.shape[0]
    # assert joined.shape[0] == len(parallel["input_id"].unique())

    PREVIEW_COLS = sorted(
        bench_cols + bench_input_cols + SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [c + "_parallel" for c in SIMULATE_EXECUTION_CONFIG_COLS]
        + [c + "_parallel" for c in metric_cols]
        + [c + "_serial" for c in metric_cols]
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

    print(joined[[
        "benchmark", "input_mode_parallel", "input_threads_parallel", "input_run_ahead_parallel",
    ]].drop_duplicates())
    # print(joined[[
    #     "benchmark", "input_mode_parallel", "input_threads_parallel", "input_run_ahead_parallel", "cycles_rmse",
    # ]].drop_duplicates())

    # functional_config = {
    #     "input_memory_only": False,
    #     "input_num_clusters": 20,
    #     "input_cores_per_cluster": 1,
    # }
    # assert set(functional_config.keys()) == set(SIMULATE_FUNCTIONAL_CONFIG_COLS)

    # table_values = dict()
    # table_df = []

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
        bench_cols + bench_input_cols + SIMULATE_FUNCTIONAL_CONFIG_COLS
        + [col + "_parallel" for col in SIMULATE_EXECUTION_CONFIG_COLS]
        + [col + "_serial" for col in SIMULATE_EXECUTION_CONFIG_COLS]
    )
    # table_df = joined.groupby(group_cols).apply(test_func)
    # .set_index(group_cols)
    # print(table_df.index)

    non_numeric = {
       'Host Name': "first",
       'Process Name': "first",
       'device': "first",
       'is_release_build': "first",
       'kernel_function_signature': "first",
       'kernel_name': "first",
       'kernel_name_mangled': "first",
    }
    # joined.loc[:,list(set(joined.columns) - set(non_numeric.keys()))] = joined[:,list(set(joined.columns) - set(non_numeric.keys()))].astype(float)
    # pprint(sorted(table_df.columns.tolist()))
    # print(len(table_df))
    aggregations = {
        **{c: "mean" for c in sorted(set(joined.columns) - set(group_cols))},
        **{c + "_parallel": agg for c, agg in non_numeric.items()},
        **{c + "_serial": agg for c, agg in non_numeric.items()},
    }
    # pprint(sorted(aggregations.items(), key=lambda x: (
    #     x[1] if isinstance(x[1], str) else x[1].__name__, x[0]
    # )))

    # non_numeric_cols = sorted(joined.select_dtypes(include=["object"]).columns.tolist())
    # print(joined[non_numeric_cols])
    # return


    if set(joined.columns.tolist()) - set(group_cols) != set(aggregations.keys()):
        pprint((set(joined.columns.tolist()) - set(group_cols)).symmetric_difference(set(aggregations.keys())))
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
        return (values - true_values).abs() / true_values

    def rmse(true_values, values):
        return((values - true_values) ** 2).mean() ** .5

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

    # compute per config metrics
    aggregated["exec_time_sec_speedup"] = grouped.apply(
        lambda df: speedup(baseline=df["exec_time_sec_serial"], values=df["exec_time_sec_parallel"]).mean())
    aggregated["cycles_rel_err"] = grouped.apply(
        lambda df: rel_err(true_values=df["cycles_serial"], values=df["cycles_parallel"]).mean())
    aggregated["cycles_rmse"] = grouped.apply(
        lambda df: rmse(true_values=df["cycles_serial"], values=df["cycles_parallel"]).mean())
    aggregated["cycles_mae"] = grouped.apply(
        lambda df: mae(true_values=df["cycles_serial"], values=df["cycles_parallel"]).mean())


    # aggregated["exec_time_sec_speedup"] = joined.groupby(group_cols, dropna=False)[
    #     ["exec_time_sec_serial", "exec_time_sec_parallel"]].apply(compute_speedup).mean()
    # aggregated["cycles_rel_err"] = joined.groupby(group_cols, dropna=False).apply(compute_cycles_rel_err).mean()
    # [["cycles_parallel", "cycles_serial"]].apply(compute_speedup).mean()
    # joined = joined.groupby(group_cols, dropna=False).apply(compute_per_config_metrics).reset_index()

    aggregated = aggregated.reset_index()
    print(aggregated.shape)
    # return
    # print(joined.T)
    # print(aggregated.loc[0:4, PREVIEW_COLS].T.drop_duplicates())
    print(aggregated.reset_index().loc[0:4,
        PREVIEW_COLS + ["cycles_rel_err", "exec_time_sec_speedup"]].T.drop_duplicates())

    # print(aggregated.reset_index().loc[0:4,
    #     ["cycles_rel_err", "cycles_serial", "cycles_parallel"]].T) #.drop_duplicates())


    # table_df = joined.groupby(group_cols, dropna=False).agg(aggregations, squeeze=False).reset_index()
    # table_df["test"] = 1
    # table_df = table_df.agg(aggregations).reset_index()
    # table_df = table_df
    # print(table_df)
    return
    # table_df = table_df.apply(test_func).agg(aggregations, squeeze=False).reset_index()
    # table_df = joined.groupby(group_cols, dropna=False).apply(test_func).agg({
    #     k: first for k in aggregations.keys()
    # }).reset_index()

    # assert "target" not in df.columns
    # assert "input_mode" not in df.columns

    # out = df.copy().agg(aggregations)

    # print(df.reset_index().loc[0:4, PREVIEW_COLS].T.drop_duplicates())
    # print(df.reset_index().loc[0:4, ["cycles_rel_err", "cycles_rmse"]].T.drop_duplicates())
    # print(df)

    # table_df.append(out)

        # per_config = df[ 
    # for (mode, benchmark, threads, run_ahead) in itertools.product(
    #         mode_values, benchmark_values, thread_values, run_ahead_values):
    #     print(
    #         "\n==> collecting [ mode={:<35} benchmark={:<15} threads={:<3} run ahead={:<3} ]".format(
    #         str(color(mode, fg="cyan")),
    #         str(color(benchmark, fg="cyan")),
    #         str(color(threads, fg="cyan")),
    #         str(color(run_ahead, fg="cyan"),
    #     )))
    #     mask = joined["benchmark"] == benchmark
    #     mask &= joined["input_mode_parallel"] == mode
    #     mask &= joined["input_threads_parallel"] == threads
    #     mask &= joined["input_run_ahead_parallel"] == run_ahead
    #     # for col, value in functional_config.items():
    #     #     df &= joined[col] == value
    #     df = joined.loc[mask,:].copy()
    #     for df in df.groupby()
    #     # print(len(df))
    #     # print(df.reset_index().loc[0:3, PREVIEW_COLS].T.drop_duplicates())

    
    # table_df = pd.concat(table_df, ignore_index=True)
    print(table_df[[
        "benchmark", "input_mode_parallel", "input_threads_parallel", "input_run_ahead_parallel", # "cycles_rmse",
    ]].drop_duplicates())


    # pprint(table_values)
    return

    

    # joined = joined[joined["input_id_parallel"] == 41]
    # print(joined["cycles_parallel"] - joined["cycles_serial"])
    # joined[[
    #     "benchmark", "input_dtype", "input_length", "input_memory_only", "input_cores_per_cluster", "input_id_parallel", "input_id_serial",
    #     "input_threads_parallel", "input_run_ahead_parallel", "parallelization_method_parallel",
    #     "exec_time_sec_parallel", "exec_time_sec_serial",
    #     "cycles_parallel", "cycles_serial",
    #     "l2_accesses_parallel", "l2_accesses_serial",
    #
    #     # computed
    #     "exec_time_sec_speedup", "cycles_rel_err", "cycles_rmse", "cycles_mae",
    # ]].T


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
# @click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
# @click.option("--input", "input_idx", type=int, help="Input index")
def view(path, bench_name, nvprof):
    # load the materialized benchmark config
    if bench_name is None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.csv"
    else:
        stats_file = REPO_ROOT_DIR / f"results/combined.stats.{bench_name}.csv"

    sim_df = pd.read_csv(stats_file, header=0)
    # assert (sim_df["input_mode"] == "serial").sum() > 0

    # print(sim_df)

    per_target = benchmark_results(sim_df, bench_name)
    stat_cols = [
        "num_blocks",
        "exec_time_sec",
        "cycles",
        "instructions",
        "dram_reads",
        "dram_writes",
        # l2 stats
        "l2_accesses",
        "l2_reads",
        "l2_writes",
    ]

    if nvprof:
        stat_cols += [
            "l2_read_hit_rate",
            "l2_write_hit_rate",
            "l2_read_hits",
            "l2_write_hits",
            # "l2_hits",
            # "l2_misses",
            # l1 stats
            "l1_accesses",
            # "l1_reads",
            # "l1_hits",
            # "l1_misses",
        ]
    else:
        stat_cols += [
            "l2_hits",
            "l2_misses",
            "l2_hit_rate",
            # "l2_miss_rate",
            "l1_hit_rate",
        ]

    per_target = per_target[stat_cols]
    print(per_target.T.to_string())
    # print(per_target.T.head(n=100))


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--input", "input_idx", type=int, help="Input index")
@click.option("--limit", "limit", type=int, help="Limit number of benchmark configs generated")
@click.option("--quick", "quick", type=bool, is_flag=True,
              help="Fast mode: only collect baseline benchmark configurations")
@click.option("--verbose", "verbose", type=bool, is_flag=True, help="verbose output")
@click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
@click.option("--out", "output_path", help="Output path for combined stats")
def generate(path, config_path, bench_name, input_idx, limit, quick, verbose, output_path, nvprof):
    benches = []

    b = Benchmarks(path)
    results_dir = Path(b.config["results_dir"])

    for target in [
        Target.Profile,
        Target.Simulate,
        Target.AccelsimSimulate,
        Target.PlaygroundSimulate,
    ]:
        if bench_name is None:
            for bench_configs in b.benchmarks[target.value].values():
                benches.extend(bench_configs)
        else:
            benches.extend(b.benchmarks[target.value][bench_name])

    if limit is not None:
        benches = benches[:limit]

    print(f"processing {len(benches)} benchmark configurations")

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    all_stats = []
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
            if input_values.get("memory_only") not in [False, None]:
                continue
            if input_values.get("cores_per_cluster") not in [1, None]:
                continue
            if input_values.get("num_clusters") not in [20, None]:
                continue

        try:
            match target.lower():
                case "profile" if nvprof:
                    target_name += "[nvprof]"
                    bench_stats = native.NvprofStats(config, bench_config)
                case "profile":
                    target_name += "[nsight]"
                    bench_stats = native.NsightStats(config, bench_config)
                case "simulate":
                    # if bench_config["values"]["mode"] != "serial":
                    #     continue
                    bench_stats = stats.Stats(config, bench_config)
                case "accelsimsimulate":
                    bench_stats = accelsim.Stats(config, bench_config)
                case "playgroundsimulate":
                    bench_stats = playground.Stats(config, bench_config)
                case other:
                    print(color(f"WARNING: {name} has unknown target {other}", fg="red"))
                    continue
        except Exception as e:
            print(" ===> {:>20} {:>15}@{:<4} {}".format(target_name, name, input_idx, input_values))
            raise e

        print(" ===> {:>20} {:>15}@{:<4} {}".format(target_name, name, input_idx, input_values))
        # print(f" ===> {target_name} \t {name}@{input_idx} \t {input_values}")

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

    if bench_name is None:
        all_stats_output_path = results_dir / "combined.stats.csv"
    else:
        all_stats_output_path = results_dir / f"combined.stats.{bench_name}.csv"

    if output_path is not None:
        all_stats_output_path = Path(output_path)

    print(f"saving to {all_stats_output_path}")
    all_stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(all_stats_output_path, index=False)

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
