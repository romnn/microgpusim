import copy
import typing
import numpy as np
import pandas as pd
from pprint import pprint

import gpucachesim.benchmarks as benchmarks
from gpucachesim.benchmarks import (
    Target,
)


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
    # trace_recon = selected_df["target"] == Target.ExecDrivenSimulate.value
    # print(
    #     selected_df.loc[
    #         trace_recon,
    #         ["target", "benchmark", "input_id", "kernel_name", "kernel_name_mangled"],
    #     ]
    # )

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

    # valid_kernel = selected_df["kernel_name"].isin(kernels)
    valid_kernel = True

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
    preview_cols += ["mean_blocks_per_sm"]
    # preview_cols += ["mean_blocks_per_sm_all_kernels"]
    preview_cols += ["cycles"]
    # print(df.loc[:, preview_cols][:])

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
        # pprint(aggregations)

        grouped = df.groupby(group_cols, dropna=False)

        def _inspect_per_config(df):
            is_babelstream = (df["benchmark"] == "babelstream").all()
            is_accelsim = (df["target"] == Target.AccelsimSimulate.value).all()
            # if is_babelstream and is_accelsim:
            print("\nINSPECT: metrics (per input config, PER RUN)")
            print(df.loc[:, preview_cols][:10])
            pass

        if inspect:
            grouped[df.columns].apply(_inspect_per_config)
        df = grouped.agg(aggregations).reset_index()

        # we no longer have kernels now
        df["kernel_launch_id"] = np.nan
        df["kernel_name"] = np.nan
        df["kernel_name_mangled"] = np.nan

    # print(
    #     df.loc[
    #         (df["benchmark"] == "babelstream") & (df["target"] == Target.AccelsimSimulate.value),
    #         preview_cols,
    #     ]
    # )

    # compute mean per input_id and kernel launch id over all runs
    group_cols = input_config_group_cols + ["kernel_launch_id", "kernel_name"]

    if mean:
        aggregations = {
            **{c: "mean" for c in sorted(df.columns)},
            # **{c: "sum" for c in ["exec_time_sec"]},
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
            print(df.loc[:, preview_cols][:10])
            pass

        if inspect:
            grouped[df.columns].apply(_inspect_per_config_per_kernel)
        df = grouped.agg(aggregations).reset_index()

    return df.copy(), group_cols


class TargetDataframes(typing.NamedTuple):
    native_df: pd.DataFrame
    accelsim_df: pd.DataFrame
    serial_gpucachesim_df: pd.DataFrame
    serial_gpucachesim_mem_only_df: pd.DataFrame
    serial_gpucachesim_exec_driven_df: pd.DataFrame
    parallel_gpucachesim_df: pd.DataFrame


class FunctionalConfig(typing.TypedDict):
    num_clusters: int
    cores_per_cluster: int


def split_into_target_dfs(
    df,
    per_kernel=False,
    mean=False,
    functional_config: typing.Optional[FunctionalConfig] = None,
    inspect=False,
) -> TargetDataframes:
    df = df.reset_index()

    if functional_config is None:
        # use baseline functional config
        baseline_cores_per_cluster = benchmarks.BASELINE["cores_per_cluster"]
        baseline_num_clusters = benchmarks.BASELINE["num_clusters"]
        functional_config = FunctionalConfig(
            cores_per_cluster=baseline_cores_per_cluster,
            num_clusters=baseline_num_clusters,
        )

    def _label(label, shape):
        return "{:>50}\t{}".format(label, shape)

    # native
    native_mask = df["target"] == Target.Profile.value
    native_df = df[native_mask]
    native_df, _ = aggregate_mean_input_config_stats(
        native_df, per_kernel=per_kernel, mean=mean, inspect=inspect
    )
    print(_label("native", native_df.shape))

    # accelsim
    accelsim_mask = df["target"] == Target.AccelsimSimulate.value
    accelsim_df = df[accelsim_mask]
    accelsim_df, _ = aggregate_mean_input_config_stats(
        accelsim_df, per_kernel=per_kernel, mean=mean, inspect=inspect
    )
    print(_label("accelsim", accelsim_df.shape))

    # gpucachesim (serial)
    serial_gpucachesim_mask = df["target"] == Target.Simulate.value
    serial_gpucachesim_mask &= df["input_mode"].isin(["serial", np.nan])
    serial_gpucachesim_mask &= df["input_memory_only"] == False

    assert functional_config is not None
    serial_gpucachesim_mask &= (
        df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
    )
    serial_gpucachesim_mask &= (
        df["input_num_clusters"] == functional_config["num_clusters"]
    )

    serial_gpucachesim_df = df[serial_gpucachesim_mask]
    serial_gpucachesim_df, _ = aggregate_mean_input_config_stats(
        serial_gpucachesim_df, per_kernel=per_kernel, mean=mean, inspect=inspect
    )
    print(_label("serial gpucachesim", serial_gpucachesim_df.shape))

    # gpucachesim (serial, mem only)
    serial_gpucachesim_mem_only_mask = df["target"] == Target.Simulate.value
    serial_gpucachesim_mem_only_mask &= df["input_memory_only"] == True
    serial_gpucachesim_mem_only_mask &= df["input_mode"].isin(["serial", np.nan])
    # if functional_config is not None:
    serial_gpucachesim_mem_only_mask &= (
        df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
    )
    serial_gpucachesim_mem_only_mask &= (
        df["input_num_clusters"] == functional_config["num_clusters"]
    )

    serial_gpucachesim_mem_only_df = df[serial_gpucachesim_mem_only_mask]
    serial_gpucachesim_mem_only_df, _ = aggregate_mean_input_config_stats(
        serial_gpucachesim_mem_only_df,
        per_kernel=per_kernel,
        mean=mean,
        inspect=inspect,
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
        serial_gpucachesim_exec_driven_df,
        per_kernel=per_kernel,
        mean=mean,
        inspect=inspect,
    )
    print(
        _label(
            "serial gpucachesim (exec driven)", serial_gpucachesim_exec_driven_df.shape
        )
    )

    # gpucachesim (parallel)
    parallel_gpucachesim_mask = df["target"] == Target.Simulate.value
    parallel_gpucachesim_mask &= df["input_mode"] != "serial"
    parallel_gpucachesim_mask &= df["input_memory_only"] == False
    # if functional_config is not None:
    parallel_gpucachesim_mask &= (
        df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
    )
    parallel_gpucachesim_mask &= (
        df["input_num_clusters"] == functional_config["num_clusters"]
    )
    parallel_gpucachesim_df = df[parallel_gpucachesim_mask]
    parallel_gpucachesim_df, _ = aggregate_mean_input_config_stats(
        parallel_gpucachesim_df, per_kernel=per_kernel, mean=mean, inspect=inspect
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
