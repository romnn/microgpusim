import copy
import typing
import numpy as np
import pandas as pd

import gpucachesim.benchmarks as benchmarks
from gpucachesim.benchmarks import (
    Target,
)


def aggregate_mean_input_config_stats(
    df: pd.DataFrame,
    per_kernel=True,
    mean=True,
    inspect=False,
) -> typing.Tuple[pd.DataFrame, typing.List[str]]:
    bench_input_cols = copy.deepcopy(list(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    input_cols = copy.deepcopy(benchmarks.SIMULATE_INPUT_COLS)
    input_config_group_cols = list(benchmarks.BENCH_TARGET_INDEX_COLS + input_cols + bench_input_cols + ["input_id"])
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
        aggregations = {col: agg for col, agg in aggregations.items() if col in df and not col in group_cols}

        grouped = df.groupby(group_cols, dropna=False)

        def _inspect_per_config(df):
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

    # compute mean per input_id and kernel launch id over all runs
    group_cols = input_config_group_cols + ["kernel_launch_id", "kernel_name"]

    if mean:
        aggregations = {
            **{c: "mean" for c in sorted(df.columns)},
            **{c: "first" for c in bench_input_cols + input_cols},
            **benchmarks.NON_NUMERIC_COLS,
        }
        aggregations = {col: agg for col, agg in aggregations.items() if col in df and not col in group_cols}
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
) -> TargetDataframes:
    df = df.reset_index()

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
        serial_gpucachesim_mask &= df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
        serial_gpucachesim_mask &= df["input_num_clusters"] == functional_config["num_clusters"]
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
        serial_gpucachesim_mem_only_mask &= df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
        serial_gpucachesim_mem_only_mask &= df["input_num_clusters"] == functional_config["num_clusters"]
    serial_gpucachesim_mem_only_df = df[serial_gpucachesim_mem_only_mask]
    serial_gpucachesim_mem_only_df, _ = aggregate_mean_input_config_stats(
        serial_gpucachesim_mem_only_df, per_kernel=per_kernel, mean=mean
    )
    print(_label("serial gpucachesim (mem only)", serial_gpucachesim_mem_only_df.shape))

    # gpucachesim (serial, exec-driven)
    serial_gpucachesim_exec_driven_mask = df["target"] == Target.ExecDrivenSimulate.value
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
        parallel_gpucachesim_mask &= df["input_cores_per_cluster"] == functional_config["cores_per_cluster"]
        parallel_gpucachesim_mask &= df["input_num_clusters"] == functional_config["num_clusters"]
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
