import enum
import typing
import itertools
import numpy as np
import pandas as pd
from pprint import pprint
from wasabi import color
from pathvalidate import sanitize_filename

from gpucachesim.stats.agg import (
    FunctionalConfig,
    TargetDataframes,
    split_into_target_dfs,
)
import gpucachesim.stats.metrics as metric_funcs
import gpucachesim.benchmarks as benchmarks
import gpucachesim.utils as utils
import gpucachesim.plot as plot
from gpucachesim.benchmarks import BENCHMARK_INPUT_COLS, Target


class ErrorMetricID(enum.Enum):
    MAPE = "MAPE"
    SMAPE = "SMAPE"
    RMSPE = "RMSPE"
    RMSE = "RMSE"
    MAE = "MAE"
    Correlation = "Corr."
    EMALE = "EMALE"
    ERMSLE = "ERMSLE"
    # RelErr = "Rel err."

    # MAPE = ("mape", "MAPE")
    # Correlation = ("corr", "Corr.")
    # RelErr = ("rel_err", "Rel err.")


class ErrorMetric(typing.NamedTuple):
    column: str
    is_percent: bool
    metric: ErrorMetricID


# class Metric(typing.TypedDict):
class Metric(typing.NamedTuple):
    key: str
    label: str
    error_metrics: typing.Sequence[ErrorMetric]


ALL_METRICS = [
    Metric(
        key=r"dramreads",
        label=r"DRAM\\reads",
        error_metrics=[
            # ("dram_reads", ErrorMetric.EMALE),
            # ("dram_reads_percent", ErrorMetric.MAPE),
            # ("dram_reads_percent", ErrorMetric.Correlation),
            ErrorMetric(
                column="dram_reads_percent", is_percent=True, metric=ErrorMetricID.RMSE
            ),
            ErrorMetric(
                column="dram_reads_percent", is_percent=True, metric=ErrorMetricID.MAE
            ),
            ErrorMetric(
                column="dram_reads", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key=r"dramwrites",
        label=r"DRAM\\writes",
        # is_percent=False,
        error_metrics=[
            # ErrorMetric(column="dram_writes", is_percent=False, metric=ErrorMetricID.EMALE),
            # (column="dram_writes_percent", metric=ErrorMetric.MAPE),
            # (column="dram_writes_percent", metric=ErrorMetric.Correlation),
            ErrorMetric(
                column="dram_writes_percent", is_percent=True, metric=ErrorMetricID.RMSE
            ),
            ErrorMetric(
                column="dram_writes_percent", is_percent=True, metric=ErrorMetricID.MAE
            ),
            ErrorMetric(
                column="dram_writes", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key=r"l1daccesses",
        label=r"L1D\\Accesses",
        # is_percent=False,
        error_metrics=[
            # ErrorMetric(column="l1_accesses", is_percent=False,metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l1_accesses", is_percent=False, metric=ErrorMetricID.RMSPE
            ),
            ErrorMetric(
                column="l1_accesses", is_percent=False, metric=ErrorMetricID.MAPE
            ),
            ErrorMetric(
                column="l1_accesses", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key=r"l2daccesses",
        label=r"L2D\\Accesses",
        # is_percent=False,
        error_metrics=[
            # ErrorMetric(column="l2_accesses", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l2_accesses", is_percent=False, metric=ErrorMetricID.RMSPE
            ),
            ErrorMetric(
                column="l2_accesses", is_percent=False, metric=ErrorMetricID.MAPE
            ),
            ErrorMetric(
                column="l2_accesses", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key=r"l2dreads",
        label=r"L2D\\reads",
        # is_percent=False,
        error_metrics=[
            # ErrorMetric(column="l2_reads", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l2_reads", is_percent=False, metric=ErrorMetricID.RMSPE
            ),
            ErrorMetric(column="l2_reads", is_percent=False, metric=ErrorMetricID.MAPE),
            ErrorMetric(
                column="l2_reads", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key=r"l2dwrites",
        label=r"L2D\\writes",
        # is_percent=False,
        error_metrics=[
            # ErrorMetric(column="l2_writes", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l2_writes", is_percent=False, metric=ErrorMetricID.RMSPE
            ),
            ErrorMetric(
                column="l2_writes", is_percent=False, metric=ErrorMetricID.MAPE
            ),
            ErrorMetric(
                column="l2_writes", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key="l1dhitrate",
        label=r"L1D\\hit rate",
        # is_percent=True,
        error_metrics=[
            # ErrorMetric(column="l1_global_hit_rate", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l1_global_hit_rate", is_percent=True, metric=ErrorMetricID.RMSE
            ),
            ErrorMetric(
                column="l1_global_hit_rate", is_percent=True, metric=ErrorMetricID.MAE
            ),
            ErrorMetric(
                column="l1_global_hit_rate",
                is_percent=True,
                metric=ErrorMetricID.Correlation,
            ),
        ],
    ),
    Metric(
        key=r"l2dhitrate",
        label=r"L2D\\hitrate",
        # is_percent=True,
        error_metrics=[
            # ErrorMetric(column="l2_hit_rate", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l2_hit_rate", is_percent=True, metric=ErrorMetricID.RMSE
            ),
            ErrorMetric(
                column="l2_hit_rate", is_percent=True, metric=ErrorMetricID.MAE
            ),
            ErrorMetric(
                column="l2_hit_rate", is_percent=True, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
    Metric(
        key=r"l2dreadhitrate",
        label=r"L2D read\\hit rate",
        # is_percent=True,
        error_metrics=[
            # ErrorMetric(column="l2_read_hit_rate", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l2_read_hit_rate", is_percent=True, metric=ErrorMetricID.RMSE
            ),
            ErrorMetric(
                column="l2_read_hit_rate", is_percent=True, metric=ErrorMetricID.MAE
            ),
            ErrorMetric(
                column="l2_read_hit_rate",
                is_percent=True,
                metric=ErrorMetricID.Correlation,
            ),
        ],
    ),
    Metric(
        key=r"l2dwritehitrate",
        label=r"L2D write\\hit rate",
        # is_percent=True,
        error_metrics=[
            # ErrorMetric(column="l2_write_hit_rate", is_percent=False, metric=ErrorMetricID.EMALE),
            ErrorMetric(
                column="l2_write_hit_rate", is_percent=True, metric=ErrorMetricID.RMSE
            ),
            ErrorMetric(
                column="l2_write_hit_rate", is_percent=True, metric=ErrorMetricID.MAE
            ),
            ErrorMetric(
                column="l2_write_hit_rate",
                is_percent=True,
                metric=ErrorMetricID.Correlation,
            ),
        ],
    ),
    Metric(
        key=r"cycles",
        label=r"Cycles",
        # is_percent=False,
        error_metrics=[
            # ErrorMetric(column="cycles", metric=ErrorMetric.RelErr),
            # ErrorMetric(column="cycles", metric=ErrorMetric.EMALE),
            # ErrorMetric(column="cycles", metric=ErrorMetric.ERMSLE),
            # ErrorMetric(column="cycles", metric=ErrorMetric.SMAPE),
            ErrorMetric(column="cycles", is_percent=False, metric=ErrorMetricID.RMSPE),
            ErrorMetric(column="cycles", is_percent=False, metric=ErrorMetricID.MAPE),
            ErrorMetric(
                column="cycles", is_percent=False, metric=ErrorMetricID.Correlation
            ),
        ],
    ),
]


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


def join_targets(
    target_dfs: TargetDataframes,
    sim_targets: typing.Dict[str, pd.DataFrame],
    large=False,
    verbose=False,
):

    joined_df = target_dfs.native_df.copy()

    for target, sim_df in sim_targets.items():
        if verbose:
            print("computing =>", target)
        # print(sim_df[benchmarks.PREVIEW_COLS][:4].T)
        join_cols = list(
            # we do NOT join based on target
            ["benchmark", "kernel_launch_id"]
            + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
            # we do NOT join based on input_memory_only
            + ["input_num_clusters", "input_cores_per_cluster"],
        )
        join_cols = [col for col in join_cols if col in sim_df]
        # pprint(join_cols)

        missing_df = (
            joined_df[join_cols]
            # native_df[join_cols]
            .merge(
                sim_df[join_cols],
                how="left",
                indicator=True,
            ).loc[lambda x: x["_merge"] != "both"]
        )
        if len(missing_df) > 0:
            # if target == "_gpucachesim_parallel":
            #     # temp: ignore for now
            #     pass
            if large:
                # when selecting only large inputs some native input
                # configs are missing from the filtered simulator inputs
                pass
            elif target == "gpucachesim_exec_driven":
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

        # _joined_df = native_df.merge(
        _joined_df = joined_df.merge(
            sim_df,
            on=join_cols,
            how="left",
            suffixes=(None, "_" + target),
        )
        # assert _joined_df.shape[0] == native_df.shape[0]
        assert _joined_df.shape[0] == joined_df.shape[0]
        if len(_joined_df) == 0:
            raise ValueError("joined dataframe is empty")

        joined_df = _joined_df

    return joined_df


def result_table(
    df,
    bench_name: typing.Optional[str] = None,
    metrics: typing.Optional[
        typing.Union[str, typing.List[typing.Optional[str]]]
    ] = None,
    combined_only=False,
    no_combined=False,
    scaled_clusters=False,
    scaled_cores=False,
    large=False,
    verbose=False,
    batch=False,
    png=False,
):
    # exec_driven = df["target"] == Target.ExecDrivenSimulate.value
    # print(df.loc[exec_driven, ["target", "benchmark", "input_id", "total_cores", "num_blocks", "mean_blocks_per_sm"]])

    # remove non-kernel results
    df = df[~df["kernel_name"].isna()]

    if large:
        profile = df["target"] == Target.Profile.value
        # df = df[profile | (df["mean_blocks_per_sm"] > 1.0)]
        df = df[profile | (df["mean_blocks_per_sm_all_kernels"] > 1.0)]
        if len(df) < 1:
            print(color("have no large configurations with blocks/SM > 1", fg="red"))
            return

    benches = sorted(df["benchmark"].unique().tolist())

    # target benchmark histogram
    target_bench_input_count_hist = (
        df[["target", "benchmark", "input_id"]]
        .drop_duplicates()
        .value_counts(["target", "benchmark"], dropna=False)
        .sort_index()
    )
    if verbose:
        print(target_bench_input_count_hist)

    if metrics is None:
        metrics_keys = []
    elif isinstance(metrics, str):
        metrics_keys = [metrics]
    elif isinstance(metrics, list):
        metrics_keys = metrics
    else:
        raise ValueError(
            "metrics must be either a string or list of strings, have {}".format(
                metrics
            )
        )

    metrics_keys = [
        metric.replace(" ", "").lower() for metric in metrics_keys if metric is not None
    ]

    if len(metrics_keys) == 0:
        # only show cycles by default
        selected_metrics = [ALL_METRICS[-1]]
    else:
        selected_metrics = [
            m for m in ALL_METRICS if m.key.replace(" ", "").lower() in metrics_keys
        ]
        if len(selected_metrics) == 0:
            raise ValueError(
                "invalid metrics {} (keys={}), have {}".format(
                    metrics,
                    metrics_keys,
                    [m.key.replace(" ", "").lower() for m in ALL_METRICS],
                ),
            )

    if verbose:
        print("\n")
        print(
            "computing {} metrics: {} for {} benches: {}".format(
                len(selected_metrics),
                [m.key for m in selected_metrics],
                len(benches),
                benches,
            )
        )

    baseline_cores_per_cluster = benchmarks.BASELINE["cores_per_cluster"]
    baseline_num_clusters = benchmarks.BASELINE["num_clusters"]
    if scaled_clusters:
        functional_config = FunctionalConfig(
            cores_per_cluster=baseline_cores_per_cluster,
            num_clusters=baseline_num_clusters * 4,
        )
    elif scaled_cores:
        functional_config = FunctionalConfig(
            cores_per_cluster=baseline_cores_per_cluster * 4,
            num_clusters=baseline_num_clusters,
        )
    else:
        functional_config = FunctionalConfig(
            cores_per_cluster=baseline_cores_per_cluster,
            num_clusters=baseline_num_clusters,
        )

    pprint(functional_config)
    target_dfs = split_into_target_dfs(
        df, per_kernel=False, mean=True, functional_config=functional_config
    )

    # native_df = target_dfs.native_df
    # accelsim_df = target_dfs.accelsim_df
    # serial_gpucachesim_df = target_dfs.serial_gpucachesim_df
    # serial_gpucachesim_mem_only_df = target_dfs.serial_gpucachesim_mem_only_df
    # serial_gpucachesim_exec_driven_df = target_dfs.serial_gpucachesim_exec_driven_df

    # dtypes = {
    #     **{col: "float64" for col in native_df.columns},
    #     **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    # }
    # dtypes = {col: dtype for col, dtype in dtypes.items() if col in native_df}
    # native_df = native_df.astype(dtypes)

    dtypes = dict()
    sim_targets = {
        "accelsim": target_dfs.accelsim_df.astype(dtypes),
        "gpucachesim": target_dfs.serial_gpucachesim_df.astype(dtypes),
        "gpucachesim_mem_only": target_dfs.serial_gpucachesim_mem_only_df.astype(
            dtypes
        ),
        "gpucachesim_exec_driven": target_dfs.serial_gpucachesim_exec_driven_df.astype(
            dtypes
        ),
    }

    joined_df = join_targets(target_dfs, sim_targets, large=large, verbose=verbose)

    # remove nan rows
    joined_df = joined_df[~joined_df["input_id_gpucachesim"].isna()]
    assert len(joined_df) > 0

    for target in list(sim_targets.keys()) + [""]:
        suffix = ("_" + target) if target != "" else ""
        num_global_loads = joined_df["num_global_loads"]
        num_global_stores = joined_df["num_global_stores"]

        joined_df["dram_reads_percent" + suffix] = joined_df[
            "dram_reads" + suffix
        ].fillna(0.0)
        # scale = (
        #     joined_df[["num_global_loads", "num_global_stores"]].max(axis=1) + 0.00001
        # )
        joined_df["dram_reads_percent" + suffix] /= num_global_loads

        joined_df["dram_writes_percent" + suffix] = joined_df[
            "dram_writes" + suffix
        ].fillna(0.0)
        joined_df["dram_writes_percent" + suffix] /= num_global_stores

        # assert (joined_df["dram_writes_percent" + suffix] <= 1.0).all()
        # assert (joined_df["dram_reads_percent" + suffix] <= 1.0).all()

    if verbose:
        for target in list(sim_targets.keys()):
            print(target)
            suffix = "_" + target
            print(
                joined_df[
                    [
                        "num_global_loads",
                        "dram_reads",
                        "dram_reads_percent",
                        "dram_reads" + suffix,
                        "dram_reads_percent" + suffix,
                    ]
                ]
            )
            print(
                joined_df[
                    [
                        "num_global_stores",
                        "dram_writes",
                        "dram_writes_percent",
                        "dram_writes" + suffix,
                        "dram_writes_percent" + suffix,
                    ]
                ]
            )

    assert all(
        [
            err.column in joined_df
            for err in utils.flatten([m.error_metrics for m in selected_metrics])
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

    if verbose:
        for metric in selected_metrics:
            metric_cols = sorted(
                list(set([err.column for err in metric.error_metrics]))
            )
            print("==> PREVIEW: {}".format(metric_cols))
            preview_cols = [
                "benchmark",
                "input_id",
                # "num_global_loads",
                # "num_global_stores",
            ] + [
                col + "_" + target
                for col, target in itertools.product(
                    metric_cols,
                    [""] + list(sim_targets.keys()),
                    # ["", "_accelsim", "_gpucachesim"],
                )
            ]
            print(joined_df[preview_cols])

    if (bench_name is None and combined_only) and not no_combined:
        selected_benches = [None]
    elif bench_name is None and no_combined:
        selected_benches = benches
    elif bench_name is None:
        selected_benches = benches + [None]
    else:
        selected_benches = [bench_name]

    table = ""
    for bench in selected_benches:
        if bench is None:
            label = "Average"
        else:
            label = benchmarks.benchmark_name_human_readable(bench)

        if bench is not None:
            bench_df = joined_df[joined_df["benchmark"] == bench]
        else:
            bench_df = joined_df

        # print(bench_df[["target", "benchmark", "input_id", "input_id_accelsim", "input_id_gpucachesim", "input_id_gpucachesim_mem_only", "input_id_gpucachesim_exec_driven"]])
        assert len(bench_df) > 0
        total_cores = bench_df["total_cores_gpucachesim"].dropna().unique()
        # print(total_cores)
        # if not scaled_cores:
        assert len(total_cores) == 1
        total_cores = int(total_cores[0])

        num_unique_bench_configs = len(
            bench_df[["benchmark", "input_id_gpucachesim"]].dropna().drop_duplicates()
        )

        label += " ({} benchmark configurations) @ {} SM's".format(
            num_unique_bench_configs, total_cores
        )
        if large:
            label += " [blocks/SM > 1]"

        table += r"\rowcolor{gray!10}"
        table += r"\multicolumn{6}{c}{\textbf{" + label + r"}} \\"
        if bench is None:
            table += r"\hline \hline"
        else:
            table += r"\hline"
        table += "\n"

        for metric in selected_metrics:
            if verbose:
                print(bench, metric.label)

            table += r"\multirow{"
            table += str(len(metric.error_metrics))
            table += r"}{*}{\shortstack[r]{"
            table += " ".join(str(metric.label).split("_"))
            table += "}} \n"

            for err in metric.error_metrics:
                preview_cols = ["benchmark"] + [
                    col + "_" + target
                    for col, target in itertools.product(
                        [err.column], [""] + list(sim_targets.keys())
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

                bench_df = bench_df.sort_values("input_id")
                if bench is not None:
                    print(bench_df[["target", "benchmark", "input_id"]])
                    assert bench_df["input_id"].nunique() == len(bench_df["input_id"])

                # print(bench_df[["num_global_loads", "dram_reads", "dram_reads_percent", "dram_reads_gpucachesim", "dram_reads_percent_gpucachesim", "dram_reads_accelsim", "dram_reads_percent_accelsim"]])

                metric_is_percent = err.is_percent
                value_scale = 100.0 if metric_is_percent else 1.0

                match err.metric:
                    case ErrorMetricID.Correlation:
                        error_values = []
                        for target in sim_targets.keys():
                            if False:
                                print(target)
                                print(
                                    bench_df[
                                        ["target", "benchmark", "input_id"]
                                        + benchmarks.BENCHMARK_INPUT_COLS[bench or ""]
                                        + [err.column, err.column + "_" + target]
                                    ]
                                )

                            # print(err.column + "_" + target)
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            atol = 1.0 if metric_is_percent else 0.1
                            error = metric_funcs.correlation(
                                true_values=true_values, values=values, atol=atol
                            )
                            bench_df[
                                err.column + "_" + err.metric.name.lower() + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df = error_values_df.mean(axis=1)

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

                    case ErrorMetricID.EMALE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.emale(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df = error_values_df.mean(axis=1)

                    case ErrorMetricID.ERMSLE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.ermsle(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df = error_values_df.mean(axis=1)

                    case ErrorMetricID.MAE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.abs_err(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error[:] = np.nan
                            # print(target, err.column + "_" + target)
                            # print(error)
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        # print("MAE: ")
                        # print(error_values_df)
                        error_values_df = error_values_df.mean(axis=1)
                        # print(error_values_df)

                    case ErrorMetricID.SMAPE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.smape(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df *= 100.0
                        error_values_df = error_values_df.mean(axis=1)

                    case ErrorMetricID.MAPE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.mape(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df *= 100.0
                        error_values_df = error_values_df.mean(axis=1)
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

                    case ErrorMetricID.RMSPE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.rmspe(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df *= 100.0
                        error_values_df = error_values_df.mean(axis=1)
                    case ErrorMetricID.RMSE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[err.column] * value_scale
                            values = bench_df[err.column + "_" + target] * value_scale
                            error = metric_funcs.rmse(
                                true_values=true_values, values=values
                            )
                            bench_df[
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                            ] = error
                            # if bench_df["cycles" + "_" + target].isna().all():
                            #     error = np.nan
                            error_values.append(error)
                        error_values_df = pd.DataFrame(error_values)
                        error_values_df *= 100.0
                        error_values_df = error_values_df.mean(axis=1)

                    case _:
                        raise ValueError(
                            "unknown error metric {}".format(err.metric.name)
                        )

                assert isinstance(error_values_df, (np.ndarray, pd.Series))
                for col, target in enumerate(sim_targets.keys()):
                    target_valid = not np.isnan(bench_df["cycles" + "_" + target]).all()
                    valid = not np.isnan(bench_df[err.column + "_" + target]).all()
                    if not (valid and target_valid):
                        error_values_df[col] = np.nan

                table += r" & {} ".format(err.metric.value)
                if verbose:
                    print(err.metric.name)
                    print(error_values_df)
                for value in error_values_df:
                    table += " & "
                    if np.isnan(value):
                        continue
                    match err.metric:
                        case ErrorMetricID.Correlation:
                            # if value == np.nanmax(error_values_df):
                            precision = 3
                            # print("value", plot.round_to_precision(value, precision))
                            # print("best", plot.round_to_precision(np.nanmax(error_values_df), precision))
                            if plot.round_to_precision(
                                value, precision
                            ) == plot.round_to_precision(
                                np.nanmax(error_values_df), precision
                            ):
                                # if np.allclose([value], [np.nanmax(error_values_df)], atol=1e-4):
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
                        case (
                            ErrorMetricID.SMAPE
                            | ErrorMetricID.MAPE
                            | ErrorMetricID.RMSPE
                        ):
                            precision = 2
                            # atol = 10**-(precision+1)
                            # print("atol", atol)
                            if plot.round_to_precision(
                                value, precision
                            ) == plot.round_to_precision(
                                np.nanmin(error_values_df), precision
                            ):
                                # np.allclose([value], [np.nanmin(error_values_df)], atol=atol):
                                # if value == np.nanmin(error_values_df):
                                table += r"\boldmath"
                            table += "${}\\%$".format(
                                plot.human_format_thousands(value, round_to=precision)
                            )
                        case (
                            ErrorMetricID.EMALE
                            | ErrorMetricID.RMSE
                            | ErrorMetricID.ERMSLE
                            | ErrorMetricID.MAE
                        ):
                            precision = 2
                            # atol = 10**-(precision+1)
                            # print("atol", atol)
                            # if np.allclose([value], [np.nanmin(error_values_df)], atol=atol):
                            if plot.round_to_precision(
                                value, precision
                            ) == plot.round_to_precision(
                                np.nanmin(error_values_df), precision
                            ):
                                # if value == np.nanmin(error_values_df):
                                table += r"\boldmath"
                            if metric_is_percent:
                                table += "${:5.2f}\\%$".format(value)
                            else:
                                table += "${}$".format(
                                    plot.human_format_thousands(
                                        value, round_to=precision
                                    )
                                )
                        case other:
                            raise ValueError("unhandled metric {}".format(other))

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
                                err.column
                                + "_"
                                + err.metric.name.lower()
                                + "_"
                                + target
                                for target in ["accelsim", "gpucachesim"]
                            ]
                            # + [sim + "_rpd" for sim in ["accelsim", "gpucachesim"]]
                        ].fillna(0.0)
                    )

            # if bench is not None:
            table += r" \hline"
            table += "\n"

        table += "%\n%\n"

    if not batch:
        print("")
        print(table)
        utils.copy_to_clipboard(table)
        print("copied table to clipboard")

    tex_code = r"""
\documentclass[preview]{standalone}
"""
    tex_code += utils.TEX_PACKAGES
    tex_code += r"""
\begin{document}
"""

    tex_code += r"""
\begin{table}[htbp]
\fontsize{8}{10}\selectfont
\footnotesize
"""
    caption = "Results"
    tex_code += r"\caption{\small " + caption + "}"
    tex_code += r"""
\centering
% \setlength\extrarowheight{2pt}
% \rowcolors{2}{white}{gray!20}
{\renewcommand{\arraystretch}{1.5}%
\begin{tabularx}{\textwidth}{ss|z|z|z|z}
& & \shortstack[t]{\textsc{AccelSim}} 
  & \shortstack[t]{\textsc{gpucachesim}} 
  & \shortstack[c]{\textsc{gpucachesim}\\\textit{(memory only)}} 
  & \shortstack[c]{\textsc{gpucachesim}\\\textit{(trace reconstr.)}} \\ 
\hline
"""
    tex_code += table
    tex_code += r"""
%
\end{tabularx}}
\end{table}
"""
    tex_code += r"""
\end{document}
"""

    filename = "result_table"
    if bench_name is None:
        filename += "_all"
    else:
        filename += "_{}".format(bench_name)
    if combined_only:
        filename += "_combined_only"
    elif no_combined:
        filename += "_no_combined"
    if large:
        filename += "_large"
    if scaled_clusters:
        filename += "_scaled_clusters"
    elif scaled_cores:
        filename += "_scaled_cores"

    filename += "_{}".format(
        "_".join(
            [metric.label.lower().replace(" ", "_") for metric in selected_metrics]
        )
    )
    filename = sanitize_filename(filename)
    pdf_output_path = (plot.TABLE_DIR / filename).with_suffix(".pdf")
    try:
        utils.render_latex(tex_code, output_path=pdf_output_path)
        pass
    except Exception as e:
        print(tex_code)
        raise e
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))

    if png:
        png_output_path = (plot.TABLE_DIR / "png" / filename).with_suffix(".png")
        utils.convert_to_png(input_path=pdf_output_path, output_path=png_output_path)
        print(color("wrote {}".format(png_output_path), fg="cyan"))
