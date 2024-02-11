import enum
import typing
import itertools
import numpy as np
import pandas as pd
from wasabi import color
from pathvalidate import sanitize_filename

import gpucachesim.stats.agg
import gpucachesim.stats.metrics as metric_funcs
import gpucachesim.benchmarks as benchmarks
import gpucachesim.utils as utils
import gpucachesim.plot as plot


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


def result_table(df, bench_name: typing.Optional[str]=None, metrics: typing.Optional[typing.Union[str, typing.List[typing.Optional[str]]]]=None, combined_only=False, verbose=False, batch=False, png=False):
    # remove non-kernel results
    df = df[~df["kernel_name"].isna()]

    # target benchmark histogram
    target_bench_input_count_hist = (
        df[["target", "benchmark", "input_id"]]
        .drop_duplicates()
        .value_counts(["target", "benchmark"], dropna=False)
        .sort_index()
    )
    if verbose:
        print(target_bench_input_count_hist)

    target_dfs = gpucachesim.stats.agg.split_into_target_dfs(df, per_kernel=False, mean=True)
    native_df = target_dfs.native_df
    accelsim_df = target_dfs.accelsim_df
    serial_gpucachesim_df = target_dfs.serial_gpucachesim_df
    serial_gpucachesim_mem_only_df = target_dfs.serial_gpucachesim_mem_only_df
    serial_gpucachesim_exec_driven_df = target_dfs.serial_gpucachesim_exec_driven_df

    class Metric(typing.TypedDict):
        label: str
        is_percent: bool
        error_metrics: typing.Sequence[typing.Tuple[str, ErrorMetric]]

    benches = sorted(df["benchmark"].unique().tolist())
    all_metrics = [
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
        ),
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
    ]

    if metrics is None:
        metrics_keys = []
    elif isinstance(metrics, str):
        metrics_keys = [metrics]
    elif isinstance(metrics, list):
        metrics_keys = metrics
    else:
        raise ValueError("metrics must be either a string or list of strings, have {}".format(metrics))

    metrics_keys = [metric.replace(" ", "").lower() for metric in metrics_keys if metric is not None]

    if len(metrics_keys) == 0:
        # only show cycles by default
        selected_metrics = [all_metrics[-1]]
    else:
        selected_metrics = [
            m for m in all_metrics if m["label"].replace(" ", "").lower() in metrics_keys
        ]
        if len(selected_metrics) == 0:
            raise ValueError(
                "invalid metrics {} ({}), have {}",
                metrics,
                metrics_keys,
                [m["label"].replace(" ", "").lower() for m in all_metrics],
            )

    if verbose:
        print("\n")
        print(
            "computing {} metrics: {} for {} benches: {}".format(
                len(selected_metrics),
                [m["label"] for m in selected_metrics],
                len(benches),
                benches,
            )
        )

    # dtypes = {
    #     **{col: "float64" for col in native_df.columns},
    #     **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    # }
    # dtypes = {col: dtype for col, dtype in dtypes.items() if col in native_df}
    # native_df = native_df.astype(dtypes)

    dtypes = dict()
    sim_targets = {
        "accelsim": accelsim_df.astype(dtypes),
        "gpucachesim": serial_gpucachesim_df.astype(dtypes),
        "gpucachesim_mem_only": serial_gpucachesim_mem_only_df.astype(dtypes),
        "gpucachesim_exec_driven": serial_gpucachesim_exec_driven_df.astype(dtypes),
    }

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
        join_cols = [col for col in join_cols if col in df]
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
            # if target == "_gpucachesim_parallel":
            #     # temp: ignore for now
            #     pass
            if target == "gpucachesim_exec_driven":
                # we do not have an exec driven version of babelstream
                missing_exec_driven_benches = sorted(missing_df["benchmark"].unique().tolist())
                if missing_exec_driven_benches != ["babelstream"]:
                    print("MISSING {}".format(missing_df.shape))
                    print(missing_df)
                    raise ValueError(
                        "missing exec driven {} but should only miss babelstream".format(missing_exec_driven_benches)
                    )
            else:
                print("MISSING {}".format(missing_df.shape))
                print(missing_df)
                assert len(missing_df) == 0

        joined_df = native_df.merge(
            sim_df,
            on=join_cols,
            how="left",
            suffixes=(None, "_" + target),
        )
        assert joined_df.shape[0] == native_df.shape[0]
        if len(joined_df) == 0:
            raise ValueError("joined dataframe is empty")

        native_df = joined_df
        # break

    for target in list(sim_targets.keys()) + [""]:
        suffix = ("_" + target) if target != "" else ""
        native_df["dram_reads_percent" + suffix] = native_df["dram_reads" + suffix].fillna(0.0)
        scale = native_df[["num_global_loads", "num_global_stores"]].max(axis=1) + 0.00001
        native_df["dram_reads_percent" + suffix] /= scale
        native_df["dram_writes_percent" + suffix] = native_df["dram_writes" + suffix].fillna(0.0)
        native_df["dram_writes_percent" + suffix] /= scale
        assert (native_df["dram_writes_percent" + suffix] <= 1.0).all()
        assert (native_df["dram_reads_percent" + suffix] <= 1.0).all()

    assert all([col in native_df for col, _ in utils.flatten([m["error_metrics"] for m in selected_metrics])])

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
            metric_cols = sorted(list(set([metric_col for metric_col, _ in metric["error_metrics"]])))
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
            print(native_df[preview_cols])


    if bench_name is None and combined_only:
        selected_benches = [None]
    elif bench_name is None:
        selected_benches = benches + [None]
    else:
        selected_benches = benches

    table = ""
    for bench in selected_benches:
        if bench is None:
            header_label = "Combined"
        else:
            header_label = benchmarks.benchmark_name_human_readable(bench)

        table += r"\rowcolor{gray!10}"
        table += r"\multicolumn{6}{c}{\textbf{" + header_label + r"}} \\"
        if bench is None:
            table += r"\hline \hline"
        else:
            table += r"\hline"
        table += "\n"

        for metric in selected_metrics:
            if verbose:
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
                    col + "_" + target
                    for col, target in itertools.product([metric_col], [""] + list(sim_targets.keys()))
                ]

                bench_df = bench_df.copy()
                if bench is not None and verbose:
                    print(bench_df[preview_cols + benchmarks.BENCHMARK_INPUT_COLS[bench]].fillna(0.0))
                    print(bench_df.shape)

                error_values: pd.DataFrame

                metric_is_percent = metric["is_percent"]
                value_scale = 100.0 if metric_is_percent else 1.0

                match error_metric:
                    case ErrorMetric.Correlation:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + "_" + target] * value_scale
                            atol = 1.0 if metric_is_percent else 0.1
                            error = metric_funcs.correlation(true_values=true_values, values=values, atol=atol)
                            bench_df[metric_col + "_" + error_metric.name.lower() + target] = error
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
                        for target in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + "_" + target] * value_scale
                            error = metric_funcs.emale(true_values=true_values, values=values)
                            bench_df[metric_col + "_" + error_metric.name.lower() + "_" + target] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.ERMSLE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + "_" + target] * value_scale
                            error = metric_funcs.ermsle(true_values=true_values, values=values)
                            bench_df[metric_col + "_" + error_metric.name.lower() + "_" + target] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.MAE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + "_" + target] * value_scale
                            error = metric_funcs.abs_err(true_values=true_values, values=values)
                            bench_df[metric_col + "_" + error_metric.name.lower() + "_" + target] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.SMAPE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + "_" + target] * value_scale
                            error = metric_funcs.smape(true_values=true_values, values=values)
                            bench_df[metric_col + "_" + error_metric.name.lower() + "_" + target] = error
                            error_values.append(error)
                        error_values = pd.DataFrame(error_values)
                        error_values *= 100.0
                        error_values = error_values.mean(axis=1)

                    case ErrorMetric.MAPE:
                        error_values = []
                        for target in sim_targets.keys():
                            true_values = bench_df[metric_col] * value_scale
                            values = bench_df[metric_col + "_" + target] * value_scale
                            error = metric_funcs.mape(true_values=true_values, values=values)
                            bench_df[metric_col + "_" + error_metric.name.lower() + "_" + target] = error
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
                        raise ValueError("unknown error metric {}".format(error_metric.name))

                # assert isinstance(error_values, (np.ndarray, pd.Series))
                for col, target in enumerate(sim_targets.keys()):
                    valid = not np.isnan(bench_df[metric_col + "_" + target]).all()
                    if not valid:
                        error_values[col] = np.nan

                table += r" & {} ".format(error_metric.value)
                if verbose:
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
                            table += "${}\\%$".format(plot.human_format_thousands(value))
                        case ErrorMetric.EMALE | ErrorMetric.ERMSLE | ErrorMetric.MAE:
                            if value == np.nanmin(error_values):
                                table += r"\boldmath"
                            if metric_is_percent:
                                table += "${:5.2f}\\%$".format(value)
                            else:
                                table += "${}$".format(plot.human_format_thousands(value))

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
                                metric_col + "_" + error_metric.name.lower() + "_" + target
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
    filename += "_{}".format("_".join([metric["label"].lower().replace(" ", "_") for metric in selected_metrics]))
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
