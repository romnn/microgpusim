import copy
import numpy as np
import pandas as pd
import itertools
from pprint import pprint
from wasabi import color

from gpucachesim.stats.agg import TargetDataframes, split_into_target_dfs
import gpucachesim.plot as plot
import gpucachesim.stats.metrics as metrics
import gpucachesim.benchmarks as benchmarks
import gpucachesim.tex as tex
import gpucachesim.utils as utils
import gpucachesim.stats.result_table
from gpucachesim.benchmarks import Target


def choose_fastest_parallel_implementation(df) -> pd.DataFrame:
    bench_input_cols = copy.deepcopy(list(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    # note, we do NOT group by SIMULATE_EXECUTION_CONFIG_COLS or SIMULATE_INPUT_COLS.
    # this means we do NOT group on input_mode, input_run_ahead, or input_threads
    functinoal_input_cols = copy.deepcopy(benchmarks.SIMULATE_FUNCTIONAL_CONFIG_COLS)
    input_config_group_cols = ["target", "benchmark"] + functinoal_input_cols + bench_input_cols
    input_config_group_cols = [col for col in input_config_group_cols if col in df]

    group_cols = input_config_group_cols + ["run"]
    min_exec_times = df.groupby(group_cols, dropna=False)["exec_time_sec"].transform("min")
    df = df[df["exec_time_sec"] == min_exec_times]
    return df


def speed_table(
    df,
    bench_name,
    include_mean_time=False,
    large=False,
    combined_only=False,
    no_combined=False,
    verbose=False,
    batch=False,
    inspect=False,
    png=False,
):
    # remove non-kernel results
    no_kernel_mask = df["kernel_name"].isna()
    df = df[~no_kernel_mask]

    if large:
        # print(
        #     df.loc[
        #         (df["benchmark"] == "babelstream") & (df["target"] == Target.AccelsimSimulate.value),
        #         ["mean_blocks_per_sm", "mean_blocks_per_sm_all_kernels"],
        #     ]
        # )

        profile = df["target"] == Target.Profile.value
        # df = df[profile | (df["mean_blocks_per_sm"] > 1.0)]
        df = df[profile | (df["mean_blocks_per_sm_all_kernels"] > 1.0)]

        if len(df) < 1:
            print(
                color(
                    "{} has no large configurations with blocks/SM > 1".format(bench_name),
                    fg="red",
                )
            )
            return

    # print(df.loc[
    #     (df["target"] == Target.Simulate.value)
    #         & (df["input_id"] == 210),
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    # print(df.loc[
    #     (df["target"] == Target.AccelsimSimulate.value)
    #         & (df["input_id"] == 3),
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    target_dfs = split_into_target_dfs(df, per_kernel=False, mean=True, inspect=inspect)

    # print(target_dfs.serial_gpucachesim_df.loc[
    #     target_dfs.serial_gpucachesim_df["input_id"] == 210,
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    # print(target_dfs.accelsim_df.loc[
    #     target_dfs.accelsim_df["input_id"] == 3,
    #     benchmarks.PREVIEW_COLS + ["cycles", "exec_time_sec"]].T)

    # native_df = target_dfs.native_df
    # accelsim_df = target_dfs.accelsim_df
    # serial_gpucachesim_df = target_dfs.serial_gpucachesim_df
    # serial_gpucachesim_mem_only_df = target_dfs.serial_gpucachesim_mem_only_df
    # serial_gpucachesim_exec_driven_df = target_dfs.serial_gpucachesim_exec_driven_df
    parallel_gpucachesim_df = choose_fastest_parallel_implementation(target_dfs.parallel_gpucachesim_df)
    print("{:>50}\t{}".format("fastest parallel gpucachesim", parallel_gpucachesim_df.shape))

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
        "gpucachesim_mem_only": target_dfs.serial_gpucachesim_mem_only_df.astype(dtypes),
        "gpucachesim_exec_driven": target_dfs.serial_gpucachesim_exec_driven_df.astype(dtypes),
        "gpucachesim_parallel": parallel_gpucachesim_df.astype(dtypes),
    }

    # mask = sim_targets["accelsim"]["benchmark"] == "babelstream"
    # print(sim_targets["accelsim"].loc[mask, ["target", "benchmark", "exec_time_sec"]])

    # if verbose:
    #     print("\n")
    #
    # for target, sim_df in sim_targets.items():
    #     if verbose:
    #         print("computing =>", target)
    #     # print(sim_df[benchmarks.PREVIEW_COLS][:4].T)
    #     join_cols = list(
    #         # we do NOT join based on target
    #         ["benchmark", "kernel_launch_id"]
    #         + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
    #         # we do NOT join based on input_memory_only
    #         + ["input_num_clusters", "input_cores_per_cluster"],
    #     )
    #     join_cols = [col for col in join_cols if col in df]
    #     # pprint(join_cols)
    #
    #     missing_df = (
    #         native_df[join_cols]
    #         .merge(
    #             sim_df[join_cols],
    #             how="left",
    #             indicator=True,
    #         )
    #         .loc[lambda x: x["_merge"] != "both"]
    #     )
    #     if len(missing_df) > 0:
    #         if large:
    #             pass
    #         # if target == "gpucachesim_parallel":
    #         #     # temp: ignore for now
    #         #     pass
    #         elif target == "gpucachesim_exec_driven":
    #             # we do not have an exec driven version of babelstream
    #             missing_exec_driven_benches = sorted(missing_df["benchmark"].unique().tolist())
    #             if missing_exec_driven_benches != ["babelstream"]:
    #                 print("MISSING {}".format(missing_df.shape))
    #                 print(missing_df)
    #                 raise ValueError(
    #                     "missing exec driven {} but should only miss babelstream".format(missing_exec_driven_benches)
    #                 )
    #         else:
    #             print("MISSING {}".format(missing_df.shape))
    #             print(missing_df)
    #             assert len(missing_df) == 0
    #
    #     joined_df = native_df.merge(
    #         sim_df,
    #         on=join_cols,
    #         how="left",
    #         suffixes=(None, "_" + target),
    #     )
    #     assert joined_df.shape[0] == native_df.shape[0]
    #     if len(joined_df) == 0:
    #         raise ValueError("joined dataframe is empty")
    #
    #     native_df = joined_df

    joined_df = gpucachesim.stats.result_table.join_targets(target_dfs, sim_targets, large=large, verbose=verbose)

    # remove nan rows
    joined_df = joined_df[~joined_df["input_id_gpucachesim"].isna()]

    joined_df["exec_time_nsec"] = joined_df["exec_time_sec"] * 1e9
    # preview_metrics = ["cycles", "instructions", "exec_time_sec", "input_id"]
    preview_metrics = ["input_id", "kernel_name", "exec_time_sec"]
    preview_cols = ["benchmark", "exec_time_nsec"] + [
        col + "_" + target for col, target in itertools.product(preview_metrics, [""] + list(sim_targets.keys()))
    ]

    benches = sorted(df["benchmark"].unique().tolist())

    all_slowdowns_over_native = []

    if (bench_name is None and combined_only) and not no_combined:
        selected_benches = [None]
    elif bench_name is None and no_combined:
        selected_benches = benches
    elif bench_name is None:
        selected_benches = benches + [None]
    else:
        selected_benches = benches

    table = ""
    for bench in selected_benches:
        if verbose:
            print(bench)
        if bench is not None:
            bench_df = joined_df[joined_df["benchmark"] == bench]
        else:
            bench_df = joined_df

        bench_df = bench_df.copy()

        if bench is None:
            primary_label = "Average"
        else:
            primary_label = benchmarks.benchmark_name_human_readable(bench)

        assert len(bench_df) > 0

        total_cores = bench_df["total_cores_gpucachesim"].dropna().unique()
        assert len(total_cores) == 1
        total_cores = int(total_cores[0])

        num_unique_bench_configs = len(bench_df[["benchmark", "input_id_gpucachesim"]].dropna().drop_duplicates())

        secondary_label = " ({} benchmark configurations) @ {} SM's".format(num_unique_bench_configs, total_cores)
        if large:
            secondary_label += r" [blocks/SM $> 1$]"

        table += r"\rowcolor{gray!10}"
        table += r"\multicolumn{6}{c}{\textbf{" + primary_label + r"}" + secondary_label + r"} \\"
        if bench is None:
            table += r" \hline \hline"
        else:
            table += r" \hline"
        table += "\n"

        if verbose:
            print(bench_df[preview_cols + benchmarks.BENCHMARK_INPUT_COLS[bench or ""]])
            print(bench_df.shape)

        table += r"Slowdown"

        bench_df["slowdown_accelsim"] = metrics.slowdown(
            baseline=bench_df["exec_time_sec"],
            values=bench_df["exec_time_sec_accelsim"],
        )
        bench_df["slowdown_gpucachesim"] = metrics.slowdown(
            baseline=bench_df["exec_time_sec"],
            values=bench_df["exec_time_sec_gpucachesim"],
        )
        bench_df["slowdown_gpucachesim_mem_only"] = metrics.slowdown(
            baseline=bench_df["exec_time_sec"],
            values=bench_df["exec_time_sec_gpucachesim_mem_only"],
        )
        bench_df["slowdown_gpucachesim_exec_driven"] = metrics.slowdown(
            baseline=bench_df["exec_time_sec"],
            values=bench_df["exec_time_sec_gpucachesim_exec_driven"],
        )
        bench_df["slowdown_gpucachesim_exec_driven_excl_trace"] = metrics.slowdown(
            baseline=bench_df["exec_time_sec"],
            values=bench_df["exec_time_sec_gpucachesim_exec_driven"]
            - bench_df["trace_time_sec_gpucachesim_exec_driven"],
        )
        bench_df["slowdown_gpucachesim_parallel"] = metrics.slowdown(
            baseline=bench_df["exec_time_sec"],
            values=bench_df["exec_time_sec_gpucachesim_parallel"],
        )

        slowdowns_over_native = bench_df[
            [
                "slowdown_accelsim",
                "slowdown_gpucachesim",
                "slowdown_gpucachesim_mem_only",
                "slowdown_gpucachesim_exec_driven",
                "slowdown_gpucachesim_parallel",
            ]
        ]
        # slowdowns_over_native = [
        #     metrics.slowdown(
        #         baseline=bench_df["exec_time_sec"],
        #         values=bench_df["exec_time_sec_accelsim"],
        #     ),
        #     metrics.slowdown(
        #         baseline=bench_df["exec_time_sec"],
        #         values=bench_df["exec_time_sec_gpucachesim"],
        #     ),
        #     metrics.slowdown(
        #         baseline=bench_df["exec_time_sec"],
        #         values=bench_df["exec_time_sec_gpucachesim_mem_only"],
        #     ),
        #     metrics.slowdown(
        #         baseline=bench_df["exec_time_sec"],
        #         values=bench_df["exec_time_sec_gpucachesim_exec_driven"],
        #     ),
        #     metrics.slowdown(
        #         baseline=bench_df["exec_time_sec"],
        #         values=bench_df["exec_time_sec_gpucachesim_parallel"],
        #     ),
        # ]
        # assert all([len(s) == len(slowdowns_over_native[0]) for s in slowdowns_over_native])

        if False and bench == "babelstream":
            print(
                bench_df[
                    [
                        "exec_time_sec",
                        "exec_time_sec_accelsim",
                        "exec_time_sec_gpucachesim",
                    ]
                ]
            )

        USE_MEDIAN = True

        def mean_or_median(series):
            if USE_MEDIAN:
                return series.median()
            else:
                return series.mean()

        # if USE_MEDIAN:
        #     slowdowns_over_native = slowdowns_over_native.median()
        # else:
        #     slowdowns_over_native = slowdowns_over_native.mean(axis=0)
        slowdowns_over_native = mean_or_median(slowdowns_over_native)

        # if bench is None:
        #     slowdowns_over_native = np.nanmean(slowdowns_over_native, axis=1)
        # else:
        #     slowdowns_over_native = np.mean(slowdowns_over_native, axis=1)

        # print(slowdowns_over_native.index)
        # print(slowdowns_over_native)
        # print(slowdowns_over_native.values)
        # print(slowdowns_over_native.reset_index())
        # print(slowdowns_over_native.to_frame())
        # print(slowdowns_over_native.T)
        all_slowdowns_over_native.append(slowdowns_over_native.to_frame().T)

        for target, slowdown_value in slowdowns_over_native.items():
            table += " & "
            if np.isnan(slowdown_value):
                continue
            bold = np.isfinite(slowdown_value) and slowdown_value == np.nanmin(slowdowns_over_native)
            if bold:
                table += r"\boldmath"

            if target.endswith("gpucachesim_exec_driven"):
                # also show the value without trace recon
                slowdown_excl_trace = mean_or_median(bench_df["slowdown_gpucachesim_exec_driven_excl_trace"])

                table += "${}$".format(plot.human_format_thousands(slowdown_excl_trace))
                # table += "${}$".format(plot.human_format_thousands(slowdown_excl_trace))
                # table += "$({} [{}])$".format(
                #     plot.human_format_thousands(slowdown_value),
                #     plot.human_format_thousands(slowdown_excl_trace),
                # )
            else:
                table += "${}$".format(plot.human_format_thousands(slowdown_value))

        table += r"\\" + "\n"

        table += r"KIPS"
        native_kilo_instructions = bench_df["instructions"] / 1000.0

        # compute accelsim KIPS
        bench_df["kips_accelsim"] = native_kilo_instructions / bench_df["exec_time_sec_accelsim"]

        # compute serial gpucachesim KIPS
        bench_df["kips_gpucachesim"] = native_kilo_instructions / bench_df["exec_time_sec_gpucachesim"]

        # compute memory only gpucachesim KIPS
        bench_df["kips_gpucachesim_mem_only"] = (bench_df["instructions_gpucachesim_mem_only"] / 1000.0) / bench_df[
            "exec_time_sec_gpucachesim_mem_only"
        ]

        # compute exec driven gpucachesim KIPS
        bench_df["kips_gpucachesim_exec_driven"] = (
            bench_df["instructions_gpucachesim_exec_driven"] / 1000.0
        ) / bench_df["exec_time_sec_gpucachesim_exec_driven"]

        # compute exec driven gpucachesim KIPS excluding trace reconstruction time
        bench_df["kips_gpucachesim_exec_driven_excl_trace"] = (
            bench_df["instructions_gpucachesim_exec_driven"] / 1000.0
        ) / (bench_df["exec_time_sec_gpucachesim_exec_driven"] - bench_df["trace_time_sec_gpucachesim_exec_driven"])

        # compute parallel gpucachesim KIPS
        bench_df["kips_gpucachesim_parallel"] = (
            native_kilo_instructions / bench_df["exec_time_sec_gpucachesim_parallel"]
        )

        # find weird results
        accelsim_faster_kips_mask = bench_df["kips_accelsim"] > bench_df["kips_gpucachesim"]
        accelsim_faster_slowdown_mask = bench_df["slowdown_accelsim"] < bench_df["slowdown_gpucachesim"]

        accelsim_should_be_faster_kips_mask = bench_df["kips_accelsim"] > bench_df["kips_gpucachesim"]
        accelsim_should_be_faster_slowdown_mask = bench_df["slowdown_accelsim"] < bench_df["slowdown_gpucachesim"]

        gpucachesim_faster_kips_mask = bench_df["kips_gpucachesim"] > bench_df["kips_accelsim"]
        gpucachesim_faster_slowdown_mask = bench_df["slowdown_gpucachesim"] < bench_df["slowdown_accelsim"]

        gpucachesim_should_be_faster_kips_mask = bench_df["kips_gpucachesim"] > bench_df["kips_accelsim"]
        gpucachesim_should_be_faster_slowdown_mask = bench_df["slowdown_gpucachesim"] < bench_df["slowdown_accelsim"]

        if (accelsim_should_be_faster_kips_mask & ~accelsim_should_be_faster_slowdown_mask).sum() > 0:
            print(bench_df[accelsim_should_be_faster_kips_mask & ~accelsim_should_be_faster_slowdown_mask])

            raise ValueError("debug this")

        if (gpucachesim_should_be_faster_kips_mask & ~gpucachesim_should_be_faster_slowdown_mask).sum() > 0:
            print(bench_df[gpucachesim_should_be_faster_kips_mask & ~gpucachesim_should_be_faster_slowdown_mask])

            raise ValueError("debug this")

        kips = bench_df[
            [
                "kips_accelsim",
                "kips_gpucachesim",
                "kips_gpucachesim_mem_only",
                "kips_gpucachesim_exec_driven",
                "kips_gpucachesim_parallel",
            ]
        ]

        # kips = np.array(
        #     [
        #         bench_df["kips_accelsim"],
        #         bench_df["kips_gpucachesim"],
        #         bench_df["kips_gpucachesim_mem_only"],
        #         bench_df["kips_gpucachesim_exec_driven"],
        #         bench_df["kips_gpucachesim_parallel"],
        #     ]
        # )

        if verbose:
            print(
                bench_df[
                    [
                        "benchmark",
                        "input_id",
                        "instructions",
                        "exec_time_sec_accelsim",
                        "kips_accelsim",
                        "exec_time_sec_gpucachesim",
                        "kips_gpucachesim",
                    ]
                ]
            )

        # kips uses mean
        # if USE_MEDIAN:
        #     kips = kips.median()
        # else:
        #     kips = kips.mean()

        kips = mean_or_median(kips)

        # if bench is None:
        #     kips = np.nanmean(kips, axis=1)
        # else:
        #     kips = np.mean(kips, axis=1)

        for target, kips_value in kips.items():
            table += " & "
            if np.isnan(kips_value):
                continue
            bold = np.isfinite(kips_value) and kips_value == np.nanmax(kips)
            if bold:
                table += r"\boldmath"

            # use parenthesis for KIPS that cannot be compared
            if target.endswith("gpucachesim_mem_only"):
                table += "$({})$".format(plot.human_format_thousands(kips_value))
                # table += "${}$".format(plot.human_format_thousands(kips_value))

            elif target.endswith("gpucachesim_exec_driven"):
                # also show the kips value without trace recon
                kips_excl_trace = mean_or_median(bench_df["kips_gpucachesim_exec_driven_excl_trace"])

                table += "$({})$".format(plot.human_format_thousands(kips_excl_trace))
                # table += "$({} [{}])$".format(
                #     plot.human_format_thousands(kips_value),
                #     plot.human_format_thousands(kips_excl_trace),
                # )
            else:
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
                bold = np.isfinite(mean_time_value) and mean_time_value == np.nanmin(mean_time)
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

    # pprint(list(all_slowdowns_over_native))
    # all_slowdowns_over_native = pd.DataFrame(
    #     all_slowdowns_over_native,
    #     # np.stack(all_slowdowns_over_native, axis=0),
    #     columns=list(sim_targets.keys()),
    # )

    # print(all_slowdowns_over_native[0])
    all_slowdowns_over_native = pd.concat(all_slowdowns_over_native)
    # print(all_slowdowns_over_native)

    speedup_over_accel = (
        all_slowdowns_over_native["slowdown_accelsim"].iloc[-1]
        / all_slowdowns_over_native["slowdown_gpucachesim_parallel"].iloc[-1]
    )
    print(
        color(
            "Mean speedup over accelsim: {:>6.3f}x".format(speedup_over_accel),
            fg="green",
        )
    )

    filename = "speed_table"
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

    tex_document_code = r"""
\documentclass[preview]{standalone}
"""
    tex_document_code += tex.TEX_PACKAGES
    tex_document_code += r"""
\begin{document}
"""

    # caption = r"Average relative speedup and percentage error for serial and parallel simulation using \simName{} on selected simulation output metrics using $t$ threads."
    #     caption = r"""
    # Simulation speed for different \simName{} modes and the
    # popular \textsc{AccelSim} simulator per benchmark.
    # Measured are relative slowdown over native execution on the
    # NVIDIA TitanX (Pascal) and absolute simulation rate in kilo
    # instructions per second (KIPS)."""

    # \begin{tabularx}{\textwidth}{zz|z|z|z|z}
    tabular_columns = [" " + tex.r(width="15mm")]  # measurement
    tabular_columns += [" " + tex.r()]  # accelsim
    tabular_columns += ["|" + tex.r() for _ in range(len(sim_targets) - 1)]

    tex_code = r"\begin{tabularx}{\textwidth}{" + "\n"
    tex_code += "\n".join(tabular_columns) + "\n"
    tex_code += "}"
    tex_code += r"""
% Native
& \textsc{AccelSim}
& \shortstack[c]{\simName{}\\(serial)}
& \shortstack[c]{\simName{}\\(mem-only)}
& \shortstack[c]{\simName{}\\(trace recon.)}
& \shortstack[c]{\simName{}\\(parallel)} \\ \hline
%
"""
    tex_code += table
    tex_code += r"""
\end{tabularx}
"""

    tex_document_code += r"""
\begin{table}[htbp]
\fontsize{8}{10}\selectfont
\footnotesize
\centering
{\renewcommand{\arraystretch}{1.5}%
    """
    tex_document_code += tex_code
    tex_document_code += r"""
}
\end{table}
\end{document}
"""

    if not batch:
        print(tex_code)
        utils.copy_to_clipboard(tex_code)
        print("copied table to clipboard")

    # write latex
    tex_output_path = (plot.TABLE_DIR / filename).with_suffix(".tex")
    with open(tex_output_path, "w") as f:
        f.write(tex_code)
    print(color("wrote {}".format(tex_output_path), fg="cyan"))

    pdf_output_path = (plot.TABLE_DIR / filename).with_suffix(".pdf")
    try:
        tex.render_latex(tex_document_code, output_path=pdf_output_path)
    except Exception as e:
        print(tex_code)
        raise e
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))

    if png:
        png_output_path = (plot.TABLE_DIR / "png" / filename).with_suffix(".png")
        utils.convert_to_png(input_path=pdf_output_path, output_path=png_output_path)
        print(color("wrote {}".format(png_output_path), fg="cyan"))
