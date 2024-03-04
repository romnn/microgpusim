import copy
import re
import typing
import numpy as np
import itertools
import pandas as pd
from pprint import pprint
from os import PathLike
from wasabi import color
import matplotlib
import matplotlib.pyplot as plt
from functools import partial

import gpucachesim.stats.load
import gpucachesim.stats.native
import gpucachesim.stats.agg
import gpucachesim.benchmarks as benchmarks
import gpucachesim.plot as plot
import gpucachesim.utils as utils

from gpucachesim.benchmarks import (
    Target,
    Benchmarks,
    GPUConfig,
    DEFAULT_BENCH_FILE,
)


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


def prepare_figure(size, fontsize, font_family, grid=True):
    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
    plt.rcParams["hatch.linewidth"] = 0.5
    matplotlib.rcParams["hatch.linewidth"] = 0.5

    fig = plt.figure(
        figsize=size,
        layout="constrained",
    )
    ax = plt.axes()

    ax.grid(
        grid,
        axis="y",
        linestyle="-",
        linewidth=1,
        color="black",
        alpha=0.1,
        zorder=1,
    )

    return fig, ax


def finish_figure(
    fig,
    ax,
    size,
    profiler,
    benchmark,
    stat_col,
    stat_config,
    all_values,
    ylabel,
    group_width,
    group_spacing,
    # yticks,
    # yticklabels,
    # xticks,
    # xticklabels,
    num_blocks,
    labels,
    normalized=False,
    large=False,
    png=False,
    png_density=600,
):
    ax.set_ylabel(ylabel)
    ax.axes.set_zorder(10)

    # ax.set_yticks(yticks, yticklabels)
    # ax.set_xticks(xticks, xticklabels)

    new_width, height = size[0], size[1]

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

    ymax = all_values.max()
    yticks = np.linspace(0, ymax, 6)

    if normalized:
        ax.set_ylim(0.0, max(1.1 * ymax, 1.5))
    else:
        if stat_config.log_y_axis:
            assert not stat_config.percent
            ymax_log = np.ceil(np.log10(ymax))
            yticks = np.arange(0, ymax_log + 1, step=int(np.ceil(ymax_log / 6)))
            yticks = np.power(10, yticks)
            print(stat_col, ymax_log, yticks)
            ax.set_yscale("log", base=10)
            ax.set_ylim(0.01, max(10 * ymax, 10))
            # ylim = (0.01, max(10 * ymax, 10))
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
            yticks = np.linspace(0, ymax, 6)

    if not normalized:
        ytick_labels = [
            plot.human_format_thousands(v, round_to=0).rjust(6, " ") for v in yticks
        ]
        ax.set_yticks(yticks, ytick_labels)

    xticklabels = [
        "{}\n{} {}".format(label, int(blocks), "blocks" if blocks > 1 else "block")
        for label, blocks in zip(labels, num_blocks)
    ]
    assert len(xticklabels) == len(labels)
    assert len(xticklabels) > 0

    xticks = np.arange(0, len(xticklabels), dtype=np.float64)
    xticks *= group_width
    xticks += 0.5 * float((group_width - group_spacing))
    # print("xvalues", xtick_values)
    # print("xlables", xtick_labels)
    xmargin = 0.5 * group_spacing
    ax.set_xlim(-xmargin, len(xticklabels) * group_width - xmargin)

    # plot without legend or xticks (middle)
    ax.set_xticks(xticks, ["" for _ in range(len(xticks))], rotation=0)
    fig.set_size_inches(new_width, height)

    plot_dir = plot.PLOT_DIR / "validation"

    filename = "{}.{}.{}{}_no_xticks_no_legend.pdf".format(
        profiler, benchmark, "norm." if normalized else "", stat_col
    )
    pdf_output_path = plot_dir / filename
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output_path)
    if png:
        png_output_path = (plot_dir / "png" / filename).with_suffix(".png")
        utils.convert_to_png(
            input_path=pdf_output_path,
            output_path=png_output_path,
            density=png_density,
        )

    # plot with xticks but without legend (bottom)
    ax.set_xticks(xticks, xticklabels, rotation=0)
    fig.set_size_inches(new_width, height)

    filename = "{}.{}.{}{}_with_xticks_no_legend.pdf".format(
        profiler, benchmark, "norm." if normalized else "", stat_col
    )
    pdf_output_path = plot_dir / filename
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output_path)

    if png:
        png_output_path = (plot_dir / "png" / filename).with_suffix(".png")
        utils.convert_to_png(
            input_path=pdf_output_path,
            output_path=png_output_path,
            density=png_density,
        )

    # plot with legend and xticks (default)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
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

    fig.set_size_inches(new_width, 1.6 * height)

    filename = "{}.{}.{}{}.pdf".format(
        profiler, benchmark, "norm." if normalized else "", stat_col
    )
    pdf_output_path = plot_dir / filename
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output_path)
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))

    if png:
        png_output_path = (plot_dir / "png" / filename).with_suffix(".png")
        utils.convert_to_png(
            input_path=pdf_output_path,
            output_path=png_output_path,
            density=png_density,
        )
        print(color("wrote {}".format(png_output_path), fg="cyan"))

    if large:
        # plot with legend and xticks (default) but LARGE
        fig.set_size_inches(new_width, 3 * height)

        filename = "{}.{}.{}{}_large.pdf".format(
            profiler, benchmark, "norm." if normalized else "", stat_col
        )
        pdf_output_path = plot_dir / filename
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output_path)
        print(color("wrote {}".format(pdf_output_path), fg="cyan"))

        if png:
            png_output_path = (plot_dir / "png" / filename).with_suffix(".png")
            utils.convert_to_png(
                input_path=pdf_output_path,
                output_path=png_output_path,
                density=png_density,
            )

    # plot with legend but without xticks (top)
    ax.set_xticks(xticks, ["" for _ in range(len(xticks))], rotation=0)
    fig.set_size_inches(new_width, height)

    filename = "{}.{}.{}{}_no_xticks_with_legend.pdf".format(
        profiler, benchmark, "norm." if normalized else "", stat_col
    )
    pdf_output_path = plot_dir / filename
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output_path)

    if png:
        png_output_path = (plot_dir / "png" / filename).with_suffix(".png")
        utils.convert_to_png(
            input_path=pdf_output_path,
            output_path=png_output_path,
            density=png_density,
        )


def filter_stat_cols(stat_cols: typing.Sequence[str], names):
    if not (isinstance(names, list) and len(names) > 0):
        return stat_cols

    # filter stats
    filtered_stat_cols = [col for col in stat_cols if col.lower() in names]

    requested = len(set(names))
    found = len(filtered_stat_cols)
    if found != requested:
        pprint([col.lower() for col in stat_cols])
        raise ValueError(
            "requested {} stats but only found {}".format(requested, found)
        )
    return filtered_stat_cols


def compute_label_for_benchmark_df(df, per_kernel=False):
    assert isinstance(df, pd.Series)

    benchmark = df["benchmark"]
    bench_input_cols = copy.deepcopy(benchmarks.BENCHMARK_INPUT_COLS[benchmark])
    assert all([c in df for c in bench_input_cols])

    kernel_name = str(df["kernel_name"]).replace("_kernel", "").strip()

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
    print("stat cols", stat_cols)

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


def plot_stats(
    per_config,
    stat_cols,
    all_input_cols,
    profiler,
    # per_kernel=False,
    plot_trace_reconstruction=False,
    normalized=False,
    large=False,
    png=False,
    png_density=600,
    verbose=False,
):
    # remove some stat_cols that should not be plotted
    stat_cols = sorted(list(set(stat_cols) - set(["num_blocks", "input_id"])))

    # print(per_config)
    # print(per_config.index)
    # targets = sorted(per_config["target"].unique().tolist())
    # benches = sorted(per_config["benchmark"].unique().tolist())
    benches = sorted(list(per_config.index.get_level_values(1).unique()))
    plot_targets = [Target.Profile.value, Target.Simulate.value]
    if plot_trace_reconstruction:
        plot_targets += [Target.ExecDrivenSimulate.value]
    plot_targets += [Target.AccelsimSimulate.value]

    fontsize = plot.FONT_SIZE_PT - 4
    font_family = "Helvetica"

    bar_width = 10
    spacing = 2
    group_spacing = 2 * bar_width

    group_width = len(plot_targets) * (bar_width + spacing) + group_spacing

    figsize = (
        1.0 * plot.DINA4_WIDTH_INCHES,
        0.10 * plot.DINA4_HEIGHT_INCHES,
    )

    bar_group_cols = [
        # "benchmark",
        "input_id",
        "kernel_launch_id",
    ]
    group_cols = ["target", "benchmark"]
    group_cols += all_input_cols
    group_cols += ["target_name", "label"]

    if normalized:
        # normalized plots
        for stat_col, benchmark in itertools.product(stat_cols, benches):
            print(stat_col, benchmark)
            stat_config = STAT_CONFIGS.get(stat_col) or StatConfig(
                **{**DEFAULT_STAT_CONFIG._asdict(), **dict(label=stat_col)}
            )
            ylabel = "Normalized\n{}".format(stat_config.label)
            # remove any unit
            ylabel = ylabel.replace(" (s)", "")
            ylabel = ylabel.replace(" (%)", "")
            # fontsize = plot.FONT_SIZE_PT - 4
            # font_family = "Helvetica"
            #
            # bar_width = 10
            # spacing = 2
            # group_spacing = 2 * bar_width
            #
            # group_width = len(plot_targets) * (bar_width + spacing) + group_spacing
            #
            # plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
            #
            # figsize = (
            #     1.0 * plot.DINA4_WIDTH_INCHES,
            #     0.10 * plot.DINA4_HEIGHT_INCHES,
            # )
            # fig = plt.figure(
            #     figsize=figsize,
            #     layout="constrained",
            # )
            # ax = plt.axes()
            #
            # ax.grid(
            #     stat_config.grid,
            #     axis="y",
            #     linestyle="-",
            #     linewidth=1,
            #     color="black",
            #     alpha=0.1,
            #     zorder=1,
            # )
            fig, ax = prepare_figure(
                size=figsize,
                fontsize=fontsize,
                font_family=font_family,
                grid=stat_config.grid,
            )

            # pprint(list(per_config.index.names))
            benchmark_df = per_config.loc[pd.IndexSlice[:, benchmark], :].reset_index()

            bench_group_cols = [col for col in group_cols if col in benchmark_df]
            # pprint(group_cols)

            per_targets_stat_df = benchmark_df.groupby(
                bench_group_cols, dropna=False, sort=False
            )[stat_col].agg(["mean", "median", "std"])
            # per_targets_stat_df= benchmark_df.groupby(group_cols, dropna=False, sort=False)[benchmark_df.columns].agg({stat_col: ["mean", "median", "std"], "target_name": "first"})
            # pprint(list(per_targets_stat_df.columns))
            # test_df = benchmark_df.groupby(group_cols, dropna=False, sort=False)[stat_col].agg(["mean" if stat_config.percent else "median", "std"])
            # .reset_index()
            # print(test_df)

            # print(test_df.index)
            # test_df = test_df.set_index(["target"])
            def _norm(df):
                norm_df = per_targets_stat_df.loc[Target.Profile.value, :]
                df = df.droplevel("target")

                # norm_df = norm_df.droplevel(
                #     [lvl for lvl in norm_df.index.names if lvl not in all_input_cols]
                # )
                norm_index = norm_df.index.droplevel(
                    [lvl for lvl in norm_df.index.names if lvl not in all_input_cols]
                )

                # df = df.droplevel(["target", "target_name", "label"])
                # df = df.droplevel(["target", "target_name", "label"])
                # df = df.droplevel(
                #     # ["target"]
                #     [lvl for lvl in df.index.names if lvl not in all_input_cols]
                # )
                df_index = df.index.droplevel(
                    [lvl for lvl in df.index.names if lvl not in all_input_cols]
                )

                # drop target index col as we drop this for norm_df too
                # df = df.droplevel(0) + eps

                # assert (norm_df.index[all_input_cols] == df.index[all_input_cols]).all()
                # cols = [stat_col + "_mean", stat_col + "_median", stat_col + "_std"]
                # df[cols] /= norm_df[cols]
                # df.loc[stat_col,:] /= norm_df.loc[stat_col,:]

                # pprint(df_index.tolist())
                # pprint(norm_index.tolist())
                # print(df_index)
                # print(norm_index)
                # print(df_index.to_frame().to_numpy())
                # print(norm_index.to_frame().to_numpy())
                # assert np.array_equal(
                #     df_index.to_frame().to_numpy(),
                #     norm_index.to_frame().to_numpy(),
                #     equal_nan=True,
                # )
                assert df_index.to_frame().equals(norm_index.to_frame())
                # assert (df_index == norm_index).all()
                # assert df_index.tolist() == norm_index.tolist()

                # avoid divide by zero
                eps = 1e-10
                for col in ["mean", "median"]:
                    norm_df[col] = norm_df[col] + eps
                    df[col] = df[col] + eps

                # assert (df.columns == norm_df.columns).all()
                # assert (df.index == norm_df.index).all()

                # print("DF:")
                # print(df)
                # print("NORM DF:")
                # print(norm_df)

                # this is unsafe, but we have asserted the order of
                # the index matches before
                df["mean"] /= norm_df["mean"].values
                df["median"] /= norm_df["median"].values

                # normalized standard deviation
                df["std"] /= norm_df["mean"].values
                return df
                # df[df.columns] / norm_df[df.columns]
                # return df
                # return df[df.columns] / norm_df[df.columns]

            # norm_df = per_targets_stat_df.groupby(["target", "target_name", "label"], dropna=False, sort=False).apply(
            norm_df = per_targets_stat_df.groupby(
                ["target"], dropna=False, sort=False
            ).apply(_norm)

            norm_df["input_idx"] = norm_df.groupby("target", sort=False).cumcount()

            if stat_col.lower() in [
                "cycles",
                "exec_time_sec",
                "l1_accesses",
                "l1_global_hit_rate",
                "l2_hit_rate",
            ]:
                print(per_targets_stat_df)
                print(norm_df)
                assert (norm_df.loc[Target.Profile.value, "mean"] == 1.0).all()
                assert (norm_df.loc[Target.Profile.value, "median"] == 1.0).all()

            norm_df = norm_df.fillna(0.0).reset_index()
            # norm_df["input_id"] = norm_df.groupby(["Product','SubProd']).cumcount()
            # print(norm_df)

            value_col = "median" if stat_config.percent else "mean"

            for row_idx, row in list(norm_df.iterrows()):
                # for target, target_df in list(norm_df.groupby("target")):
                target = row["target"]
                if target not in plot_targets:
                    # print("skip {}".format(target))
                    continue

                # for row_idx, row in list(norm_df.iterrows()):
                target_name = row["target_name"]
                target_idx = plot_targets.index(target)
                input_idx = row["input_idx"]
                # input_idx = row_idx % ((target_idx + 1) * len(plot_targets))
                idx = input_idx * group_width + (target_idx + 0.5) * (
                    bar_width + spacing
                )

                print(target, input_idx)
                # print(target, target_idx, input_idx, idx)
                # print(target, target_name, target_idx, input_idx, idx)
                # print(row_idx, row)

                x = [idx]
                y = row[value_col]
                ystd = row["std"]

                # print("mean ", row[value_col])
                # print("std  ", row["std"])
                # y = np.nan_to_num(y, nan=0.0)
                # ystd = np.nan_to_num(ystd, nan=0.0)

                bar_color = plot.plt_rgba(*plot.SIM_RGB_COLOR[target.lower()], 1.0)
                hatch = plot.SIM_HATCH.get(target.lower())

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

            # print(benchmark_df.index)
            simulate_df = benchmark_df.loc[
                benchmark_df["target"] == Target.Simulate.value
            ]
            # print(simulate_df[group_cols])

            # labels = simulate_df["label"]
            # num_blocks = simulate_df["num_blocks"]

            simulate_grouped = simulate_df.groupby(
                bar_group_cols, dropna=False, sort=False
            )
            labels = simulate_grouped["label"].first().to_numpy()
            num_blocks = simulate_grouped["num_blocks"].max().to_numpy()
            # num_blocks = simulate_df["num_blocks"].values
            # print(labels.tolist())
            assert len(labels) == len(num_blocks)
            assert len(labels) > 0

            all_values = norm_df[value_col] + norm_df["std"]
            # all_values = norm_df.loc[pd.IndexSlice[:, benchmark], stat_col]
            assert len(all_values) > 0
            print(all_values.values)

            finish_figure(
                fig,
                ax,
                size=figsize,
                profiler=profiler,
                benchmark=benchmark,
                stat_col=stat_col,
                stat_config=stat_config,
                all_values=all_values,
                ylabel=ylabel,
                group_width=group_width,
                group_spacing=group_spacing,
                # yticks=ytick_values,
                # yticklabels=ytick_labels,
                # xticks=xtick_values,
                # xticklabels=xtick_labels,
                num_blocks=num_blocks,
                labels=labels,
                normalized=True,
                large=large,
                png=png,
                png_density=png_density,
            )

    # absolute
    for stat_col, benchmark in itertools.product(stat_cols, benches):
        print(stat_col, benchmark)
        stat_config = STAT_CONFIGS.get(stat_col) or StatConfig(
            **{**DEFAULT_STAT_CONFIG._asdict(), **dict(label=stat_col)}
        )
        ylabel = stat_config.label

        fig, ax = prepare_figure(
            size=figsize,
            fontsize=fontsize,
            font_family=font_family,
            grid=stat_config.grid,
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
            # print(target_idx, target)
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

            target_df = per_config.loc[(target, benchmark), :]
            assert len(target_df) > 0
            if target != Target.ExecDrivenSimulate.value:
                assert target_df["run"].nunique() > 1

            assert "input_size" in target_df.index.names

            target_df = target_df.reset_index()
            # leave the sorting manual, e.g. when we have different dtype
            # target_df = target_df.sort_values(["num_blocks", "input_id"])
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
            # pprint([col for col in group_cols if col in target_df])
            # print(target_df[[col for col in group_cols if col in target_df]])

            target_group_cols = [col for col in group_cols if col in target_df]
            input_dfs = list(
                target_df.groupby(target_group_cols, dropna=False, sort=False)
            )
            assert len(input_dfs) > 0
            for input_idx, (_, input_values_df) in enumerate(input_dfs):
                # for input_idx, (_input_id, input_values_df) in enumerate(target_df.groupby("input_id")):

                # key = (target, benchmark) + tuple(input_values.values)

                # print(input_idx, input_values)
                # print(input_values_df[[col for col in all_input_cols if col in input_values_df]].drop_duplicates())

                input_values = (
                    input_values_df[
                        [col for col in all_input_cols if col in input_values_df]
                    ]
                    .drop_duplicates()
                    .dropna(axis="columns")
                )
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
                idx = input_idx * group_width + (target_idx + 0.5) * (
                    bar_width + spacing
                )

                # target = target_df["target"].first().values[0]
                # assert target == target_df["target"].first().values[0]
                assert input_values_df["target_name"].nunique() == 1
                target_name = input_values_df["target_name"].iloc[0]
                # target_name = target_df["target_name"].first().values[0]

                x = [idx]
                raw_y = input_values_df[stat_col]  # .fillna(0.0)
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
                    y = raw_y.mean()  # .fillna(0.0)

                ystd = raw_y.std()  # .fillna(0.0)

                y = np.nan_to_num(y, nan=0.0)
                ystd = np.nan_to_num(ystd, nan=0.0)

                bar_color = plot.plt_rgba(*plot.SIM_RGB_COLOR[target.lower()], 1.0)
                hatch = plot.SIM_HATCH.get(target.lower())

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

        # ax.set_ylabel(ylabel)
        # ax.axes.set_zorder(10)

        # simulate_df_mask = per_config["target"] == Target.Simulate.value
        # simulate_df_mask &= per_config["benchmark"] == benchmark
        # simulate_df = per_config.loc[simulate_df_mask, :]
        # simulate_df = simulate_df.merge(bench_input_values, how="inner")
        simulate_df = per_config.loc[(Target.Simulate.value, benchmark), :]

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
        simulate_grouped = simulate_df.reset_index().groupby(
            bar_group_cols, dropna=False, sort=False
        )
        # simulate_grouped = simulate_df.groupby([col for col in bar_group_cols if col in simulate_df], dropna=False)

        # print(simulate_grouped["label"].first())
        # print(simulate_grouped["label"].apply(lambda df: print(df)))

        # labels = simulate_grouped["label"].to_numpy()
        # print(simulate_df)
        # print(simulate_grouped)
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
        # all_values_df = per_config.loc[pd.IndexSlice[:, benchmark], :]
        all_values = per_config.loc[pd.IndexSlice[:, benchmark], stat_col]
        assert len(all_values) > 0

        # ymax = all_values_df[stat_col].max()

        # if stat_config.log_y_axis:
        #     assert not stat_config.percent
        #     ymax_log = np.ceil(np.log10(ymax))
        #     ytick_values = np.arange(0, ymax_log + 1, step=int(np.ceil(ymax_log / 6)))
        #     ytick_values = np.power(10, ytick_values)
        #     print(stat_col, ymax_log, ytick_values)
        #     ax.set_yscale("log", base=10)
        #     ax.set_ylim(0.01, max(10 * ymax, 10))
        #     # ylim = (0.01, max(10 * ymax, 10))
        # else:
        #     if stat_config.percent:
        #         ymax *= 100.0
        #         assert ymax <= 101.0
        #         ymax = utils.round_to_multiple_of(1.5 * ymax, multiple_of=25.0)
        #         ymax = np.clip(ymax, a_min=25.0, a_max=100.0)
        #         ax.set_ylim(0, ymax + 10.0)
        #     else:
        #         ymax = max(2 * ymax, 1)
        #         ax.set_ylim(0, ymax)
        #     ytick_values = np.linspace(0, ymax, 6)
        #
        # ytick_labels = [
        #     plot.human_format_thousands(v, round_to=0).rjust(6, " ")
        #     for v in ytick_values
        # ]
        # ax.set_yticks(ytick_values, ytick_labels)
        #
        # xtick_labels = [
        #     "{}\n{} {}".format(label, int(blocks), "blocks" if blocks > 1 else "block")
        #     for label, blocks in zip(labels, num_blocks)
        # ]
        # assert len(xtick_labels) == len(labels)
        # assert len(xtick_labels) > 0
        #
        # xtick_values = np.arange(0, len(xtick_labels), dtype=np.float64)
        # xtick_values *= group_width
        # xtick_values += 0.5 * float((group_width - group_spacing))
        # # print("xvalues", xtick_values)
        # # print("xlables", xtick_labels)
        # xmargin = 0.5 * group_spacing
        # ax.set_xlim(-xmargin, len(xtick_labels) * group_width - xmargin)

        finish_figure(
            fig,
            ax,
            size=figsize,
            profiler=profiler,
            benchmark=benchmark,
            stat_col=stat_col,
            stat_config=stat_config,
            all_values=all_values,
            ylabel=ylabel,
            group_width=group_width,
            group_spacing=group_spacing,
            # yticks=ytick_values,
            # yticklabels=ytick_labels,
            # xticks=xtick_values,
            # xticklabels=xtick_labels,
            normalized=False,
            num_blocks=num_blocks,
            labels=labels,
            large=large,
            png=png,
            png_density=png_density,
        )


def view(
    path: PathLike,
    bench_name: typing.Optional[str] = None,
    should_plot=True,
    nsight=False,
    mem_only=False,
    verbose=False,
    strict=True,
    per_kernel: typing.Optional[bool] = None,
    normalized=False,
    trace_reconstruction=True,
    playground=True,
    plot_trace_reconstruction=True,
    stat_names: typing.Optional[typing.Sequence[str]] = None,
    inspect=False,
    png=False,
):
    profiler = "nsight" if nsight else "nvprof"
    selected_df = gpucachesim.stats.load.load_stats(
        bench_name=bench_name, profiler=profiler, path=path
    )

    if per_kernel is None:
        per_kernel = False

    # gpucachesim stats include "no kernel" (e.g. memcopies) stats
    assert selected_df["kernel_name"].isna().sum() > 0

    target_bench_input_count_hist = (
        selected_df[["target", "benchmark", "input_id"]]
        .drop_duplicates()
        .value_counts(["target", "benchmark"], dropna=False)
        .sort_index()
    )
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

    stat_cols = gpucachesim.stats.native.stat_cols_for_profiler(profiler)
    stat_cols = filter_stat_cols(stat_cols, stat_names)

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

    per_config, _ = gpucachesim.stats.agg.aggregate_benchmark_results(
        selected_df,
        memory_only=mem_only,
        per_kernel=per_kernel,
        inspect=inspect,
        mean=False,
    )

    print(
        per_config[
            ["target", "benchmark"]
            + (["input_id", "kernel_name"] if per_kernel else [])
        ].drop_duplicates()
    )

    all_input_cols = list(copy.deepcopy(benchmarks.ALL_BENCHMARK_INPUT_COLS))
    if per_kernel:
        all_input_cols += ["kernel_launch_id", "kernel_name"]
    all_input_cols = [col for col in all_input_cols if col in selected_df]
    all_input_cols = sorted(all_input_cols)

    # # all_input_cols = sorted(list([col for col in all_input_cols if col in per_config]))
    #     all_input_cols = [col for col in all_input_cols if col in per_config]

    per_config_group_cols = utils.dedup(
        [
            "target",
            "benchmark",
            "input_id",
            "kernel_name",
            "kernel_name_mangled",
            "run",
        ]
        + benchmarks.SIMULATE_INPUT_COLS
        + all_input_cols
    )

    def _inspect(df):
        if len(df) > 1:
            print(df[per_config_group_cols].T)
            print(df.T)
            raise ValueError("must have exactly one row per config/run")

    rows_per_config_grouper = per_config.groupby(
        per_config_group_cols,
        as_index=False,
        dropna=False,
        sort=False,
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
    test = per_config[
        [
            "target",
            "benchmark",
            "input_id",
            "kernel_launch_id",
            "kernel_name",
            "run",
        ]
    ]
    print(test)
    print(per_config.drop_duplicates())
    print(
        test.loc[
            test.duplicated(),
            [
                "target",
                "benchmark",
                "input_id",
                "kernel_launch_id",
                "kernel_name",
                "run",
            ],
        ]
    )
    print(len(test.drop_duplicates()), len(per_config))

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
    per_config_grouped = per_config.groupby(group_cols, dropna=False, sort=False)

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

    def _inspect(df):
        print("\nINSPECT")
        print(df.loc[:, preview_cols][:10])
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
    preview_per_config_pivoted = per_config_pivoted.T.copy()

    selected_targets = [
        Target.Profile.value,
        Target.Simulate.value,
    ]
    if trace_reconstruction:
        selected_targets += [Target.ExecDrivenSimulate.value]

    if playground:
        selected_targets += [Target.PlaygroundSimulate.value]

    selected_targets += [Target.AccelsimSimulate.value]

    preview_per_config_pivoted = preview_per_config_pivoted.loc[
        pd.IndexSlice[:, selected_targets], :
    ]
    preview_target_name = {
        Target.Simulate.value.lower(): "Ours",
        Target.ExecDrivenSimulate.value.lower(): "TR",
        Target.AccelsimSimulate.value.lower(): "Accel",
        Target.PlaygroundSimulate.value.lower(): "Play",
        Target.Profile.value.lower(): "Native",
    }
    print(preview_per_config_pivoted.index)
    preview_per_config_pivoted.index = preview_per_config_pivoted.index.set_levels(
        [
            preview_target_name[target.lower()]
            for target in preview_per_config_pivoted.index.levels[1].values
        ],
        level=1,
    )
    print(preview_per_config_pivoted.index)
    print(preview_per_config_pivoted.loc[pd.IndexSlice[stat_cols, :], :])

    table_stat_cols = gpucachesim.stats.native.table_stat_cols_for_profiler(profiler)
    table_stat_cols = filter_stat_cols(table_stat_cols, stat_names)

    # table_stat_cols = [
    #     col
    #     for col in table_stat_cols_for_profiler(profiler)
    #     if col not in ["input_id", "mean_blocks_per_sm", "l1_local_hit_rate", "l1_hit_rate"]
    # ]

    # filter benchmarks that should be in the table
    selected_table_benchmarks = [
        # babelstream
        pd.DataFrame.from_records(
            (
                [
                    ("babelstream", 10240.0),
                    ("babelstream", 102400.0),
                ]
                if per_kernel
                else [
                    ("babelstream", 1024.0),
                    ("babelstream", 10240.0),
                    ("babelstream", 102400.0),
                ]
            ),
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
                ("simple_matrixmul", 32, 32, 32, 32),
                ("simple_matrixmul", 32, 128, 128, 128),
                ("simple_matrixmul", 32, 32, 64, 128),
                ("simple_matrixmul", 32, 128, 32, 32),
                # extra configs
                ("simple_matrixmul", 32, 128, 512, 128),
                ("simple_matrixmul", 32, 512, 32, 512),
            ],
            columns=["benchmark", "input_dtype", "input_m", "input_n", "input_p"],
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

    if bench_name is not None:
        # do not show statistics table for all benchmarks combined
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

    # add plot labels
    per_config.loc[:, "label"] = per_config.apply(
        partial(compute_label_for_benchmark_df, per_kernel=per_kernel), axis=1
    )
    per_config.loc[
        per_config["target"] == Target.ExecDrivenSimulate.value, "target_name"
    ] = "gpucachesim (TR)"
    per_config.loc[per_config["target"] == Target.Simulate.value, "target_name"] = (
        "gpucachesim"
    )
    per_config.loc[
        per_config["target"] == Target.AccelsimSimulate.value, "target_name"
    ] = "AccelSim"
    per_config.loc[per_config["target"] == Target.Profile.value, "target_name"] = (
        per_config.loc[~per_config["device"].isna(), "device"].apply(
            gpucachesim.stats.native.normalize_nvprof_device_name
        )
    )

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
    plot_per_config = per_config.reset_index(drop=True).merge(
        selected_table_benchmarks, how="inner"
    )
    assert len(plot_per_config) <= len(per_config)
    assert "input_size" in plot_per_config

    plot_per_config = plot_per_config.set_index(
        list(
            ["target", "benchmark"]
            + list(selected_table_benchmarks.columns)
            + ["target_name", "label"]
        )
    )
    plot_per_config = plot_per_config.sort_index()
    # print(sorted(group_cols))
    # print(sorted(["target", "benchmark"] + list(selected_table_benchmarks.columns)))
    assert "input_size" in plot_per_config.index.names

    # print(plot_per_config[[col for col in preview_cols if col in plot_per_config]])

    plot_stats(
        per_config=plot_per_config,
        stat_cols=stat_cols,
        all_input_cols=all_input_cols,
        profiler=profiler,
        plot_trace_reconstruction=plot_trace_reconstruction,
        # per_kernel=per_kernel,
        normalized=normalized,
        png=png,
        verbose=verbose,
        large=False,
    )
