import typing
import numpy as np
import pandas as pd
from pathlib import Path
from os import PathLike
from wasabi import color
import matplotlib.pyplot as plt

import gpucachesim.benchmarks as benchmarks
import gpucachesim.stats.stats
import gpucachesim.plot as plot
import gpucachesim.utils as utils

from gpucachesim.benchmarks import (
    Target,
    Benchmarks,
)

TIMING_COLS_SUMMING_TO_FULL_CYCLE = [
    "cycle::core",
    "cycle::dram",
    "cycle::interconn",
    "cycle::issue_block_to_core",
    "cycle::l2",
    "cycle::subpartitions",
]


def _build_timings_pie(ax, timings_df, sections, colors, title=None, validate=False):
    if validate:
        # for _, df in timings_df.groupby(["benchmark", "input_id", "target", "run"]):
        #     print(df)
        #     exec_time_sec = df["exec_time_sec"]
        #     print(exec_time_sec)
        #     assert len(exec_time_sec.unique()) == 1
        #
        #     total = df.loc[cols_summing_to_full_cycle, "total"].sum()
        #     # total = df.T[cols_summing_to_full_cycle].T["total"] # [cols_summing_to_full_cycle].sum()
        #     print(total)
        #     df["abs_diff"] = (total - exec_time_sec).abs()
        #     # abs_diff = (total - exec_time_sec).abs()
        #     df["rel_diff"] = (1 - (total / exec_time_sec)).abs()
        #     # rel_diff = (1 - (total / exec_time_sec)).abs()
        #
        #     valid_rel = df["rel_diff"] <= 0.2
        #     valid_abs = df["abs_diff"] <= 0.1
        #     # print(timings_df[timings_df["total"] > timings_df["exec_time_sec"]])
        #     # print(timings_df[~(valid_rel | valid_abs)])
        #     if not (valid_rel | valid_abs).all():
        #         invalid = ~(valid_rel | valid_abs)
        #         print(df.loc[invalid, ["total", "exec_time_sec", "abs_diff", "rel_diff"]])
        #
        #     assert (valid_rel | valid_abs).all()
        pass

    def stderr(df):
        return df.std() / np.sqrt(len(df))

    averaged = timings_df.groupby("name")[
        [
            "total_sec",
            "share",
            "mean_sec",
            "mean_micros",
            "mean_millis",
            "exec_time_sec",
            # "total_cores",
        ]
    ].agg(["min", "max", "mean", "median", "std", "sem", stderr])

    # make sure sem is correct
    all_sem = averaged.iloc[:, averaged.columns.get_level_values(1) == "sem"]
    all_sem.columns = all_sem.columns.droplevel(1)
    all_stderr = averaged.iloc[:, averaged.columns.get_level_values(1) == "stderr"]
    all_stderr.columns = all_stderr.columns.droplevel(1)
    assert ((all_sem - all_stderr).abs() > 0.001).sum().sum() == 0

    # total does not really say much, because we are averaging for different
    # benchmark configurations
    # print("\n\n=== TOTAL")
    # pd.options.display.float_format = "{:.2f}".format
    # print(averaged["total"])

    def compute_gustafson_speedup(p, n):
        s = 1 - p
        assert 1 + (n - 1) * p == s + p * n
        return 1 + (n - 1) * p

    def compute_amdahl_speedup(p, n):
        """p is the fraction of parallelizeable work. n is the speedup of that parallel part, i.e. number of processors."""
        return 1 / ((1 - p) + p / n)

    threads = 8
    parallel_frac = float(averaged.loc["cycle::core", ("share", "median")])
    amdahl_speedup = compute_amdahl_speedup(p=parallel_frac, n=threads)
    print("AMDAHL SPEEDUP = {:>6.3f}x for {:>2} threads (p={:>5.2f})".format(amdahl_speedup, threads, parallel_frac))

    gustafson_speedup = compute_gustafson_speedup(p=parallel_frac, n=threads)
    print(
        "GUSTAFSON SPEEDUP = {:>6.3f}x for {:>2} threads (p={:>5.2f})".format(gustafson_speedup, threads, parallel_frac)
    )

    print("\n\n=== MEAN MICROSECONS")
    pd.options.display.float_format = "{:.6f}".format
    print(averaged["mean_micros"])
    print("\n\n=== SHARE")
    pd.options.display.float_format = "{:.2f}".format
    print(averaged["share"] * 100.0)

    # validate averaged values
    total_cycle_share = averaged["share", "mean"].T["cycle::total"]

    computed_total_cycle_share = averaged["share", "mean"].T[TIMING_COLS_SUMMING_TO_FULL_CYCLE]
    if computed_total_cycle_share.sum() > total_cycle_share:
        print(total_cycle_share, computed_total_cycle_share.sum())
    assert computed_total_cycle_share.sum() <= total_cycle_share

    unit = "mean_micros"
    agg = "median"
    idx = pd.MultiIndex.from_product((["share", unit], [agg, "std"]))
    # print(averaged[idx])

    # sort based on share
    shares = averaged.loc[sections, idx]
    shares = shares.sort_values([("share", agg)], ascending=False, kind="stable")

    # compute other
    other = 1.0 - shares["share", agg].sum()
    # print("other:", other)
    shares.loc["other", :] = 0.0
    shares.loc["other", ("share", agg)] = other
    print(shares)

    values = shares["share", agg].values * 100.0
    wedges, texts, autotexts = ax.pie(
        values,
        # labels=shares.index,
        # autopct=compute_label,
        autopct="",
        colors=[colors[s] for s in shares.index],
        # labeldistance=1.2,
        pctdistance=1.0,
    )
    # textprops=dict(color="w"))

    labels = shares.index
    # labels = [r"{} (${:4.1f}\%$)".format(label, values[i])
    #           for i, label in enumerate(shares.index)]
    # # labels = [label.removeprefix("cycle::").replace("_", " ").capitalize() for label in shares.index]
    # legend = ax.legend(wedges, labels,
    #       # title="Ingredients",
    #       loc="center left",
    #       bbox_to_anchor=(1, 0, 0.5, 1))
    #
    # bbox_extra_artists.append(legend)

    for i, a in enumerate(autotexts):
        share = values[i]
        col = shares.index[i].lower()

        # compute desired pct distance
        if share >= 40.0:
            label_dist = 0.5
        else:
            label_dist = 0.7
        xi, yi = a.get_position()
        ri = np.sqrt(xi**2 + yi**2)
        phi = np.arctan2(yi, xi)
        x = label_dist * ri * np.cos(phi)
        y = label_dist * ri * np.sin(phi)
        a.set_position((x, y))
        # print(col, share, label_dist)

        if share < 5.0 or col == "other":
            a.set_text("")
        else:
            label = r"${:>4.1f}\%".format(share)
            share_std = shares["share", "std"].iloc[i] * 100.0
            # label += r" \pm {:>4.1f}\%".format(share_std)
            label += "$"
            label += "\n"

            dur = shares[unit, agg].iloc[i]
            dur_std = shares[unit, "std"].iloc[i]
            label += r"${:4.1f}\mu s$".format(dur)
            # label += r"$({:4.1f}ms \pm {:4.2f}ms)$".format(dur_mean, dur_std)

            if col == "cycle::core":
                dur_per_sm = averaged.loc["core::cycle", (unit, agg)]
                # temp fix
                dur_per_sm = dur / 28
                label += "\n"
                label += r"${:4.1f}\mu s$ per core".format(dur_per_sm)
            a.set_text(label)

    # plt.setp(autotexts, size=fontsize, weight="bold")

    if title is not None:
        ax.set_title(title)

    return wedges, list(labels), texts, autotexts


def timings(
    path: PathLike,
    bench_name: typing.Optional[str] = None,
    baseline=False,
    strict=True,
    validate=True,
    png=False,
):
    print("loading", path)
    b = Benchmarks(path)
    benches = b.benchmarks[Target.Simulate.value]

    if bench_name is not None:
        selected_benchmarks = [bench_name for bench_name in benches.keys() if bench_name.lower() == bench_name.lower()]
    else:
        selected_benchmarks = list(benches.keys())

    timings_dfs = []
    for valid_bench_name in selected_benchmarks:
        bench_configs = benches[valid_bench_name]

        def is_baseline(config):
            return not baseline or all(
                [
                    config["values"].get("memory_only") in [False, None],
                    config["values"].get("num_clusters") in [int(benchmarks.BASELINE["num_clusters"]), None],
                    config["values"].get("cores_per_cluster") in [int(benchmarks.BASELINE["cores_per_cluster"]), None],
                    config["values"].get("mode") in ["serial", None],
                ]
            )

        baseline_bench_configs = [config for config in bench_configs if is_baseline(config)]

        for bench_config in baseline_bench_configs:
            repetitions = int(bench_config["common"]["repetitions"])
            target_config = bench_config["target_config"].value
            stats_dir = Path(target_config["stats_dir"])
            assert bench_config["values"]["mode"] == "serial"

            num_clusters = bench_config["values"].get("num_clusters", benchmarks.BASELINE["num_clusters"])
            cores_per_cluster = bench_config["values"].get(
                "cores_per_cluster", benchmarks.BASELINE["cores_per_cluster"]
            )
            total_cores = num_clusters * cores_per_cluster
            assert total_cores == 28

            for r in range(repetitions):
                sim_df = pd.read_csv(
                    stats_dir / f"stats.sim.{r}.csv",
                    header=0,
                )
                sim_df["run"] = r
                grouped_sim_including_no_kernel = sim_df.groupby(gpucachesim.stats.stats.INDEX_COLS, dropna=False)
                grouped_sim_excluding_no_kernel = sim_df.groupby(gpucachesim.stats.stats.INDEX_COLS, dropna=True)

                timings_path = stats_dir / f"timings.{r}.csv"
                # timings_path = stats_dir / f"timings.csv"
                # print(timings_path)
                if not strict and not timings_path.is_file():
                    continue

                assert timings_path.is_file()

                timing_df = pd.read_csv(timings_path, header=0)
                timing_df = timing_df.rename(columns={"total": "total_sec"})
                timing_df["benchmark"] = bench_config["name"]
                timing_df["input_id"] = bench_config["input_idx"]
                timing_df["target"] = bench_config["target"]
                timing_df["run"] = r

                timing_df["total_cores"] = total_cores
                timing_df["mean_blocks_per_sm"] = (
                    grouped_sim_excluding_no_kernel["num_blocks"].mean().mean() / total_cores
                )
                timing_df["exec_time_sec"] = grouped_sim_including_no_kernel["elapsed_millis"].sum().sum()
                timing_df["exec_time_sec"] /= 1000.0

                timings_dfs.append(timing_df)

    timings_df = pd.concat(timings_dfs)
    timings_df = timings_df.set_index(["name"])
    # timings_df = timings_df.set_index(["target", "benchmark", "input_id", "run", "name"])
    index_cols = ["target", "benchmark", "input_id", "run"]
    timings_df["max_total_sec"] = timings_df.groupby(index_cols)["total_sec"].transform("max")

    def compute_exec_time_sec(df) -> float:
        time = df.loc[TIMING_COLS_SUMMING_TO_FULL_CYCLE, "total_sec"].sum()
        return time

    computed_exec_time_sec = (
        timings_df.groupby(index_cols)[timings_df.columns].apply(compute_exec_time_sec).rename("computed_exec_time_sec")
    )

    before = len(timings_df)
    timings_df = timings_df.reset_index().merge(computed_exec_time_sec, on=index_cols, how="left")
    assert len(timings_df) == before

    if "computed_exec_time_sec" in timings_df:
        timings_df["abs_diff_to_real"] = (timings_df["computed_exec_time_sec"] - timings_df["exec_time_sec"]).abs()
        timings_df["rel_diff_to_real"] = (
            1 - (timings_df["computed_exec_time_sec"] / timings_df["exec_time_sec"])
        ).abs()

    # exec time sec is usually more efficient when timing is disabled.
    # while its not quite the real thing, we normalize to max total timing
    timings_df["exec_time_sec"] = timings_df["max_total_sec"]

    # timings_df["exec_time_sec"] = timings_df[["max_total", "exec_time_sec"]].max(axis=1)
    timings_df["mean_sec"] = timings_df["total_sec"] / timings_df["count"]
    timings_df["mean_millis"] = timings_df["mean_sec"] * 1000.0
    timings_df["mean_micros"] = timings_df["mean_millis"] * 1000.0
    timings_df["share"] = timings_df["total_sec"] / timings_df["exec_time_sec"]
    timings_df = timings_df.set_index("name")

    # filter
    sufficient_size_mask = timings_df["mean_blocks_per_sm"] > 1.0
    sufficient_size_timings_df = timings_df[sufficient_size_mask]

    print(timings_df.head(n=10).T)
    print(timings_df.head(n=30))

    fontsize = plot.FONT_SIZE_PT - 4
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    figsize = (
        0.8 * plot.DINA4_WIDTH_INCHES,
        0.2 * plot.DINA4_HEIGHT_INCHES,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    sections = [
        "cycle::core",
        "cycle::dram",
        "cycle::interconn",
        "cycle::issue_block_to_core",
        "cycle::l2",
        "cycle::subpartitions",
    ]
    cmap = plt.get_cmap("tab20")
    # cmap = plt.get_cmap('tab20c')
    # cmap = plt.get_cmap('Set3')

    colors = cmap(np.linspace(0, 1.0, len(sections)))
    colors = [
        "lightskyblue",
        "gold",
        "yellowgreen",
        "lightcoral",
        "violet",
        "palegreen",
    ]
    assert len(colors) == len(sections)

    colors = {section: colors[i] for i, section in enumerate(sections)}
    colors["other"] = "whitesmoke"

    args = dict(sections=sections, colors=colors, validate=validate)

    print("=============== blocks/core <= 1 =============")
    title = r"$N_{\text{blocks}}$/SM $\leq 1$"
    samples = len(timings_df[index_cols].drop_duplicates())
    title += "\n({} benchmark configurations)".format(samples)

    total_micros = timings_df.loc["cycle::total", :].groupby(index_cols)["mean_micros"].first().median()
    title += "\n" + r"${:4.1f}\mu s$ total".format(total_micros)

    wedges1, labels1, texts1, autotexts1 = _build_timings_pie(ax1, timings_df, title=title, **args)

    print("=============== blocks/core > 1 =============")
    title = r"$N_{\text{blocks}}$/SM $>1$"
    samples = len(sufficient_size_timings_df[index_cols].drop_duplicates())
    title += "\n({} benchmark configurations)".format(samples)

    total_micros = sufficient_size_timings_df.loc["cycle::total", :].groupby(index_cols)["mean_micros"].first().median()
    title += "\n" + r"${:4.1f}\mu s$ total".format(total_micros)

    wedges2, labels2, texts2, autotexts2 = _build_timings_pie(ax2, sufficient_size_timings_df, title=title, **args)

    handles = wedges1 + wedges2
    labels = labels1 + labels2
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend = fig.legend(
        *zip(*unique),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        edgecolor="none",
        frameon=False,
        fancybox=False,
        shadow=False,
    )
    bbox_extra_artists = [legend]

    filename = "timings_pie"
    pdf_output_path = (plot.PLOT_DIR / filename).with_suffix(".pdf")
    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    plt.tight_layout()
    fig.savefig(filename, bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")
    print(color("wrote {}".format(pdf_output_path), fg="cyan"))

    if png:
        png_output_path = (plot.PLOT_DIR / "png" / filename).with_suffix(".png")
        png_output_path.parent.mkdir(parents=True, exist_ok=True)

        utils.convert_to_png(input_path=pdf_output_path, output_path=png_output_path)

        print(color("wrote {}".format(png_output_path), fg="cyan"))
