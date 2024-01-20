import click
import pandas as pd
import numpy as np
from pathlib import Path
from os import PathLike
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gpucachesim import MB, KB
import gpucachesim.plot as plot

pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 1000)
np.seterr(all="raise")
np.set_printoptions(suppress=True)


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


@main.command()
@click.option("--input", help="input file path")
# @click.option("--output", help="output file path")
@click.option("--states", "plot_states", type=bool, default=True, help="Plot cache line states")
@click.option(
    "--allocations",
    "plot_states",
    type=bool,
    default=True,
    help="Plot cache line allocation ids",
)
@click.option(
    "--allocations",
    "plot_allocations",
    type=bool,
    default=True,
    help="Plot cache line allocation ids",
)
@click.option(
    "--alloc-times",
    "plot_alloc_times",
    type=bool,
    default=True,
    help="Plot cache line allocation times",
)
@click.option(
    "--access-times",
    "plot_last_access_time",
    type=bool,
    default=True,
    help="Plot cache line last access times",
)
@click.option(
    "--mem",
    "mem_arg",
    type=str,
    help="Memory (L1 or L2)",
)
@click.option(
    "--allocation-colorbar",
    "allocation_colorbar",
    type=bool,
    is_flag=True,
    help="Use color bar for allocation plot",
)
@click.option(
    "--state-colorbar",
    "state_colorbar",
    type=bool,
    is_flag=True,
    help="Use color bar for cache state plot",
)
def cache(
    input,
    plot_states,
    plot_allocations,
    plot_alloc_times,
    plot_last_access_time,
    mem_arg,
    allocation_colorbar,
    state_colorbar,
):
    if input is None:
        raise ValueError("need input path")
    input = Path(input)
    if input.is_dir():
        inputs = sorted(list(input.glob("*.csv")))
    else:
        inputs = [input]

    for input in inputs:
        # guess the memory
        mem = "l2"
        if mem_arg is None:
            # guess here
            if "l1" in input.name.lower():
                mem = "l1"
            elif "l2" in input.name.lower():
                mem = "l2"
            else:
                print("WARNING: failed to guess cache, assuming L2 geometry")

        print("### PROCESSING [mem={}] {}".format(mem, input))
        df = pd.read_csv(
            input,
            header=0,
        )
        match mem.lower():
            case "l2":
                partitions, sets = (12, 128)
                # partitions, sets = (24, 64)
                nx, ny = 6, 2
                assert nx * ny == partitions
                sector_size = 32
                line_size = 128
                assoc = 16
                partition_size = sets * line_size * assoc
                cache_size = partition_size * partitions
                assert cache_size == 3 * MB

            case "l1":
                partitions, sets = (28, 4)
                nx, ny = 7, 4
                assert nx * ny == partitions
                sector_size = 32
                line_size = 128
                assoc = 48
                partition_size = sets * line_size * assoc
                cache_size = partition_size * partitions
                assert cache_size == 28 * 24 * KB

            case other:
                raise ValueError("unknown memory {}".format(other))

        def title(id) -> str:
            if mem.lower() == "l2":
                return "subpartition {}".format(id + 1)
            else:
                return "SM {}".format(id + 1)

        df["partition"] = df.index // (partition_size / sector_size)
        df["set_id"] = df["line_id"] // assoc
        df["assoc_id"] = df["line_id"] % assoc
        print(df)

        assert len(df["partition"].unique()) == partitions
        assert len(df["line_id"].unique()) == sets * assoc
        assert len(df["set_id"].unique()) == sets
        assert len(df["assoc_id"].unique()) == assoc
        assert len(df[["partition", "line_id", "sector"]].drop_duplicates()) == cache_size / sector_size

        status_values = ["INVALID", "RESERVED", "VALID", "MODIFIED"]
        status_hist = df.value_counts(["partition", "status"], dropna=False).sort_index(level=0)
        allocation_hist = df.value_counts(["partition", "allocation_id"], dropna=False).sort_index(level=0)
        print(status_hist)
        print(allocation_hist)

        sectors = int(line_size / sector_size)
        shape = (partitions, sets, assoc, sectors)
        states = np.zeros(shape=shape)
        allocations = np.zeros(shape=shape)
        alloc_times = np.zeros(shape=shape)
        last_access_times = np.zeros(shape=shape)
        cache_lines_per_set = assoc * sectors

        for (partition_id, set_id, assoc_id), row_df in df.groupby(["partition", "set_id", "assoc_id"]):
            row_df.sort_values(["line_id", "sector"])

            states[int(partition_id), int(set_id), int(assoc_id), :] = (
                row_df["status"].apply(lambda status: status_values.index(status)).to_numpy()
            )
            allocations[int(partition_id), int(set_id), int(assoc_id), :] = row_df["allocation_id"].to_numpy()
            alloc_times[int(partition_id), int(set_id), int(assoc_id), :] = row_df["sector_alloc_time"].to_numpy()
            last_access_times[int(partition_id), int(set_id), int(assoc_id), :] = row_df[
                "last_sector_access_time"
            ].to_numpy()

        states = states.reshape((partitions, sets, -1))
        allocations = allocations.reshape((partitions, sets, -1))
        alloc_times = alloc_times.reshape((partitions, sets, -1))
        last_access_times = last_access_times.reshape((partitions, sets, -1))
        assert states.shape[-1] == cache_lines_per_set

        # flip sectors in cache set to more intuitive left-to-right order
        states = np.flip(states, axis=-1)
        allocations = np.flip(allocations, axis=-1)
        alloc_times = np.flip(alloc_times, axis=-1)
        last_access_times = np.flip(last_access_times, axis=-1)

        fontsize = 9
        font_family = "Helvetica"
        xlabel = "sector in cache set"
        ylabel = "cache set"

        def find_step_size(val, num_ticks=4):
            target_step_size = val / num_ticks
            print("target_step_size", target_step_size)
            closest_power_of_2 = np.floor(np.log2(target_step_size))
            print("closest_power_of_2 ", closest_power_of_2)
            closest_frac_power_of_2 = np.floor(np.log2(target_step_size / 3))
            print("closest_frac_power_of_2 ", closest_frac_power_of_2)
            power_of_2_step_size = int(2**closest_power_of_2)
            print("power_of_2_step_size ", power_of_2_step_size)
            frac_power_of_2_step_size = 3 * int(2**closest_frac_power_of_2)
            print("frac_power_of_2_step_size ", frac_power_of_2_step_size)

            if abs(target_step_size - power_of_2_step_size) <= abs(target_step_size - frac_power_of_2_step_size):
                return power_of_2_step_size
            else:
                return frac_power_of_2_step_size

        step_size = find_step_size(sets)
        yticks = np.arange(0, sets + step_size - 1, step=step_size)
        yticks = yticks.astype(int)
        yticklabels = yticks
        print(yticks, yticklabels)

        step_size = find_step_size(cache_lines_per_set)
        xticks = np.arange(0, cache_lines_per_set + step_size - 1, step=step_size)
        xticks = xticks.astype(int)
        xticklabels = xticks
        print(xticks, xticklabels)

        xrotation = 45

        plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

        fig_options = dict(
            figsize=(1.25 * plot.DINA4_WIDTH_INCHES, 0.35 * plot.DINA4_HEIGHT_INCHES),
            layout="constrained",
            sharex=True,
            sharey=True,
        )

        imshow_options = dict(
            # none and nearest should be the same,
            # but apple preview only likes nearest
            # interpolation="none",
            interpolation="nearest",
            origin="upper",
            aspect="auto",
        )

        colorbar_options = dict(
            orientation="horizontal",
            location="top",
            # location="bottom",
        )

        if plot_states:
            fig, axes = plt.subplots(ny, nx, **fig_options)

            # status
            status_colors = {
                "INVALID": "white",
                "RESERVED": "yellow",
                "VALID": "green",
                "MODIFIED": "red",
            }

            status_labels = np.array(status_values)
            status_cmap = matplotlib.colors.ListedColormap([status_colors[status] for status in status_values])
            # status_labels = np.array(list(status_colors.keys()))

            status_norm_bins = np.arange(len(status_labels)).astype(float) + 0.5
            status_norm_bins = np.insert(status_norm_bins, 0, np.min(status_norm_bins) - 1.0)
            status_norm = matplotlib.colors.BoundaryNorm(status_norm_bins, len(status_labels), clip=True)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    states[partition_id],
                    cmap=status_cmap,
                    norm=status_norm,
                    **imshow_options,
                )

                ax.set_title(title(partition_id), fontsize=6)

            bbox_extra_artists = []
            if state_colorbar:
                # use colorbar
                status_fmt = matplotlib.ticker.FuncFormatter(lambda x, _: status_labels[status_norm(x)])
                diff = status_norm_bins[1:] - status_norm_bins[:-1]
                status_ticks = status_norm_bins[:-1] + diff / 2
                cbar = fig.colorbar(
                    im,
                    ax=axes.ravel().tolist(),
                    format=status_fmt,
                    ticks=status_ticks,
                    label="cache sector state",
                    **colorbar_options,
                )
                cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=0)
            else:
                # use legend
                patches = [
                    mpatches.Patch(
                        label=label,
                        facecolor=im.cmap(status_values.index(label)),
                        edgecolor="black",
                        linewidth=1.0,
                    )
                    for label in status_labels
                ]
                legend = plt.figlegend(
                    handles=patches,
                    loc="outside upper center",
                    borderpad=0.2,
                    labelspacing=0.2,
                    columnspacing=2.5,
                    edgecolor="none",
                    frameon=False,
                    fancybox=False,
                    shadow=False,
                    ncols=len(status_labels),
                )
                bbox_extra_artists.append(legend)

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)
            for ax in axes.flatten():
                ax.set_yticks(yticks, yticklabels)
                ax.set_xticks(xticks, xticklabels, rotation=xrotation)

            output = input.with_stem(input.stem + "_states").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, bbox_extra_artists=bbox_extra_artists)
            plt.close(fig)

        if plot_allocations:
            MAX_ALLOCATIONS = 16

            # allocations
            fig, axes = plt.subplots(ny, nx, **fig_options)

            allocation_counts = df["allocation_id"].value_counts()
            allocation_ids = np.nan_to_num(allocation_counts.index.to_numpy(), nan=0.0)
            # print("allocation_counts:")
            # print(allocation_counts)

            threshold = 4 * partitions

            valid_allocation_ids = allocation_ids[allocation_counts > threshold]
            valid_allocation_ids = np.sort(valid_allocation_ids).astype(int)

            if len(valid_allocation_ids) < 1:
                min_allocation_id, max_allocation_id = 0, 0
            else:
                min_allocation_id = np.amin(valid_allocation_ids)
                max_allocation_id = np.amax(valid_allocation_ids)
            print(min_allocation_id, max_allocation_id)
            print("valid allocations", valid_allocation_ids)

            # ensure that we have at least one allocation for normalization
            allocation_ids = np.arange(1, np.amax([2, max_allocation_id + 1]))
            print("allocations", allocation_ids)

            num_allocations = len(allocation_ids)
            if num_allocations > MAX_ALLOCATIONS:
                raise ValueError("too many ({}) allocations".format(num_allocations))

            if num_allocations > 0:
                allocation_boundaries = allocation_ids.astype(float)
            else:
                allocation_boundaries = np.array([1.0])

            allocation_boundaries += 0.5
            allocation_boundaries = np.insert(allocation_boundaries, 0, 0.5)
            print("boundaries", allocation_boundaries)
            allocation_norm = matplotlib.colors.BoundaryNorm(
                boundaries=allocation_boundaries,
                ncolors=np.max([1, num_allocations]),
                clip=True,
            )

            def get_alloc_fmt(x, pos):
                allocation_bin = allocation_norm(x)
                allocation = allocation_ids[allocation_bin]
                # print("fmt for cmap: x={} pos={} -> allocation_ids[{}]={}".format(x, pos, allocation_bin, allocation))
                return allocation

            allocation_fmt = matplotlib.ticker.FuncFormatter(get_alloc_fmt)

            allocation_cmap = plt.get_cmap("hsv", MAX_ALLOCATIONS)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    allocations[partition_id],
                    cmap=allocation_cmap,
                    norm=allocation_norm,
                    **imshow_options,
                )
                ax.set_title(title(partition_id), fontsize=6)

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)
            for ax in axes.flatten():
                ax.set_yticks(yticks, yticklabels)
                ax.set_xticks(xticks, xticklabels, rotation=xrotation)

            bbox_extra_artists = []
            if allocation_colorbar:
                # use color map
                cbar = fig.colorbar(
                    im,
                    ax=axes.ravel().tolist(),
                    format=allocation_fmt,
                    ticks=allocation_ids,
                    label="allocation ID",
                    # aspect=20,
                    **colorbar_options,
                )
            else:
                # use legend
                patches = [
                    mpatches.Patch(
                        color=im.cmap(im.norm(allocation_id)),
                        label="Allocation {}".format(int(allocation_id)),
                    )
                    for allocation_id in allocation_ids
                ]
                legend = plt.figlegend(
                    handles=patches,
                    loc="outside upper center",
                    borderpad=0.2,
                    labelspacing=0.2,
                    columnspacing=2.5,
                    edgecolor="none",
                    frameon=False,
                    fancybox=False,
                    shadow=False,
                    ncols=4,
                )
                bbox_extra_artists.append(legend)

            output = input.with_stem(input.stem + "_allocations").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            # fig.tight_layout()
            fig.savefig(output, bbox_extra_artists=bbox_extra_artists)
            plt.close(fig)

        def format_value_in_thousands(x, pos):
            return plot.human_format_thousands(x, round_to=3, variable_precision=True)

        thousands_fmt = matplotlib.ticker.FuncFormatter(format_value_in_thousands)

        if plot_alloc_times:
            # alloc_times
            fig, axes = plt.subplots(ny, nx, **fig_options)
            # print("alloc_times:")
            # print(alloc_times)

            min_time = alloc_times.min()
            max_time = alloc_times.max() + 10
            print("time (min/max)", min_time, max_time)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    alloc_times[partition_id],
                    vmin=min_time,
                    vmax=max_time,
                    **imshow_options,
                )
                ax.set_title(title(partition_id), fontsize=6)

            fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                label="allocation time (cycle)",
                format=thousands_fmt,
                **colorbar_options,
            )

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)
            for ax in axes.flatten():
                ax.set_yticks(yticks, yticklabels)
                ax.set_xticks(xticks, xticklabels, rotation=xrotation)

            output = input.with_stem(input.stem + "_alloc_times").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)

        if plot_last_access_time:
            # last access time
            fig, axes = plt.subplots(ny, nx, **fig_options)
            # print("last_access_times:")
            # print(last_access_times)

            min_time = last_access_times.min()
            max_time = last_access_times.max() + 10
            print("time (min/max)", min_time, max_time)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    last_access_times[partition_id],
                    vmin=min_time,
                    vmax=max_time,
                    **imshow_options,
                )
                ax.set_title(title(partition_id), fontsize=6)

            fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                label="last access time (cycle)",
                format=thousands_fmt,
                **colorbar_options,
            )

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)
            for ax in axes.flatten():
                ax.set_yticks(yticks, yticklabels)
                ax.set_xticks(xticks, xticklabels, rotation=xrotation)

            output = input.with_stem(input.stem + "_last_access_times").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)


if __name__ == "__main__":
    main()
