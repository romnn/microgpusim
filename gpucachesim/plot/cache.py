import click
import pandas as pd
import numpy as np
from pathlib import Path
from os import PathLike
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt

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
def cache(input, plot_states, plot_allocations, plot_alloc_times, plot_last_access_time):
    if input is None:
        raise ValueError("need input path")
    input = Path(input)
    if input.is_dir():
        inputs = sorted(list(input.glob("*.csv")))
    else:
        inputs = [input]

    for input in inputs:
        print("### PROCESSING {}".format(input))
        df = pd.read_csv(
            input,
            header=0,
        )
        sector_size = 32
        sets = 64
        line_size = 128
        assoc = 16
        partitions = 24
        partition_size = sets * line_size * assoc
        cache_size = partition_size * partitions

        assert cache_size == 3 * MB

        df["partition"] = df.index // (partition_size / sector_size)
        df["set_id"] = df["line_id"] // assoc
        df["assoc_id"] = df["line_id"] % assoc
        print(df)

        assert len(df["partition"].unique()) == partitions
        assert len(df["line_id"].unique()) == sets * assoc
        assert len(df["set_id"].unique()) == sets
        assert len(df["assoc_id"].unique()) == assoc
        assert len(df[["partition", "line_id", "sector"]].drop_duplicates()) == 3 * MB / sector_size

        status_values = ["INVALID", "RESERVED", "VALID", "MODIFIED"]
        print(df.value_counts(["partition", "status"], dropna=False))
        print(df.value_counts(["partition", "allocation_id"], dropna=False))

        states = np.zeros(shape=(partitions, sets, assoc, int(line_size / sector_size)))
        allocations = np.zeros(shape=(partitions, sets, assoc, int(line_size / sector_size)))
        alloc_times = np.zeros(shape=(partitions, sets, assoc, int(line_size / sector_size)))
        last_access_times = np.zeros(shape=(partitions, sets, assoc, int(line_size / sector_size)))

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

        nx, ny = 6, 4
        assert nx * ny == 24

        fontsize = 9
        font_family = "Helvetica"
        xlabel = "cache lines in set"
        ylabel = "cache set"

        plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

        fig_options = dict(
            figsize=(1.25 * plot.DINA4_WIDTH_INCHES, 0.35 * plot.DINA4_HEIGHT_INCHES),
            layout="constrained",
            sharex=True,
            sharey=True,
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

            status_cmap = matplotlib.colors.ListedColormap(list(status_colors.values()))
            status_labels = np.array(list(status_colors.keys()))

            # Make normalizer and formatter
            status_norm_bins = np.arange(len(status_colors)).astype(float) + 0.5
            status_norm_bins = np.insert(status_norm_bins, 0, np.min(status_norm_bins) - 1.0)
            status_norm = matplotlib.colors.BoundaryNorm(status_norm_bins, len(status_labels), clip=True)
            status_fmt = matplotlib.ticker.FuncFormatter(lambda x, _: status_labels[status_norm(x)])

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    states[partition_id],
                    cmap=status_cmap,
                    norm=status_norm,
                    interpolation="none",
                    origin="upper",
                    aspect="auto",
                )
                ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

            diff = status_norm_bins[1:] - status_norm_bins[:-1]
            status_ticks = status_norm_bins[:-1] + diff / 2
            cbar = fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                format=status_fmt,
                ticks=status_ticks,
                orientation="horizontal",
                location="bottom",
                label="cache line state",
            )
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=0)

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)

            output = input.with_stem(input.stem + "_states").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)

        if plot_allocations:
            # allocations
            fig, axes = plt.subplots(ny, nx, **fig_options)

            first_allocation_id = 1
            last_allocation_id = int(np.amax(np.nan_to_num(allocations, nan=0.0)))
            print("allocations: {} to {}".format(first_allocation_id, last_allocation_id))

            allocation_ids = np.arange(first_allocation_id, last_allocation_id + 1)
            print(allocation_ids)
            num_allocations = len(allocation_ids)
            print("number of allocations={}".format(num_allocations))

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

            MAX_ALLOCATION_COLORS = 16
            allocation_cmap = plt.get_cmap("hsv", MAX_ALLOCATION_COLORS)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    allocations[partition_id],
                    cmap=allocation_cmap,
                    norm=allocation_norm,
                    interpolation="none",
                    origin="upper",
                    aspect="auto",
                )
                ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

            cbar = fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                format=allocation_fmt,
                ticks=allocation_ids,
                orientation="horizontal",
                location="bottom",
                label="allocation",
                aspect=20,
            )

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)

            output = input.with_stem(input.stem + "_allocations").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)

        if plot_alloc_times:
            # alloc_times
            fig, axes = plt.subplots(ny, nx, **fig_options)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    alloc_times[partition_id],
                    interpolation="none",
                    origin="upper",
                    aspect="auto",
                )
                ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

            fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                location="bottom",
                label="cache line allocation time (cycle)",
            )

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)

            output = input.with_stem(input.stem + "_alloc_times").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)

        if plot_last_access_time:
            # last access time
            fig, axes = plt.subplots(ny, nx, **fig_options)

            im = None
            for partition_id in range(partitions):
                sy, sx = divmod(partition_id, nx)
                ax = axes[sy, sx]
                im = ax.imshow(
                    last_access_times[partition_id],
                    interpolation="none",
                    origin="upper",
                    aspect="auto",
                )
                ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

            fig.colorbar(
                im,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                location="bottom",
                label="cache line last access time (cycle)",
            )

            plt.grid(False)
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)

            output = input.with_stem(input.stem + "_last_access_times").with_suffix(".pdf")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output)
            plt.close(fig)


if __name__ == "__main__":
    main()
