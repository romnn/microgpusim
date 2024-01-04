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
# @click.option("--baseline", type=bool, default=True, help="Baseline configurations")
def cache(input):
    if input is None:
        raise ValueError("need input path")
    input = Path(input)
    # if output is None:
    #     output = Path(input).with_suffix(".pdf")
    # print(input)
    # print(output)
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
    assert (
        len(df[["partition", "line_id", "sector"]].drop_duplicates())
        == 3 * MB / sector_size
    )

    status_values = ["INVALID", "RESERVED", "VALID", "MODIFIED"]
    print(df.value_counts(["partition", "status"], dropna=False))
    print(df.value_counts(["partition", "allocation_id"], dropna=False))
    # return

    # dx, dy = int(sets * assoc), int(line_size / sector_size)
    # dx, dy = int(sets), int(sector_size * assoc)
    # data = np.zeros(shape=(partitions, dx, dy))
    states = np.zeros(shape=(partitions, sets, assoc, int(line_size / sector_size)))
    allocations = np.zeros(
        shape=(partitions, sets, assoc, int(line_size / sector_size))
    )
    alloc_times = np.zeros(
        shape=(partitions, sets, assoc, int(line_size / sector_size))
    )
    last_access_times = np.zeros(
        shape=(partitions, sets, assoc, int(line_size / sector_size))
    )

    for (partition_id, set_id, assoc_id), row_df in df.groupby(
        ["partition", "set_id", "assoc_id"]
    ):
        # row_df.sort_values(["line_id", "sector"])
        row_df.sort_values(["line_id", "sector"])
        # print(row_df.head(n=100))
        # break

        # data[set_id * assoc + assoc_id,] = data[]
        # print(set_id)
        # assoc_size = int(assoc * (line_size / sector_size))
        # assoc_size = int(assoc * sector_size)
        # start_idx = (line_size / sector_size)
        # print(int(assoc_id) * assoc_size, (int(assoc_id) + 1) * assoc_size)
        # print(row_df.shape)
        # print(values.shape)
        states[int(partition_id), int(set_id), int(assoc_id), :] = (
            row_df["status"]
            .apply(lambda status: status_values.index(status))
            .to_numpy()
        )
        allocations[int(partition_id), int(set_id), int(assoc_id), :] = row_df[
            "allocation_id"
        ].to_numpy()
        alloc_times[int(partition_id), int(set_id), int(assoc_id), :] = row_df[
            "sector_alloc_time"
        ].to_numpy()
        last_access_times[int(partition_id), int(set_id), int(assoc_id), :] = row_df[
            "last_sector_access_time"
        ].to_numpy()

        # int(assoc_id) * assoc_size : (int(assoc_id) + 1) * assoc_size,
        # print(row_df)
        # print(row_df.shape)

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
    status_norm = matplotlib.colors.BoundaryNorm(
        status_norm_bins, len(status_labels), clip=True
    )
    status_fmt = matplotlib.ticker.FuncFormatter(
        lambda x, _: status_labels[status_norm(x)]
    )

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

    # allocations
    fig, axes = plt.subplots(ny, nx, **fig_options)

    first_allocation_id = int(np.amin(np.nan_to_num(allocations, nan=0.0)))
    last_allocation_id = int(np.amax(np.nan_to_num(allocations, nan=0.0)))
    print("allocations: {} to {}".format(first_allocation_id, last_allocation_id))

    allocation_ids = np.arange(first_allocation_id - 1, last_allocation_id + 1)
    num_allocations = len(allocation_ids) - 1
    allocation_norm_bins = allocation_ids.astype(float)
    allocation_norm = matplotlib.colors.BoundaryNorm(
        allocation_norm_bins, num_allocations, clip=True
    )
    # allocation_norm_bins = np.insert(allocation_norm_bins, 0, np.min(allocation_norm_bins) - 1.0)
    allocation_fmt = matplotlib.ticker.FuncFormatter(lambda x, _: allocation_norm(x))

    diffs = allocation_ids[1:] - allocation_ids[:-1]
    allocation_tick_values = allocation_ids[:-1].astype(float) + diffs / 2.0
    allocation_tick_labels = allocation_ids[:-1].astype(int)
    # print(diffs)
    # print(allocation_tick_values)
    # print(allocation_tick_labels)
    allocation_cmap = plt.get_cmap("jet", num_allocations)
    # cmap.set_under("gray")

    im = None
    for partition_id in range(partitions):
        sy, sx = divmod(partition_id, nx)
        ax = axes[sy, sx]
        im = ax.imshow(
            allocations[partition_id],
            cmap=allocation_cmap,
            # norm=allocation_norm,
            interpolation="none",
            origin="upper",
            aspect="auto",
        )
        ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        # format=allocation_fmt,
        # ticks=allocation_tick_values,
        orientation="horizontal",
        location="bottom",
        label="allocation",
        aspect=20,
    )
    # cbar.ax.set_xticklabels(allocation_tick_labels)
    # cbar.ax.set_width(20)
    # cbar.ax.set_aspect(2)
    # cbar.ax.set_xticklabels(allocation_tick_labels)

    plt.grid(False)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    output = input.with_stem(input.stem + "_allocations").with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)

    # alloc_times
    fig, axes = plt.subplots(ny, nx, **fig_options)

    im = None
    for partition_id in range(partitions):
        sy, sx = divmod(partition_id, nx)
        ax = axes[sy, sx]
        im = ax.imshow(
            alloc_times[partition_id],
            # cmap=status_cmap,
            # norm=status_norm,
            interpolation="none",
            origin="upper",
            aspect="auto",
        )
        ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

    # diff = status_norm_bins[1:] - status_norm_bins[:-1]
    # tickz = status_norm_bins[:-1] + diff / 2
    # fig.colorbar(im, format=status_fmt, ticks=tickz, ax=axes.ravel().tolist())
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        location="bottom",
        label="cache line allocation time (cycle)",
    )  # , format=status_fmt, ticks=tickz, )

    plt.grid(False)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    output = input.with_stem(input.stem + "_alloc_times").with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)

    # last access time
    fig, axes = plt.subplots(ny, nx, **fig_options)

    im = None
    for partition_id in range(partitions):
        sy, sx = divmod(partition_id, nx)
        ax = axes[sy, sx]
        im = ax.imshow(
            last_access_times[partition_id],
            # cmap=status_cmap,
            # norm=status_norm,
            interpolation="none",
            origin="upper",
            aspect="auto",
        )
        ax.set_title("subpartition {}".format(partition_id + 1), fontsize=6)

    # diff = status_norm_bins[1:] - status_norm_bins[:-1]
    # tickz = status_norm_bins[:-1] + diff / 2
    # fig.colorbar(im, format=status_fmt, ticks=tickz, ax=axes.ravel().tolist())
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        location="bottom",
        label="cache line last access time (cycle)",
    )  # , format=status_fmt, ticks=tickz, )

    plt.grid(False)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    output = input.with_stem(input.stem + "_last_access_times").with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # if im is not None:
    #     fig.colorbar(im, ax=cbar_ax)
    # cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    # cmap_values = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    # cmap_values[0] = (0.5, 0.5, 0.5, 1.0)

    # create the new map
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cache states", cmap_values, cmap.N)

    # for ax in axes.flat:
    # im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)
    #
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # for ax in axs.flat:
    #     ax.set(xlabel="x-label", ylabel="y-label")
    #
    # # hide x labels and tick labels for top plots and y ticks for right plots
    # for ax in axs.flat:
    #     ax.label_outer()


if __name__ == "__main__":
    main()
