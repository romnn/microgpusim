import click
import humanize
import typing
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from io import StringIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from scipy.stats import zscore
from wasabi import color


from gpucachesim.benchmarks import REPO_ROOT_DIR
import gpucachesim.cmd as cmd_utils

P_CHASE_EXECUTABLE = REPO_ROOT_DIR / "test-apps/microbenches/chxw/p_chase_l1"

# suppress scientific notation by setting float_format
# pd.options.display.float_format = "{:.3f}".format
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.seterr(all="raise")
np.set_printoptions(suppress=True)


"""
Generations:
Tesla < Fermi < Kepler < Maxwell < Pascal < Turing < Ampere < Lovelace

"""

"""
L1D cache: 48 KB
# from GTX1080 whitepaper:
96 KB shared memory unit, 48 KB of total L1 cache storage, and eight texture units.
64 KB of L1 cache for each SM plus a special 48 KB texture unit memory (can be read-only cache)
S:64:128:6,L:L:m:N:H,A:128:8,8
sets            = 64
line size       = 128B
assoc           = 6
"""


"""
# 64 sets, each 128 bytes 16-way for each memory sub partition (128 KB per memory sub partition).
# This gives 3MB L2 cache
# EDIT: GTX 1080 has 2MB L2 instead of 3MB (no change here since 8 instead of 11 mem controllers)
# 256 KB per mem controller
# -gpgpu_cache:dl2 N:64:128:16,L:B:m:W:L,S:1024:1024,4:0,32 # used to be 128:4
# 128B cache line * 64sets * 16ways * 8mem_ctrl * 2sub_part_per_mem_ctrl = 2097152
-gpgpu_cache:dl2 S:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32 # used to be 128:4
-gpgpu_cache:dl2_texture_only 0 
"""

KB = 1024
MB = 1024**2


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


def compute_dbscan_clustering(values):
    values = np.array(values)
    labels = DBSCAN(eps=2, min_samples=3).fit_predict(values.reshape(-1, 1))
    # clustering_df = pd.DataFrame(
    #     np.array([values.ravel(), labels.ravel()]).T,
    #     columns=["latency", "cluster"],
    # )
    # print(clustering_df)
    return labels


def predict_is_hit(latencies, fit=None):
    km = KMeans(n_clusters=3, random_state=1, n_init=5)
    # km = KMedoids(n_clusters=2, random_state=0)
    if fit is None:
        km.fit(latencies)
    else:
        km.fit(fit)

    predicted_clusters = km.predict(latencies)

    cluster_centroids = km.cluster_centers_.ravel()
    sorted_cluster_centroid_indices = np.argsort(cluster_centroids)
    # sorted_cluster_centroid_indices = np.array([2, 0, 1])
    sorted_cluster_centroid_indices_inv = np.argsort(sorted_cluster_centroid_indices)
    # print(sorted_cluster_centroid_indices_inv)
    # print(np.arange(3))
    # print(np.arange(3)[sorted_cluster_centroid_indices[::-1]])
    # print(np.arange(3).where(np.arange(3) == sorted_cluster_centroid_indices))
    # sorted_cluster_centroid_new_indices = cluster_centroids.where()
    sorted_cluster_centroids = cluster_centroids[sorted_cluster_centroid_indices]
    # print(km.cluster_centers_)
    # print(cluster_centroids)
    # print(predicted_clusters)
    #
    # print(sorted_cluster_centroid_indices)
    # print(sorted_cluster_centroids)
    # print(np.sort(cluster_centroids))
    assert (np.sort(cluster_centroids) == sorted_cluster_centroids).all()

    # hit_latency_centroid = sorted_cluster_centroids[0]
    # miss_latency_centroids = sorted_cluster_centroids[1:]
    # assert len(miss_latency_centroids) == len(cluster_centroids) - 1

    # print(predicted_clusters)
    # sorted_predicted_clusters = np.put(
    #     predicted_clusters,
    #     cluster_centroid_indices * sorted_cluster_centroid_indices,
    # )
    sorted_predicted_clusters = sorted_cluster_centroid_indices_inv[predicted_clusters]
    # print(sorted_predicted_clusters)
    assert sorted_predicted_clusters.shape == predicted_clusters.shape

    return pd.Series(sorted_predicted_clusters), sorted_cluster_centroids
    # (
    #     hit_latency_centroid,
    #     miss_latency_centroids,
    # )


def compute_hits(df, force_misses=True):
    latencies = df["latency"].to_numpy()
    fit_latencies = latencies.copy()

    if force_misses or True:
        hit_latencies_df, _ = p_chase(mem="l1data", size_bytes=128, stride_bytes=4, warmup=1)
        hit_latencies = hit_latencies_df["latency"].to_numpy()

        miss_latencies_df, _ = p_chase(mem="l1data", size_bytes=512 * KB, stride_bytes=32, warmup=1)
        miss_latencies = miss_latencies_df["latency"].to_numpy()

        long_miss_latencies_df, _ = p_chase(mem="l2", size_bytes=2 * MB, stride_bytes=128, warmup=0)
        long_miss_latencies = long_miss_latencies_df["latency"].to_numpy()

        fit_latencies = np.hstack([latencies, hit_latencies, miss_latencies, long_miss_latencies])

    latencies = np.abs(latencies)
    fit_latencies = np.abs(fit_latencies)

    latencies = latencies.reshape(-1, 1)
    fit_latencies = fit_latencies.reshape(-1, 1)

    bins = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, np.inf])
    bin_cols = ["{:>5} - {:<5}".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

    pred_hist, _ = np.histogram(latencies, bins=bins)
    pred_hist_df = pd.DataFrame(pred_hist.reshape(1, -1), columns=bin_cols).T

    fit_hist, _ = np.histogram(fit_latencies, bins=bins)
    fit_hist_df = pd.DataFrame(fit_hist.reshape(1, -1), columns=bin_cols).T

    print("=== LATENCY HISTOGRAM (prediction)")
    print(pred_hist_df)
    print("")
    print("=== LATENCY HISTOGRAM (fitting)")
    print(fit_hist_df)
    print("")

    clustering_bins = fit_hist[bins[:-1] <= 1000.0]
    print(clustering_bins)

    # find the top 3 bins and infer the bounds for outlier detection
    # hist_percent = hist / np.sum(hist)
    # valid_latency_bins = hist[hist_percent > 0.5]
    # latency_cutoff = np.min(valid_latency_bins)

    print("BEFORE: mean=%4.2f min=%4.2f max=%4.2f" % (fit_latencies.mean(), fit_latencies.min(), fit_latencies.max()))

    # latency_abs_z_score = np.abs(latencies - np.median(fit_latencies))
    # outliers = latency_abs_z_score > 1000
    outliers = fit_latencies > 1000

    num_outliers = outliers.sum()
    print(num_outliers)
    print(len(df))
    print("found {} outliers ({:1.4}%)".format(num_outliers, float(num_outliers) / float(len(df))))
    fit_latencies[outliers] = np.amax(fit_latencies[~outliers])
    print("AFTER: mean=%4.2f min=%4.2f max=%4.2f" % (fit_latencies.mean(), fit_latencies.min(), fit_latencies.max()))

    # df["hit_cluster"], (
    #     hit_latency_centroid,
    #     miss_latency_centroids,
    # ) = predict_is_hit(latencies, fit=fit_latencies)
    df["hit_cluster"], latency_centroids = predict_is_hit(latencies, fit=fit_latencies)

    # df["is_hit"] = df["hit_cluster"] == 0
    # df["is_miss"] = ~df["is_hit"]

    print("latency_centroids = {}".format(latency_centroids))
    # print("hit_latency_centroid   = {}".format(np.array(hit_latency_centroid)))
    # print("miss_latency_centroids = {}".format(np.array(miss_latency_centroids)))
    return df


def p_chase(mem, stride_bytes, warmup, size_bytes=None, start_size_bytes=None, end_size_bytes=None, step_size_bytes=None):
    cmd = [
        str(P_CHASE_EXECUTABLE.absolute()),
        str(mem.lower()),
    ]
    if end_size_bytes is None:
        # run for single size
        assert size_bytes is not None
        cmd += [ str(int(size_bytes)) ]
    else:
        # run for multiple sizes
        assert start_size_bytes is not None
        assert end_size_bytes is not None
        assert step_size_bytes is not None
        cmd += [
            str(int(start_size_bytes)),
            str(int(end_size_bytes)),
            str(int(step_size_bytes)),
        ]

    cmd += [
        str(int(stride_bytes)),
        str(int(warmup)),
    ]

    cmd = " ".join(cmd)
    print(cmd)

    _, stdout, stderr, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=10 * 60,
    )

    stdout_reader = StringIO(stdout)
    df = pd.read_csv(
        stdout_reader,
        header=0,
        dtype=float,
    )
    # base units are 32 bit integers
    # df["n"] = df["n"] * 4
    # df["index"] = df["index"] * 4  
    return df, (stdout, stderr)


@main.command()
@click.option("--warmup", "warmup", type=int, help="cache warmup")
@click.option("--repetitions", "repetitiions", type=int, default=10, help="repetitions")
def find_l2_prefetch_size(warmup, repetitiions):
    warmup = warmup or 0

    total_l2_cache_size_bytes = 2 * MB
    step_bytes = 100 * KB
    start_cache_size_bytes = step_bytes
    # start_cache_size_bytes = 800 * KB
    end_cache_size_bytes = 8 * MB
    # end_cache_size_bytes = 900 * KB
    # stride_bytes = 10 * KB

    # strides = [2**i for i in range(int(np.floor(np.log2(start_cache_size_bytes))))]
    # print(strides)

    total_l2_cache_size_bytes = 2 * MB
    start_cache_size_bytes = 32 * KB
    end_cache_size_bytes = 3 * MB
    end_cache_size_bytes = 1 * MB
    # end_cache_size_bytes = 512 * KB
    step_bytes = 32 * KB

    combined = []
    for repetition in range(repetitiions):
        for n in range(start_cache_size_bytes, end_cache_size_bytes, step_bytes):
            # compute stride that is sufficient to traverse N twice
            # iterations = 48 * KB
            # min_stride = np.floor(np.log2((2 * n) / (iterations * 4)))
            #
            # stride_bytes = 2**min_stride * 4
            # rounds = float(iterations) / (float(n) / float(stride_bytes))
            # assert rounds >= 1.0
            # print(rounds)

            stride_bytes = 128

            # stride_bytes = ((2 * n / 4) / (6 * 1024)) * 4
            df, (_, stderr) = p_chase(mem="l2", size_bytes=n, stride_bytes=stride_bytes, warmup=warmup)
            print(stderr)
            df["n"] = n
            df["r"] = repetition
            combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    # compute the mean latency
    combined = combined.groupby(["n", "r", "index"]).mean().reset_index()
    combined = compute_hits(combined)
    combined = compute_rounds(combined)

    # # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index
    #
    # combined = combined[combined["round"].isin(full_rounds)]

    for n, df in combined.groupby("n"):
        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        hits = df["hit_cluster"] <= 1  # l1 hits or l1 miss & l2 hit
        misses = df["hit_cluster"] > 1  # l1 miss & l2 miss

        num_hits = hits.sum()
        num_misses = misses.sum()

        human_size = humanize.naturalsize(n, binary=True)
        miss_rate = float(num_misses) / float(len(df))
        hit_rate = float(num_hits) / float(len(df))
        print(
            # \t\thits={:<4} ({}) misses={:<4} ({})".format(
            "size {:>15} ({:>3.1f}%)".format(
                human_size,
                float(n) / float(total_l2_cache_size_bytes) * 100.0,
                # color(num_hits, fg="green", bold=True),
                # color("{:>3.1f}%".format(hit_rate * 100.0), fg="green"),
                # color(num_misses, fg="red", bold=True),
                # color("{:>3.1f}%".format(miss_rate * 100.0), fg="red"),
            )
        )

        for round, round_df in df.groupby("round"):
            # count hits and misses
            hits = round_df["hit_cluster"] <= 1  # l1 hits or l1 miss & l2 hit
            misses = round_df["hit_cluster"] > 1  # l1 miss & l2 miss

            num_hits = hits.sum()
            num_misses = misses.sum()

            human_size = humanize.naturalsize(n, binary=True)
            miss_rate = float(num_misses) / float(len(round_df))
            hit_rate = float(num_hits) / float(len(round_df))
            print(
                "\t\tround={:>2} \t\thits={:<4} ({}) misses={:<4} ({})".format(
                    round,
                    color(num_hits, fg="green", bold=True),
                    color("{:>3.1f}%".format(hit_rate * 100.0), fg="green"),
                    color(num_misses, fg="red", bold=True),
                    color("{:>3.1f}%".format(miss_rate * 100.0), fg="red"),
                )
            )
        print("\n")


def compute_rounds(df):
    df["round"] = np.nan
    if "r" not in df:
        df["r"] = 0
    for (n, r), _ in df.groupby(["n", "r"]):
        mask = (df["n"] == n) & (df["r"] == r)
        arr_indices = df.loc[mask, "index"].values

        intra_round_start_index = 0

        round_start_index = df[mask].index.min()
        round_end_index = df[mask].index.max()

        intra_round_start_indices = df[mask].index[arr_indices == intra_round_start_index]
        round_start_indices = np.hstack([round_start_index, intra_round_start_indices])
        # shift right and add end index
        round_end_indices = np.hstack([round_start_indices[1:], round_end_index])
        assert len(round_start_indices) == len(round_end_indices)

        for round, (start_idx, end_idx) in enumerate(zip(round_start_indices, round_end_indices)):
            round_size = end_idx - start_idx
            assert round_size >= 0
            if round_size == 0:
                continue
            # print("n={: >7}, start_idx={: >7}, end_idx={: >7}, len={: <7} round={: <3}".format(n, start_idx, end_idx, end_idx - start_idx, round))
            df.loc[start_idx:end_idx, "round"] = round

    # print(df[["n", "set", "round"]].drop_duplicates())
    return df


def compute_number_of_sets(combined):
    # use DBscan clustering to find the number of sets
    combined["is_miss"] = combined["hit_cluster"] != 0
    num_misses_per_n = combined.groupby("n")["is_miss"].sum().reset_index()
    num_misses_per_n = num_misses_per_n.rename(columns={"is_miss": "num_misses"})
    num_misses_per_n["set_cluster"] = compute_dbscan_clustering(num_misses_per_n["num_misses"])

    misses_per_set = num_misses_per_n.groupby(["set_cluster"])["num_misses"].mean().reset_index()
    print(misses_per_set)

    set_clusters = misses_per_set["set_cluster"]
    set_clusters = set_clusters[set_clusters >= 0]
    num_clusters = len(set_clusters.unique())
    num_sets = num_clusters - 1
    print("DBSCAN clustering found ", color(f"{num_sets} sets", fg="blue", bold=True))
    return num_sets, misses_per_set 



@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option("--repetitions", "repetitions", default=1, type=int, help="number of repetitions")
@click.option("--mem", "mem", default="l1data", type=str, help="memory to microbenchmark")
def find_cache_replacement_policy(repetitions,mem):
    """
    Determine cache replacement policy.

    As mentioned before, if the cache replacement policy is LRU, then the
    memory access process should be periodic and all the
    cache ways in the cache set are missed. If memory
    access process is aperiodic, then the replacement policy
    cannot be LRU.

    Under this circumstance, we set N = C + b, s = b with a
    considerable large k (k >> N/s) so that we can traverse
    the array multiple times. All cache misses are from one cache set.
    Every cache miss is caused by its former cache replacement because we
    overflow the cache by only one cache line. We have the
    accessed data indices thus we can reproduce the full
    memory access process and find how the cache lines
    are updated.
    """
    repetitions = max(1, repetitions)

    known_cache_size_bytes = 24 * KB
    known_cache_line_bytes = 128
    derived_total_cache_lines = known_cache_size_bytes / known_cache_line_bytes

    # 768 cache lines
    print("total cache lines = {:<3}".format(derived_total_cache_lines))  

    known_num_sets = 4

    # 48 ways
    # terminology: num ways == cache lines per set == associativity
    # terminology: way size == num sets
    derived_num_ways = known_cache_size_bytes / (known_cache_line_bytes * known_num_sets)
    print("num ways = {:<3}".format(derived_num_ways)) 

    assert known_cache_size_bytes == known_num_sets * derived_num_ways * known_cache_line_bytes

    derived_cache_lines_per_set = derived_total_cache_lines / known_num_sets

    # stride_bytes = known_cache_line_bytes
    stride_bytes = 32

    match mem.lower():
        case "l1readonly":
            stride_bytes = known_cache_line_bytes
            pass


    if False:
        combined = []
        for repetition in range(repetitions):
            for set_idx in range(1, known_num_sets + 1):
                # overflow by mulitples of the cache line
                n = known_cache_size_bytes + set_idx * known_cache_line_bytes
                # n = known_cache_size_bytes + set_idx * derived_cache_lines_per_set
                # n = known_cache_size_bytes + set_idx * derived_cache_lines_per_set * known_cache_line_bytes
                # n = known_cache_size_bytes + set_idx * derived_cache_lines_per_set * 32
                # n = known_cache_size_bytes + set_idx * derived_num_ways * known_cache_line_bytes
                # n = known_cache_size_bytes + set_idx * derived_num_ways
                df, (_, stderr) = p_chase(mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=2)
                print(stderr)
                df["n"] = n
                df["r"] = repetition
                df["set"] = set_idx
                combined.append(df)

        combined = pd.concat(combined, ignore_index=True)
    else:
        step_size_bytes = known_cache_line_bytes
        start_size_bytes = known_cache_size_bytes + 1 * step_size_bytes
        end_size_bytes = known_cache_size_bytes + known_num_sets * step_size_bytes
        combined, (_, stderr) = p_chase(
                mem=mem,
                start_size_bytes=start_size_bytes,
                end_size_bytes=end_size_bytes,
                step_size_bytes=step_size_bytes,
                stride_bytes=stride_bytes,
                warmup=5)
        print(stderr)
        combined["r"] = 0
        combined["set"] = (combined["n"] % known_cache_size_bytes) // step_size_bytes

    # print(combined[["n", "set"]].drop_duplicates())

    # print(combined.columns)
    # combined = combined.groupby(["n", "set", "latency", "index"]).mean().reset_index()
    combined = compute_hits(combined)
    combined = compute_rounds(combined)

    # combined["cache_line"] = combined["index"] // cache_line_bytes
    # combined["cache_line"] = (combined["index"] // cache_line_bytes) % num_total_cache_lines
    # combined["cache_line"] = ((combined.index * stride_bytes) // cache_line_bytes) % num_total_cache_lines
    #
    # combined["cache_line"] = (combined["index"] // cache_line_bytes) % num_total_cache_lines
    # combined["cache_line"] = combined["cache_line"] % num_total_cache_lines

    # we have more cache lines because  we overflow the cache by some cache lines
    # either we remove them or we use mod on the cache line computation
    # print(combined["cache_line"].unique())
    # print(len(combined["cache_line"].unique()))
    # print(num_total_cache_lines)
    # assert len(combined["cache_line"].unique()) >= num_total_cache_lines

    # filter out the overflow cache lines
    # combined = combined[combined["cache_line"] < num_total_cache_lines]


    # # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index
    # # num_full_rounds = len(full_rounds)
    #
    # combined = combined[combined["round"].isin(full_rounds)]
    # # combined = combined[combined["round"] == 0]
    # # combined = combined[combined["round"] > num_full_rounds / 2]
    # # combined = combined[combined["round"] < num_full_rounds]

    line_size_address_bits = int(np.floor(np.log2(known_cache_line_bytes)))
    set_address_bits = known_num_sets - 1
    print(np.binary_repr(set_address_bits))
    predicted_set = combined["index"].astype(int).to_numpy() >> line_size_address_bits
    combined["predicted_set"] = predicted_set & set_address_bits

    sector_size = 32
    combined["cache_line"] = combined["index"] // sector_size
    # assert known_cache_size_bytes == known_num_sets * derived_num_ways * known_cache_line_bytes
    # combined["cache_line"] = combined["index"] // known_cache_line_bytes # works but random
    # combined["cache_line"] = combined["index"] % (known_num_sets * derived_num_ways) # wrong
    # combined["cache_line"] = combined["index"] % (derived_num_ways * derived_cache_lines_per_set) # wrong
    # combined["cache_line"] = combined["index"] % (known_num_sets * derived_cache_lines_per_set) # wrong
    combined["cache_line"] = (combined["index"] % known_cache_size_bytes) // sector_size

    for round, _ in combined.groupby("round"):
        # round_mask = combined["round"] == round
        # lines = combined.loc[round_mask, "index"] // known_cache_line_bytes
        # combined.loc[round_mask, "cache_line"] = 
        # combined.loc[round_mask, "new_index"] = np.arange(len(combined[round_mask])) * stride_bytes
        # combined.loc[round_mask, "cache_line"] = combined.loc[round_mask, "new_index"] // known_cache_line_bytes
        pass

    for n, df in combined.groupby("n"):
        print("number of unique indices = {:<4}".format(len(df["index"].unique().tolist())))
        # unique_cache_lines = df["cache_line"].unique()
        # print("number of unique cache lines = {}".format(len(unique_cache_lines)))
        # print("unique cache lines = {}".format(unique_cache_lines))
        # assert len(unique_cache_lines) <= num_total_cache_lines

        # count hits and misses
        num_hits = (df["hit_cluster"] == 0).sum()
        num_misses = (df["hit_cluster"] != 0).sum()
        # hit_pattern = df.index[df["hit_cluster"] == 0].tolist()

        # extract miss pattern
        # miss_pattern = df.index[df["hit_cluster"] != 0].tolist()
        # assert len(miss_pattern) == num_misses

        # miss_pattern1 = df.index[df["hit_cluster"] == 1].tolist()
        # miss_pattern2 = df.index[df["hit_cluster"] == 2].tolist()

        human_size = humanize.naturalsize(n, binary=True)

        print(
            "size={:<10} cache lines={:<3.1f} hits={:<4} ({:2.2f}%) misses={:<4} ({:2.2f}%)".format(
                human_size,
                float(n) / float(known_cache_line_bytes),
                color(num_hits, fg="green", bold=True),
                float(num_hits) / float(len(df)) * 100.0,
                color(num_misses, fg="red", bold=True),
                float(num_misses) / float(len(df)) * 100.0,
            )
        )

        # compute mean occurences per index
        mean_rounds = df["index"].value_counts().mean()
        print(mean_rounds)
        print("mean occurences per index (ROUNDS) = {:3.3f}".format(mean_rounds))
        assert mean_rounds >= 1.0

    # check if pattern is periodic
    for (set_idx, n), df in combined.groupby(["set", "n"]):
        human_size = humanize.naturalsize(n, binary=True)
        miss_indices_per_round = []
        miss_cache_lines_per_round = []
        miss_rounds = []
        num_rounds = len(df["round"].unique())
        max_round = df["round"].max()
        print("################ set={: <2} has {: >2} rounds".format(set_idx, num_rounds))

        for round, round_df in df.groupby("round"):
            if (round == 0 or round == max_round) and num_rounds > 2:
                # skip 
                print(color("skip set={: <2} round={: <2}".format(set_idx, round), fg="info"))
                continue

            human_size = humanize.naturalsize(n, binary=True)
            misses = round_df[round_df["hit_cluster"] != 0]
            miss_indices = misses["index"].astype(int)
            miss_cache_lines = misses["cache_line"].astype(int)
            print("set={: <2} round={: <2} has {: >4} misses".format(set_idx, round, len(miss_cache_lines)))

            if False:
                print("num misses = {}".format(len(misses)))
                print("miss cache indices = {}..".format(miss_indices.tolist()[:25]))
                print("miss cache lines = {}..".format(miss_cache_lines.tolist()[:25]))
                print("max missed cache line = {}".format(miss_cache_lines.max()))
                print("mean missed cache line = {}".format(miss_cache_lines.mean()))
                print("min missed cache line = {}".format(miss_cache_lines.min()))

            miss_rounds.append(round)
            miss_indices_per_round.append(miss_indices.tolist())
            miss_cache_lines_per_round.append(miss_cache_lines.tolist())

        def is_periodic(patterns, strict=True, max_rel_err=0.05) -> typing.Tuple[list, int]:
            matches = []
            for pattern1 in patterns:
                num_matches = 0
                for pattern2 in patterns:
                    l = np.amin(np.array([len(pattern1), len(pattern2)]))
                    if pattern1[:l] == pattern2[:l]:
                        num_matches += 1
                    elif strict == False:
                        # try soft match: allow removing up to rel_err percent items from pattern2
                        # this will not affet the ordering of elements
                        remove = set(pattern2) - set(pattern1)
                        occurences = len([i for i in pattern2 if i in remove])
                        rel_err = occurences / len(pattern2)
                        # print("soft match: need to remove {:2.4f}% elements".format(rel_err))
                        if rel_err <= max_rel_err:
                            new_pattern2 = [i for i in pattern2 if i not in remove]
                            l = np.amin(np.array([len(pattern1), len(new_pattern2)]))
                            if pattern1[:l] == new_pattern2[:l]:
                                num_matches += 1

                matches.append((pattern1, num_matches))

            matches = sorted(matches, key=lambda m: m[1], reverse=True)
            assert matches[0][1] >= matches[-1][1]
            (best_match, num_matches) = matches[0]
            return sorted(best_match), num_matches

        # miss_patterns = miss_indices_per_round
        miss_patterns = miss_cache_lines_per_round
        assert len(miss_patterns) > 0, "set {} has no miss patterns".format(set_idx)

        strict_pattern_match, num_strict_matches = is_periodic(miss_patterns, strict=True)
        soft_pattern_match, num_soft_matches = is_periodic(miss_patterns, strict=False)
        assert num_soft_matches >= num_strict_matches
        valid_miss_count = len(strict_pattern_match)

        print("set={: <2} best match has {: >4} misses".format(set_idx, valid_miss_count))

        # filter out bad rounds
        before = len(combined)
        invalid = 0
        for round, misses in zip(miss_rounds, miss_patterns):
            if sorted(misses) != sorted(strict_pattern_match):
            # if len(misses) != valid_miss_count:
                print(color("set={: <2} round={: <2} SKIP (too many misses)".format(set_idx, round), fg="warn"))
                combined = combined.loc[~((combined["set"] == set_idx) & (combined["round"] == round)),:]
                invalid += 1
        after = len(combined)
        if invalid > 0:
            assert before > after

        info = "[{: <2}/{: <2} strict match, {: <2}/{: <2} soft match, {: >4} unqiue miss lines]\n".format(num_strict_matches, len(miss_patterns), num_soft_matches, len(miss_patterns), len(combined.loc[(combined["set"] == set_idx) & (combined["hit_cluster"] != 0), "cache_line"].unique()))


        # if num_strict_matches == len(miss_patterns):
        if len(miss_patterns) > 1 and float(num_strict_matches) / float(len(miss_patterns)) > 0.7:
            # # filter out bad rounds
            # before = len(combined)
            # invalid = 0
            # for round, misses in zip(miss_rounds, miss_patterns):
            #     if len(misses) != valid_miss_count:
            #         print(color("set={: <2} round={: <2} SKIP (too many misses)".format(set_idx, round), fg="warn"))
            #         combined = combined.loc[~((combined["set"] == set_idx) & (combined["round"] == round)),:]
            #         invalid += 1
            # after = len(combined)
            # if invalid > 0:
            #     assert before > after

            print(
                "set={: >2} size={: <10}".format(set_idx, human_size), 
                color("IS PERIODIC (LRU) \t", fg="green"),
                info,
                np.array(strict_pattern_match[:32]),
            )
        elif num_soft_matches == len(miss_patterns):
            # # filter out bad rounds
            # before = len(combined)
            # invalid = 0
            # for round, misses in zip(miss_rounds, miss_patterns):
            #     if len(misses) != valid_miss_count:
            #         print("set={: <2} round={: <2} SKIP".format(set_idx, round))
            #         combined = combined.loc[~((combined["set"] == set_idx) & (combined["round"] == round)),:]
            #         invalid += 1
            # after = len(combined)
            # if invalid > 0:
            #     assert before > after

            print(
                "size={: <10}".format(human_size), 
                color("IS PERIODIC (LRU) \t", fg="warn"),
                info,
                np.array(soft_pattern_match[:32]),
            )
        else:
            print(
                "size={: <10}".format(human_size),
                color("IS NON-PERIODIC (NOT LRU) \t", fg="red"), 
                info
            )
            for round, miss_lines in enumerate(miss_patterns):
                print("========== set={: <2} round[{: <2}] ===========\t {: >5} missed lines:\n{}".format(set_idx, round, len(miss_lines), miss_lines))

    for (set_idx, n), set_df in combined.groupby(["set", "n"]):
        set_misses = set_df.loc[set_df["hit_cluster"] != 0, "cache_line"]
        print("set={: <2} has {: <4} missed cache lines ({: >4} unique)".format(set_idx, len(set_misses), len(set_misses.unique())))

    # combined = combined[combined["round"] == 1]

    # reverse engineer the cache set mapping
    combined["mapped_set"] = np.nan
    combined["is_hit"] = combined["hit_cluster"] == 0

    set_misses = combined.groupby("set")["is_hit"].mean().reset_index()
    set_misses = set_misses.sort_values(["is_hit"], ascending=True)
    print(set_misses)

    total_unique_indices = combined["index"].unique()
    total_unique_cache_lines = combined["cache_line"].unique()
    print("total unique indices {}".format(len(total_unique_indices)))
    print("total unique cache lines {}".format(len(total_unique_cache_lines)))

    cache_line_set_mapping = pd.DataFrame(
            np.array(combined["cache_line"].unique()), columns=["cache_line"])
    # cache_line_set_mapping = combined["cache_line"].unique().to_frame()
    # print(cache_line_set_mapping)
    cache_line_set_mapping["mapped_set"] = np.nan
    cache_line_set_mapping = cache_line_set_mapping.sort_values("cache_line")
    cache_line_set_mapping = cache_line_set_mapping.reset_index()

    # print(cache_line_set_mapping)
    # print(cache_line_set_mapping.index)

    last_unique_miss_indices = None
    last_unique_miss_cache_lines = None
    for set_idx in set_misses["set"]:
        # print("SET {}".format(set_idx))
        # print(combined.loc[combined["set"] == set_idx, ["set", "index", "cache_line"]].head(10))
        set_mask = combined["set"] == set_idx
        miss_mask = combined["is_hit"] == False

        miss_indices = combined.loc[set_mask & miss_mask, "index"].astype(int).unique()
        miss_cache_lines = combined.loc[set_mask & miss_mask, "cache_line"].astype(int).unique()
        print("\n=== set={: <2}\t {: >4} miss lines {: >4} miss indices".format(set_idx, len(miss_cache_lines), len(miss_indices)))
        # print("miss indices = {}".format(sorted(miss_indices)))
        print("miss lines   = {}".format(sorted(miss_cache_lines)))

        unique_miss_indices = combined.loc[set_mask & miss_mask, "index"].unique()
        # print("set {} unique miss indices {}".format(set_idx, len(unique_miss_indices)))
        if last_unique_miss_indices is not None:
            diff = set(unique_miss_indices).difference(set(last_unique_miss_indices))
            # print("OVERLAP: {}".format(len(diff)))

        unique_miss_cache_lines = combined.loc[set_mask & miss_mask, "cache_line"].unique()
        # print("set {} unique miss cache lines {}".format(set_idx, len(unique_miss_cache_lines)))
        if last_unique_miss_cache_lines is not None:
            diff = set(unique_miss_cache_lines).difference(set(last_unique_miss_cache_lines))
            # print("OVERLAP: {}".format(len(diff)))

        last_unique_miss_indices = unique_miss_indices.copy()
        last_unique_miss_cache_lines = unique_miss_cache_lines.copy()

        for miss_cache_line in unique_miss_cache_lines:
            combined.loc[combined["cache_line"] == miss_cache_line, "mapped_set"] = set_idx

            cache_line_set_mapping.loc[cache_line_set_mapping["cache_line"] == miss_cache_line, "mapped_set"] = set_idx

            # cache_line_set_mapping.loc[cache_line_set_mapping["cache_line"] == miss_cache_line, "mapped_set"] = set_idx
        # set_df = combined.loc[set_mask, "is_hit"].copy()
        # set_df = set_df.reset_index()
        # print(set_df.index)
        # print(set_idx)
        # print(cache_line_set_mapping)
        # print(set_df["is_hit"])
        # cache_line_set_mapping.loc[~set_df["is_hit"], "mapped_set"] = set_idx

        # print(cache_line_set_mapping["mapped_set"].value_counts())

        # for miss_index in unique_miss_indices:
        #     combined.loc[combined["index"] == miss_index, "mapped_set"] = set_idx

    if False:
        miss_mask = combined["is_hit"] == False
        # combined["set4_miss"] = (combined["set"] == 4) & miss_mask
        # combined["set3_miss"] = (combined["set"] == 4) & miss_mask
        set4_miss_lines = combined.loc[(combined["set"] == 4) & miss_mask, "cache_line"].unique()
        set3_miss_lines = combined.loc[(combined["set"] == 3) & miss_mask, "cache_line"].unique()
        set2_miss_lines = combined.loc[(combined["set"] == 2) & miss_mask, "cache_line"].unique()
        set1_miss_lines = combined.loc[(combined["set"] == 1) & miss_mask, "cache_line"].unique()
        set4_3_diff = sorted(list(set(set4_miss_lines) - set(set3_miss_lines)))
        set3_2_diff = sorted(list(set(set3_miss_lines) - set(set2_miss_lines)))
        set2_1_diff = sorted(list(set(set2_miss_lines) - set(set1_miss_lines)))
        print("set 4 - 3", len(set4_3_diff))
        print("set 3 - 2", len(set3_2_diff))
        print("set 2 - 1", len(set2_1_diff))

        print("set 4 - 3", set4_3_diff)
        print("set 3 - 2", set3_2_diff)
        print("set 2 - 1", set2_1_diff)

    # print(cache_line_set_mapping)
    set_probability = cache_line_set_mapping["mapped_set"].value_counts().reset_index()
    set_probability["prob"] = set_probability["count"] / set_probability["count"].sum()
    print(set_probability)
    print("sum of cache lines per set = {: >4}/{: <4}".format(set_probability["count"].sum(), len(total_unique_cache_lines)))
    assert set_probability["count"].sum() == len(total_unique_cache_lines)


    mapped_sets = combined.loc[~combined["is_hit"], "mapped_set"]
    unmapped_sets_percent = mapped_sets.isnull().sum() / float(len(mapped_sets))
    print(unmapped_sets_percent)
    assert unmapped_sets_percent <= 0.05

    # TEMP try
    # combined = combined[combined["set"] == 4]
    # combined = combined[combined["round"] == combined["round"].max()]

    # brute force the set-mapping
    # set_mapping = combined.loc[~combined["is_hit"], ["index", "cache_line", "set", "mapped_set"]]
    set_mapping = combined.loc[
        ~combined["is_hit"],
        ["index", "cache_line", "set", "mapped_set", "predicted_set"],
    ]
    # print(set_mapping[["cache_line", "mapped_set"]].drop_duplicates().head(10))
    # print(set_mapping[["index", "mapped_set"]].drop_duplicates().head(10))

    # check if the set-mapping is random
    # print(set_mapping[["index", "mapped_set"]].value_counts().reset_index()["mapped_set"].value_counts())
    # print(set_mapping["mapped_set"].value_counts())
    # set_probability = set_mapping["mapped_set"].value_counts().reset_index()
    if False:
        for set_col in ["set", "mapped_set", "predicted_set"]:
            set_probability = set_mapping[set_col].value_counts().reset_index()
            set_probability["prob"] = set_probability["count"] / set_probability["count"].sum()
            print(set_col)
            print(set_probability)

    # print(full_round_size)
    # print(int(set_probability["count"].sum()))
    # assert full_round_size == int(set_probability["count"].sum())

    # print(set_mapping["mapped_set"].value_counts())
    # print(set_mapping.groupby("index")["mapped_set"].value_counts())
    # ["mapped_set"].value_counts())

    min_set_bits = int(np.ceil(np.log2(known_num_sets)))
    max_set_bits = 64  # 64 bit memory addresses
    max_set_bits = min_set_bits + 1  # 64 bit memory addresses
    for set_bits in range(min_set_bits, max_set_bits):
        for i in range(64 - set_bits, 0, -1):
            # print("bits {:<2} to {:<2}".format(i, i + set_bits))

            # hash
            # set_mapping
            pass

    # set_df = combined[combined["set"] == set_idx]
    # set_df = combined[combined["set"] == set_idx]
    # set_df_misses
    # for set_idx, set_df in (
    #     combined.groupby("set")["is_hit"].mean().reset_index().sort_values(["is_hit"], ascending=True)
    # ):
    #     print(set_idx)
    #     print(set_df)
    # for set_idx, set_df in combined.sort_values("is_hit", ascending=True)["set"].unique():
    # for set_idx in combined.sort_values("is_hit", ascending=True)["set"].unique():
    #     print(set_idx, combined[combined["set"] == set_idx]["is_hit"].sum())
    #     # combined["mapped_set"] =

    return
    for n, _ in combined.groupby(["n", "set"]):
        # print("short miss pattern = {}..".format(miss_pattern1[:25]))
        # print("long miss pattern  = {}..".format(miss_pattern2[:25]))
        # print("hit pattern", hit_pattern[:35])
        # print("miss pattern", miss_pattern1[:35])
        # print(miss_pattern1)
        # print(miss_pattern2)

        # print(df.index)
        repetition_start_indices = np.array(df.index[df["index"] == df["index"][0]])
        # print("repetition indices", repetition_start_indices.tolist())

        # indices = df["index"]
        # cache_lines = df["cache_line"]
        # print("short miss pattern = {}..".format(indices[miss_pattern1[:25]].tolist()))
        # print("long miss pattern  = {}..".format(indices[miss_pattern2[:25]].tolist()))

        # print(df.index[~df["is_hit"]].max())
        # misses = ~df["is_hit"]
        # miss_indices = indices[misses]
        # miss_cache_lines = cache_lines[misses]
        # print("unique miss indices = {}..".format(miss_indices.unique()))

        # for i in range(len(repetition_indices) - 1):
        # print(repetition_indices[1:] + [df.index.stop])

        repetition_end_indices = np.hstack([repetition_start_indices[1:], df.index.stop])
        for repetition, (start_idx, end_idx) in enumerate(zip(repetition_start_indices, repetition_end_indices)):
            print("\n========== repetition[{:>2}] {:>4} to {:<4} ===========".format(repetition, start_idx, end_idx))
            df.iloc[start_idx:end_idx]["round"] = repetition
            repetition_df = df.iloc[start_idx:end_idx]

            misses = repetition_df[repetition_df["hit_cluster"] != 0]
            print("num misses = {}".format(len(misses)))
            # print("miss cache lines = {}".format(misses["cache_line"].tolist()))
            print("unique miss cache indices = {}".format(misses["index"].unique().tolist()))
            print("unique miss cache lines = {}".format(misses["cache_line"].unique().tolist()))

        # compute the

        # missed_cache_lines = indices[~df["is_hit"]] / stride_bytes
        # print("miss indices = {}..".format(missed_cache_lines.unique()))

        # compute_number_of_sets(df)


"""
Set-associative cache.
The whole cache is divided into sets and each set contains n cache lines (hence n-way cache).
So the relationship stands like this:
    cache size = number of sets in cache * number of cache lines in each set * cache line size
"""


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option("--mem", "mem", type=str, default="l1data", help="memory to microbenchmark")
def find_cache_sets(mem):
    """
    Determine number of cache sets T.

    We set s to b.
    We then start with N = C and increase N at the granularity of b.
    Every increment causes cache misses of a new cache set.
    When N > C + (T − 1)b, all cache sets are missed.
    We can then deduce T from cache miss patterns accordingly.
    """
    stride_bytes = 8
    step_bytes = 8

    predicted_max_sets = 8

    # L1/TEX and L2 have 128B cache lines.
    # Cache lines consist of 4 32B sectors.
    # The tag lookup is at 128B granularity.
    # A miss does not imply that all sectors in the cache line will be filled.

    # 16KB is fully direct-mapped? equal number of misses for consecutive 32B
    # cache_size_bytes = 16 * KB - 1 * line_size_bytes
    # 24KB tag lookup is at 128B granularity, have a total of 4 sets
    # after 24KB + 4 * 128B all cache sets are missed and the number of misses does not change
    known_cache_line_bytes = 128
    known_cache_size_bytes = 24 * KB

    match mem.lower():
        case "l1readonly":
            stride_bytes = 16
            pass


    start_cache_size_bytes = known_cache_size_bytes - 1 * known_cache_line_bytes
    end_cache_size_bytes = known_cache_size_bytes + predicted_max_sets * known_cache_line_bytes

    combined = []
    for n in range(start_cache_size_bytes, end_cache_size_bytes, step_bytes):
        df, (_, stderr) = p_chase(mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        print(stderr)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    for n, df in combined.groupby("n"):
        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        num_hits = (df["hit_cluster"] == 0).sum()
        num_misses = (df["hit_cluster"] != 0).sum()
        num_l1_misses = (df["hit_cluster"] == 1).sum()
        num_l2_misses = (df["hit_cluster"] == 2).sum()

        hit_rate = float(num_hits) / float(len(df)) * 100.0
        miss_rate = float(num_misses) / float(len(df)) * 100.0

        # extract miss pattern
        miss_pattern = df.index[df["hit_cluster"] != 0].tolist()
        assert len(miss_pattern) == num_misses

        l1_misses = df.index[df["hit_cluster"] == 1].tolist()
        l2_misses = df.index[df["hit_cluster"] == 2].tolist()

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "size={: <10} lsbs={: <3} hits={: <4} ({}) l1 misses={: <4} l2 misses={: <4} (miss rate={}) l1 misses={}.. l2 misses= {}..".format(
                human_size,
                n % 128,
                str(color(num_hits, fg="green", bold=True)),
                color("{: >2.2f}%".format(hit_rate), fg="green"),
                str(color(num_l1_misses, fg="red", bold=True)),
                str(color(num_l2_misses, fg="red", bold=True)),
                color("{: >2.2f}%".format(miss_rate), fg="red"),
                l1_misses[:10],
                l2_misses[:10],
            )
        )

        if n % KB == 0:
            print("==> {} KB".format(n / KB))
        if n % known_cache_line_bytes == 0:
            human_cache_line_bytes = humanize.naturalsize(known_cache_line_bytes, binary=True)
            print("==> start of predicted cache line ({})".format(human_cache_line_bytes ))

    compute_number_of_sets(combined)



@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
def find_cache_line_size(mem):
    """
    Step 2.

    Determine cache line size b. We set s to 1.
    We begin with `N = C + 1` and increase N gradually again.
    When `N < C + b + 1`, the numbers of cache misses are close.
    When N is increased to `C + b + 1`, there is a sudden
    increase on the number of cache misses, despite that
    we only increase N by 1. Accordingly we can find b.
    Based on the memory access patterns, we can also have
    a general idea on the cache replacement policy.
    """
    known_cache_size_bytes = 24 * KB
    predicted_cache_line_bytes = 128

    step_bytes = 8
    stride_bytes = 8
    num_lines = 8

    match mem.lower():
        case "l1readonly":
            # stride_bytes = 8
            pass
    
    start_size_bytes = known_cache_size_bytes - 2 * predicted_cache_line_bytes
    end_size_bytes = known_cache_size_bytes + num_lines * predicted_cache_line_bytes

    combined = []
    for n in range(start_size_bytes, end_size_bytes, step_bytes):
        df, (_, stderr) = p_chase(mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        print(stderr)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)
    combined = compute_rounds(combined)

    num_unique_indices = len(combined["index"].unique())

    # # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index
    # print("have {: >3} rounds (full round size is {: <5})".format(len(full_rounds), full_round_size))
    #
    # combined = combined[combined["round"].isin(full_rounds)]
    # # combined = combined[combined["round"].isin([0, 1])]

    i = 0
    for n, df in combined.groupby("n"):
        if n % KB == 0:
            print("==> {} KB".format(n / KB))

        if n % predicted_cache_line_bytes == 0:
            human_cache_line_bytes = humanize.naturalsize(predicted_cache_line_bytes, binary=True)
            print("==> start of predicted cache line ({})".format(human_cache_line_bytes))


        # reset numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        if True:
            num_hits = (df["hit_cluster"] == 0).sum()
            num_misses = (df["hit_cluster"] != 0).sum()
            num_l1_misses = (df["hit_cluster"] == 1).sum()
            num_l2_misses = (df["hit_cluster"] == 2).sum()

            hit_rate = float(num_hits) / float(len(df)) * 100.0
            miss_rate = float(num_misses) / float(len(df)) * 100.0
        else:
            per_round = df.groupby("round")["hit_cluster"].value_counts().reset_index()
            mean_hit_clusters = per_round.groupby("hit_cluster")["count"].median().reset_index()

            num_hits = int(mean_hit_clusters.loc[mean_hit_clusters["hit_cluster"] == 0, "count"].sum())
            num_misses = int(mean_hit_clusters.loc[mean_hit_clusters["hit_cluster"] != 0, "count"].sum())
            num_l1_misses = int(mean_hit_clusters.loc[mean_hit_clusters["hit_cluster"] == 1, "count"].sum())
            num_l2_misses = int(mean_hit_clusters.loc[mean_hit_clusters["hit_cluster"] == 2, "count"].sum())

            hit_rate = float(num_hits) / float(num_unique_indices) * 100.0
            miss_rate = float(num_misses) / float(num_unique_indices) * 100.0

        # extract miss patterns
        # miss_pattern = df.index[df["hit_cluster"] != 0].tolist()
        # miss_pattern1 = df.index[df["hit_cluster"] == 1].tolist()
        # miss_pattern2 = df.index[df["hit_cluster"] == 2].tolist()

        human_size = humanize.naturalsize(n, binary=True)
        print(
             # short miss pattern ={}.. long miss pattern = {}..".format(
            "i={: >3} size={: >10} lsbs={: <3} hits={: <4} ({}) l1 misses={: <4} l2 misses={: <4} (miss rate={})".format(
                i,
                human_size,
                n % 128,
                str(color(num_hits, fg="green", bold=True)),
                color("{: >2.2f}%".format(hit_rate), fg="green"),
                str(color(num_l1_misses, fg="red", bold=True)),
                str(color(num_l2_misses, fg="red", bold=True)),
                color("{: >2.2f}%".format(miss_rate), fg="red"),
                # miss_pattern1[:6],
                # miss_pattern2[:6],
            )
        )
        i += 1

    # print(combined[combined["hit_cluster"] != 0])


def round_to_multiple_of(x, multiple_of):
    return multiple_of * round(x/multiple_of)

def quantize_latency(latency, bin_size=50):
    return round_to_multiple_of(latency, multiple_of=bin_size)

@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
def latency_n_graph(mem):
    """
    Compute latency-N graph.

    This is not by itself sufficient to deduce cache parameters but our simulator should match
    this behaviour.
    """
    known_cache_line_bytes = 128
    known_cache_size_bytes = 24 * KB
    known_cache_sets = 4

    stride_bytes = 16
    step_size_bytes = 32

    match mem.lower():
        case "l1readonly":
            pass

    start_cache_size_bytes = known_cache_size_bytes - 1 * known_cache_line_bytes
    end_cache_size_bytes = known_cache_size_bytes + (known_cache_sets + 1) * known_cache_line_bytes

    combined = []
    for n in range(start_cache_size_bytes, end_cache_size_bytes, step_size_bytes):
        df, (_, stderr) = p_chase(mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        print(stderr)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)
    combined = compute_rounds(combined)

    # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index

    iter_size = int(combined["n"].value_counts().mean())
    assert (combined["n"].value_counts() == iter_size).all()

    # for n in range(start_cache_size_bytes, end_cache_size_bytes, step_size_bytes):
    #     one_round_size = n / stride_bytes
    #     expected_num_rounds = iter_size / one_round_size
    #     print("expected num rounds for n={: >7} is {:<3} (one full round is {: >5})".format(n, expected_num_rounds, one_round_size))
    #
    #     round_sizes = combined.loc[combined["n"] == n, "round"].value_counts()
    #     print(round_sizes)
    #     full_rounds = round_sizes[round_sizes == one_round_size].index
    #     print(full_rounds)
    #     assert len(full_rounds) > 0
    #
    #     combined = combined[(combined["n"] != n) ^ (combined["round"].isin(full_rounds))]

    # combined["bin_latency"] = combined["latency"].apply(lambda latency: quantize_latency(latency))
    # print(combined[["n", "bin_latency"]].value_counts().reset_index().sort_values("n"))

    grouped = combined.groupby("n")
    mean_latency = grouped["latency"].mean().reset_index().sort_values("n")
    print(mean_latency)


@main.command()
@click.option("--start", "start_size", type=int, help="start cache size in bytes")
@click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
def find_cache_size(start_size, end_size, mem):
    """
    Step 1.

    Determine cache size C. We set s to 1. We then initialize
    N with a small value and increase it gradually until
    the first cache miss appears. C equals the maximum N
    where all memory accesses are cache hits.
    """
    
    predicted_cache_size_bytes = 24 * KB

    step_size_bytes = 1 * KB
    stride_bytes = 4

    match mem.lower():
        case "l1readonly":
            stride_bytes = 16

    search_interval_bytes = 4 * KB
    start_size = start_size or predicted_cache_size_bytes - search_interval_bytes
    end_size = end_size or predicted_cache_size_bytes + search_interval_bytes

    combined = []
    for n in range(start_size, end_size, step_size_bytes):
        df, (_, stderr) = p_chase(mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        print(stderr)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)
    combined = compute_rounds(combined)
    print("number of unqiue indices = {: <4}".format(len(combined["index"].unique())))

    for n, df in combined.groupby("n"):
        if n % KB == 0:
            print("==> {} KB".format(n / KB))

        num_hits = (df["hit_cluster"] == 0).sum()
        num_misses = (df["hit_cluster"] != 0).sum()

        hit_rate = float(num_hits) / float(len(df)) * 100.0
        miss_rate = float(num_misses) / float(len(df)) * 100.0

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "size={: >10} hits={: <4} ({}) misses={: <4} ({})".format(
                human_size,
                str(color(num_hits, fg="green", bold=True)),
                color("{: >2.2f}%".format(hit_rate), fg="green"),
                str(color(num_misses, fg="red", bold=True)),
                color("{: >2.2f}%".format(miss_rate), fg="red"),
            )
        )

@main.command()
def test():
    print(16 * KB, 128 * 32 * 4)


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="memory to microbenchmark")
@click.option("--warmup", type=int, help="number of warmup interations")
@click.option("--size", type=int, help="size in bytes")
@click.option("--stride", type=int, help="stride in bytes")
@click.option("--verbose", type=bool, is_flag=True, help="verbose output")
# @click.option("--stride", type=int, is_flag=True, help="use nvprof")
def run(mem, warmup, size, stride, verbose):
    if warmup is None:
        warmup = 1
    if stride is None:
        stride = 32
    if size is None:
        size = 16 * KB

    df, (stdout, stderr) = p_chase(mem=mem, size_bytes=size, stride_bytes=stride, warmup=warmup)
    # print(stdout)
    print(stderr)
    print(df)


if __name__ == "__main__":
    main()
