import click
import humanize
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
    labels = DBSCAN(eps=3, min_samples=2).fit_predict(values.reshape(-1, 1))
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
        hit_latencies_df = l1_p_chase(size_bytes=128, stride_bytes=4, warmup=1)
        hit_latencies = hit_latencies_df["latency"].to_numpy()

        miss_latencies_df = l1_p_chase(size_bytes=512 * KB, stride_bytes=32, warmup=1)
        miss_latencies = miss_latencies_df["latency"].to_numpy()

        long_miss_latencies_df = l1_p_chase(size_bytes=2 * MB, stride_bytes=128, warmup=0)
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
    print("found {} outliers ({:1.4}%)".format(num_outliers, num_outliers / len(df)))
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


def l1_p_chase(size_bytes, stride_bytes, warmup, verbose=False):
    cmd = [
        str(P_CHASE_EXECUTABLE.absolute()),
        str(int(size_bytes)),
        str(int(stride_bytes)),
        str(int(warmup)),
    ]
    cmd = " ".join(cmd)
    print(cmd)

    _, stdout, stderr, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=10 * 60,
    )
    # print("stdout:")
    # print(stdout)
    # print("stderr:")
    if verbose:
        print(stderr)

    stdout_reader = StringIO(stdout)
    df = pd.read_csv(
        stdout_reader,
        header=0,
    )
    return df


@main.command()
@click.option("--warmup", "warmup", type=int, help="cache warmup")
def find_l2_prefetch_size(warmup):
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

    combined = []
    for n in range(start_cache_size_bytes, end_cache_size_bytes, step_bytes):
        # compute stride that is sufficient to traverse N twice
        num_loads = 6 * 1024
        min_stride = np.floor(np.log2((2 * n) / (num_loads * 4)))
        # print(2**stride)
        # print(2**stride)
        # stride = n / (6 * 1024 * 4)
        stride_bytes = 2**min_stride * 4
        print(stride_bytes)
        rounds = float(num_loads) / (float(n) / float(stride_bytes))
        assert rounds >= 1.0
        # print(num_loads / (n / stride_bytes))

        # stride_bytes = ((2 * n / 4) / (6 * 1024)) * 4
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, warmup=warmup)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    for n, df in combined.groupby("n"):
        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        # print(df[["hit_cluster", "latency"]])

        hits = df["hit_cluster"] == 1  # l1 miss & l2 hit
        misses = df["hit_cluster"] > 1  # l1 miss & l2 miss

        num_hits = hits.sum()
        num_misses = misses.sum()

        human_size = humanize.naturalsize(n, binary=True)
        miss_rate = float(num_misses) / float(len(df))
        hit_rate = float(num_hits) / float(len(df))
        print(
            "size={:>15} ({:>3.1f}%) \t\thits={:<4} ({}) misses={:<4} ({})".format(
                human_size,
                float(n) / float(total_l2_cache_size_bytes) * 100.0,
                color(num_hits, fg="green", bold=True),
                color("{:>3.1f}%".format(hit_rate * 100.0), fg="green"),
                color(num_misses, fg="red", bold=True),
                color("{:>3.1f}%".format(miss_rate * 100.0), fg="red"),
            )
        )


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
def find_cache_replacement_policy():
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
    cache_size_bytes = 16 * KB
    # cache_size_bytes = 19 * KB
    cache_size_bytes = 24 * KB
    cache_line_bytes = 32
    num_total_cache_lines = cache_size_bytes / cache_line_bytes
    num_sets = 4
    # terminology: num ways == cache lines per set == associativity
    # terminology: way size == num sets
    num_ways = cache_size_bytes / (cache_line_bytes * num_sets)
    print("num ways = {:<3}".format(num_ways))

    assert cache_size_bytes == num_sets * num_ways * cache_line_bytes
    # assert cache_lines_per_set == 128
    # assert num_ways == 192

    # size_bytes = (cache_lines_per_set + 1) * num_sets * cache_line_bytes
    cache_size_bytes = num_sets * num_ways * cache_line_bytes

    # overflow by one cache line
    # stride_bytes = 16  # 4 * 32, so end up in the same set
    stride_bytes = 32  # 4 * 32, so end up in the same set
    # stride_bytes = 128  # 4 * 32, so end up in the same set

    num_sets = 4
    # num_sets = 20

    combined = []
    for set_idx in range(1, num_sets + 1):
        n = cache_size_bytes + set_idx * num_sets * cache_line_bytes
        # (num_ways / 2)
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, warmup=1, verbose=True)
        # df = compute_hits(df, force_misses=False)
        df["n"] = n
        df["set"] = set_idx
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    combined["round"] = 0
    combined["cache_line"] = combined["index"] // cache_line_bytes

    for n, df in combined.groupby("n"):
        print("number of unique indices = {:<4}".format(len(df["index"].unique().tolist())))
        unique_cache_lines = df["cache_line"].unique()
        print("number of unique cache lines = {}".format(len(unique_cache_lines)))
        print("unique cache lines = {}".format(unique_cache_lines))
        assert len(unique_cache_lines) <= num_total_cache_lines

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
            "size={:<10} cache lines={:<3} hits={:<4} ({:2.2f}%) misses={:<4} ({:2.2f}%)".format(
                human_size,
                n / (num_sets * cache_line_bytes),
                color(num_hits, fg="green", bold=True),
                float(num_hits) / float(len(df)) * 100.0,
                color(num_misses, fg="red", bold=True),
                float(num_misses) / float(len(df)) * 100.0,
            )
        )

        # compute mean occurences per index
        mean_rounds = df["index"].value_counts().mean()
        print("mean occurences per index (ROUNDS) = {:3.3f}".format(mean_rounds))

    for n, _ in combined.groupby("n"):
        mask = combined["n"] == n
        df = combined[mask]
        arr_indices = df["index"].values
        round_start_indices = np.array(df.index[arr_indices == arr_indices[0]])
        round_end_indices = np.hstack([round_start_indices[1:], df.index.max()])
        for round, (start_idx, end_idx) in enumerate(zip(round_start_indices, round_end_indices)):
            # print("n={:>5}, start_idx={:>5}, end_idx={:>5}, round={:>3}".format(n, start_idx, end_idx, round))
            combined.loc[start_idx:end_idx, "round"] = round

    # print(combined[["n", "set", "round"]].drop_duplicates())

    # check if pattern is periodic
    for n, df in combined.groupby("n"):
        unique_miss_indices_per_round = []
        unique_miss_cache_lines_per_round = []

        for round, round_df in df.groupby("round"):
            human_size = humanize.naturalsize(n, binary=True)
            print("\n========== {:<15} round[{:>2}] ===========".format(human_size, round))

            misses = round_df[round_df["hit_cluster"] != 0]
            print("num misses = {}".format(len(misses)))
            miss_indices = misses["index"].tolist()
            miss_cache_lines = misses["cache_line"].tolist()
            print("miss cache indices = {}..".format(miss_indices[:25]))
            print("miss cache lines = {}..".format(miss_cache_lines[:25]))

            unique_miss_indices_per_round.append(miss_indices)
            unique_miss_cache_lines_per_round.append(miss_cache_lines)

        def is_periodic(patterns) -> bool:
            for pattern1 in patterns:
                for pattern2 in patterns:
                    l = np.amin(np.array([len(pattern1), len(pattern2)]))
                    if pattern1[:l] != pattern2[:l]:
                        return False
            return True

        if is_periodic(unique_miss_indices_per_round):
            print(
                "{} {}".format(
                    color("IS PERIODIC (LRU)", fg="green"),
                    unique_miss_cache_lines_per_round[0][:25],
                )
            )
        else:
            print(color("IS NON-PERIODIC (NOT LRU)", fg="red"))
            for round, miss_lines in enumerate(unique_miss_cache_lines_per_round):
                print("round {:>2}: miss lines={}".format(round, miss_lines))

    # reverse engineer the cache set mapping
    combined["mapped_set"] = np.nan
    combined["is_hit"] = combined["hit_cluster"] == 0

    set_misses = combined.groupby("set")["is_hit"].mean().reset_index()
    set_misses = set_misses.sort_values(["is_hit"], ascending=True)
    print(set_misses)

    for set_idx in set_misses["set"]:
        set_mask = combined["set"] == set_idx
        miss_mask = combined["is_hit"] == False
        combined.loc[set_mask & miss_mask, "mapped_set"] = set_idx

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

        # find_number_of_sets(df)


"""
Set-associative cache.
The whole cache is divided into sets and each set contains n cache lines (hence n-way cache).
So the relationship stands like this:
    cache size = number of sets in cache * number of cache lines in each set * cache line size
"""


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
def find_cache_sets():
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
    line_size_bytes = 128
    max_sets = 8

    # L1/TEX and L2 have 128B cache lines.
    # Cache lines consist of 4 32B sectors.
    # The tag lookup is at 128B granularity.
    # A miss does not imply that all sectors in the cache line will be filled.

    # 16KB is fully direct-mapped? equal number of misses for consecutive 32B
    cache_size_bytes = 16 * KB - 1 * line_size_bytes
    # 24KB tag lookup is at 128B granularity, have a total of 4 sets
    # after 24KB + 4 * 128B all cache sets are missed and the number of misses does not change
    cache_size_bytes = 24 * KB - 1 * line_size_bytes

    combined = []
    for n in range(cache_size_bytes, cache_size_bytes + max_sets * line_size_bytes, step_bytes):
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    i = 0
    for n, df in combined.groupby("n"):
        if n % KB == 0:
            print("==> {} KB".format(n / KB))

        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        num_hits = (df["hit_cluster"] == 0).sum()
        num_misses = (df["hit_cluster"] != 0).sum()

        # extract miss pattern
        miss_pattern = df.index[df["hit_cluster"] != 0].tolist()
        assert len(miss_pattern) == num_misses

        miss_pattern1 = df.index[df["hit_cluster"] == 1].tolist()
        miss_pattern2 = df.index[df["hit_cluster"] == 2].tolist()

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "i={:<3} size={:<10} lsbs={:<3} hits={:<4} misses={:<4} short miss pattern={}.. long miss pattern = {}..".format(
                i,
                human_size,
                n % 128,
                color(num_hits, fg="green", bold=True),
                color(num_misses, fg="red", bold=True),
                miss_pattern1[:10],
                miss_pattern2[:10],
            )
        )
        i += 1

    find_number_of_sets(combined)


def find_number_of_sets(combined):
    # use DBscan clustering to find the number of sets
    num_misses_per_n = combined.groupby("n")["is_miss"].sum().reset_index()
    num_misses_per_n = num_misses_per_n.rename(columns={"is_miss": "num_misses"})
    num_misses_per_n["set_cluster"] = compute_dbscan_clustering(num_misses_per_n["num_misses"])

    misses_per_set = num_misses_per_n.groupby(["set_cluster"])["num_misses"].mean().reset_index()
    print(misses_per_set)

    set_clusters = misses_per_set["set_cluster"]
    set_clusters = set_clusters[set_clusters >= 0]
    num_clusters = len(set_clusters.unique())
    num_sets = num_clusters - 1
    print("num sets = {:<3}".format(num_sets))


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
def find_cache_line_size():
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
    cache_size_bytes = 16 * KB - 128
    # cache_size_bytes = 24 * KB - 128
    end_size_bytes = cache_size_bytes

    step_bytes = 4
    stride_bytes = 4
    num_lines = 3
    combined = []

    for n in range(cache_size_bytes, end_size_bytes + num_lines * 128, step_bytes):
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    i = 0
    for n, df in combined.groupby("n"):
        if n % KB == 0:
            print("==> {} KB".format(n / KB))

        # reset numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        num_hits = (df["hit_cluster"] == 0).sum()
        num_misses = (df["hit_cluster"] != 0).sum()

        # extract miss pattern
        miss_pattern = df.index[df["hit_cluster"] != 0].tolist()
        assert len(miss_pattern) == num_misses

        miss_pattern1 = df.index[df["hit_cluster"] == 1].tolist()
        miss_pattern2 = df.index[df["hit_cluster"] == 2].tolist()

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "i={:<3} size={:<10} lsbs={:<3} hits={:<4} misses={:<4} short miss pattern ={}.. long miss pattern = {}..".format(
                i,
                human_size,
                n % 128,
                color(num_hits, fg="green", bold=True),
                color(num_misses, fg="red", bold=True),
                miss_pattern1[:12],
                miss_pattern2[:12],
            )
        )
        i += 1

    # print(combined[combined["hit_cluster"] != 0])


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
# def compute_latency_n_graph(start_size, end_size):
def latency_n_graph():
    """
    Compute latency-N graph.

    This is not by itself sufficient to deduce cache parameters but our simulator should match
    this behaviour.
    """
    cache_line_bytes = 128
    # size_bytes = 16 * KB - cache_line_bytes
    cache_size_bytes = 24 * KB - cache_line_bytes
    # size_bytes = 48 * KB - cache_line_bytes
    stride_bytes = 4
    max_ways = 10

    combined = []
    for n in range(cache_size_bytes, cache_size_bytes + max_ways * cache_line_bytes, 16):
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, warmup=1)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    # print(combined)

    # group by n and compute mean latency
    # print(combined[["index"]].drop_duplicates())
    # combined = combined[(combined["index"] * stride_bytes) > (cache_size_bytes - cache_line_bytes)]
    grouped = combined.groupby("n")
    mean_latency = grouped["latency"].mean().reset_index()
    print(mean_latency)


@main.command()
@click.option("--start", "start_size", type=int, help="start cache size in bytes")
@click.option("--end", "end_size", type=int, help="end cache size in bytes")
def find_cache_size(start_size, end_size):
    """
    Step 1.

    Determine cache size C. We set s to 1. We then initialize
    N with a small value and increase it gradually until
    the first cache miss appears. C equals the maximum N
    where all memory accesses are cache hits.
    """

    # check 1KB to 20KB
    default_range = (1 * KB, 20 * KB)
    start_size = start_size or default_range[0]
    end_size = end_size or default_range[1]

    combined = []
    for size in range(start_size, end_size, KB):
        df = l1_p_chase(size_bytes=size, stride_bytes=4, warmup=1)
        df["size"] = size
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)

    # combined.loc[:, hit_cols] = predict_is_hit(latencies).reset_index()
    # after = combined.copy()
    # assert (before["is_hit"] != after["is_hit"]).any()
    # new_df = predict_is_hit(latencies)

    # print(new_df.reset_index().groupby(hit_cols).count().reset_index())
    # print(combined.reset_index().groupby(hit_cols).count().reset_index())

    # print(combined[hit_cols].drop_duplicates())

    # combined.loc[:, ["is_hit", "hit_latency_centroid", "miss_latency_centroid"]] = predict_is_hit(latencies)
    # after = combined.copy()
    # assert (before["is_hit"] != after["is_hit"]).any()
    # new_df = predict_is_hit(latencies)
    # print(new_df[["is_hit", "hit_latency_centroid", "miss_latency_centroid"]].drop_duplicates())
    # print(combined[["is_hit", "hit_latency_centroid", "miss_latency_centroid"]].drop_duplicates())

    # hit_confidence = (combined["hit_latency_centroid"] - combined["miss_latency_centroid"]).abs()
    # bad_hits = (hit_confidence < 10) ^ (hit_confidence > 1000)
    # combined.loc[bad_hits, "is_hit"] = np.nan
    # combined["is_hit"] = combined["is_hit"].bfill()

    combined = compute_hits(combined)

    for size, df in combined.groupby("size"):
        num_hits = (df["hit_cluster"] == 0).sum()
        num_misses = (df["hit_cluster"] != 0).sum()
        human_size = humanize.naturalsize(size, binary=True)
        print(
            "size={:<10} hits={:<4} misses={:<4}".format(
                human_size,
                color(num_hits, fg="green", bold=True),
                color(num_misses, fg="red", bold=True),
            )
        )

    # for size in range(start_size, end_size, KB):
    #     combined[combined["size"] == size]
    # print(combined)

    # 1. overflow cache with 1 element. stride=1, N=4097
    # for (N = 16 * 256; N <= 24 * 256; N += stride) {
    # printf("\n=====%10.4f KB array, warm TLB, read NUM_LOADS element====\n",
    #          sizeof(unsigned int) * (float)N / 1024);
    #   printf("Stride = %d element, %d byte\n", stride,
    #          stride * sizeof(unsigned int));
    #   parametric_measure_global(N, iterations, stride);
    #   printf("===============================================\n\n");
    # }


@main.command()
def test():
    print(16 * KB, 128 * 32 * 4)


@main.command()
@click.option("--warmup", type=int, help="number of warmup interations")
@click.option("--size", type=int, help="size in bytes")
@click.option("--stride", type=int, help="stride in bytes")
# @click.option("--stride", type=int, is_flag=True, help="use nvprof")
def run(warmup, size, stride):
    if warmup is None:
        warmup = 1
    if stride is None:
        stride = 4  # byte
    if size is None:
        size = 16 * 1024  # byte

    # try:
    df = l1_p_chase(size_bytes=size, stride_bytes=stride, warmup=warmup)
    print(df)
    # except cmd_utils.ExecError as e:
    #     raise e


if __name__ == "__main__":
    main()
