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


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


def predict_is_hit(latencies, fit=None):
    km = KMeans(n_clusters=3, random_state=1, n_init=5)
    # km = KMedoids(n_clusters=2, random_state=0)
    if fit is None:
        km.fit(latencies)
    else:
        km.fit(fit)

    predicted_clusters = km.predict(latencies).ravel()

    # sorted_cluster_centroids = np.sort(km.cluster_centers_)
    cluster_centroids = km.cluster_centers_.ravel()
    # print(cluster_centroids)
    sorted_cluster_centroid_indices = np.argsort(cluster_centroids)
    sorted_cluster_centroids = cluster_centroids[sorted_cluster_centroid_indices]
    # print(sorted_cluster_centroid_indices)
    # print(sorted_cluster_centroids)

    hit_latency_centroid = sorted_cluster_centroids[0]
    miss_latency_centroids = sorted_cluster_centroids[1:]
    assert len(miss_latency_centroids) == len(cluster_centroids) - 1

    # hit_cluster_idx = np.argmin(km.cluster_centers_)
    # hit_latency_centroid = np.min(km.cluster_centers_)
    # miss_latency_centroids = np.sort(np.delete(km.cluster_centers_, [hit_cluster_idx]))
    # assert len(miss_latency_centroids) == len(km.cluster_centers_) - 1

    # print(predicted_clusters)
    # print(predicted_clusters.shape)
    sorted_predicted_clusters = sorted_cluster_centroid_indices[predicted_clusters]
    # print(sorted_predicted_clusters)
    # print(sorted_predicted_clusters.shape)
    assert sorted_predicted_clusters.shape == predicted_clusters.shape
    # print(np.unique(predicted_clusters))
    # print(np.unique(sorted_predicted_clusters))

    # df = pd.DataFrame()
    # df["hit_cluster"] =
    # df["hit_cluster"] = df["is_hit"].apply(lambda cluster_idx: cluster_idx == hit_cluster_idx)
    # df["hit_cluster"] = df["is_hit"].apply(lambda cluster_idx: cluster_idx == hit_cluster_idx)
    # .apply(lambda cluster_idx: cluster_idx == hit_cluster_idx)
    return pd.Series(sorted_predicted_clusters), (
        hit_latency_centroid,
        miss_latency_centroids,
    )


def compute_hits(df, force_misses=True):
    latencies = df["latency"].to_numpy()
    fit_latencies = latencies.copy()

    if force_misses:
        miss_latencies_df = l1_p_chase(size_bytes=2 * MB, stride_bytes=128, iters=1)
        miss_latencies = miss_latencies_df["latency"].to_numpy()
        fit_latencies = np.hstack([latencies, miss_latencies])
        # fit_latencies = pd.concat([latencies, miss_latencies_df["latency"]])

    latencies = np.abs(latencies)
    fit_latencies = np.abs(fit_latencies)

    latencies = latencies.reshape(-1, 1)
    fit_latencies = fit_latencies.reshape(-1, 1)

    # print("BEFORE: latency median", df["latency"].median())
    # print("BEFORE: latency min", df["latency"].min())
    # print("BEFORE: latency max", df["latency"].max())

    bins = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, np.inf])
    bin_cols = ["{:>5} - {:<5}".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    hist, _ = np.histogram(fit_latencies, bins=bins)
    hist_df = pd.DataFrame(hist.reshape(1, -1), columns=bin_cols).T
    print("=== LATENCY HISTOGRAM (fitting)")
    print(hist_df)
    # print(hist_df / hist_df.sum())
    print("===")

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
    # df.loc[outliers, "latency"] = np.nan
    #
    # df["latency"] = df["latency"].bfill()
    # assert df["latency"].isnull().sum() == 0

    print("AFTER: mean=%4.2f min=%4.2f max=%4.2f" % (fit_latencies.mean(), fit_latencies.min(), fit_latencies.max()))

    df["hit_cluster"], (
        hit_latency_centroid,
        miss_latency_centroids,
    ) = predict_is_hit(latencies, fit=fit_latencies)
    df["is_hit"] = df["hit_cluster"] == 0
    print(df)

    print("hit_latency_centroid   = {}".format(np.array(hit_latency_centroid)))
    print("miss_latency_centroids = {}".format(np.array(miss_latency_centroids)))

    # hit_confidences = combined["hit_confidence"]
    # hit_confidences = hit_confidences[hit_confidences > 10
    # median_confidence = combined["hit_confidence"].mean()
    # median_confidence = combined["hit_confidence"].mean()
    # print(median_confidence)

    # remove outliers (important)
    # print(combined["latency"].min())
    # print(combined["latency"].max())
    # print(combined["latency"].mean())

    # latency_z_score = np.abs(zscore(combined["latency"]))
    # outliers = latency_z_score > 100

    # print(latency_z_score.min())
    # print(latency_z_score.max())
    # print(combined["latency"].mean())
    # print(combined["latency"].median())
    # print(latency_z_score.mean())
    # print(combined[["latency"]].drop_duplicates())
    # print("AFTER: latency mean", df["latency"].mean())
    # print("AFTER: latency median", df["latency"].median())
    # print("AFTER: latency min", df["latency"].min())
    # print("AFTER: latency max", df["latency"].max())

    # z = np.abs(stats.zscore(df_diabetics["age"]))

    # print(combined[["latency"]].drop_duplicates())

    # latencies = combined["latency"].to_numpy().reshape(-1, 1)
    # kmedoids = KMedoids(n_clusters=2, random_state=0)
    # kmedoids.fit(latencies)

    # dbscan = DBSCAN(eps=5.0, min_samples=20)
    # dbscan.fit(latencies)
    # labels = dbscan.labels_
    # num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print("num clusters", num_clusters)
    # print("outliers", (labels == -1).sum())
    # return

    # Q1 = np.percentile(combined["latency"], 49, method="midpoint")
    # Q3 = np.percentile(combined["latency"], 50, method="midpoint")
    # Q1 = np.percentile(combined["latency"], 1, method="linear")
    # Q3 = np.percentile(combined["latency"], 99, method="linear")
    # IQR = Q3 - Q1
    # print(Q1, Q3)
    # print(Q1, Q3, IQR)
    #
    # lower = Q1 - 1.5 * IQR
    # upper = Q3 + 1.5 * IQR
    # print(upper, lower)
    #
    # return

    # print(combined)
    # print(combined.index)

    # lets recompute hits and misses here, now that we have more data
    # latencies = df["latency"].to_numpy().reshape(-1, 1)

    # df["n"] = n
    # combined.append(df)
    # combined = pd.concat(combined, ignore_index=True)

    # print(np.min(latencies))

    # print(np.mean(latencies))
    # print(np.max(latencies))
    # before = combined.copy()
    # hit_cols = ["is_hit", "hit_latency_centroid", "miss_latency_centroid"]

    # print("hit_latency_centroid   = {:10.3f}".format(hit_latency_centroid))
    # , ",".join({:10.3f}".format(miss_latency_centroids)
    # print("miss_latency_centroid_2 = {:10.3f}".format(miss_latency_centroid_2))
    return df


def l1_p_chase(size_bytes, stride_bytes, iters):
    cmd = [
        str(P_CHASE_EXECUTABLE.absolute()),
        str(int(size_bytes)),
        str(int(stride_bytes)),
        str(int(iters)),
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
    # print(stderr)

    stdout_reader = StringIO(stdout)
    df = pd.read_csv(
        stdout_reader,
        header=0,
    )

    # print(df)
    # latencies = df["latency"].to_numpy().reshape(-1, 1)
    # df[["is_hit", "hit_latency_centroid", "miss_latency_centroid"]] = predict_is_hit(latencies)
    # df[["is_hit", "hit_confidence"]] = predict_is_hit(latencies)
    return df


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
    pass


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
    iters = 1
    stride_bytes = 4
    line_size_bytes = 128
    max_sets = 10

    # cache_size_bytes = 16 * KB - 2 * line_size_bytes
    cache_size_bytes = 24 * KB - 2 * line_size_bytes
    # cache_size_bytes = 12 * KB

    combined = []
    for n in range(cache_size_bytes, cache_size_bytes + max_sets * line_size_bytes, 32):
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, iters=iters)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    i = 0
    for n, df in combined.groupby("n"):
        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        num_hits = df["is_hit"].sum()
        num_misses = (~df["is_hit"]).sum()

        # extract miss pattern
        miss_pattern = df.index[df["is_hit"] == False].tolist()
        # assert len(miss_pattern) == num_misses

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "i={:<3} size={:<10} lsbs={:<3} hits={:<4} misses={:<4} miss pattern={}".format(
                i,
                human_size,
                n % 128,
                num_hits,
                num_misses,
                miss_pattern[:10] + [".."] + miss_pattern[-10:],
            )
        )
        i += 1

    print(combined[combined["is_hit"] == False])


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
    iters = 1
    cache_size_bytes = 16 * KB - 32
    end_size_bytes = cache_size_bytes

    step_bytes = 16  # 4
    stride_bytes = 4  # 1 float
    num_lines = 1
    combined = []

    for n in range(cache_size_bytes, end_size_bytes + num_lines * 128, step_bytes):
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, iters=iters)
        df["n"] = n
        combined.append(df)

    combined = pd.concat(combined, ignore_index=True)
    combined = compute_hits(combined)

    i = 0
    for n, df in combined.groupby("n"):
        # reset numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        num_hits = df["is_hit"].sum()
        num_misses = (~df["is_hit"]).sum()

        # extract miss pattern
        miss_pattern = df.index[df["is_hit"] == False].tolist()
        # assert len(miss_pattern) == num_misses

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "i={:<3} size={:<10} lsbs={:<3} hits={:<4} misses={:<4} miss pattern={}..".format(
                i,
                human_size,
                n % 128,
                num_hits,
                num_misses,
                miss_pattern[:10],
            )
        )
        i += 1

    # print(combined[combined["is_hit"] == False])


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
        df = l1_p_chase(size_bytes=n, stride_bytes=stride_bytes, iters=1)
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
        df = l1_p_chase(size_bytes=size, stride_bytes=4, iters=1)
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
        # print(df)
        num_hits = df["is_hit"].sum()
        num_misses = (~df["is_hit"]).sum()
        human_size = humanize.naturalsize(size, binary=True)
        print("size={:<10} hits={:<4} misses={:<4}".format(human_size, num_hits, num_misses))

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
@click.option("--iters", type=int, help="number of interations")
@click.option("--size", type=int, help="size in bytes")
@click.option("--stride", type=int, help="stride in bytes")
# @click.option("--stride", type=int, is_flag=True, help="use nvprof")
def run(iters, size, stride):
    if iters is None:
        iters = 1
    if stride is None:
        stride = 4  # byte
    if size is None:
        size = 16 * 1024  # byte

    # try:
    df = l1_p_chase(size_bytes=size, stride_bytes=stride, iters=iters)
    print(df)
    # except cmd_utils.ExecError as e:
    #     raise e


if __name__ == "__main__":
    main()
