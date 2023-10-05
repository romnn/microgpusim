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


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


def predict_is_hit(latencies):
    km = KMeans(n_clusters=2, random_state=1, n_init=5)
    # km = KMedoids(n_clusters=2, random_state=0)
    km.fit(latencies)
    predicted_clusters = km.predict(latencies)

    # print(km.cluster_centers_)
    assert km.cluster_centers_[0] != km.cluster_centers_[1]
    hit_latency_centroid = np.min(km.cluster_centers_)
    miss_latency_centroid = np.max(km.cluster_centers_)
    hit_cluster_idx = 0 if km.cluster_centers_[0] < km.cluster_centers_[1] else 1

    df = pd.DataFrame()
    df["is_hit"] = pd.Series(predicted_clusters)
    df["is_hit"] = df["is_hit"].apply(lambda cluster_idx: cluster_idx == hit_cluster_idx)
    df["hit_latency_centroid"] = hit_latency_centroid
    df["miss_latency_centroid"] = miss_latency_centroid
    # df["hit_confidence"] = np.abs(km.cluster_centers_[0][0] - km.cluster_centers_[1][0])
    # df[["is_hit", "hit_latency_centroid", "miss_latency_centroid"]] = predict_is_hit(latencies)
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

    # check 45KB to 50KB
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

    latency_abs_z_score = np.abs(combined["latency"] - combined["latency"].mean())
    outliers = latency_abs_z_score > 1000

    num_outliers = outliers.sum()
    print("found {} outliers ({:1.4}%)".format(num_outliers, num_outliers / len(combined)))
    combined.loc[outliers, "latency"] = np.nan

    combined["latency"] = combined["latency"].bfill()
    assert combined["latency"].isnull().sum() == 0

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
    latencies = combined["latency"].to_numpy().reshape(-1, 1)
    # print(np.min(latencies))
    # print(np.mean(latencies))
    # print(np.max(latencies))
    # before = combined.copy()
    hit_cols = ["is_hit", "hit_latency_centroid", "miss_latency_centroid"]
    combined[hit_cols] = predict_is_hit(latencies)
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
