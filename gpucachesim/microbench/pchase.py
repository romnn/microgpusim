import click
import humanize
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from io import StringIO
from sklearn.cluster import KMeans


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
    km.fit(latencies)
    predicted_clusters = km.predict(latencies)

    assert km.cluster_centers_[0] != km.cluster_centers_[1]
    hit_cluster_idx = 0 if km.cluster_centers_[0] < km.cluster_centers_[1] else 1

    is_hit = pd.Series(predicted_clusters)
    is_hit = is_hit.apply(lambda cluster_idx: cluster_idx == hit_cluster_idx)
    print(np.abs(km.cluster_centers_[0] - km.cluster_centers_[1]))
    return is_hit


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
    print(stderr)

    stdout_reader = StringIO(stdout)
    df = pd.read_csv(
        stdout_reader,
        header=0,
    )

    # print(df)
    latencies = df["latency"].to_numpy().reshape(-1, 1)
    df["is_hit"] = predict_is_hit(latencies)
    return df


"""
L1D cache: 48 KB
S:64:128:6,L:L:m:N:H,A:128:8,8
sets            = 64
line size       = 128B
assoc           = 6
"""


"""
# 64 sets, each 128 bytes 16-way for each memory sub partition (128 KB per memory sub partition).
# This gives 3MB L2 cache
# EDIT: GTX 1080 has 2MB L2 instead of 3MB (no change here since 8 instead of 11 mem controllers)
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
    if start_size is None:
        # start_size = 45 * KB
        start_size = 1 * KB
    if end_size is None:
        end_size = 64 * KB

    for size in range(start_size, end_size, KB):
        df = l1_p_chase(size_bytes=size, stride_bytes=4, iters=1)
        num_hits = df["is_hit"].sum()
        num_misses = (~df["is_hit"]).sum()
        human_size = humanize.naturalsize(size, binary=True)
        print("size={:<10} hits={:<4} misses={:<4}".format(human_size, num_hits, num_misses))

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
