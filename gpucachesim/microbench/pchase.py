import click
import math
import humanize
import typing
import numpy as np
import pandas as pd
import itertools
import pyeda
import pyeda.boolalg
import pyeda.boolalg.expr
import pyeda.boolalg.minimization
import time
import sympy as sym
import logicmin
import re
import bitarray
import bitarray.util
import tempfile
import gpucachesim.remote as remote
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from io import StringIO, BytesIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from scipy.stats import zscore
from wasabi import color
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gpucachesim.asm import (
    solve_mapping_table_xor_fast,
    solve_mapping_table_xor,
    solve_mapping_table,
)
from gpucachesim.benchmarks import REPO_ROOT_DIR
from gpucachesim.plot import PLOT_DIR
import gpucachesim.cmd as cmd_utils
import gpucachesim.plot as plot

NATIVE_P_CHASE = REPO_ROOT_DIR / "test-apps/microbenches/chxw/pchase"
NATIVE_SET_MAPPING = REPO_ROOT_DIR / "test-apps/microbenches/chxw/set_mapping"

SIM_P_CHASE = REPO_ROOT_DIR / "target/release/pchase"
SIM_SET_MAPPING = REPO_ROOT_DIR / "target/release/pchase"

CACHE_DIR = PLOT_DIR / "cache"

SEC = 1
MIN = 60 * SEC

# valid GPU device names in DAS6 cluster
VALID_GPUS = [None, "A4000", "A100"]

# suppress scientific notation by setting float_format
# pd.options.display.float_format = "{:.3f}".format
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 1000)
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

def round_up_to_next_power_of_two(x):
    return np.power(2, np.ceil(np.log2(x)))

def round_down_to_next_power_of_two(x):
    return np.power(2, np.floor(np.log2(x)))

def round_to_multiple_of(x, multiple_of):
    return multiple_of * np.round(x / multiple_of)


def round_up_to_multiple_of(x, multiple_of):
    return multiple_of * np.ceil(x / multiple_of)

def round_down_to_multiple_of(x, multiple_of):
    return multiple_of * np.floor(x / multiple_of)

def quantize_latency(latency, bin_size=50):
    return round_to_multiple_of(latency, multiple_of=bin_size)


def compute_dbscan_clustering(values):
    values = np.array(values)
    labels = DBSCAN(eps=2, min_samples=3).fit_predict(values.reshape(-1, 1))
    # clustering_df = pd.DataFrame(
    #     np.array([values.ravel(), labels.ravel()]).T,
    #     columns=["latency", "cluster"],
    # )
    # print(clustering_df)
    return labels


def predict_is_hit(latencies, fit=None, num_clusters=3):
    km = KMeans(n_clusters=num_clusters, random_state=1, n_init=15)
    # km = KMedoids(n_clusters=2, random_state=0)
    if fit is None:
        km.fit(latencies)
    else:
        km.fit(fit)

    predicted_clusters = km.predict(latencies)

    cluster_centroids = km.cluster_centers_.ravel()
    sorted_cluster_centroid_indices = np.argsort(cluster_centroids)
    sorted_cluster_centroid_indices_inv = np.argsort(sorted_cluster_centroid_indices)
    sorted_cluster_centroids = cluster_centroids[sorted_cluster_centroid_indices]
    assert (np.sort(cluster_centroids) == sorted_cluster_centroids).all()

    sorted_predicted_clusters = sorted_cluster_centroid_indices_inv[predicted_clusters]
    assert sorted_predicted_clusters.shape == predicted_clusters.shape

    assert len(sorted_cluster_centroids) == num_clusters
    return pd.Series(sorted_predicted_clusters), sorted_cluster_centroids


def get_latency_distribution(latencies, bins=None):
    if bins is None:
        bins = np.array(
            [
                0,
                50,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                600,
                700,
                800,
                1000,
                2000,
                np.inf,
            ]
        )
    bin_cols = [
        "{:>5} - {:<5}".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)
    ]

    hist, _ = np.histogram(latencies, bins=bins)
    hist_df = pd.DataFrame(hist.reshape(1, -1), columns=bin_cols).T.reset_index()
    hist_df.columns = ["bin", "count"]

    hist_df["bin_start"] = [bins[i] for i in range(len(bins) - 1)]
    hist_df["bin_end"] = [bins[i] for i in range(1, len(bins))]
    hist_df["bin_mid"] = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    return hist_df


class PChaseConfig(typing.NamedTuple):
    mem: str
    sim: bool
    start_size_bytes: int
    end_size_bytes: int
    step_size_bytes: int
    stride_bytes: int
    warmup: typing.Optional[int]
    max_rounds: typing.Optional[int]
    iter_size: typing.Optional[int]
    repetitions: typing.Optional[int]


def collect_full_latency_distribution(sim, gpu=None, force=False, configs=None):
    if configs is None:
        configs = []

        # include 64 l1 hit (size_bytes < l1 size)
        size_bytes = 256
        stride_bytes = 4
        assert size_bytes / stride_bytes <= 64
        configs.append(
            PChaseConfig(
                mem="l1data",
                start_size_bytes=size_bytes,
                end_size_bytes=size_bytes,
                step_size_bytes=1,
                stride_bytes=stride_bytes,
                warmup=1,
                iter_size=64,
                max_rounds=None,
                repetitions=1,
                sim=sim,
            )
        )

        # include 64 l1 miss + l2 hit (l1 size < size_bytes < l2 size)
        size_bytes = 256
        stride_bytes = 4
        assert size_bytes / stride_bytes <= 64
        configs.append(
            PChaseConfig(
                mem="l2",
                start_size_bytes=size_bytes,
                end_size_bytes=size_bytes,
                step_size_bytes=1,
                stride_bytes=stride_bytes,
                warmup=1,
                repetitions=1,
                max_rounds=None,
                iter_size=64,
                sim=sim,
            )
        )

        # include 64 l1 miss + l2 miss (l2 size < size_bytes)
        size_bytes = 2 * get_known_cache_size_bytes(mem="l2", gpu=gpu)
        stride_bytes = 128
        assert size_bytes / stride_bytes >= 64
        configs.append(
            PChaseConfig(
                mem="l2",
                start_size_bytes=size_bytes,
                end_size_bytes=size_bytes,
                step_size_bytes=1,
                stride_bytes=stride_bytes,
                warmup=0,
                repetitions=1,
                iter_size=64,
                max_rounds=None,
                sim=sim,
            )
        )

    latencies = []
    for config in configs:
        hit_latencies_df, _ = pchase(**config._asdict(), gpu=gpu, force=force)
        hit_latencies = hit_latencies_df["latency"].to_numpy()
        latencies.append(hit_latencies)
    # if gpu is None:
    # else:
        # run remote
        # das6 = remote.DAS6()
        # try:
        #     for config in configs:
        #         stdout, stderr = das6.run_pchase_sync(
        #             cmd, gpu=gpu,
        #             executable=das6.remote_scratch_dir / "pchase",
        #             force=False)
        #         hit_latencies_df, _ = pchase(**config._asdict())
        #         hit_latencies = hit_latencies_df["latency"].to_numpy()
        #         latencies.append(hit_latencies)
        #
        # except Exception as e:
        #     das6.close()
        #     raise e

    # if True:
    #     size_bytes=128
    #     # max_rounds=None if fair else 2
    #     hit_latencies_df, _ = pchase(
    #         mem="l1data",
    #         start_size_bytes=size_bytes,
    #         end_size_bytes=size_bytes,
    #         step_size_bytes=1,
    #         stride_bytes=4,
    #         warmup=0,
    #         max_rounds=max_rounds,
    #         repetitions=repetitions,
    #         sim=sim,
    #     )
    #     hit_latencies = hit_latencies_df["latency"].to_numpy()
    #     latencies.append(hit_latencies)
    #
    # if True:
    #     # include l2 hits (l1 size < size_bytes < l2 size)
    #     size_bytes=256 * KB
    #     # max_rounds = None if fair else 2
    #     miss_latencies_df, _ = pchase(
    #         mem="l1data",
    #         start_size_bytes=size_bytes,
    #         end_size_bytes=size_bytes,
    #         step_size_bytes=1,
    #         stride_bytes=128,
    #         warmup=0,
    #         max_rounds=max_rounds,
    #         repetitions=repetitions,
    #         sim=sim,
    #     )
    #     miss_latencies = miss_latencies_df["latency"].to_numpy()
    #     latencies.append(miss_latencies)
    #
    # # include l2 misses (l2 size < size_bytes)
    # if True:
    #     size_bytes=2 * MB
    #     # iter_size = None if fair else 512
    #     long_miss_latencies_df, _ = pchase(
    #         mem="l2",
    #         start_size_bytes=size_bytes,
    #         end_size_bytes=size_bytes,
    #         step_size_bytes=1,
    #         stride_bytes=128,
    #         warmup=0,
    #         iter_size=iter_size,
    #         repetitions=repetitions,
    #         sim=sim,
    #     )
    #     long_miss_latencies = long_miss_latencies_df["latency"].to_numpy()
    #     latencies.append(long_miss_latencies)

    return np.hstack(latencies)


def compute_hits(df, sim, gpu=None, force_misses=True):
    latencies = df["latency"].to_numpy()
    fit_latencies = latencies.copy()
    combined_latencies = latencies.copy()

    if force_misses:
        fit_latencies = collect_full_latency_distribution(sim=sim, gpu=gpu)
        combined_latencies = np.hstack([latencies, fit_latencies])

    latencies = np.abs(latencies)
    latencies = latencies.reshape(-1, 1)

    fit_latencies = np.abs(fit_latencies)
    fit_latencies = fit_latencies.reshape(-1, 1)

    combined_latencies = np.abs(combined_latencies)
    combined_latencies = combined_latencies.reshape(-1, 1)

    if force_misses:
        pred_hist_df = get_latency_distribution(latencies)
        print("=== LATENCY HISTOGRAM (prediction)")
        print(pred_hist_df[["bin", "count"]])
        print("")

        fit_hist_df = get_latency_distribution(fit_latencies)
        print("=== LATENCY HISTOGRAM (fitting)")
        print(fit_hist_df[["bin", "count"]])
        print("")

    combined_hist_df = get_latency_distribution(combined_latencies)
    print("=== LATENCY HISTOGRAM (combined)")
    print(combined_hist_df[["bin", "count"]])
    print("")

    # clustering_bins = fit_hist[bins[:-1] <= 1000.0]
    # print(clustering_bins)

    # find the top 3 bins and infer the bounds for outlier detection
    # hist_percent = hist / np.sum(hist)
    # valid_latency_bins = hist[hist_percent > 0.5]
    # latency_cutoff = np.min(valid_latency_bins)

    print(
        "BEFORE: mean={:4.2f} min={:4.2f} max={:4.2f} #nans={}".format(
            fit_latencies.mean(), fit_latencies.min(), fit_latencies.max(),
            np.count_nonzero(np.isnan(fit_latencies))
        )
    )

    # latency_abs_z_score = np.abs(latencies - np.median(fit_latencies))
    # outliers = latency_abs_z_score > 1000
    outliers = fit_latencies > 1000

    num_outliers = outliers.sum()
    print(num_outliers)
    print(len(df))
    print(
        "found {} outliers ({:1.4}%)".format(
            num_outliers,
            float(num_outliers) / float(len(df)),
        )
    )
    fit_latencies[outliers] = np.amax(fit_latencies[~outliers])
    print(
        "AFTER: mean={:4.2f} min={:4.2f} max={:4.2f}".format(
            fit_latencies.mean(), fit_latencies.min(), fit_latencies.max()
        )
    )

    df["hit_cluster"], latency_centroids = predict_is_hit(latencies, fit=fit_latencies)

    print("latency_centroids = {}".format(latency_centroids))
    return df

# def run_remote_pchase(cmd, gpu, executable=, force=False) -> typing.Tuple[str, str]:
#     das6 = remote.DAS6()
#     try:
#         job_name = "-".join([prefix, str(gpu)] + cmd)
#         remote_stdout_path = das6.remote_pchase_results_dir / "{}.stdout".format(job_name)
#         remote_stderr_path = das6.remote_pchase_results_dir / "{}.stderr".format(job_name)
#
#         # check if job already running
#         running_job_names = das6.get_running_job_names()
#         if not force and job_name in running_job_names:
#             raise ValueError("slurm job <{}> is already running".format(job_name))
#
#         # check if results already exists
#         if force or not das6.file_exists(remote_stdout_path):
#             job_id, _, _ = das6.submit_pchase(gpu=gpu, name=job_name, args=cmd)
#             print("submitted job <{}> [ID={}]".format(job_name, job_id))
#
#             das6.wait_for_job(job_id)
#
#         # copy stdout and stderr
#         stdout = das6.read_file_contents(remote_path=remote_stdout_path).decode("utf-8")
#         stderr = das6.read_file_contents(remote_path=remote_stderr_path).decode("utf-8")
#     except Exception as e:
#         das6.close()
#         raise e
#     return stdout, stderr


def pchase(
    gpu,
    mem,
    stride_bytes,
    warmup,
    start_size_bytes,
    end_size_bytes,
    step_size_bytes,
    max_rounds=None,
    repetitions=None,
    iter_size=None,
    sim=False,
    force=False,
):
    warmup = max(0, int(warmup))
    repetitions = max(1, int(repetitions if repetitions is not None else 1))

    cmd = [
        str(mem.lower()),
        str(int(start_size_bytes)),
        str(int(end_size_bytes)),
        str(int(step_size_bytes)),
    ]
    cmd += [
        str(int(stride_bytes)),
        str(warmup),
        str(repetitions),
    ]

    derived_iter_size = None
    round_size = float(end_size_bytes) / float(stride_bytes)

    # custom limit to iter size
    if iter_size is not None:
        derived_iter_size = int(iter_size)
        cmd += [str(derived_iter_size)]
    elif max_rounds is not None:
        derived_iter_size = int(float(max_rounds) * float(round_size))
        cmd += ["R{}".format(max_rounds)]
    else:
        derived_iter_size = round_size * 4


    if gpu is None:
        # run locally
        executable = SIM_P_CHASE if sim else NATIVE_P_CHASE
        cmd = " ".join([str(executable.absolute())] + cmd)

        unit = 1 * MIN if sim else 1 * SEC
        per_size_timeout = [
            ((derived_iter_size * (1 + warmup)) / 1000) * unit
            for _ in range(start_size_bytes, end_size_bytes + 1, step_size_bytes)
        ]
        timeout_sec = repetitions * sum(per_size_timeout)
        timeout_sec = max(5, 2 * timeout_sec)

        print("[timeout {: >5.1f} sec]\t{}".format(timeout_sec, cmd))

        try:
            _, stdout, stderr, _ = cmd_utils.run_cmd(
                cmd,
                timeout_sec=int(timeout_sec),
            )
        except cmd_utils.ExecStatusError as e:
            print(e.stderr)
            raise e
    else:
        das6 = remote.DAS6()
        try:
            stdout, stderr = das6.run_pchase_sync(
                cmd, gpu=gpu,
                executable=das6.remote_scratch_dir / "pchase",
                force=force)
        #     job_name = "-".join(["pchase", str(gpu)] + cmd)
        #     remote_stdout_path = das6.remote_pchase_results_dir / "{}.stdout".format(job_name)
        #     remote_stderr_path = das6.remote_pchase_results_dir / "{}.stderr".format(job_name)
        #
        #     # check if job already running
        #     running_job_names = das6.get_running_job_names()
        #     if not force and job_name in running_job_names:
        #         raise ValueError("slurm job <{}> is already running".format(job_name))
        #
        #     # check if results already exists
        #     if force or not das6.file_exists(remote_stdout_path):
        #         job_id, _, _ = das6.submit_pchase(gpu=gpu, name=job_name, args=cmd)
        #         print("submitted job <{}> [ID={}]".format(job_name, job_id))
        #
        #         das6.wait_for_job(job_id)
        #
        #     # copy stdout and stderr
        #     stdout = das6.read_file_contents(remote_path=remote_stdout_path).decode("utf-8")
        #     stderr = das6.read_file_contents(remote_path=remote_stderr_path).decode("utf-8")
        except Exception as e:
            das6.close()
            raise e
            
    stdout_reader = StringIO(stdout)
    df = pd.read_csv(
        stdout_reader,
        header=0,
        dtype=float,
    )
    return df, (stdout, stderr)


def set_mapping(
    gpu,
    mem,
    stride_bytes,
    warmup,
    size_bytes,
    repetitions=1,
    max_rounds=None,
    iter_size=None,
    sim=False,
    force=False,
):
    cmd = [
        str(mem.lower()),
        str(int(size_bytes)),
    ]
    cmd += [
        str(int(stride_bytes)),
        str(int(warmup)),
    ]

    # repetitions
    cmd += [str(int(repetitions))]

    # derived iter size
    derived_iter_size = None

    # custom limit to iter size
    if iter_size is not None:
        derived_iter_size = int(iter_size)
        cmd += [str(derived_iter_size)]
    elif max_rounds is not None:
        round_size = size_bytes / stride_bytes
        derived_iter_size = int(float(max_rounds) * float(round_size))
        cmd += [str(derived_iter_size)]

    if gpu is None:
        # run locally
        executable = SIM_SET_MAPPING if sim else NATIVE_SET_MAPPING
        cmd = " ".join([str(executable.absolute())] + cmd)
        print(cmd)

        timeout_sec = repetitions * (20 * MIN if sim else 10 * SEC)
        _, stdout, stderr, _ = cmd_utils.run_cmd(
            cmd,
            timeout_sec=int(timeout_sec),
        )
    else:
        # connect to remote gpu
        das6 = remote.DAS6()
        try:
            stdout, stderr = das6.run_pchase_sync(
                cmd, gpu=gpu,
                executable=das6.remote_scratch_dir / "set_mapping",
                force=force)
            # job_name = "-".join(["set_mapping", str(gpu)] + cmd)
            # remote_stdout_path = das6.remote_pchase_results_dir / "{}.stdout".format(job_name)
            # remote_stderr_path = das6.remote_pchase_results_dir / "{}.stderr".format(job_name)
            #
            # # check if job already running
            # running_job_names = das6.get_running_job_names()
            # if not force and job_name in running_job_names:
            #     raise ValueError("slurm job <{}> is already running".format(job_name))
            #
            # # check if results already exists
            # if force or not das6.file_exists(remote_stdout_path):
            #     job_id, _, _ = das6.submit_pchase(
            #         gpu=gpu, name=job_name,
            #         executable=das6.remote_scratch_dir / "set_mapping",
            #         args=cmd)
            #     print("submitted job <{}> [ID={}]".format(job_name, job_id))
            #
            #     das6.wait_for_job(job_id)
            #
            # # copy stdout and stderr
            # stdout = das6.read_file_contents(remote_path=remote_stdout_path).decode("utf-8")
            # stderr = das6.read_file_contents(remote_path=remote_stderr_path).decode("utf-8")
        except Exception as e:
            das6.close()
            raise e


    stdout_reader = StringIO(stdout)
    df = pd.read_csv(
        stdout_reader,
        header=0,
        dtype=float,
    )
    return df, (stdout, stderr)


def get_known_cache_size_bytes(mem: str, gpu=None) -> int:
    match (gpu, mem.lower()):
        case (None, "l1data"):
            return 24 * KB
        case (None, "l2"):
            return 2 * MB
        case ("A4000", "l1data"):
            return 58 * KB
        case ("A4000", "l2"):
            return 4 * MB
    raise ValueError("unknown num sets for {}".format(mem))


def get_known_cache_line_bytes(mem: str, gpu=None) -> int:
    match (gpu, mem.lower()):
        case (None, "l1data"):
            return 128
        case (None, "l2"):
            return 32
        case ("A4000", "l1data"):
            return 128
        case ("A4000", "l2"):
            return 32
    raise ValueError("unknown num sets for {}".format(mem))


def get_known_cache_num_sets(mem: str, gpu=None) -> int:
    match (gpu, mem.lower()):
        case (None, "l1data"):
            return 4
        case (None, "l2"):
            return 4
        case ("A4000", "l1data"):
            return 4
        case ("A4000", "l2"):
            return 4
    raise ValueError("unknown num sets for {}".format(mem))


@main.command()
@click.option("--warmup", "warmup", type=int, help="cache warmup")
@click.option("--repetitions", "repetitions", type=int, help="repetitions")
@click.option("--mem", "mem", default="l1data", help="memory to microbenchmark")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def find_l2_prefetch_size(warmup, repetitions, mem, cached, sim, gpu, force):
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 5))
    warmup = warmup or 0

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    known_l2_cache_size_bytes = 2 * MB
    step_size_bytes = 64 * KB
    start_cache_size_bytes = step_size_bytes
    end_cache_size_bytes = 2 * MB
    stride_bytes = 128

    cache_file = get_cache_file(prefix="l2_prefetch_size", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        combined, (_, stderr) = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_cache_size_bytes,
            end_size_bytes=end_cache_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            repetitions=repetitions,
            max_rounds=1,
            warmup=warmup,
            sim=sim,
            force=force,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

    # combined = compute_hits(combined, sim=sim, gpu=gpu)
    # combined = compute_rounds(combined)

    # print(combined)
    # # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index
    #
    # combined = combined[combined["round"].isin(full_rounds)]

    for n, df in combined.groupby("n"):
        # print(n)
        # print(df)

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
            "size {:>15} ({:>5.1f}%)".format(
                human_size,
                float(n) / float(known_l2_cache_size_bytes) * 100.0,
                # color(num_hits, fg="green", bold=True),
                # color("{:>3.1f}%".format(hit_rate * 100.0), fg="green"),
                # color(num_misses, fg="red", bold=True),
                # color("{:>3.1f}%".format(miss_rate * 100.0), fg="red"),
            )
        )

        for round, round_df in df.groupby("round", dropna=False):
            # count hits and misses
            l1_hits = round_df["hit_cluster"] == 0
            l2_hits = round_df["hit_cluster"] == 1
            misses = round_df["hit_cluster"] > 1  # l1 miss & l2 miss

            num_l1_hits = l1_hits.sum()
            num_l2_hits = l2_hits.sum()
            num_misses = misses.sum()

            human_size = humanize.naturalsize(n, binary=True)
            miss_rate = float(num_misses) / float(len(round_df))
            l1_hit_rate = float(num_l1_hits) / float(len(round_df))
            l2_hit_rate = float(num_l2_hits) / float(len(round_df))
            print(
                "round={:>4} L1 hits={} ({}) L2 hits={} ({}) misses={} ({})".format(
                    str(round),
                    color("{:<6}".format(num_l1_hits), fg="green", bold=True),
                    color("{: >5.1f}%".format(l1_hit_rate * 100.0), fg="green"),
                    color("{:<6}".format(num_l2_hits), fg="green", bold=True),
                    color("{: >5.1f}%".format(l2_hit_rate * 100.0), fg="green"),
                    color("{:<6}".format(num_misses), fg="red", bold=True),
                    color("{: >5.1f}%".format(miss_rate * 100.0), fg="red"),
                )
            )

    plot_df = (
        combined.groupby("n")
        .agg(
            {
                "hit_cluster": [agg_l1_hit_rate, agg_l2_hit_rate],
            }
        )
        .reset_index()
    )
    plot_df.columns = [
        "_".join([col for col in cols if col != ""]) for cols in plot_df.columns
    ]
    print(plot_df)
    print(plot_df.columns)

    ylabel = r"hit rate ($\%$)"
    xlabel = r"$N$ (bytes)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()

    min_x = round_down_to_multiple_of(plot_df["n"].min(), 128)
    max_x = round_up_to_multiple_of(plot_df["n"].max(), 128)

    ax.axvline(
        x=0.25 * known_l2_cache_size_bytes,
        color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
        linestyle="--",
        label=r"25% L2 size",
    )

    marker_size = 10
    ax.scatter(
        plot_df["n"],
        plot_df["hit_cluster_agg_l1_hit_rate"] * 100.0,
        marker_size,
        # [3] * len(plot_df["n"]),
        # linewidth=1.5,
        # linestyle='--',
        marker="o",
        # markersize=marker_size,
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        # color=plt_rgba(*SIM_RGB_COLOR["gpucachesim" if sim else "native"], 1.0),
        # label="gpucachesim" if sim else "GTX 1080",
        label="L1",
    )

    ax.scatter(
        plot_df["n"],
        plot_df["hit_cluster_agg_l2_hit_rate"] * 100.0,
        marker_size,
        # [3] * len(plot_df["n"]),
        # linewidth=1.5,
        # linestyle='--',
        marker="x",
        # markersize=marker_size,
        color=plot.plt_rgba(*plot.RGB_COLOR["blue1"], 1.0),
        # color=plt_rgba(*SIM_RGB_COLOR["gpucachesim" if sim else "native"], 1.0),
        # label="gpucachesim" if sim else "GTX 1080",
        label="L2",
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    xticks = np.arange(min_x, max_x, step=256 * KB)
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    ax.set_xticks(xticks, xticklabels, rotation=45)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, 100.0)
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


def compute_rounds_old(df):
    df["round"] = np.nan
    if "r" not in df:
        df["r"] = 0

    for (n, r), _ in df.groupby(["n", "r"]):
        mask = (df["n"] == n) & (df["r"] == r)
        arr_indices = df.loc[mask, "index"].values

        start_index = df[mask].index.min()

        intra_round_start_index = 0

        round_start_index = df[mask].index.min()
        round_end_index = df[mask].index.max()

        intra_round_start_indices = df[mask].index[
            arr_indices == intra_round_start_index
        ]
        round_start_indices = np.hstack([round_start_index, intra_round_start_indices])
        # shift right and add end index
        round_end_indices = np.hstack([round_start_indices[1:], round_end_index])
        assert len(round_start_indices) == len(round_end_indices)

        for round, (start_idx, end_idx) in enumerate(
            zip(round_start_indices, round_end_indices)
        ):
            round_size = end_idx - start_idx
            assert round_size >= 0
            if round_size == 0:
                continue
            # print("n={: >7}, start_idx={: >7}, end_idx={: >7}, len={: <7} round={: <3}".format(n, start_idx, end_idx, end_idx - start_idx, round))
            df.loc[start_idx:end_idx, "round"] = round

    # print(df[["n", "set", "round"]].drop_duplicates())
    return df


def compute_cache_lines(df, cache_size_bytes, sector_size_bytes, cache_line_bytes):
    # df["cache_line"] = (df["index"] % cache_size_bytes) // sector_size_bytes
    # df["cache_line"] = df["index"] // sector_size_bytes
    # df["cache_line"] += 1

    # print(len(df["index"].unique()))
    # print(len(df["virt_addr"].unique()))
    assert len(df["index"].unique()) == len(df["virt_addr"].unique())
    # print(list(((df["virt_addr"] // sector_size_bytes) % 700).unique()))

    # df["cache_line"] = df["virt_addr"].astype(int) // sector_size_bytes
    # ns = list(df["n"].unique())
    # for n in ns:
    #     total_cache_lines = n / sector_size_bytes
    #     total_cache_lines = 768
    #     print("total cache lines for n={} is {}".format(n, total_cache_lines))
    #     df.loc[df["n"] == n, "cache_line"] = df.loc[df["n"] == n, "cache_line"] % int(total_cache_lines)

    # df["cache_line"] = df["virt_addr"] // sector_size_bytes
    if False:
        for (set_idx, n), _ in df.groupby(["set", "n"]):
            total_cache_lines = int(n / sector_size_bytes)
            print(total_cache_lines)

            virt_addr = df.loc[df["n"] == n, "virt_addr"] % cache_size_bytes
            df.loc[df["n"] == n, "cache_line"] = virt_addr // sector_size_bytes

            print(df.loc[df["n"] == n, "cache_line"].values)
            print((df.loc[df["n"] == n, "index"] // sector_size_bytes).values)
    if True:
        # THIS WORKS
        # df["cache_line"] = df["index"] // sector_size_bytes
        df["cache_line"] = df["index"] // cache_line_bytes

    if False:
        for (n, s), set_df in df.groupby(["n", "set"]):
            total_cache_lines = n / sector_size_bytes
            # total_cache_lines = cache_size_bytes / sector_size_bytes
            print(n, s)
            print(len(set_df))
            print(total_cache_lines)
            index = df[df["set"] == s].index
            print(index)

            # set_index = index - index[0] + len(set_df) + s * 4
            # set_index = index - index[0] + len(set_df) # - (s-1) * 2
            set_index = index - index[0] + len(set_df) + s * 32
            print(set_index)

            assert len(index) == len(set_index)
            cache_lines = set_index % total_cache_lines
            print(cache_lines)
            df.loc[df["set"] == s, "cache_line"] = cache_lines

        # sets = list(df[["n", "set"]].drop_duplicates())
        # sets = list(df[["n", "set"]].drop_duplicates())
        # for (n, s) in sets:
        #     total_cache_lines = n / sector_size_bytes
        #     # total_cache_lines = 768
        #     # print("total cache lines for n={} is {}".format(n, total_cache_lines))
        #     # df.loc[df["set"] == s, "cache_line"] -= s
        #     df.loc[df["set"] == s, "cache_line"] %= total_cache_lines
        # setdf.loc[df["n"] == n, "cache_line"] % int(total_cache_lines)

    #
    # assert len((df["index"] // sector_size_bytes).unique()) == len(df["cache_line"].unique())

    # df["cache_line"] = df["index"] // sector_size_bytes
    # df["cache_line"] += 1

    # assert known_cache_size_bytes == known_num_sets * derived_num_ways * known_cache_line_bytes
    # combined["cache_line"] = combined["index"] // known_cache_line_bytes # works but random
    # combined["cache_line"] = combined["index"] % (known_num_sets * derived_num_ways) # wrong
    # combined["cache_line"] = combined["index"] % (derived_num_ways * derived_cache_lines_per_set) # wrong
    # combined["cache_line"] = combined["index"] % (known_num_sets * derived_cache_lines_per_set) # wrong

    # correct?
    # combined["cache_line"] = (combined["index"] % known_cache_size_bytes) // sector_size
    # df["cache_line"] = (df["virt_addr"] % cache_size_bytes) // sector_size_bytes
    # df["cache_line"] = df["index"] // sector_size_bytes
    return df


def compute_rounds(df):
    df["round"] = np.nan
    if "r" not in df:
        df["r"] = 0

    for (n, r), per_size_df in df.groupby(["n", "r"]):
        # print(per_size_df)

        mask = (df["n"] == n) & (df["r"] == r)
        arr_indices = df.loc[mask, "index"].values

        start_index_value = arr_indices.min()
        # print(start_index_value)
        assert start_index_value == 0.0
        intra_round_start_indices = df[mask].index[arr_indices == start_index_value]
        # print(n, intra_round_start_indices)

        for round in range(len(intra_round_start_indices) - 1):
            start_idx = intra_round_start_indices[round]
            end_idx = intra_round_start_indices[round + 1]
            print(
                "n={: >7}, start_idx={: >7}, end_idx={: >7}, len={: <7} round={: <3}".format(
                    n, start_idx, end_idx, end_idx - start_idx, round
                )
            )
            df.loc[start_idx : end_idx - 1, "round"] = round

        # intra_round_start_index = 0
        #
        # round_start_index = df[mask].index.min()
        # round_end_index = df[mask].index.max()
        #
        # intra_round_start_indices = df[mask].index[arr_indices == intra_round_start_index]
        # round_start_indices = np.hstack([round_start_index, intra_round_start_indices])
        # # shift right and add end index
        # round_end_indices = np.hstack([round_start_indices[1:], round_end_index])
        # assert len(round_start_indices) == len(round_end_indices)
        #
        # for round, (start_idx, end_idx) in enumerate(zip(round_start_indices, round_end_indices)):
        #     round_size = end_idx - start_idx
        #     assert round_size >= 0
        #     if round_size == 0:
        #         continue
        #     # print("n={: >7}, start_idx={: >7}, end_idx={: >7}, len={: <7} round={: <3}".format(n, start_idx, end_idx, end_idx - start_idx, round))
        #     df.loc[start_idx:end_idx, "round"] = round

    return df


def compute_number_of_sets(combined):
    # use DBscan clustering to find the number of sets
    combined["is_miss"] = combined["hit_cluster"] != 0
    num_misses_per_n = combined.groupby("n")["is_miss"].sum().reset_index()
    num_misses_per_n = num_misses_per_n.rename(columns={"is_miss": "num_misses"})
    num_misses_per_n["set_cluster"] = compute_dbscan_clustering(
        num_misses_per_n["num_misses"]
    )

    misses_per_set = (
        num_misses_per_n.groupby(["set_cluster"])["num_misses"].mean().reset_index()
    )
    print(misses_per_set)

    set_clusters = misses_per_set["set_cluster"]
    set_clusters = set_clusters[set_clusters >= 0]
    num_clusters = len(set_clusters.unique())
    num_sets = num_clusters - 1
    print("DBSCAN clustering found ", color(f"{num_sets} sets", fg="blue", bold=True))
    return num_sets, misses_per_set


def compute_set_probability(df, col="set"):
    set_probability = df[col].value_counts().reset_index()
    set_probability["prob"] = set_probability["count"] / set_probability["count"].sum()
    return set_probability


def find_pattern(values, num_sets):
    patterns = []
    for pattern_start in range(0, num_sets):
        for pattern_length in range(1, len(values)):
            pattern = itertools.cycle(
                values[pattern_start : pattern_start + pattern_length]
            )
            pattern = list(itertools.islice(pattern, len(values) * 2))
            if values + values == pattern and pattern_start + len(pattern) < len(
                values
            ):
                patterns.append((pattern_start, pattern))

    return patterns


def equal_expressions(a, b):
    a = sym.logic.boolalg.to_cnf(a, simplify=True, force=True)
    b = sym.logic.boolalg.to_cnf(b, simplify=True, force=True)
    return a == b


def pyeda_minimize(f):
    (minimized,) = pyeda.boolalg.minimization.espresso_exprs(f.to_dnf())
    return minimized


def sympy_to_pyeda(f):
    return pyeda.boolalg.expr.expr(str(f))


def print_cnf_terms(cnf):
    terms = sorted(
        [
            sorted(
                [str(var) for var in term.args],
                key=lambda var: int(str(var).removeprefix("~").removeprefix("b")),
            )
            for term in cnf.args
        ]
    )
    for term in terms:
        print(" | ".join(["{: >5}".format(t) for t in term]))


def contains_var(f, var) -> bool:
    if isinstance(f, (sym.Symbol, sym.Not)):
        if str(f).removeprefix("~") == str(var).removeprefix("~"):
            return True
        else:
            return False
    elif isinstance(f, (sym.And, sym.Or)):
        return any([contains_var(ff, var=var) for ff in f.args])
    else:
        raise TypeError("unknown type {}", type(f))


def remove_variable(f, var) -> typing.Optional[typing.Any]:
    if isinstance(f, (sym.Symbol, sym.Not)):
        if str(f).removeprefix("~") == str(var).removeprefix("~"):
            return None
        else:
            return f
    elif isinstance(f, (sym.And, sym.Or)):
        terms = [remove_variable(ff, var=var) for ff in f.args]
        terms = [t for t in terms if t is not None]
        # print(f.__class__, terms)
        return f.__class__(*terms, simplify=False)
    else:
        raise TypeError("unknown type {}", type(f))

def unique_bits(f) -> typing.Set[str]:
    if isinstance(f, (sym.Symbol, sym.Not)):
        return set([str(f).removeprefix("~")])
    elif isinstance(f, (sym.And, sym.Or)):
        return set(itertools.chain.from_iterable(unique_bits(ff) for ff in f.args))
    else:
        raise TypeError("unknown type {}", type(f))


def logicmin_dnf_to_sympy_cnf(dnf: str):
    # add brackes to DNF terms
    # dnf = "b0'.b1'.b2'.b3'.b4'.b5'.b6'.b7'.b8'.b9'.b10'.b11'.b12'.b13'.b14' + b0'.b1'.b2'.b3'.b4.b5.b6'.b7'.b8'.b9'.b10'.b11'.b12'.b13'.b14'.b15'"
    # print(dnf)
    dnf = re.sub(r"\+?([^+\s]+)\+?", r"(\1)", dnf)
    # print(dnf)
    # remove spaces
    dnf = re.sub(r"\s", r"", dnf)
    # print(dnf)
    # convert negations from x' to ~x
    negations = r"([^.'+()\s]+)'"
    dnf = re.sub(negations, r"~\1", dnf)
    # print(dnf)
    # convert <or> from + to |
    dnf = re.sub(r"\+", r" | ", dnf)
    # print(dnf)
    # convert <and> from . to &
    dnf = re.sub(r"\.", r" & ", dnf)
    # print(dnf)
    set_mapping_function = sym.parsing.sympy_parser.parse_expr(dnf)
    # print(set_mapping_function)
    assert sym.logic.boolalg.is_dnf(set_mapping_function)
    set_mapping_function_cnf = sym.logic.simplify_logic(set_mapping_function)
    # print(set_mapping_function)
    # print(len(set_mapping_function.args))
    # print(len(set_mapping_function.args[0].args))
    return set_mapping_function_cnf


def split_at_indices(s, indices):
    return [s[i:j] for i, j in zip(indices, indices[1:] + [None])]


@main.command()
@click.option(
    "--mem", "mem", default="l1data", type=str, help="memory to microbenchmark"
)
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def find_cache_set_mapping(mem, gpu, warmup, repetitions, cached, sim, force):
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 5))
    warmup = warmup if warmup is not None else (1 if sim else 2)

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    known_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    known_cache_line_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)
    known_num_sets = get_known_cache_num_sets(mem=mem, gpu=gpu)

    derived_num_ways = known_cache_size_bytes // (
        known_cache_line_bytes * known_num_sets
    )
    print("num ways = {:<3}".format(derived_num_ways))

    assert (
        known_cache_size_bytes
        == known_num_sets * derived_num_ways * known_cache_line_bytes
    )

    stride_bytes = known_cache_line_bytes

    match mem.lower():
        case "l1readonly":
            stride_bytes = known_cache_line_bytes
            pass

    cache_file = get_cache_file(prefix="cache_set_mapping", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        combined, (_, stderr) = set_mapping(
            mem=mem,
            gpu=gpu,
            size_bytes=known_cache_size_bytes,
            stride_bytes=stride_bytes,
            warmup=warmup,
            repetitions=repetitions,
            sim=sim,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "overflow_index", "index", "virt_addr"])
            .median()
            .reset_index()
        )

        combined = compute_hits(combined, sim=sim, gpu=gpu)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

    total_sets = OrderedDict()
    combined = combined.sort_values(["n", "overflow_index", "k"])

    print(combined["virt_addr"].value_counts().value_counts())

    # compute misses per overflow index
    for overflow_index, _ in combined.groupby("overflow_index"):
        misses = combined["overflow_index"] == overflow_index
        misses &= combined["hit_cluster"] > 0
        # print(combined.loc[misses,:].head(n=2*derived_num_ways))

        if misses.sum() != derived_num_ways:
            print(
                color(
                    "found {:<2} ways (expected {})".format(
                        misses.sum(), derived_num_ways
                    ),
                    fg="red",
                )
            )

        total_sets[
            tuple(combined.loc[misses, "virt_addr"].astype(int).to_numpy())
        ] = True

    total_sets = list(total_sets.keys())
    num_sets = len(total_sets)
    num_sets_log2 = int(np.log2(num_sets))
    print(
        color(
            "total sets={:<2} ({:<2} bits)".format(num_sets, num_sets_log2),
            fg="green" if num_sets == known_num_sets else "red",
        )
    )

    total_sets = sorted([list(s) for s in total_sets])
    set_addresses = np.array(total_sets)
    base_addr = np.amin(total_sets)
    print("base addr={}".format(base_addr))
    print(set_addresses.shape)
    print(set_addresses[:, 0:2])
    assert set_addresses.shape == (num_sets, derived_num_ways)
    offsets = np.argmin(set_addresses, axis=0)
    print(offsets)
    assert offsets.shape == (derived_num_ways,)

    def check_duplicates(needle):
        count = 0
        for s in total_sets:
            for addr in s:
                if addr - base_addr == needle:
                    count += 1
            if count > 1:
                break
        if count > 1:
            return str(color(str(needle), fg="red"))
        return str(needle)

    for set_id, s in enumerate(total_sets):
        print(
            "set {: <4}\t [{}]".format(
                set_id, ", ".join([check_duplicates(addr - base_addr) for addr in s])
            )
        )
        combined.loc[combined["virt_addr"].astype(int).isin(s), "set"] = set_id

    combined = combined[["virt_addr", "set"]].astype(int).drop_duplicates()
    combined = combined.sort_values("virt_addr")

    assert len(combined) == num_sets * derived_num_ways
    combined["set_offset"] = 0
    for way_id in range(derived_num_ways):
        way = combined.index[way_id * num_sets : (way_id + 1) * num_sets]
        sets = combined.loc[way, "set"].to_numpy()
        combined.loc[way, "offset"] = sets[0]

    print(combined.head(n=10))
    print(combined.shape)

    print(compute_set_probability(combined))

    # return

    max_bits = 64
    num_bits = 64
    line_size_log2 = int(np.log2(known_cache_line_bytes))

    def get_set_bit_mapping(bit: int):
        return [ 
            (
                int(addr),
                bitarray.util.int2ba(
                    int(set_id), length=num_sets_log2, endian="little"
                )[bit],
            )
            for addr, set_id in combined[["virt_addr", "set"]].to_numpy()
        ]

    def get_offset_bit_mapping(bit: int):
        return [
            (
                int(addr),
                bitarray.util.int2ba(
                    int(offset), length=num_sets_log2, endian="little"
                )[bit],
            )
            for addr, offset in combined[["virt_addr", "offset"]].to_numpy()
        ]


    found_mapping_functions = dict()
    for set_bit in range(num_sets_log2):
        bit_mapping = get_set_bit_mapping(bit=set_bit)
        bit_mapping = get_offset_bit_mapping(bit=set_bit)

        for min_bits in range(1, num_bits):
            bits_used = [line_size_log2 + bit for bit in range(min_bits)]

            print("testing bits {:<30}".format(str(bits_used)))
            t = logicmin.TT(min_bits, 1)
            validation_table = []

            # set_boundary_addresses = range(0, known_cache_size_bytes, known_cache_line_bytes)
            # assert len(set_boundary_addresses) == len(set_bits)
            # for index, offset in zip(set_boundary_addresses, set_bits):
            for addr, target_bit in bit_mapping:
                index_bits = bitarray.util.int2ba(
                    addr, length=max_bits, endian="little"
                )
                new_index_bits = bitarray.bitarray(
                    [index_bits[b] for b in reversed(bits_used)]
                )
                new_index = bitarray.util.ba2int(new_index_bits)
                new_index_bits_str = np.binary_repr(new_index, width=min_bits)
                t.add(new_index_bits_str, str(target_bit))
                validation_table.append((index_bits, target_bit))

            sols = t.solve()
            dnf = str(
                sols[0].expr(xnames=[f"b{b}" for b in reversed(bits_used)], syntax=None)
            )
            set_mapping_function = logicmin_dnf_to_sympy_cnf(dnf)

            # validate set mapping function
            valid = True
            for index_bits, offset in validation_table:
                vars = {
                    sym.symbols(f"b{b}"): index_bits[b] for b in reversed(bits_used)
                }
                predicted = set_mapping_function.subs(vars)
                predicted = int(bool(predicted))
                if predicted != offset:
                    valid = False

            if valid:
                print(
                    color(
                        "found valid set mapping function for bit {:<2}: {}".format(
                            set_bit, set_mapping_function
                        ),
                        fg="green",
                    )
                )
                found_mapping_functions[set_bit] = set_mapping_function
                break

    no_set_mapping_functions = [
        set_bit
        for set_bit in range(num_sets_log2)
        if found_mapping_functions.get(set_bit) is None
    ]

    if len(no_set_mapping_functions) > 0:
        for set_bit in no_set_mapping_functions:
            print(
                color(
                    "no minimal set mapping function found for set bit {:<2}".format(
                        set_bit
                    ),
                    fg="red",
                )
            )
        return


    for set_bit, f in found_mapping_functions.items():
        # if set_bit == 0:
        #     continue
        print("==== SET BIT {:<2}".format(set_bit))

        bit_mapping = get_offset_bit_mapping(bit=set_bit)
        bit_pos_used = sorted([int(str(bit).removeprefix("~").removeprefix("b")) for bit in unique_bits(f)])
        bits_used = [sym.symbols(f"b{pos}") for pos in bit_pos_used]

        # bits_used = sorted(
        #     [sym.symbols(bit) for bit in unique_bits(f)],
        #     key=lambda b: int(str(b).removeprefix("~").removeprefix("b")))

        print("unique bits: {}".format(bits_used))
        print_cnf_terms(f)

        f = sym.logic.boolalg.to_dnf(f, simplify=True, force=True)
        print("original", f)

        optimized = True
        while optimized:
            optimized = False
            for bit in bits_used:
                if str(bit) != "b10":
                    continue
                ff = remove_variable(f, var=bit)
                print("remove {:<5} => {}".format(str(bit), ff))

                assert isinstance(ff, sym.Or)
                assert len(ff.args) == len(f.args)

                substitutions = {
                    # "xor_bit": sym.logic.boolalg.Or(*[term for term in ff.args]),
                    # "xor_bit": sym.logic.boolalg.Or(*[sym.logic.boolalg.And(term, sym.symbols(f"b{bit}")) for term in ff.args]),
                    "xor_bit": sym.logic.boolalg.Or(*[
                        sym.logic.boolalg.Xor(term, bit) if contains_var(f.args[i], var=bit) else term
                        for i, term in enumerate(ff.args)
                    ]),
                    # "xor_bit": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, bit) for term in ff.args]),
                    # "xor_not_bit": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit) for term in ff.args]),
                    # "xor_bit_twice": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, bit, bit) for term in ff.args]),
                    # "xor_not_bit_twice": sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit, ~bit) for term in ff.args]),
                }
                # substitutions = {
                #     # "remove_bit": ff,
                #     "xor_bit": sym.logic.boolalg.Xor(ff, bit),
                #     # "xor_not_bit": sym.logic.boolalg.Xor(ff, ~bit),
                #     # "xor_bit_twice": sym.logic.boolalg.Xor(ff, bit, bit),
                #     # "xor_not_bit_twice": sym.logic.boolalg.Xor(ff, ~bit, ~bit),
                # }
                for sub_name, sub_f in substitutions.items():
                    print(sub_f)
                    valid = True
                    for addr, target_bit in bit_mapping:
                        index_bits = bitarray.util.int2ba(
                            addr, length=64, endian="little"
                        )
                        vars = {
                            bit: index_bits[pos] for bit, pos in reversed(list(zip(bits_used, bit_pos_used)))
                        }
                        ref_pred = int(bool(f.subs(vars)))
                        sub_pred = int(bool(sub_f.subs(vars)))
                        assert ref_pred == target_bit
                        print(
                            np.binary_repr(addr, width=64),
                            vars,
                            color(sub_pred, fg="red" if sub_pred != target_bit else "green"),
                            target_bit,
                        )
                        if sub_pred != target_bit:
                            valid = False

                    is_valid_under_simplification = equal_expressions(f, sub_f)
                    assert valid == is_valid_under_simplification
                    print(sym.logic.boolalg.to_dnf(sub_f, simplify=True, force=True))
                    print(sub_name, valid)

                    if valid:
                        f = sub_f
                        optimized = True
                        break

                print("current function {}".format(f))

            # print("final function {}".format(f))
            break

            # ff_xor_bit = 
            # print(ff_xor_bit)
            # print(sym.logic.boolalg.to_dnf(ff_xor_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_bit))
            #
            # ff_xor_not_bit = sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit) for term in ff.args])
            # print(sym.logic.boolalg.to_dnf(ff_xor_not_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_not_bit))
            #
            # ff_xor_xor_double_bit = sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, bit, bit) for term in ff.args])
            # print(sym.logic.boolalg.to_dnf(ff_xor_xor_double_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_xor_double_bit))
            #
            # ff_xor_xor_double_not_bit = sym.logic.boolalg.Or(*[sym.logic.boolalg.Xor(term, ~bit, ~bit) for term in ff.args])
            # print(sym.logic.boolalg.to_dnf(ff_xor_xor_double_not_bit, simplify=True, force=True))
            # print(equal_expressions(f, ff_xor_xor_double_not_bit))

    return

    for set_bit in range(1, num_sets_log2):
        bit_mapping = [
            (
                int(addr),
                bitarray.util.int2ba(
                    int(set_id), length=num_sets_log2, endian="little"
                )[set_bit],
            )
            for addr, set_id in combined[["virt_addr", "set"]].to_numpy()
        ]
        is_offset = True
        bit_mapping = [
            (
                int(addr),
                bitarray.util.int2ba(
                    int(offset), length=num_sets_log2, endian="little"
                )[set_bit],
            )
            for addr, offset in combined[["virt_addr", "offset"]].to_numpy()
        ]

        print("==== SET BIT {:<2}".format(set_bit))
        for addr, target_bit in bit_mapping:
            full_index_bits = bitarray.util.int2ba(
                addr, length=max_bits, endian="little"
            )

            # even xor => 0
            # uneven xor => 1

            vars = {sym.symbols(f"b{b}"): full_index_bits[b] for b in range(max_bits)}
            predicted = found_mapping_functions[set_bit].subs(vars)
            predicted = int(bool(predicted))

            marks = set([])
            # marks = set([3584, 7680, 11776, 19968, 24064])
            # marks = set([1536])

            # CORRECT OFFSET BIT 0
            bit0 = bool(full_index_bits[9])
            bit0 = bit0 ^ bool(full_index_bits[9])
            bit0 = bit0 ^ bool(full_index_bits[10])

            bit0 = bit0 ^ bool(full_index_bits[12])
            bit0 = bit0 ^ bool(full_index_bits[14])
            # if bool(full_index_bits[9]) ^ bool(full_index_bits[10]):
            #     # predicted = not predicted
            #     pass

            if is_offset and set_bit == 0:
                predicted = bit0

            if is_offset and set_bit == 1:
                # CORRECT OFFSET BIT 1
                predicted = bool(full_index_bits[9])
                predicted = predicted ^ (not bit0)
                predicted = predicted ^ bool(full_index_bits[10])
                predicted = predicted ^ bool(full_index_bits[11])
                predicted = predicted ^ bool(full_index_bits[12])
                predicted = predicted ^ bool(full_index_bits[13])
                predicted = predicted ^ bool(full_index_bits[14])

            if not is_offset:
                # PREDICTED 0
                predicted0 = bool(full_index_bits[7])
                # predicted = predicted ^ bool(full_index_bits[9])
                # predicted = predicted ^ (not bool(full_index_bits[9]))
                predicted0 = predicted0 ^ bool(full_index_bits[10])
                # predicted = predicted ^ bool(full_index_bits[11])
                predicted0 = predicted0 ^ bool(full_index_bits[12])
                # predicted = predicted ^ bool(full_index_bits[13])
                predicted0 = predicted0 ^ bool(full_index_bits[14])

                if set_bit == 0:
                    predicted = predicted0

                if set_bit == 1:
                    # this is for the offset only:
                    if False:
                        predicted = bool(full_index_bits[9]) ^ bool(full_index_bits[11])
                        predicted = predicted ^ bool(full_index_bits[13])
                        if False:
                            if full_index_bits[9] & full_index_bits[10]:
                                predicted = bool(not predicted)
                                # print([full_index_bits[b] for b in [11, 12, 13, 14]])
                                # lol = bool(full_index_bits[11]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13]) ^ bool(full_index_bits[14]) ^ bool(full_index_bits[15])
                                # print(predicted, lol)
                                # predicted = predicted ^ lol
                            if (
                                full_index_bits[9]
                                & full_index_bits[10]
                                & full_index_bits[11]
                                & full_index_bits[12]
                            ):
                                predicted = bool(not predicted)

                        if (
                            full_index_bits[9]
                            & full_index_bits[10]
                            & ~(full_index_bits[11] & full_index_bits[12])
                        ):
                            predicted = bool(not predicted)

                    # predicted = bool(full_index_bits[7])
                    predicted = bool(full_index_bits[8])
                    # predicted = predicted ^ bool(full_index_bits[8])
                    # predicted = predicted ^ bool(full_index_bits[7])
                    predicted = predicted ^ (not bool(full_index_bits[7]))

                    predicted = predicted ^ bool(full_index_bits[9])
                    predicted = predicted ^ bool(full_index_bits[10])
                    predicted = predicted ^ bool(full_index_bits[11])
                    # predicted = predicted ^ (not bool(full_index_bits[12]))
                    predicted = predicted ^ bool(full_index_bits[12])
                    predicted = predicted ^ bool(full_index_bits[13])
                    predicted = predicted ^ bool(full_index_bits[14])

                    predicted = not predicted

                    inverter = False

                    section1 = full_index_bits[9:11] == bitarray.bitarray(
                        "11", endian="big"
                    ) or full_index_bits[9:11] == bitarray.bitarray("01", endian="big")
                    section2 = full_index_bits[11:13] != bitarray.bitarray(
                        "11", endian="big"
                    )
                    # section2 = (
                    #     full_index_bits[11:13] == bitarray.bitarray("11", endian="big")
                    #     or full_index_bits[11:13] == bitarray.bitarray("01", endian="big")
                    # )

                    # if (
                    #     full_index_bits[9:11] == bitarray.bitarray("11", endian="big")
                    #     or full_index_bits[9:11] == bitarray.bitarray("01", endian="big")
                    # ):
                    # if section1 and section2:
                    t1 = full_index_bits[10:12] != bitarray.bitarray("01", endian="big")
                    t2 = full_index_bits[10:12] != bitarray.bitarray("11", endian="big")
                    t3 = full_index_bits[10:12] != bitarray.bitarray("00", endian="big")
                    t4 = full_index_bits[10:12] != bitarray.bitarray("10", endian="big")
                    # print(full_index_bits[10:12])

                    # 000
                    # 010
                    # 101
                    # 111
                    t1 = full_index_bits[10:13] == bitarray.bitarray(
                        "000", endian="big"
                    )
                    t2 = full_index_bits[10:13] == bitarray.bitarray(
                        "010", endian="big"
                    )
                    t3 = full_index_bits[10:13] == bitarray.bitarray(
                        "101", endian="big"
                    )
                    t4 = full_index_bits[10:13] == bitarray.bitarray(
                        "111", endian="big"
                    )
                    print(full_index_bits[10:13])

                    t5 = full_index_bits[13:15] != bitarray.bitarray("11", endian="big")
                    # 11
                    # 00
                    # 01
                    if not (t1 | t2 | t3 | t4):
                        # predicted = predicted ^ bool(full_index_bits[7])
                        inverter = True

                    offset = bool(full_index_bits[9]) ^ bool(full_index_bits[11])
                    offset = offset ^ bool(full_index_bits[13])
                    # if full_index_bits[9] & full_index_bits[10] & ~(full_index_bits[11] & full_index_bits[12]):
                    #     predicted = bool(not predicted)

                    # if (bool(full_index_bits[9]) & bool(full_index_bits[10])) | ((not bool(full_index_bits[9])) & bool(full_index_bits[10])):
                    # 9 & 10 & ~(11 & 12)
                    # if bool(full_index_bits[7]):
                    #     predicted = not predicted

                    # mask = bool(full_index_bits[11]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13]) ^ bool(full_index_bits[14]) ^ bool(full_index_bits[7])
                    # if mask:
                    #     print(mask)

                    # sector1 = (not bool(full_index_bits[10])) & (not bool(full_index_bits[11]))
                    # sector2 = bool(full_index_bits[11]) & bool(full_index_bits[12])

                    # have:
                    # 00000
                    # 00010
                    # 00101
                    # 00111
                    # 01000
                    # 01010
                    # 01101
                    # 01111
                    # 10001
                    # 10011
                    # 10100
                    # 10110

                    # 000
                    # 010
                    # 101
                    # 111

                    # 000
                    # 010
                    # 101
                    # 111

                    # 001
                    # 011
                    # 100
                    # 110

                    # not 001
                    # not 011
                    # not 100
                    # not 110

                    # not 000
                    # not 010
                    # not 101
                    # not 111

                    # not 11---

                    # 11000
                    # 11000
                    # 10111
                    # 10111
                    # 10101
                    # 10101
                    # 10010
                    # 10010
                    # 10000
                    # 10000
                    # 01110
                    # 01110
                    # 01100
                    # 01100
                    # 01011
                    # 01011
                    # 00110
                    # 00110
                    # 00100
                    # 00100
                    # 00011
                    # 00011
                    # 00001
                    # 00001

                    # 110000
                    # 110001
                    # 101110
                    # 101111
                    # 101010
                    # 101011
                    # 100100
                    # 100101
                    # 100000
                    # 100001
                    # 011100
                    # 011101
                    # 011000
                    # 011001
                    # 010110
                    # 010111
                    # 001100
                    # 001101
                    # 001000
                    # 001001
                    # 000110
                    # 000111
                    # 000010
                    # 000011
                    # if (not bool(full_index_bits[10])) & (not bool(full_index_bits[13])):
                    #     pass
                    # if sector1 | sector2:
                    # if sector1:
                    #     predicted = not predicted

                    if False:
                        predicted = predicted ^ bool(full_index_bits[10])
                        # predicted = predicted ^ bool(full_index_bits[10])
                        predicted = predicted ^ bool(full_index_bits[11])
                        predicted = predicted ^ bool(full_index_bits[12])
                        predicted = predicted ^ bool(full_index_bits[13])
                        predicted = predicted ^ bool(full_index_bits[14])
                        # predicted = predicted ^ bool(full_index_bits[15])
                        # predicted = predicted ^ bool(full_index_bits[11])

                    # special = bool(full_index_bits[11]) ^ bool(full_index_bits[13])
                    # if special:
                    #     # predicted = bool(~predicted)
                    #     pass
                    # ^ full_index_bits[14]
                    # predicted |= full_index_bits[10] ^ full_index_bits[12] ^ full_index_bits[14]

                    # predicted = bool(not predicted)
                    # predicted &= ~(full_index_bits[9] & full_index_bits[10])
                    # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                    # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                    # predicted &= special_case
                    # special_case2 = ~(full_index_bits[10] & full_index_bits[11])
                    # predicted |= full_index_bits[10] ^ full_index_bits[11]

            print(
                "{}\t\t{} => {:>2} {:>2} {} \t bit0={:<1} inverter={:<1}".format(
                    addr,
                    "|".join(
                        split_at_indices(
                            np.binary_repr(
                                bitarray.util.ba2int(full_index_bits), width=num_bits
                            ),
                            indices=[
                                0,
                                num_bits - 15,
                                num_bits - line_size_log2 - num_sets_log2,
                                num_bits - line_size_log2,
                            ],
                        )
                    ),
                    target_bit,
                    str(
                        color(
                            int(predicted),
                            fg="green"
                            if bool(predicted) == bool(target_bit)
                            else "red",
                        )
                    ),
                    str(color("<==", fg="blue")) if addr in marks else "",
                    str(0),
                    str(0),
                    # predicted0,
                    # str(color(str(int(inverter)), fg="cyan")) if inverter else str(int(inverter)),
                )
            )

    return

    offset_bit_0 = [int(np.binary_repr(o, width=2)[0]) for o in offsets]
    offset_bit_1 = [int(np.binary_repr(o, width=2)[1]) for o in offsets]

    if False:
        print(offset_bit_0)
        print(offset_bit_1)

        for name, values in [
            ("offset bit 0", offset_bit_0),
            ("offset bit 1", offset_bit_1),
        ]:
            patterns = find_pattern(values=values, num_sets=num_sets)
            if len(patterns) < 0:
                print("NO pattern found for {:<10}".format(name))
            for pattern_start, pattern in patterns:
                print(
                    "found pattern for {:<10} (start={: <2} length={: <4}): {}".format(
                        name, pattern_start, len(pattern), pattern
                    )
                )

        print(
            len(
                list(
                    range(0, known_cache_size_bytes, num_sets * known_cache_line_bytes)
                )
            )
        )
        print(len(offsets))
        assert len(
            list(range(0, known_cache_size_bytes, num_sets * known_cache_line_bytes))
        ) == len(offsets)

    if True:
        found_mapping_functions = dict()
        for set_bit in range(num_sets_log2):
            offset_bits = [
                bitarray.util.int2ba(int(o), length=num_sets_log2, endian="little")[
                    set_bit
                ]
                for o in offsets
            ]
            for min_bits in range(1, num_bits):
                bits_used = [
                    line_size_log2 + num_sets_log2 + bit for bit in range(min_bits)
                ]

                print("testing bits {:<30}".format(str(bits_used)))
                t = logicmin.TT(min_bits, 1)
                validation_table = []

                way_boundary_addresses = range(
                    0, known_cache_size_bytes, num_sets * known_cache_line_bytes
                )
                for index, offset in zip(way_boundary_addresses, offset_bits):
                    index_bits = bitarray.util.int2ba(
                        index, length=max_bits, endian="little"
                    )
                    new_index_bits = bitarray.bitarray(
                        [index_bits[b] for b in reversed(bits_used)]
                    )
                    new_index = bitarray.util.ba2int(new_index_bits)
                    new_index_bits_str = np.binary_repr(new_index, width=min_bits)
                    t.add(new_index_bits_str, str(offset))
                    validation_table.append((index_bits, offset))

                sols = t.solve()
                # dnf = sols.printN(xnames=[f"b{b}" for b in range(num_bits)], ynames=['offset'], syntax=None)
                # dnf = sols[0].printSol("offset",xnames=[f"b{b}" for b in range(num_bits)],syntax=None)
                dnf = str(
                    sols[0].expr(
                        xnames=[f"b{b}" for b in reversed(bits_used)], syntax=None
                    )
                )
                set_mapping_function = logicmin_dnf_to_sympy_cnf(dnf)
                print(set_mapping_function)

                # validate set mapping function
                valid = True
                for index_bits, offset in validation_table:
                    vars = {
                        sym.symbols(f"b{b}"): index_bits[b] for b in reversed(bits_used)
                    }
                    predicted = set_mapping_function.subs(vars)
                    predicted = int(bool(predicted))
                    if predicted != offset:
                        valid = False

                if valid:
                    # set_mapping_function = sym.logic.boolalg.to_dnf(set_mapping_function, simplify=True, force=True)
                    print(
                        color(
                            "found valid set mapping function for bit {:<2}: {}".format(
                                set_bit, set_mapping_function
                            ),
                            fg="green",
                        )
                    )
                    found_mapping_functions[set_bit] = set_mapping_function
                    break

            if found_mapping_functions.get(set_bit) is None:
                print(
                    color(
                        "no minimal set mapping function found for set bit {:<2}".format(
                            set_bit
                        ),
                        fg="red",
                    )
                )

        assert (
            str(found_mapping_functions[0])
            == "(~b13 | ~b14) & (b10 | b12 | b14 | b9) & (b10 | b12 | ~b14 | ~b9) & (b10 | b14 | ~b12 | ~b9) & (b10 | b9 | ~b12 | ~b14) & (b12 | b9 | ~b10 | ~b14) & (b14 | b9 | ~b10 | ~b12) & (b12 | ~b11 | ~b14 | ~b9) & (b11 | b12 | b14 | ~b10 | ~b9) & (b13 | b14 | ~b11 | ~b12 | ~b9) & (b11 | ~b10 | ~b12 | ~b14 | ~b9)"
        )
        assert (
            str(found_mapping_functions[1])
            == "(b10 & b13 & ~b11) | (b11 & b12 & b13 & b9) | (b11 & ~b13 & ~b9) | (b13 & ~b11 & ~b9) | (b11 & b13 & b9 & ~b10) | (b10 & b11 & ~b12 & ~b13) | (b9 & ~b10 & ~b11 & ~b13)"
        )

        found_mapping_functions_pyeda = {
            k: sympy_to_pyeda(v) for k, v in found_mapping_functions.items()
        }

        if False:
            for set_bit, f in found_mapping_functions_pyeda.items():
                minimized = pyeda_minimize(f)
                # (minimized,) = pyeda.boolalg.minimization.espresso_exprs(f.to_dnf())
                print(
                    "minimized function for set bit {:<2}: {}".format(
                        set_bit, minimized
                    )
                )

        for set_bit, f in found_mapping_functions.items():
            print("==== SET BIT {:<2}".format(set_bit))
            print_cnf_terms(f)

        for xor_expr in ["a ^ b", "~(a ^ b)"]:
            print(
                "\t{:>20}  =>  {}".format(
                    str(xor_expr),
                    str(
                        sym.logic.boolalg.to_cnf(
                            sym.parsing.sympy_parser.parse_expr(xor_expr),
                            simplify=True,
                            force=True,
                        )
                    ),
                )
            )

        if False:
            print("==== SET BIT 2 SIMPLIFIED")
            simplified1 = sym.logic.boolalg.to_cnf(
                sym.parsing.sympy_parser.parse_expr(
                    (
                        "(b10 & b13 & ~b11)"
                        "| (b11 & b12 & b13 & b9)"
                        "| (b11 ^ b13 ^ b9)"
                        "| (b11 & ~b13 & ~b9)"
                        "| (b13 & ~b11 & ~b9)"
                        "| (b10 & b11 & ~b12 & ~b13)"
                        "| (b9 & ~b10)"
                        # "| (b11 & b13 & b9 & ~b10)"
                        # "| (b9 & ~b10 & ~b11 & ~b13)"
                        # "| ((b9 | ~b10) & (b11 ^ b13))"
                    )
                ),
                simplify=True,
                force=True,
            )
            print_cnf_terms(simplified1)
            assert equal_expressions(simplified1, found_mapping_functions[1])

    # t = logicmin.TT(num_bits, 1);
    # way_boundary_addresses = range(0, known_cache_size_bytes, num_sets * known_cache_line_bytes)
    # for index, offset in zip(way_boundary_addresses, offset_bit_1):
    #     index_bits_str = np.binary_repr(index, width=num_bits)
    #     # print(index_bits_str, offset)
    #     t.add(index_bits_str, str(offset))
    #
    # sols = t.solve()
    # # print(sols.printInfo("test"))
    # # dnf = sols.printN(xnames=[f"b{b}" for b in range(num_bits)], ynames=['offset'], syntax=None)
    # # dnf = sols[0].printSol("offset",xnames=[f"b{b}" for b in range(num_bits)],syntax=None)
    # dnf = str(sols[0].expr(xnames=[f"b{b}" for b in reversed(range(num_bits))], syntax=None))
    # set_mapping_function = logicmin_dnf_to_sympy_cnf(dnf)
    # print(set_mapping_function)

    # to cnf never completes, try usign pyeda minimization
    # set_mapping_function = sym.logic.boolalg.to_cnf(set_mapping_function, simplify=True, force=True)
    # minimized = pyeda_minimize(sympy_to_pyeda(set_mapping_function))
    # print(minimized)

    for set_bit in range(num_sets_log2):
        way_boundary_addresses = range(
            0, known_cache_size_bytes, num_sets * known_cache_line_bytes
        )
        offset_bits = [
            bitarray.util.int2ba(int(o), length=num_sets_log2, endian="little")[set_bit]
            for o in offsets
        ]
        print("==== SET BIT {:<2}".format(set_bit))
        for index, offset in zip(way_boundary_addresses, offset_bits):
            full_index_bits = bitarray.util.int2ba(
                index, length=max_bits, endian="little"
            )

            # index_bits = bitarray.util.int2ba(index >> line_size_log2, length=num_bits, endian="little")
            # even xor => 0
            # uneven xor => 1

            vars = {sym.symbols(f"b{b}"): full_index_bits[b] for b in range(max_bits)}
            predicted = found_mapping_functions[set_bit].subs(vars)
            predicted = int(bool(predicted))

            # predicted = index_bits[13] + (index_bits[12] ^ index_bits[11])
            # predicted = index_bits[10] ^ index_bits[9]
            # predicted |= index_bits[8]
            # predicted = index_bits[11] ^ (index_bits[3] ^ index_bits[2])
            marks = set([3584, 7680, 11776, 19968, 24064])
            marks = set([1536])

            if False and set_bit == 0:
                predicted = (
                    index_bits[5] ^ (index_bits[4] ^ index_bits[3]) ^ index_bits[2]
                )
                # predicted = index_bits[11] ^ predicted
                if True:
                    predicted = (index_bits[4] + predicted) % 2
                    predicted = (index_bits[7] + predicted) % 2
                # ( + (index_bits[10] ^ index_bits[9])) % 2

            if set_bit == 1:
                # predicted = bool(full_index_bits[9]) ^ bool(full_index_bits[10])
                # predicted = predicted ^ bool(full_index_bits[11])
                # predicted = predicted ^ bool(full_index_bits[13])

                predicted = bool(full_index_bits[9]) ^ bool(full_index_bits[11])
                predicted = predicted ^ bool(full_index_bits[13])
                # predicted = predicted ^ bool(full_index_bits[14])
                # predicted = predicted ^ bool(full_index_bits[11])
                # predicted = predicted ^ bool(full_index_bits[13])
                # predicted |= full_index_bits[9] & full_index_bits[11])
                # predicted |= full_index_bits[10]
                # special = bool(full_index_bits[9]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13])
                if False:
                    if full_index_bits[9] & full_index_bits[10]:
                        predicted = bool(not predicted)
                        # print([full_index_bits[b] for b in [11, 12, 13, 14]])
                        # lol = bool(full_index_bits[11]) ^ bool(full_index_bits[12]) ^ bool(full_index_bits[13]) ^ bool(full_index_bits[14]) ^ bool(full_index_bits[15])
                        # print(predicted, lol)
                        # predicted = predicted ^ lol
                    if (
                        full_index_bits[9]
                        & full_index_bits[10]
                        & full_index_bits[11]
                        & full_index_bits[12]
                    ):
                        predicted = bool(not predicted)

                if (
                    full_index_bits[9]
                    & full_index_bits[10]
                    & ~(full_index_bits[11] & full_index_bits[12])
                ):
                    predicted = bool(not predicted)

                # special = bool(full_index_bits[11]) ^ bool(full_index_bits[13])
                # if special:
                #     # predicted = bool(~predicted)
                #     pass
                # ^ full_index_bits[14]
                # predicted |= full_index_bits[10] ^ full_index_bits[12] ^ full_index_bits[14]

                # predicted = bool(not predicted)
                # predicted &= ~(full_index_bits[9] & full_index_bits[10])
                # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                # special_case = ~full_index_bits[10] | full_index_bits[11] # | full_index_bits[13]
                # predicted &= special_case
                # special_case2 = ~(full_index_bits[10] & full_index_bits[11])
                # predicted |= full_index_bits[10] ^ full_index_bits[11]

            print(
                "{}\t\t{} => {:>2} {:>2} {}".format(
                    index,
                    np.binary_repr(
                        bitarray.util.ba2int(full_index_bits), width=num_bits
                    ),
                    offset,
                    str(
                        color(
                            int(predicted),
                            fg="green" if bool(predicted) == bool(offset) else "red",
                        )
                    ),
                    str(color("<==", fg="blue")) if index in marks else "",
                )
            )

    return

    offsets_df = pd.DataFrame(offsets, columns=["offset"])
    print(compute_set_probability(offsets_df, "offset"))

    offset_mapping_table = combined[["virt_addr", "set"]].copy()
    offset_mapping_table = offset_mapping_table.drop_duplicates()
    print(len(offset_mapping_table), num_sets * len(offsets_df))

    assert len(offset_mapping_table) == num_sets * len(offsets_df)
    for i in range(len(offset_mapping_table)):
        set_id = i % num_sets
        way_id = i // num_sets
        # derived_num_ways
        # print(i, set_id, way_id)
        # print(sets[:,way_id])
        set_offsets = np.argsort(sets[:, way_id])
        # print(set_offsets)
        # offsets_df
        offset_mapping_table.loc[offset_mapping_table.index[i], "set"] = int(
            set_offsets[set_id]
        )

    offset_mapping_table = offset_mapping_table.astype(int)

    # print(offset_mapping_table)
    # print("===")
    # print([int(np.binary_repr(int(s), width=2)[0]) for s in offset_mapping_table["set"]])
    # print("===")
    # print([int(np.binary_repr(int(s), width=2)[1]) for s in offset_mapping_table["set"]])

    # for way_id in range(len(offsets_df) // num_sets):
    #     way_offset = offsets_df["offset"][way_id]
    #     # print(way_id*num_sets, (way_id+1)*num_sets)
    #     rows = offset_mapping_table.index[way_id*num_sets:(way_id+1)*num_sets]
    #     offset_mapping_table.loc[rows, "set"] = way_offset

    def build_set_mapping_table(df, addr_col="virt_addr", num_sets=None, offset=None):
        set_mapping_table = df.copy()
        if offset is not None and num_sets is not None:
            set_mapping_table["set"] = (set_mapping_table["set"] + int(offset)) % int(
                num_sets
            )

        set_mapping_table = set_mapping_table[[addr_col, "set"]].astype(int)
        set_mapping_table = set_mapping_table.rename(columns={addr_col: "addr"})
        set_mapping_table = set_mapping_table.drop_duplicates()
        return set_mapping_table

    if True:
        set_mapping_table = build_set_mapping_table(offset_mapping_table)
    if False:
        # compute possible set id mappings
        # for offset in range(num_sets):
        set_mapping_table = build_set_mapping_table(combined)

    compute_set_probability(set_mapping_table)

    num_bits = 64
    print(color(f"SOLVE FOR <AND> MAPPING [bits={num_bits}]", fg="cyan"))
    sols = solve_mapping_table(set_mapping_table, use_and=True, num_bits=num_bits)
    print(color(f"SOLVE FOR <OR> MAPPING [bits={num_bits}]", fg="cyan"))
    sols = solve_mapping_table(set_mapping_table, use_and=False, num_bits=num_bits)

    # for offset in range(num_sets):
    # set_mapping_table = build_set_mapping_table(combined, num_sets=num_sets, offset=offset)

    num_bits = 64
    for degree in range(2, 4):
        print(
            color(
                f"SOLVE FOR <XOR> MAPPING [degree={degree}, bits={num_bits}]", fg="cyan"
            )
        )
        sols = solve_mapping_table_xor(
            set_mapping_table, num_bits=num_bits, degree=degree
        )
    # print(sols)

    # remove incomplete rounds
    # combined = combined[~combined["round"].isna()]

    # combined = compute_cache_lines(
    #     combined,
    #     cache_size_bytes=known_cache_size_bytes,
    #     sector_size_bytes=sector_size_bytes,
    #     cache_line_bytes=known_cache_line_bytes,
    #     )


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option(
    "--mem", "mem", default="l1data", type=str, help="memory to microbenchmark"
)
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="warmup iterations")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def find_cache_replacement_policy(mem, gpu, repetitions, warmup, cached, sim, force):
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
    repetitions = max(1, repetitions or (1 if sim else 4))
    warmup = warmup or (1 if sim else 5)

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    known_cache_size_bytes = 24 * KB
    known_cache_line_bytes = 128
    sector_size_bytes = 32
    known_num_sets = 4

    # 48 ways
    # terminology: num ways == cache lines per set == associativity
    # terminology: way size == num sets

    stride_bytes = known_cache_line_bytes
    # stride_bytes = sector_size_bytes

    # derived_total_cache_lines = known_cache_size_bytes / known_cache_line_bytes
    derived_total_cache_lines = known_cache_size_bytes / stride_bytes
    derived_cache_lines_per_set = int(derived_total_cache_lines // known_num_sets)

    # 768 cache lines
    print("expected cache lines = {:<3}".format(derived_total_cache_lines))

    derived_num_ways = known_cache_size_bytes // (
        known_cache_line_bytes * known_num_sets
    )
    print("num ways = {:<3}".format(derived_num_ways))

    assert (
        known_cache_size_bytes
        == known_num_sets * derived_num_ways * known_cache_line_bytes
    )

    match mem.lower():
        case "l1readonly":
            stride_bytes = known_cache_line_bytes
            pass

    cache_file = get_cache_file(prefix="cache_replacement_policy", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        # combined = []
        # for repetition in range(repetitions):
        #     for set_idx in range(1, known_num_sets + 1):
        #         # overflow by mulitples of the cache line
        #         n = known_cache_size_bytes + set_idx * known_cache_line_bytes
        #         # n = known_cache_size_bytes + set_idx * derived_cache_lines_per_set
        #         # n = known_cache_size_bytes + set_idx * derived_cache_lines_per_set * known_cache_line_bytes
        #         # n = known_cache_size_bytes + set_idx * derived_cache_lines_per_set * 32
        #         # n = known_cache_size_bytes + set_idx * derived_num_ways * known_cache_line_bytes
        #         # n = known_cache_size_bytes + set_idx * derived_num_ways
        #         df, (_, stderr) = pchase(executable, mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=2)
        #         print(stderr)
        #         df["n"] = n
        #         df["r"] = repetition
        #         df["set"] = set_idx
        #         combined.append(df)
        #
        step_size_bytes = known_cache_line_bytes
        # step_size_bytes = 32
        start_size_bytes = known_cache_size_bytes + 1 * step_size_bytes
        end_size_bytes = known_cache_size_bytes + known_num_sets * step_size_bytes
        combined, (_, stderr) = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_size_bytes,
            end_size_bytes=end_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            repetitions=repetitions,
            warmup=warmup,
            sim=sim,
            force=force,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        )

        combined["set"] = (combined["n"] % known_cache_size_bytes) // step_size_bytes

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        # for (set_idx, n), set_df in combined.groupby(["set", "n"]):
        #     print(set_idx)
        #     print(list(set_df["index"][:20]))

        if False:
            for (set_idx, n), set_df in combined.groupby(["set", "n"]):
                total_cache_lines = int(n / sector_size_bytes)
                # total_cache_lines = derived_total_cache_lines
                # combined.loc[combined["set"] == set_idx, "cache_line"] = range((combined["set"] == set_idx).sum())
                # combined.loc[combined["set"] == set_idx, "cache_line"] -= known_num_sets - set_idx - 1) * 1
                # print(n)
                # print(combined.loc[combined["set"] == set_idx, "index"][:10])
                # print(combined.loc[combined["set"] == set_idx, "index"].max())
                combined.loc[combined["set"] == set_idx, "index"] -= (set_idx) * 32 * 4
                # combined.loc[combined["set"] == set_idx, "cache_line"] += (set_idx * 4)
                combined.loc[combined["set"] == set_idx, "index"] %= n

        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

    # remove incomplete rounds
    combined = combined[~combined["round"].isna()]

    combined = compute_cache_lines(
        combined,
        cache_size_bytes=known_cache_size_bytes,
        sector_size_bytes=sector_size_bytes,
        cache_line_bytes=known_cache_line_bytes,
    )

    if False:
        for (set_idx, n), set_df in combined.groupby(["set", "n"]):
            total_cache_lines = int(n / sector_size_bytes)
            # total_cache_lines = derived_total_cache_lines
            # combined.loc[combined["set"] == set_idx, "cache_line"] = range((combined["set"] == set_idx).sum())
            # combined.loc[combined["set"] == set_idx, "cache_line"] -= known_num_sets - set_idx - 1) * 1
            combined.loc[combined["set"] == set_idx, "cache_line"] -= (
                (set_idx - 1) * known_num_sets * cache_line_size_bytes
            )
            # combined.loc[combined["set"] == set_idx, "cache_line"] += (set_idx * 4)
            combined.loc[combined["set"] == set_idx, "cache_line"] %= total_cache_lines

    for (set_idx, n), set_df in combined.groupby(["set", "n"]):
        print(set_idx)
        print(list(set_df["cache_line"][:20]))

    # print(combined[["n", "set"]].drop_duplicates())

    # print(combined.columns)
    # combined = combined.groupby(["n", "set", "latency", "index"]).mean().reset_index()

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

    num_actual_cache_lines = len(combined["cache_line"].unique())
    num_unique_indices = len(combined["index"].unique())
    num_unique_addresses = len(combined["virt_addr"].unique())
    print(
        "have {} cache lines (want {}) from {} indices or {} virt. addresses".format(
            num_actual_cache_lines,
            derived_total_cache_lines,
            num_unique_indices,
            num_unique_addresses,
        )
    )

    # we can have more too
    if False and mem != "l1readonly":
        print(len(combined["cache_line"].unique()))
        print(derived_total_cache_lines)
        assert len(combined["cache_line"].unique()) >= derived_total_cache_lines

    for (set_idx, n), set_df in combined.groupby(["set", "n"]):
        print(set_idx)
        print(list(set_df["cache_line"][:20]))
        # assert (set_df["cache_line"] == 1).sum() >= 2
        # assert (set_df["cache_line"] == 1).sum() == len(set_df["round"].unique())
        print(list(set_df.loc[set_df["round"] == 0, "cache_line"][:20]))
        print(list(set_df.loc[set_df["round"] == 0, "cache_line"][-20:]))
        # print(set_df.head(n=1000))
        # return

    # for round, _ in combined.groupby("round"):
    #     # round_mask = combined["round"] == round
    #     # lines = combined.loc[round_mask, "index"] // known_cache_line_bytes
    #     # combined.loc[round_mask, "cache_line"] =
    #     # combined.loc[round_mask, "new_index"] = np.arange(len(combined[round_mask])) * stride_bytes
    #     # combined.loc[round_mask, "cache_line"] = combined.loc[round_mask, "new_index"] // known_cache_line_bytes
    #     pass

    for n, df in combined.groupby("n"):
        print(
            "number of unique indices = {:<4}".format(
                len(df["index"].unique().tolist())
            )
        )
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

        # compute mean occurences per index
        mean_rounds = df["index"].value_counts().mean()
        assert mean_rounds >= 2.0

        print(
            "size={:<10} cache lines={:<3.1f} hits={} ({}) misses={} ({}) rounds={:3.3f}".format(
                human_size,
                float(n) / float(known_cache_line_bytes),
                color("{: <5}".format(num_hits), fg="green", bold=True),
                color(
                    "{: >6.2f}%".format(float(num_hits) / float(len(df)) * 100.0),
                    fg="green",
                    bold=True,
                ),
                color("{: <5}".format(num_misses), fg="red", bold=True),
                color(
                    "{: >6.2f}%".format(float(num_misses) / float(len(df)) * 100.0),
                    fg="red",
                    bold=True,
                ),
                mean_rounds,
            )
        )

    # check if pattern is periodic
    for (set_idx, n), df in combined.groupby(["set", "n"]):
        human_size = humanize.naturalsize(n, binary=True)
        miss_indices_per_round = []
        miss_cache_lines_per_round = []
        miss_rounds = []
        num_rounds = len(df["round"].unique())
        max_round = df["round"].max()
        print(
            "################ set={: <2} has {: >2} rounds".format(set_idx, num_rounds)
        )

        for round, round_df in df.groupby("round"):
            # if (round == 0 or round == max_round) and num_rounds > 2:
            #     # skip
            #     print(color("skip set={: <2} round={: <2}".format(set_idx, round), fg="info"))
            #     continue

            human_size = humanize.naturalsize(n, binary=True)
            misses = round_df[round_df["hit_cluster"] != 0]
            miss_indices = misses["index"].astype(int)
            miss_cache_lines = misses["cache_line"].astype(int)
            print(
                "set={: <2} round={: <2} has {: >4} misses".format(
                    set_idx, round, len(miss_cache_lines)
                )
            )

            # if False:
            #     print("num misses = {}".format(len(misses)))
            #     print("miss cache indices = {}..".format(miss_indices.tolist()[:25]))
            #     print("miss cache lines = {}..".format(miss_cache_lines.tolist()[:25]))
            #     print("max missed cache line = {}".format(miss_cache_lines.max()))
            #     print("mean missed cache line = {}".format(miss_cache_lines.mean()))
            #     print("min missed cache line = {}".format(miss_cache_lines.min()))

            miss_rounds.append(round)
            miss_indices_per_round.append(miss_indices.tolist())
            miss_cache_lines_per_round.append(miss_cache_lines.tolist())

        def is_periodic(
            patterns, strict=True, max_rel_err=0.05
        ) -> typing.Tuple[list, int]:
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

        strict_pattern_match, num_strict_matches = is_periodic(
            miss_patterns, strict=True
        )
        soft_pattern_match, num_soft_matches = is_periodic(miss_patterns, strict=False)
        assert num_soft_matches >= num_strict_matches
        valid_miss_count = len(strict_pattern_match)

        print(
            "set={: <2} best match has {: >4} misses".format(set_idx, valid_miss_count)
        )

        # filter out bad rounds
        # before = len(combined)
        # invalid = 0
        # for round, misses in zip(miss_rounds, miss_patterns):
        #     if sorted(misses) != sorted(strict_pattern_match):
        #     # if len(misses) != valid_miss_count:
        #         print(color("set={: <2} round={: <2} SKIP (too many misses)".format(set_idx, round), fg="warn"))
        #         combined = combined.loc[~((combined["set"] == set_idx) & (combined["round"] == round)),:]
        #         invalid += 1
        # after = len(combined)
        # if invalid > 0:
        #     assert before > after

        info = "[{: <2}/{: <2} strict match, {: <2}/{: <2} soft match, {: >4} unqiue miss lines]\n".format(
            num_strict_matches,
            len(miss_patterns),
            num_soft_matches,
            len(miss_patterns),
            len(
                combined.loc[
                    (combined["set"] == set_idx) & (combined["hit_cluster"] != 0),
                    "cache_line",
                ].unique()
            ),
        )

        # if num_strict_matches == len(miss_patterns):
        if (
            len(miss_patterns) > 1
            and float(num_strict_matches) / float(len(miss_patterns)) > 0.7
        ):
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
                info,
            )
            for round, miss_lines in enumerate(miss_patterns):
                print(
                    "========== set={: <2} round[{: <2}] ===========\t {: >5} missed lines:\n{}".format(
                        set_idx, round, len(miss_lines), miss_lines
                    )
                )

    return

    for (set_idx, n), set_df in combined.groupby(["set", "n"]):
        set_misses = set_df.loc[set_df["hit_cluster"] != 0, "cache_line"]
        print(
            "set={: <2} has {: <4} missed cache lines ({: >4} unique)".format(
                set_idx, len(set_misses), len(set_misses.unique())
            )
        )
        print(
            "set={: <2} first cache lines: {}".format(
                set_idx, list(set_df["cache_line"][:20])
            )
        )
        print(
            "set={: <2} last cache lines: {}".format(
                set_idx, list(set_df["cache_line"][-20:])
            )
        )

        # print("set={: <2} first cache lines: {}" .format(set_idx, list(set_df["cache_line"][:20])))
        # print("set={: <2} last cache lines: {}" .format(set_idx, list(set_df["cache_line"][-20:])))

    # round_sizes = combined.groupby(["n", "set"])["round"].value_counts()
    # min_round_size = round_sizes.min()
    # # print(round_sizes)
    print("derived total cache lines", derived_total_cache_lines)
    # print("min round", min_round_size)
    # # return

    # combined = combined[(combined["round"] == 0) & (combined["cache_line"] <= derived_total_cache_lines) & (combined["cache_line"] > 0)]
    combined = combined[combined["cache_line"] < int(derived_total_cache_lines)]
    combined = combined[combined["cache_line"] > 0]

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
        np.array(combined["cache_line"].unique()), columns=["cache_line"]
    )
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
        miss_cache_lines = (
            combined.loc[set_mask & miss_mask, "cache_line"].astype(int).unique()
        )
        print(
            "\n=== set={: <2}\t {: >4} miss lines {: >4} miss indices".format(
                set_idx, len(miss_cache_lines), len(miss_indices)
            )
        )
        # print("miss indices = {}".format(sorted(miss_indices)))
        print("miss lines   = {}".format(sorted(miss_cache_lines)))

        unique_miss_indices = combined.loc[set_mask & miss_mask, "index"].unique()
        # print("set {} unique miss indices {}".format(set_idx, len(unique_miss_indices)))
        if last_unique_miss_indices is not None:
            diff = set(unique_miss_indices).difference(set(last_unique_miss_indices))
            # print("OVERLAP: {}".format(len(diff)))

        unique_miss_cache_lines = combined.loc[
            set_mask & miss_mask, "cache_line"
        ].unique()
        # print("set {} unique miss cache lines {}".format(set_idx, len(unique_miss_cache_lines)))
        if last_unique_miss_cache_lines is not None:
            diff = set(unique_miss_cache_lines).difference(
                set(last_unique_miss_cache_lines)
            )
            # print("OVERLAP: {}".format(len(diff)))

        last_unique_miss_indices = unique_miss_indices.copy()
        last_unique_miss_cache_lines = unique_miss_cache_lines.copy()

        for miss_cache_line in unique_miss_cache_lines:
            combined.loc[
                combined["cache_line"] == miss_cache_line, "mapped_set"
            ] = set_idx

            cache_line_set_mapping.loc[
                cache_line_set_mapping["cache_line"] == miss_cache_line, "mapped_set"
            ] = set_idx

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

    # if False:
    #     miss_mask = combined["is_hit"] == False
    #     # combined["set4_miss"] = (combined["set"] == 4) & miss_mask
    #     # combined["set3_miss"] = (combined["set"] == 4) & miss_mask
    #     set4_miss_lines = combined.loc[(combined["set"] == 4) & miss_mask, "cache_line"].unique()
    #     set3_miss_lines = combined.loc[(combined["set"] == 3) & miss_mask, "cache_line"].unique()
    #     set2_miss_lines = combined.loc[(combined["set"] == 2) & miss_mask, "cache_line"].unique()
    #     set1_miss_lines = combined.loc[(combined["set"] == 1) & miss_mask, "cache_line"].unique()
    #     set4_3_diff = sorted(list(set(set4_miss_lines) - set(set3_miss_lines)))
    #     set3_2_diff = sorted(list(set(set3_miss_lines) - set(set2_miss_lines)))
    #     set2_1_diff = sorted(list(set(set2_miss_lines) - set(set1_miss_lines)))
    #     print("set 4 - 3", len(set4_3_diff))
    #     print("set 3 - 2", len(set3_2_diff))
    #     print("set 2 - 1", len(set2_1_diff))
    #
    #     print("set 4 - 3", set4_3_diff)
    #     print("set 3 - 2", set3_2_diff)
    #     print("set 2 - 1", set2_1_diff)

    # print(cache_line_set_mapping)
    set_probability = cache_line_set_mapping["mapped_set"].value_counts().reset_index()
    set_probability["prob"] = set_probability["count"] / set_probability["count"].sum()
    print(set_probability)
    print(
        "sum of cache lines per set = {: >4}/{: <4}".format(
            set_probability["count"].sum(), len(total_unique_cache_lines)
        )
    )
    assert set_probability["count"].sum() == len(total_unique_cache_lines)

    mapped_sets = combined.loc[~combined["is_hit"], "mapped_set"]
    unmapped_sets_percent = mapped_sets.isnull().sum() / float(len(mapped_sets))
    print(unmapped_sets_percent)
    assert unmapped_sets_percent <= 0.05

    line_table = pd.DataFrame(
        np.zeros(shape=(int(derived_cache_lines_per_set), int(known_num_sets))),
        columns=[f"set {set+1}" for set in range(known_num_sets)],
    )

    for mapped_set, set_df in cache_line_set_mapping.groupby("mapped_set"):
        cache_lines = set_df["cache_line"]
        # cache_lines //= known_cache_line_bytes / sector_size_bytes
        # cache_lines = cache_lines[cache_lines < derived_num_ways]

        # cache_lines = sorted(cache_lines.unique().tolist())
        print("=== {:<2} === [{: >4}]".format(int(mapped_set), len(cache_lines)))
        print(sorted(cache_lines.astype(int).unique().tolist()))
        # for line in cache_lines.astype(int):
        #     # way = ((line -1) // 4) % derived_num_ways
        #     way = line % derived_num_ways
        #     # print(way, int(mapped_set))
        #     # line_table.iloc[way, int(mapped_set) - 1] = line

        valid = min(len(cache_lines), derived_cache_lines_per_set)
        line_table.iloc[0:valid, int(mapped_set) - 1] = cache_lines.ravel()[:valid]

    print(line_table)

    if stride_bytes < known_cache_line_bytes:
        line_table = pd.DataFrame(
            np.zeros(shape=(int(derived_num_ways), int(known_num_sets))),
            columns=[f"set {set+1}" for set in range(known_num_sets)],
        )

        for mapped_set, set_df in cache_line_set_mapping.groupby("mapped_set"):
            cache_lines = (set_df["cache_line"][::4]) // 4
            # cache_lines = (set_df["cache_line"][::4] - 1) // 4

            # cache_lines = cache_lines.unique()
            # cache_lines //= known_cache_line_bytes / sector_size_bytes
            # cache_lines = cache_lines[cache_lines < derived_num_ways]

            # cache_lines = sorted(cache_lines.unique().tolist())
            print("=== {:<2} === [{: >4}]".format(int(mapped_set), len(cache_lines)))
            print(sorted(cache_lines.astype(int).unique().tolist()))
            valid = min(len(cache_lines), derived_num_ways)
            line_table.iloc[0:valid, int(mapped_set) - 1] = cache_lines.ravel()[:valid]

        def format_cache_line(line):
            return "{: >3}-{:<3}".format(int(line * 4), int((line + 1) * 4))

        print(line_table.map(format_cache_line))

    addr_col = "virt_addr"
    # make sure a single virt addr never maps to two different sets (otherwise the mapping is random)
    set_mapping = combined[[addr_col, "mapped_set"]].astype(int)
    # map sets 1,2,3,4 to 0,1,2,3 for binary encoding to work
    set_mapping["mapped_set"] -= 1
    # make all addresses cache line aligned
    set_mapping[addr_col] = (
        set_mapping[addr_col] // known_cache_line_bytes
    ) * known_cache_line_bytes
    set_mapping = set_mapping.drop_duplicates()
    set_mapping = set_mapping.rename(columns={addr_col: "addr", "mapped_set": "set"})
    set_mapping = set_mapping.sort_values("addr")
    set_mapping["bin_addr"] = set_mapping["addr"].apply(
        lambda addr: np.binary_repr(addr)
    )
    print(set_mapping.head(20))
    assert ((set_mapping["addr"] % known_cache_line_bytes) == 0).all()

    # print(set_mapping["addr"].unique())
    # print(set_mapping["set"].unique())
    # print(set_mapping["virt_addr"].value_counts())
    # print(set_mapping.sort_values("virt_addr"))

    line_size_log2 = int(np.log2(known_cache_line_bytes))
    expected = (set_mapping["addr"].to_numpy() >> line_size_log2) & (known_num_sets - 1)
    set_mapping["expected"] = expected
    # print(set_mapping.head(20))
    if sim:
        assert (set_mapping["expected"] == set_mapping["set"]).all()

    # print(set_mapping["addr"].value_counts())
    assert set_mapping["addr"].value_counts().max() <= 1
    # assert set_mapping["addr"].value_counts().max() <= known_cache_line_bytes / sector_size_bytes

    num_bits = 20
    print("SOLVE FOR AND MAPPING")
    sols = solve_mapping_table(
        set_mapping[["addr", "set"]], use_and=True, num_bits=num_bits
    )
    print("SOLVE FOR OR MAPPING")
    sols = solve_mapping_table(
        set_mapping[["addr", "set"]], use_and=False, num_bits=num_bits
    )
    print("SOLVE FOR XOR MAPPING")
    sols = solve_mapping_table_xor(set_mapping[["addr", "set"]], num_bits=num_bits)
    # print(sols)

    return

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
            set_probability["prob"] = (
                set_probability["count"] / set_probability["count"].sum()
            )
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

    # trace back the cache ways

    # for index in range(start_size_bytes / )
    # for
    # for set in range(known_num_sets):
    #     pri

    # print(combined.groupby("index")[["mapped_set"]])
    # print(combined.groupby("mapped_set")

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

        repetition_end_indices = np.hstack(
            [repetition_start_indices[1:], df.index.stop]
        )
        for repetition, (start_idx, end_idx) in enumerate(
            zip(repetition_start_indices, repetition_end_indices)
        ):
            print(
                "\n========== repetition[{:>2}] {:>4} to {:<4} ===========".format(
                    repetition, start_idx, end_idx
                )
            )
            df.iloc[start_idx:end_idx]["round"] = repetition
            repetition_df = df.iloc[start_idx:end_idx]

            misses = repetition_df[repetition_df["hit_cluster"] != 0]
            print("num misses = {}".format(len(misses)))
            # print("miss cache lines = {}".format(misses["cache_line"].tolist()))
            print(
                "unique miss cache indices = {}".format(
                    misses["index"].unique().tolist()
                )
            )
            print(
                "unique miss cache lines = {}".format(
                    misses["cache_line"].unique().tolist()
                )
            )

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


def agg_miss_rate(hit_clusters):
    cluster_counts = hit_clusters.value_counts().reset_index()
    num_misses = cluster_counts.loc[cluster_counts["hit_cluster"] != 0, "count"].sum()
    total = cluster_counts["count"].sum()
    return num_misses / total


def agg_l1_hit_rate(hit_clusters):
    cluster_counts = hit_clusters.value_counts().reset_index()
    num_misses = cluster_counts.loc[cluster_counts["hit_cluster"] == 0, "count"].sum()
    total = cluster_counts["count"].sum()
    return num_misses / total


def agg_l2_hit_rate(hit_clusters):
    cluster_counts = hit_clusters.value_counts().reset_index()
    num_misses = cluster_counts.loc[cluster_counts["hit_cluster"] == 1, "count"].sum()
    total = cluster_counts["count"].sum()
    return num_misses / total


@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option(
    "--mem", "mem", type=str, default="l1data", help="memory to microbenchmark"
)
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def find_cache_sets(mem, gpu, cached, sim, repetitions, warmup, force):
    """
    Determine number of cache sets T.

    We set s to b.
    We then start with N = C and increase N at the granularity of b.
    Every increment causes cache misses of a new cache set.
    When N > C + (T − 1)b, all cache sets are missed.
    We can then deduce T from cache miss patterns accordingly.
    """
    repetitions = max(1, repetitions or (1 if sim else 10))
    warmup = warmup or (1 if sim else 2)

    stride_bytes = 8
    step_bytes = 8

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    predicted_num_sets = 4

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
    end_cache_size_bytes = (
        known_cache_size_bytes + (1 + predicted_num_sets) * known_cache_line_bytes
    )

    # combined = []
    # for n in range(start_cache_size_bytes, end_cache_size_bytes, step_bytes):
    #     df, (_, stderr) = pchase(mem=mem, size_bytes=n, stride_bytes=stride_bytes, warmup=1, sim=sim)
    #     print(stderr)
    #     df["n"] = n
    #     combined.append(df)
    #
    # combined = pd.concat(combined, ignore_index=True)
    # combined = compute_hits(combined, gpu=gpu)

    cache_file = get_cache_file(prefix="cache_sets", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        combined, (_, stderr) = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_cache_size_bytes,
            end_size_bytes=end_cache_size_bytes,
            step_size_bytes=step_bytes,
            stride_bytes=stride_bytes,
            max_rounds=1,
            warmup=warmup,
            repetitions=repetitions,
            sim=sim,
            force=force,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

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
            "size={: <10} lsbs={: <4} hits={} ({}) l1 misses={} l2 misses={} (miss rate={}) l1 misses={}.. l2 misses= {}..".format(
                human_size,
                int(n % 128),
                color("{: <5}".format(num_hits), fg="green", bold=True),
                color("{: >3.2f}%".format(hit_rate), fg="green"),
                color("{: <5}".format(num_l1_misses), fg="red", bold=True),
                color("{: <5}".format(num_l2_misses), fg="red", bold=True),
                color("{: >3.2f}%".format(miss_rate), fg="red"),
                l1_misses[:10],
                l2_misses[:10],
            )
        )

        if n % KB == 0:
            print("==> {} KB".format(n / KB))
        if n % known_cache_line_bytes == 0:
            human_cache_line_bytes = humanize.naturalsize(
                known_cache_line_bytes, binary=True
            )
            print(
                "==> start of predicted cache line ({})".format(human_cache_line_bytes)
            )

    derived_num_sets, _misses_per_set = compute_number_of_sets(combined)

    plot_df = combined.groupby("n").agg({"hit_cluster": [agg_miss_rate]}).reset_index()
    plot_df.columns = [
        "_".join([col for col in cols if col != ""]) for cols in plot_df.columns
    ]

    ylabel = r"miss rate ($\%$)"
    xlabel = r"$N$ (bytes)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()

    min_x = round_down_to_multiple_of(plot_df["n"].min(), known_cache_line_bytes)
    max_x = round_up_to_multiple_of(plot_df["n"].max(), known_cache_line_bytes)
    print(min_x, max_x)

    cache_line_boundaries = np.arange(min_x, max_x, step=known_cache_line_bytes)

    for i, cache_line_boundary in enumerate(cache_line_boundaries):
        ax.axvline(
            x=cache_line_boundary,
            color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
            linestyle="--",
            label="cache line boundary" if i == 0 else None,
        )

    marker_size = 5
    ax.scatter(
        plot_df["n"],
        plot_df["hit_cluster_agg_miss_rate"] * 100.0,
        marker_size,
        # linewidth=1.5,
        # linestyle='--',
        marker="x",
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label="gpucachesim" if sim else "GTX 1080",
    )

    for set_idx in range(derived_num_sets):
        x = known_cache_size_bytes + (set_idx + 0.5) * known_cache_line_bytes
        set_mask = (
            (plot_df["n"] % known_cache_size_bytes) // known_cache_line_bytes
        ) == set_idx
        y = plot_df.loc[set_mask, "miss_rate"].mean() * 100

        label = r"$S_{{{}}}$".format(set_idx)
        plt.text(x, 0.9 * y, label, ha="center", va="top")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    xticks = np.arange(known_cache_size_bytes, max_x, step=256)
    # xticks = np.linspace(
    #         round_to_multiple_of(plot_df["n"].min(), KB),
    #         round_to_multiple_of(plot_df["n"].max(), KB), num=4)
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, min(plot_df["miss_rate"].max() * 2 * 100.0, 100.0))
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--start", "start_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_size_bytes", type=int, help="end cache size in bytes")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--rounds", "rounds", type=int, default=1, help="number of rounds")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def find_cache_line_size(mem, gpu, start_size_bytes, end_size_bytes, cached, sim, repetitions, warmup, rounds, force):
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
    print(mem, gpu)
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 20))
    warmup = warmup if warmup is not None else 1

    known_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    predicted_cache_line_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)
    print("known cache size: {} bytes ({})".format(known_cache_size_bytes, humanize.naturalsize(known_cache_size_bytes, binary=True)))
    print("predicted cache line size: {} bytes ({})".format(predicted_cache_line_bytes, humanize.naturalsize(predicted_cache_line_bytes, binary=True)))

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    stride_bytes = 8
    predicted_num_lines = get_known_cache_num_sets(mem=mem, gpu=gpu)
    print("predicted num lines: {}".format(predicted_num_lines))

    match mem.lower():
        case "l1readonly":
            # stride_bytes = 8
            pass

    if mem == "l2":
        step_size_bytes = 32
        # start_size_bytes = start_size_bytes or (known_cache_size_bytes - 1 * predicted_cache_line_bytes)
        # end_size_bytes = end_size_bytes or (
        #     known_cache_size_bytes + (1 + predicted_num_lines) * predicted_cache_line_bytes
        # )
    elif mem == "l1data":
        step_size_bytes = 8
    else:
        raise NotImplementedError("mem {} not yet supported".format(mem))

    step_size_bytes = 8
    start_size_bytes = known_cache_size_bytes - 3 * predicted_cache_line_bytes
    end_size_bytes = (
        known_cache_size_bytes + (1 + predicted_num_lines) * predicted_cache_line_bytes
    )

    assert start_size_bytes % stride_bytes == 0
    assert step_size_bytes % stride_bytes == 0

    print("range: {:>10} to {:<10} step size={}".format(
        humanize.naturalsize(start_size_bytes, binary=True),
        humanize.naturalsize(end_size_bytes, binary=True),
        step_size_bytes))

    cache_file = get_cache_file(prefix="cache_line_size", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        combined, (_, stderr) = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_size_bytes,
            end_size_bytes=end_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            warmup=warmup,
            max_rounds=rounds,
            repetitions=repetitions,
            sim=sim,
            force=force,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

    num_unique_indices = len(combined["index"].unique())

    # # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index
    # print("have {: >3} rounds (full round size is {: <5})".format(len(full_rounds), full_round_size))
    #
    # combined = combined[combined["round"].isin(full_rounds)]
    # # combined = combined[combined["round"].isin([0, 1])]

    for n, df in combined.groupby("n"):
        if n % KB == 0:
            print("==> {} KB".format(n / KB))

        if n % predicted_cache_line_bytes == 0:
            human_cache_line_bytes = humanize.naturalsize(
                predicted_cache_line_bytes, binary=True
            )
            print(
                "==> start of predicted cache line ({})".format(human_cache_line_bytes)
            )

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
            mean_hit_clusters = (
                per_round.groupby("hit_cluster")["count"].median().reset_index()
            )

            num_hits = int(
                mean_hit_clusters.loc[
                    mean_hit_clusters["hit_cluster"] == 0, "count"
                ].sum()
            )
            num_misses = int(
                mean_hit_clusters.loc[
                    mean_hit_clusters["hit_cluster"] != 0, "count"
                ].sum()
            )
            num_l1_misses = int(
                mean_hit_clusters.loc[
                    mean_hit_clusters["hit_cluster"] == 1, "count"
                ].sum()
            )
            num_l2_misses = int(
                mean_hit_clusters.loc[
                    mean_hit_clusters["hit_cluster"] == 2, "count"
                ].sum()
            )

            hit_rate = float(num_hits) / float(num_unique_indices) * 100.0
            miss_rate = float(num_misses) / float(num_unique_indices) * 100.0

        # extract miss patterns
        # miss_pattern = df.index[df["hit_cluster"] != 0].tolist()
        # miss_pattern1 = df.index[df["hit_cluster"] == 1].tolist()
        # miss_pattern2 = df.index[df["hit_cluster"] == 2].tolist()

        human_size = humanize.naturalsize(n, binary=True)
        print(
            # short miss pattern ={}.. long miss pattern = {}..".format(
            "size={: >10} lsbs={: <4} hits={} ({}) l1 misses={} l2 misses={} (miss rate={})".format(
                human_size,
                int(n % 128),
                color("{: <5}".format(num_hits), fg="green", bold=True),
                color("{: >3.2f}%".format(hit_rate), fg="green"),
                color("{: <5}".format(num_l1_misses), fg="red", bold=True),
                color("{: <5}".format(num_l2_misses), fg="red", bold=True),
                color("{: >3.2f}%".format(miss_rate), fg="red"),
                # miss_pattern1[:6],
                # miss_pattern2[:6],
            )
        )

    def agg_miss_rate(hit_clusters):
        cluster_counts = hit_clusters.value_counts().reset_index()
        num_misses = cluster_counts.loc[
            cluster_counts["hit_cluster"] != 0, "count"
        ].sum()
        total = cluster_counts["count"].sum()
        return num_misses / total

    plot_df = combined.groupby("n").agg({"hit_cluster": agg_miss_rate}).reset_index()
    plot_df = plot_df.rename(columns={"hit_cluster": "miss_rate"})
    # print(plot_df)

    ylabel = r"miss rate ($\%$)"
    xlabel = r"$N$ (bytes)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()

    min_kb = round_down_to_multiple_of(plot_df["n"].min(), predicted_cache_line_bytes)
    max_kb = round_up_to_multiple_of(plot_df["n"].max(), predicted_cache_line_bytes)
    print(min_kb, max_kb)

    cache_line_boundaries = np.arange(
        round_down_to_multiple_of(plot_df["n"].min(), predicted_cache_line_bytes),
        round_up_to_multiple_of(plot_df["n"].max(), predicted_cache_line_bytes),
        step=predicted_cache_line_bytes,
    )

    for i, cache_line_boundary in enumerate(cache_line_boundaries):
        ax.axvline(
            x=cache_line_boundary,
            color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
            linestyle="--",
            label="cache line boundary" if i == 0 else None,
        )

    # ax.plot(plot_df["n"], plot_df["miss_rate"] * 100.0,
    marker_size = 3
    ax.scatter(
        plot_df["n"],
        plot_df["miss_rate"] * 100.0,
        marker_size,
        # [3] * len(plot_df["n"]),
        # linewidth=1.5,
        # linestyle='--',
        marker="x",
        # markersize=3,
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label="GTX 1080",
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    xticks = np.arange(min_kb, max_kb, step=256)
    # xticks = np.linspace(
    #         round_to_multiple_of(plot_df["n"].min(), KB),
    #         round_to_multiple_of(plot_df["n"].max(), KB), num=4)
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(min_kb, max_kb)
    ax.set_ylim(0, np.clip(plot_df["miss_rate"].max() * 2 * 100.0, 10.0, 100.0))
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)



@main.command()
# @click.option("--start", "start_size", type=int, help="start cache size in bytes")
# @click.option("--end", "end_size", type=int, help="end cache size in bytes")
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def latency_n_graph(mem, gpu, cached, repetitions, warmup, sim, force):
    """
    Compute latency-N graph.

    This is not by itself sufficient to deduce cache parameters but our simulator should match
    this behaviour.
    """
    repetitions = max(1, repetitions or (1 if sim else 10))
    warmup = warmup or (1 if sim else 2)
    known_cache_line_bytes = 128
    known_cache_size_bytes = 24 * KB
    known_cache_sets = 4

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    stride_bytes = 16
    step_size_bytes = 32

    match mem.lower():
        case "l1readonly":
            pass

    start_cache_size_bytes = known_cache_size_bytes - 1 * known_cache_line_bytes
    end_cache_size_bytes = (
        known_cache_size_bytes + (known_cache_sets + 1) * known_cache_line_bytes
    )

    cache_file = get_cache_file(prefix="latency_n_graph", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        combined, (_, stderr) = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_cache_size_bytes,
            end_size_bytes=end_cache_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            warmup=1 if sim else 2,
            max_rounds=1,
            repetitions=repetitions,
            sim=sim,
            force=force,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

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

    tick_unit = 1 * KB
    tick_unit = 256
    xtick_start = math.floor(mean_latency["n"].min() / float(tick_unit))
    xtick_end = math.ceil(mean_latency["n"].max() / float(tick_unit))
    xticks = np.array(list(range(xtick_start, xtick_end))) * tick_unit
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    print(xticks)

    ylabel = r"mean latency"
    xlabel = r"$N$ (bytes)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    # manager = matplotlib.font_manager.FontManager()
    # helvetica_prop = matplotlib.font_manager.FontProperties(family = 'Helvetica')
    # helvetica = manager.findfont(helvetica_prop)
    # print(helvetica.get_name())

    # pprint(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
    # matplotlib.font_manager._rebuild()
    # helvetica = matplotlib.font_manager.FontProperties(fname=r"")
    # print(helvetica.get_name())

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    # ax = fig.gca()
    ax = plt.axes()
    # axs = fig.subplots(1, 1) # , sharex=True, sharey=True)
    # axs[0].plot(mean_latency["n"], mean_latency["latency"], "k--", linewidth=1.5, linestyle='--', marker='o', color='b', label="GTX 1080")
    ax.plot(
        mean_latency["n"],
        mean_latency["latency"],
        linewidth=1.5,
        linestyle="--",
        marker="o",
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label="GTX 1080",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks, xticklabels)
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)
    return
    # plt.show()
    # return

    if False:
        margin = 50

        data = []
        data.append(
            go.Scatter(
                name="GTX 1080",
                x=mean_latency["n"],
                y=mean_latency["latency"],
                # mode = 'markers',
                mode="lines+markers",
                marker=dict(
                    size=int(0.8 * fontsize),
                    color=plot.rgba(*RGB_COLOR["green1"], 1.0),
                    # color = "rgba(%d, %d, %d, %f)" % (*hex_to_rgb(SIM_COLOR[sim]), 0.7),
                    symbol="circle",
                    # symbol = "x",
                ),
                # marker=dict(
                #     size=fontsize,
                #     line=dict(width=2, color='DarkSlateGrey'),
                #     symbol = "x",
                # ),
                # selector=dict(mode='markers'),
                line=dict(
                    color=plot.rgba(*plot.RGB_COLOR["green1"], 0.4),
                    # color='firebrick',
                    width=5,
                ),
            )
        )

        layout = go.Layout(
            # font_family=font_family,
            font_family="Open Sans",
            font_color="black",
            font_size=fontsize,
            plot_bgcolor="white",
            margin=dict(
                # pad=10,
                autoexpand=True,
                l=margin,
                r=margin,
                t=margin,
                b=margin,
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                tickfont=dict(size=fontsize),
                showticklabels=True,
                gridcolor="gray",
                zerolinecolor="gray",
            ),
            xaxis=go.layout.XAxis(
                title=xlabel,
                tickvals=xticks,
                ticktext=xticklabels,
                showticklabels=True,
                tickfont=dict(size=fontsize),
                ticks="outside",
                tickwidth=2,
                tickcolor="gray",
                ticklen=3,
                # dividerwidth=3,
                # dividercolor="black",
            ),
            # hoverlabel=dict(
            #     bgcolor="white",
            #     font_size=fontsize,
            #     font_family=font_family,
            # ),
            # barmode="group",
            # bargroupgap=0.02,
            # bargap=0.02,
            showlegend=True,
            width=int(DINA4_WIDTH / 4),
            # height=int(500,
            # **DEFAULT_LAYOUT_OPTIONS,
        )
        fig = go.Figure(data=data, layout=layout)
        filename = PLOT_DIR / "latency_n_plot.{}.{}.pdf".format(
            mem, "sim" if sim else "native"
        )
        for _ in range(2):
            time.sleep(1)
            fig.write_image(filename, **plot.PLOTLY_PDF_OPTS)


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def plot_latency_distribution(mem, gpu, cached, sim, repetitions, force):
    repetitions = max(1, repetitions or 1)

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    # plot latency distribution
    cache_file = get_cache_file(prefix="latency_distribution", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        latencies = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        latencies = pd.DataFrame(
            collect_full_latency_distribution(sim=sim, gpu=gpu),
            columns=["latency"],
        )

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        latencies.to_csv(cache_file, index=False)

    bin_size = 20
    bins = np.arange(0, 700, step=bin_size)
    latency_hist_df = get_latency_distribution(latencies["latency"], bins=bins)
    _, latency_centroids = predict_is_hit(
        latencies["latency"].to_numpy().reshape(-1, 1)
    )
    print(latency_centroids)
    # print(latency_hist_df)

    ylabel = "count"
    xlabel = "latency (cycles)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()
    ax.bar(
        latency_hist_df["bin_mid"],
        latency_hist_df["count"],
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        hatch="/",
        width=bin_size,
        edgecolor="black",
        label="gpucachesim" if sim else "GTX 1080",
    )
    for centroid, label in zip(latency_centroids, ["L1 Hit", "L2 Hit", "L2 Miss"]):
        centroid_bins = latency_hist_df["bin_start"] <= centroid + 2 * bin_size
        centroid_bins &= centroid - 2 * bin_size <= latency_hist_df["bin_end"]
        y = latency_hist_df.loc[centroid_bins, "count"].max()
        ax.annotate(
            "{}\n({:3.1f})".format(label, centroid),
            xy=(centroid, y),
            # xytext=(0, 4),  # 4 points vertical offset.
            # textcoords='offset points',
            ha="center",
            va="bottom",
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, latency_hist_df["count"].max() * 1.5)
    # ax.set_xticks(latency_hist_df["bin_mid"][::4]) # , xticklabels)
    # ax.set_xticks(latency_centroids) # , xticklabels)
    # ax.set_xticks(latency_hist_df.loc[np.argsort(latency_hist_df["count"]), "bin_mid"][-3:]) # , xticklabels)
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)

    return
    if False:
        data = []

        # add native data
        data.append(
            go.Bar(
                name="GTX 1080",
                x=latency_hist_df["bin_mid"],
                y=latency_hist_df["count"],
            )
        )

        # fig = go.Figure(data=[
        #     go.Bar(name='gpucachesim', x=animals, y=[20, 14, 23]),
        #     go.Bar(name='GTX 1080', x=animals, y=[12, 18, 29])
        # ])

        margin = 50

        layout = go.Layout(
            font_family=font_family,
            font_color="black",
            font_size=fontsize,
            plot_bgcolor="white",
            margin=dict(
                pad=10, autoexpand=True, l=margin, r=margin, t=margin, b=margin
            ),
            yaxis=go.layout.YAxis(
                title=ylabel,
                tickfont=dict(size=fontsize),
                gridcolor="gray",
                zerolinecolor="gray",
            ),
            xaxis=go.layout.XAxis(
                title=xlabel,
                tickfont=dict(size=fontsize),
                dividerwidth=0,
                dividercolor="white",
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=fontsize,
                font_family=font_family,
            ),
            barmode="group",
            bargroupgap=0.02,
            bargap=0.02,
            showlegend=True,
            width=int(DINA4_WIDTH / 4),
            # height=int(500,
            # **DEFAULT_LAYOUT_OPTIONS,
        )
        fig = go.Figure(data=data, layout=layout)
        filename = PLOT_DIR / "latency_distribution.{}.{}.pdf".format(
            mem, "sim" if sim else "native"
        )
        for _ in range(2):
            time.sleep(1)
            fig.write_image(filename, **plot.PLOTLY_PDF_OPTS)


def get_cache_file(prefix, mem, sim, gpu) -> Path:
    kind = "sim" if sim else "native"
    if gpu is None:
        cache_file_name = "{}.{}.{}.csv".format(prefix, mem, kind)
    else:
        cache_file_name = "{}/{}.{}.{}.csv".format(gpu, prefix, mem, kind)
    return CACHE_DIR / cache_file_name


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--start", "start_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_size_bytes", type=int, help="end cache size in bytes")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--warmup", "warmup", type=int, default=1, help="warmup iterations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--max_rounds", "max_rounds", type=int, help="maximum number of rounds")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def find_cache_size(
    mem, gpu, start_size_bytes, end_size_bytes, sim, cached, warmup, repetitions, max_rounds, force
):
    """
    Step 1.

    Determine cache size C. We set s to 1. We then initialize
    N with a small value and increase it gradually until
    the first cache miss appears. C equals the maximum N
    where all memory accesses are cache hits.
    """
    repetitions = max(1, repetitions or (1 if sim else 20))
    max_rounds = max(1, max_rounds or 1)
    predicted_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)

    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS

    print("predicted cache size: {} bytes ({})".format(
        predicted_cache_size_bytes, humanize.naturalsize(predicted_cache_size_bytes, binary=True)))

    # match mem.lower():
    #     case "l1readonly":
    #         stride_bytes = 16

    match (gpu, mem.lower()):
        case (_, "l2"):
            stride_bytes = 32
            step_size_bytes = 256 * KB
        case (_, "l1data"):
            stride_bytes = 8
            step_size_bytes = 1 * KB
        case other:
            raise ValueError("unsupported config {}".format(other))

    start_size_bytes = start_size_bytes or step_size_bytes  # or (1 * MB)
    # end_size_bytes = start_size_bytes
    # end_size_bytes = start_size_bytes + 3 * step_size_bytes
    end_size_bytes = end_size_bytes or (
        round_up_to_multiple_of(2 * predicted_cache_size_bytes, multiple_of=step_size_bytes)
    )

    # end_size_bytes = 220 * KB
    # step_size_bytes = 1 * KB
    # stride_bytes = 8
        
    start_size_bytes = max(0, start_size_bytes)
    end_size_bytes = max(0, end_size_bytes)
    
    cache_file = get_cache_file(prefix="cache_size", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(cache_file, header=0, index_col=None)
    else:
        combined, (_, stderr) = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_size_bytes,
            end_size_bytes=end_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            warmup=warmup,
            repetitions=repetitions,
            max_rounds=max_rounds,
            sim=sim,
            force=force,
        )
        print(stderr)

        combined = combined.drop(columns=["r"])
        combined = (
            combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False)

    print("number of unqiue indices = {: <4}".format(len(combined["index"].unique())))
    hit_cluster = 1 if mem == "l2" else 0

    for n, df in combined.groupby("n"):
        if n % KB == 0:
            print("==> {} KB".format(n / KB))

        num_hits = (df["hit_cluster"] <= hit_cluster).sum()
        num_misses = (df["hit_cluster"] > hit_cluster).sum()

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

    def agg_miss_rate(hit_clusters):
        cluster_counts = hit_clusters.value_counts().reset_index()
        num_misses = cluster_counts.loc[
            cluster_counts["hit_cluster"] > hit_cluster, "count"
        ].sum()
        total = cluster_counts["count"].sum()
        return num_misses / total

    plot_df = combined.groupby("n").agg({"hit_cluster": agg_miss_rate}).reset_index()
    plot_df = plot_df.rename(columns={"hit_cluster": "miss_rate"})
    # print(plot_df)

    ylabel = r"miss rate ($\%$)"
    xlabel = r"$N$ (bytes)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()
    ax.axvline(
        x=predicted_cache_size_bytes,
        color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
        linestyle="--",
        label="cache size",
    )
    ax.plot(
        plot_df["n"],
        plot_df["miss_rate"] * 100.0,
        linewidth=1.5,
        linestyle="--",
        marker="x",
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label="GTX 1080",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    num_ticks = 8
    tick_step_size_bytes = round_up_to_next_power_of_two(plot_df["n"].max() / num_ticks)
    min_kb = round_down_to_multiple_of(plot_df["n"].min(), tick_step_size_bytes)
    max_kb = round_up_to_multiple_of(plot_df["n"].max(), tick_step_size_bytes)

    xticks = np.arange(np.max([min_kb, tick_step_size_bytes]), max_kb, step=tick_step_size_bytes)
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]

    ax.set_xticks(xticks, xticklabels, rotation=45)
    ax.set_xlim(min_kb, max_kb)
    ax.set_ylim(0, 100.0)
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


@main.command()
def test():
    # print(16 * KB, 128 * 32 * 4)
    pass


@main.command()
@click.option(
    "--mem", "mem", type=str, default="l1data", help="memory to microbenchmark"
)
@click.option("--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark")
@click.option("--warmup", type=int, help="number of warmup interations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--rounds", "rounds", type=int, help="number of rounds")
@click.option("--size", type=int, help="size in bytes")
@click.option("--stride", type=int, help="stride in bytes")
@click.option("--verbose", type=bool, is_flag=True, help="verbose output")
@click.option("--sim", type=bool, is_flag=True, help="use simulator")
@click.option("--force", "force", type=bool, is_flag=True, help="force re-running experiments")
def run(mem, gpu, warmup, repetitions, rounds, size, stride, verbose, sim, force):
    gpu = gpu.upper() if gpu is not None else None
    assert gpu in VALID_GPUS
    repetitions = repetitions or 1
    warmup = warmup or 1
    stride = stride or 32
    size = size or 24 * KB
    df, (stdout, stderr) = pchase(
        gpu=gpu,
        mem=mem,
        start_size_bytes=size,
        end_size_bytes=size,
        step_size_bytes=1,
        stride_bytes=stride,
        repetitions=warmup,
        max_rounds=rounds,
        warmup=warmup,
        sim=sim,
        force=force,
    )
    if verbose:
        print(stdout)
        print(stderr)
    print(df)


if __name__ == "__main__":
    main()
