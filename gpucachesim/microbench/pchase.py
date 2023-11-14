import click
import math
import humanize
import typing
import numpy as np
import pandas as pd
import itertools
from functools import partial
import pyeda
import pyeda.boolalg
import pyeda.boolalg.expr
import pyeda.boolalg.minimization
import time
import sympy as sym
import logicmin
import re
import os
import enum
import bitarray
import bitarray.util
import tempfile
import gpucachesim.remote as remote
from collections import OrderedDict, defaultdict
from pathlib import Path
from pprint import pprint
from io import StringIO, BytesIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from scipy.stats import zscore
from wasabi import color
import matplotlib
import matplotlib.pyplot as plt

from gpucachesim.asm import (
    solve_mapping_table_xor_fast,
    solve_mapping_table_xor,
    solve_mapping_table,
)
from gpucachesim.benchmarks import REPO_ROOT_DIR
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from gpucachesim.plot import PLOT_DIR
import gpucachesim.cmd as cmd_utils
import gpucachesim.plot as plot
import gpucachesim.utils as utils

NATIVE_P_CHASE = REPO_ROOT_DIR / "test-apps/microbenches/chxw/pchase"
NATIVE_SET_MAPPING = REPO_ROOT_DIR / "test-apps/microbenches/chxw/set_mapping"
NATIVE_RANDOM_SET_MAPPING = REPO_ROOT_DIR / "test-apps/microbenches/chxw/set_mapping"

SIM_P_CHASE = REPO_ROOT_DIR / "target/release/pchase"
SIM_SET_MAPPING = REPO_ROOT_DIR / "target/release/pchase"

CACHE_DIR = PLOT_DIR / "cache"

SEC = 1
MIN = 60 * SEC

CSV_COMPRESSION = {"method": "bz2", "compresslevel": 9}

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


def quantize_latency(latency, bin_size=50):
    return utils.round_to_multiple_of(latency, multiple_of=bin_size)


def compute_dbscan_clustering(values, eps=3, min_samples=3):
    values = np.array(values)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(values.reshape(-1, 1))
    # clustering_df = pd.DataFrame(
    #     np.array([values.ravel(), labels.ravel()]).T,
    #     columns=["latency", "cluster"],
    # )
    # print(clustering_df)
    return labels


def predict_is_hit(latencies, fit=None, num_clusters=3):
    latencies = np.nan_to_num(latencies, nan=0, posinf=0)
    km = KMeans(n_clusters=num_clusters, random_state=1, n_init=15)
    # km = KMedoids(n_clusters=2, random_state=0)
    if fit is None:
        km.fit(latencies)
    else:
        fit = np.nan_to_num(fit, nan=0, posinf=0)
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


def collect_full_latency_distribution(
    sim, gpu=None, force=False, skip_l1=True, configs=None, verbose=False
):
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

        if skip_l1:
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
        else:
            # include l1 miss + l2 hit (l1 size < size_bytes < l2 size)
            size_bytes = 2 * get_known_cache_size_bytes(mem="l1data", gpu=gpu)
            stride_bytes = get_known_cache_line_bytes(mem="l1data", gpu=gpu)
            configs.append(
                PChaseConfig(
                    mem="l1data",
                    start_size_bytes=size_bytes,
                    end_size_bytes=size_bytes,
                    step_size_bytes=1,
                    stride_bytes=stride_bytes,
                    warmup=1,
                    repetitions=1,
                    max_rounds=2,
                    iter_size=None,
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
        hit_latencies_df, stderr = pchase(**config._asdict(), gpu=gpu, force=force)
        if verbose:
            print(stderr)
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
        print("=== LATENCY HISTOGRAM (prediction)".format(
            np.count_nonzero(np.isnan(latencies))))
        print(pred_hist_df[["bin", "count"]])
        print("")

        fit_hist_df = get_latency_distribution(fit_latencies)
        print("=== LATENCY HISTOGRAM (fitting)".format(
            np.count_nonzero(np.isnan(fit_latencies))))
        print(fit_hist_df[["bin", "count"]])
        print("")

    combined_hist_df = get_latency_distribution(combined_latencies)
    print("=== LATENCY HISTOGRAM (combined) #nans={}".format(
        np.count_nonzero(np.isnan(combined_latencies))))
    print(combined_hist_df[["bin", "count"]])
    print("")

    # clustering_bins = fit_hist[bins[:-1] <= 1000.0]
    # print(clustering_bins)

    # find the top 3 bins and infer the bounds for outlier detection
    # hist_percent = hist / np.sum(hist)
    # valid_latency_bins = hist[hist_percent > 0.5]
    # latency_cutoff = np.min(valid_latency_bins)

    for (name, values) in [("FIT", fit_latencies), ("", latencies)]:
        print(
            "BEFORE: {}: mean={:4.2f} min={:4.2f} max={:4.2f} #nans={}".format(
                name,
                values.mean(),
                values.min(),
                values.max(),
                np.count_nonzero(np.isnan(values)),
            )
        )

    # latency_abs_z_score = np.abs(latencies - np.median(fit_latencies))
    # outliers = latency_abs_z_score > 1000
    outliers = fit_latencies > 1000

    num_outliers = outliers.sum()
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
    gpu: typing.Optional[enum.Enum],
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
    executable: typing.Optional[os.PathLike]=None,
    force=False,
    stream_output=False,
    log_every=100_000,
) -> typing.Tuple[pd.DataFrame, str]:
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

    env = dict(os.environ)
    if isinstance(log_every, int):
        env.update({"LOG_EVERY": str(log_every)})

    if gpu is None:
        if executable is None:
            executable = SIM_P_CHASE if sim else NATIVE_P_CHASE
        elif not os.path.isabs(executable):
            executable = NATIVE_P_CHASE.parent / executable
        else:
            executable = Path(executable)

        # run locally
        cmd = " ".join([str(executable.absolute())] + cmd)

        unit = 1 * MIN if sim else 1 * SEC
        per_size_timeout = [
            ((derived_iter_size * (1 + warmup)) / 1000) * unit
            for _ in range(start_size_bytes, end_size_bytes + 1, step_size_bytes)
        ]
        timeout_sec = repetitions * sum(per_size_timeout)
        timeout_sec = max(5, 2 * timeout_sec)

        print("[stream={}, timeout {: >5.1f} sec]\t{}".format(stream_output, timeout_sec, cmd))

        try:
            _, stdout, stderr, _ = cmd_utils.run_cmd(
                cmd,
                env=env,
                timeout_sec=int(timeout_sec),
                stream_output=stream_output,
            )
            stdout_reader = StringIO(stdout)
        except cmd_utils.ExecStatusError as e:
            print(e.stderr)
            raise e
    else:
        if remote.DAS5_GPU.has(gpu):
            das = remote.DAS5()
        elif remote.DAS6_GPU.has(gpu):
            das = remote.DAS6()
        else:
            raise ValueError("cannot run on GPU {}".format(str(gpu)))

        if executable is None:
            executable = das.remote_scratch_dir / "pchase"
        elif not os.path.isabs(executable):
            executable = das.remote_scratch_dir / executable
        else:
            executable = Path(executable)

        try:
            stdout_reader, stderr_reader = das.run_pchase_sync(
                cmd,
                gpu=gpu,
                executable=executable,
                force=force,
            )
            stderr = stderr_reader.read().decode("utf-8")
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
            das.close()
            raise e

    df = pd.read_csv(
        stdout_reader,
        header=0,
        dtype=float,
    )
    stdout_reader.close()
    return df, stderr


def set_mapping(
    gpu: typing.Optional[enum.Enum],
    mem,
    stride_bytes,
    warmup,
    size_bytes,
    repetitions=1,
    max_rounds=None,
    iter_size=None,
    compute_capability=None,
    executable: typing.Optional[typing.Union[os.PathLike, str]] = None,
    # random=False,
    sim=False,
    force=False,
    debug=False,
    log_every=100_000,
) -> typing.Tuple[pd.DataFrame, str]:
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
        if executable is None:
            executable = SIM_SET_MAPPING if sim else NATIVE_SET_MAPPING
        elif not os.path.isabs(executable):
            executable = NATIVE_SET_MAPPING.parent / executable
            # if sim:
            #     executable = SIM_SET_MAPPING if random else SIM_SET_MAPPING 
            # else:
            #     executable = NATIVE_RANDOM_SET_MAPPING if random else NATIVE_SET_MAPPING
        else:
            executable = Path(executable)

        # executable = SIM_SET_MAPPING if sim else NATIVE_SET_MAPPING
        cmd = " ".join([str(executable.absolute())] + cmd)

        timeout_sec = repetitions * (20 * MIN if sim else 10 * SEC)
        print("[timeout {: >5.1f} sec]\t{}".format(timeout_sec, cmd))

        try:
            env = dict(os.environ)
            if compute_capability is not None:
                env.update({"COMPUTE_CAPABILITY": str(int(compute_capability))})
            if debug:
                env.update({"DEBUG": "1"})
            # if random:
            #     env.update({"RANDOM": "1"})
            if isinstance(log_every, int):
                env.update({"LOG_EVERY": str(log_every)})

            _, stdout, stderr, _ = cmd_utils.run_cmd(
                cmd,
                timeout_sec=int(timeout_sec),
                env=env,
            )
            stdout_reader = StringIO(stdout)
        except cmd_utils.ExecStatusError as e:
            print(e.stderr)
            raise e
    else:
        # connect to remote gpu
        if remote.DAS5_GPU.has(gpu):
            das = remote.DAS5()
        elif remote.DAS6_GPU.has(gpu):
            das = remote.DAS6()
        else:
            raise ValueError("cannot run on GPU {}".format(str(gpu)))


        if executable is None:
            executable = das.remote_scratch_dir / "set_mapping"
        elif not os.path.isabs(executable):
            executable = das.remote_scratch_dir / executable
        else:
            executable = Path(executable)

        try:
            stdout_reader, stderr_reader = das.run_pchase_sync(
                cmd,
                gpu=gpu,
                executable=executable,
                compute_capability=compute_capability,
                # random=random,
                force=force,
            )
            stderr = stderr_reader.read().decode("utf-8")
        except Exception as e:
            das.close()
            raise e

    df = pd.read_csv(
        stdout_reader,
        header=0,
        dtype=float,
    )
    stdout_reader.close()
    return df, stderr



def get_label(sim, gpu: typing.Optional[enum.Enum]):
    if sim:
        return "gpucachesim"
    if gpu is None:
        return "GTX 1080"
    return gpu.value

def get_known_l2_prefetch_percent(mem: str, gpu: typing.Optional[enum.Enum]=None) -> float:
    match (gpu, mem.lower()):
        case (None, "l2"):
            return 0.25
        case (remote.DAS6_GPU.A4000, "l2"):
            return 0.65
        case (remote.DAS5_GPU.TITANXPASCAL, "l2"):
            return 0.90
        case _:
            return 0.0

def get_known_cache_size_bytes(mem: str, gpu: typing.Optional[enum.Enum] = None) -> int:
    match (gpu, mem.lower()):
        # local gpu gtx 1080 (pascal)
        case (None, "l1data"):
            return 24 * KB
        case (None, "l2"):
            return 2 * MB
        # a 4000 (ampere)
        case (remote.DAS6_GPU.A4000, "l1data"):
            return 19 * KB
        case (remote.DAS6_GPU.A4000, "l2"):
            return 4 * MB
        # gtx 980 (maxwell)
        case (remote.DAS5_GPU.GTX980, "l1data"):
            return 24 * KB
        case (remote.DAS5_GPU.GTX980, "l2"):
            return 2 * MB
        # titan (kepler)
        case (remote.DAS5_GPU.TITAN, "l1data"):
            return 16 * KB
        case (remote.DAS5_GPU.TITAN, "l2"):
            return 1536 * KB
        # titan x (pascal / maxwell)
        case (remote.DAS5_GPU.TITANXPASCAL | remote.DAS5_GPU.TITANX, "l1data"):
            return 24 * KB
            # return 48 * KB
        case (remote.DAS5_GPU.TITANXPASCAL | remote.DAS5_GPU.TITANX, "l2"):
            return 3 * MB
    raise ValueError("unknown cache size for ({}, {})".format(gpu, mem))


def get_known_cache_line_bytes(mem: str, gpu: typing.Optional[enum.Enum]=None) -> int:
    match (gpu, mem.lower()):
        case (None, "l1data"):
            return 128
        case (None, "l2"):
            return 128
        # a 4000 (ampere)
        case (remote.DAS6_GPU.A4000, "l1data"):
            return 128
        case (remote.DAS6_GPU.A4000, "l2"):
            return 128
        # gtx 980 (maxwell)
        case (remote.DAS5_GPU.GTX980, "l1data"):
            return 128
        case (remote.DAS5_GPU.GTX980, "l2"):
            return 32
        # titan x (pascal / maxwell)
        case (remote.DAS5_GPU.TITANXPASCAL | remote.DAS5_GPU.TITANX, "l1data"):
            return 128
        case (remote.DAS5_GPU.TITANXPASCAL | remote.DAS5_GPU.TITANX, "l2"):
            return 128
    raise ValueError("unknown cache line size for ({}, {})".format(gpu, mem))


def get_known_cache_num_sets(mem: str, gpu: typing.Optional[enum.Enum]=None) -> int:
    match (gpu, mem.lower()):
        case (None, "l1data"):
            return 4
        case (None, "l2"):
            return 4
        # a 4000 (ampere)
        case (remote.DAS6_GPU.A4000.value, "l1data"):
            return 4
        case (remote.DAS6_GPU.A4000.value, "l2"):
            return 4
        # gtx 980 (maxwell)
        case (remote.DAS5_GPU.GTX980, "l1data"):
            return 4
        case (remote.DAS5_GPU.GTX980, "l2"):
            return 4
        # titan x (pascal / maxwell)
        case (remote.DAS5_GPU.TITANXPASCAL | remote.DAS5_GPU.TITANX, "l1data"):
            return 4
        case (remote.DAS5_GPU.TITANXPASCAL | remote.DAS5_GPU.TITANX, "l2"):
            return 4
    raise ValueError("unknown num sets for ({}, {})".format(gpu, mem))


@main.command()
@click.option("--warmup", "warmup", type=int, help="cache warmup")
@click.option("--repetitions", "repetitions", type=int, help="repetitions")
@click.option("--mem", "mem", default="l2", help="memory to microbenchmark")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def find_l2_prefetch_size(warmup, repetitions, mem, cached, sim, gpu, force):
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 5))
    warmup = warmup or 0

    gpu = remote.find_gpu(gpu)

    predicted_l2_prefetch_percent = get_known_l2_prefetch_percent(mem=mem, gpu=gpu)
    known_l2_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)

    match (gpu, mem.lower()):
        case (_, "l2"):
            stride_bytes = 128
            step_size_bytes = 256 * KB
        case (_, "l1data"):
            stride_bytes = 128
            step_size_bytes = 4 * KB
        case other:
            raise ValueError("unsupported config {}".format(other))

    start_cache_size_bytes = step_size_bytes
    end_cache_size_bytes = int(
        utils.round_down_to_multiple_of(1.5 * known_l2_cache_size_bytes, stride_bytes))

    assert start_cache_size_bytes % stride_bytes == 0
    assert end_cache_size_bytes % stride_bytes == 0

    cache_file = get_cache_file(prefix="l2_prefetch_size", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = pchase(
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
            stream_output=sim == True,
        )
        print(stderr)

        # cannot average like this for non-LRU cache processes
        # combined = combined.drop(columns=["r"])
        # combined = (
        #     combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        # )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

    for n, df in combined.groupby("n"):
        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        misses = df["hit_cluster"] > 1  # l1 miss & l2 miss

        num_misses = misses.sum()

        human_size = humanize.naturalsize(n, binary=True)
        miss_rate = float(num_misses) / float(len(df))
        print(
            "size {:>15} ({:>5.1f}%)".format(
                human_size,
                float(n) / float(known_l2_cache_size_bytes) * 100.0,
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

    

    ax.axvline(
        x=predicted_l2_prefetch_percent * known_l2_cache_size_bytes,
        color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
        linestyle="--",
        label=r"{:2.0f}% L2 size".format(predicted_l2_prefetch_percent * 100.0),
    )

    marker_size = 12
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
        # label=get_label(sim=sim, gpu=gpu),
        label="L1 ({})".format(get_label(sim=sim, gpu=gpu)),
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
        # label=get_label(sim=sim, gpu=gpu),
        label="L2 ({})".format(get_label(sim=sim, gpu=gpu)),
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # min_x = utils.round_down_to_multiple_of(plot_df["n"].min(), 128)
    # max_x = utils.round_up_to_multiple_of(plot_df["n"].max(), 128)

    num_ticks = 16
    tick_step_size_bytes = utils.round_up_to_next_power_of_two(plot_df["n"].max() / num_ticks)
    min_x = utils.round_down_to_multiple_of(plot_df["n"].min(), tick_step_size_bytes)
    max_x = utils.round_up_to_multiple_of(plot_df["n"].max(), tick_step_size_bytes)

    xticks = np.arange(
        np.max([min_x, tick_step_size_bytes]), max_x, step=tick_step_size_bytes
    )
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]

    # xticks = np.arange(min_x, max_x, step=256 * KB)
    # xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    ax.set_xticks(xticks, xticklabels, rotation=45)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, 100.0)
    ax.legend(loc="upper right" if predicted_l2_prefetch_percent < 0.75 else "lower left")
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


def compute_number_of_sets(combined, hit_cluster):
    # use DBscan clustering to find the number of sets
    combined["is_miss"] = combined["hit_cluster"] > hit_cluster
    num_misses_per_n = combined.groupby("n")["is_miss"].sum().reset_index()
    num_misses_per_n = num_misses_per_n.rename(columns={"is_miss": "num_misses"})

    # print(num_misses_per_n["num_misses"])
    sorted = num_misses_per_n["num_misses"].drop_duplicates().sort_values().to_numpy()
    deltas = np.array([sorted[i+1] - sorted[i] for i in range(len(sorted)-1)])
    eps = int(2 * np.amin(deltas))

    num_misses_per_n["set_cluster"] = compute_dbscan_clustering(
        num_misses_per_n["num_misses"], eps=eps,
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
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option(
    "--size",
    "known_cache_size_bytes",
    type=int,
    help="cache line size in bytes (stride)",
)
@click.option(
    "--line-size", "known_cache_line_bytes", type=int, help="cache size in bytes"
)
@click.option("--sets", "known_num_sets", type=int, help="number of cache sets")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option(
    "--max-rounds", "max_rounds", type=int, default=1, help="maximum number of rounds"
)
@click.option(
    "--average", "average", type=bool, help="average of latencies over repetitions"
)
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def find_cache_set_mapping_pchase(
    mem,
    gpu,
    known_cache_size_bytes,
    known_cache_line_bytes,
    known_num_sets,
    warmup,
    repetitions,
    max_rounds,
    average,
    cached,
    sim,
    force,
):
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 5))
    warmup = warmup if warmup is not None else (1 if sim else 2)

    gpu = remote.find_gpu(gpu)

    known_cache_size_bytes = known_cache_size_bytes or get_known_cache_size_bytes(
        mem=mem, gpu=gpu
    )
    known_cache_line_bytes = known_cache_line_bytes or get_known_cache_line_bytes(
        mem=mem, gpu=gpu
    )
    known_num_sets = known_num_sets or get_known_cache_num_sets(mem=mem, gpu=gpu)

    derived_num_ways = known_cache_size_bytes // (
        known_cache_line_bytes * known_num_sets
    )

    # compute capability 8.6 supports shared memory capacity of 0, 8, 16, 32, 64 or 100 KB per SM
    # we choose 75 percent carveout for shared memory, which gives 25KB L1
    # we also find a discrepancy of 7KB because we see misses to occur at around 18KB as well.

    print("average: {}".format(average))
    print(
        "known cache size: {} bytes ({})".format(
            known_cache_size_bytes,
            humanize.naturalsize(known_cache_size_bytes, binary=True),
        )
    )
    print(
        "known cache line size: {} bytes ({})".format(
            known_cache_line_bytes,
            humanize.naturalsize(known_cache_line_bytes, binary=True),
        )
    )
    print("num ways = {:<3}".format(derived_num_ways))

    stride_bytes = known_cache_line_bytes

    if gpu == "A4000":
        # stride_bytes = 32
        pass

    print("warmups = {:<3}".format(warmup))
    print("repetitions = {:<3}".format(repetitions))
    print("stride = {:<3} bytes".format(stride_bytes))

    match mem.lower():
        case "l1readonly":
            stride_bytes = known_cache_line_bytes
            pass

    cache_file = get_cache_file(
        prefix="cache_set_mapping_pchase", mem=mem, sim=sim, gpu=gpu, random=random
    )
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = pchase(
            mem=mem,
            gpu=gpu,
            executable="pchase",
            start_size_bytes=known_cache_size_bytes,
            end_size_bytes=known_cache_size_bytes,
            step_size_bytes=1,
            stride_bytes=stride_bytes,
            warmup=warmup,
            repetitions=repetitions,
            max_rounds=max_rounds,
            sim=sim,
            force=force,
        )
        print(stderr)

        print(combined)
        combined = compute_hits(combined, sim=sim, gpu=gpu)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

    base_addr = int(combined["virt_addr"].min())
    combined["rel_virt_addr"] = combined["virt_addr"].astype(int) - base_addr

    # compute plot matrices
    max_cols = int(len(combined["k"].astype(int).unique()))
    max_rows = len(combined["r"].astype(int).unique())
    assert max_rows == repetitions

    latencies = np.zeros((max_rows, max_cols))
    hit_clusters = np.zeros((max_rows, max_cols))
    print(latencies.shape)

    for r, round_df in combined.groupby("r"):
        round_df = round_df.sort_values(["k"])
        assert max_cols == len(round_df)

        row_idx = int(r)
        latencies[row_idx, :] = round_df["latency"].to_numpy()
        hit_clusters[row_idx, :] = round_df["hit_cluster"].to_numpy()

    fig = plot_access_process_latencies(
        combined,
        latencies,
        warmup=warmup,
        rounds=max_rounds,
        ylabel="repetition",
        size_bytes=known_cache_size_bytes,
        stride_bytes=stride_bytes,
    )
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    print("saved plot to {}".format(filename))
    fig.savefig(filename)

    # remove first round of loading
    combined = combined[combined["k"] >= known_cache_size_bytes / stride_bytes]

    total_sets = OrderedDict()
    for r, _ in combined.groupby("r"):
        mask = combined["r"] == r
        misses = mask & (combined["hit_cluster"] > 0)
        miss_rate = float(misses.sum()) / float(mask.sum()) * 100.0
        # if misses.sum() > 0:
        #     print(combined[mask])
        # print(combined.loc[misses,:].head(n=2*derived_num_ways))
        # rel_combined = combined[mask].copy()
        # rel_combined["virt_addr"] -= base_addr
        # print(rel_combined[:10])
        # print(rel_combined[-10:])

        if misses.sum() != derived_num_ways:
            print(
                color(
                    "[r={:<3}] miss rate={:<4.2f}% has {:<2} misses (expected {} ways)".format(
                        int(r), miss_rate, misses.sum(), derived_num_ways
                    ),
                    fg="red",
                )
            )

        # key = tuple(combined.loc[misses, "virt_addr"].astype(int).to_numpy())
        missed_addresses = combined.loc[misses, "virt_addr"].astype(int).tolist()
        # if unique:
        #     missed_addresses = list(np.unique(missed_addresses))
        # if sort:
        #     missed_addresses = sorted(missed_addresses)
        key = tuple(missed_addresses)
        if key not in total_sets:
            total_sets[key] = 0
        total_sets[key] += 1

    num_sets = len(total_sets)
    num_sets_log2 = int(np.log2(num_sets))
    print(
        color(
            "total sets={:<2} ({:<2} bits)".format(num_sets, num_sets_log2),
            fg="green" if num_sets == known_num_sets else "red",
        )
    )

    total_sets = sorted([(list(s), occurences) for s, occurences in total_sets.items()])

    expanded_sets = []
    for set_id, (set_addresses, occurences) in enumerate(total_sets):
        set_addresses = [addr - base_addr for addr in set_addresses]
        print(
            "=== cache set {:>3}: {:>4} addresses ( observed {:>3}x ) === {}".format(
                set_id, len(set_addresses), occurences, set_addresses
            )
        )

        found = False
        for si in range(len(expanded_sets)):
            intersection_size = len(expanded_sets[si].intersection(set(set_addresses)))
            union_size = len(expanded_sets[si].union(set(set_addresses)))
            # intersects = intersection_size / union_size > 0.8
            # intersects = len(set(set_addresses)) - intersection_size <= 4
            intersects = (
                union_size / intersection_size > 0.5 if intersection_size > 0 else False
            )
            if intersects:
                expanded_sets[si] = expanded_sets[si].union(set(set_addresses))
                found = True
                break
        if not found:
            expanded_sets.append(set(set_addresses))

    print("expanded sets: {}".format(len(expanded_sets)))
    expanded_sets = sorted([sorted(list(s)) for s in expanded_sets])
    for si, s in enumerate(expanded_sets):
        print("=> expanded set {}:\n{}".format(si, s))

    largest_set = np.amax([len(s) for s, _ in total_sets])
    print("largest set={}".format(largest_set))


def latency_colormap(combined, min_latency, max_latency, tol=0.3):
    L1_HIT = 0
    L1_MISS = 1
    L2_MISS = 2

    combined["latency"] = combined["latency"].clip(lower=0.0)
    mean_cluster_latency = combined.groupby("hit_cluster")["latency"].mean()
    min_cluster_latency = combined.groupby("hit_cluster")["latency"].min()
    max_cluster_latency = combined.groupby("hit_cluster")["latency"].max()

    white = (255, 255, 255)
    orange = (255, 140, 0)
    red = (255, 0, 0)

    def rgb_to_vec(color: typing.Tuple[int, int, int], alpha: float = 1.0):
        return np.array([color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha])

    def gradient(
        start: typing.Tuple[int, int, int], end: typing.Tuple[int, int, int], n=256
    ):
        vals = np.ones((n, 4))
        vals[:, 0] = np.linspace(start[0] / 255.0, end[0] / 255.0, n)
        vals[:, 1] = np.linspace(start[1] / 255.0, end[1] / 255.0, n)
        vals[:, 2] = np.linspace(start[2] / 255.0, end[2] / 255.0, n)
        return ListedColormap(vals)

    white_to_orange = gradient(start=white, end=orange)
    orange_to_red = gradient(start=orange, end=red)

    latency_range = max_latency - min_latency
    mean_hit_cluster_latency = mean_cluster_latency.get(L1_HIT, 0.0)
    mean_miss_cluster_latency = mean_cluster_latency.get(L1_MISS, 200.0)
    assert (
        min_latency
        <= mean_hit_cluster_latency
        <= mean_miss_cluster_latency
        <= max_latency
    )

    start = min_latency
    hit_end = min_cluster_latency.get(L1_MISS, 200.0) - tol * abs(
        min_cluster_latency.get(L1_MISS, 200.0) - max_cluster_latency.get(L1_HIT, 0.0)
    )
    # hit_end = mean_hit_cluster_latency + tol * (mean_miss_cluster_latency - mean_hit_cluster_latency)
    # miss_end = mean_miss_cluster_latency + tol * (max_latency - mean_miss_cluster_latency)
    # miss_end = np.min([mean_cluster_latency.get(MISS, 200.0) + 100.0, max_latency])
    # print(mean_cluster_latency.get(L1_MISS))
    # print(max_latency)
    # miss_end = mean_cluster_latency.get(L1_MISS, 200.0) + 100.0

    # print(min_cluster_latency.get(L1_MISS))
    # print(mean_cluster_latency.get(L1_MISS))
    # print(max_cluster_latency.get(L1_MISS))
    # print(min_cluster_latency.get(L2_MISS))

    miss_end = (
        mean_cluster_latency.get(L1_MISS, 200.0) + max_cluster_latency.get(L1_MISS, 200.0)
    ) / 2.0
    if min_cluster_latency.get(L2_MISS) is not None:
        miss_end = np.min([miss_end, min_cluster_latency[L2_MISS]])
    miss_end = np.min([miss_end, max_latency])
    end = max_latency

    points = [start, hit_end, miss_end, end]
    print("latency cmap:", points)
    widths = [points[i + 1] - points[i] for i in range(len(points) - 1)]

    assert np.sum(widths) == latency_range

    latency_cmap = np.vstack(
        [
            np.repeat(
                rgb_to_vec(white).reshape(1, 4),
                repeats=int(np.round(widths[0])),
                axis=0,
            ),
            white_to_orange(np.linspace(0, 1, int(np.round(0.5 * widths[1])))),
            orange_to_red(np.linspace(0, 1, int(np.round(0.5 * widths[1])))),
            np.repeat(
                rgb_to_vec(red).reshape(1, 4), repeats=int(np.round(widths[2])), axis=0
            ),
        ]
    )

    assert np.allclose(len(latency_cmap), int(np.round(latency_range)), atol=2)
    return ListedColormap(latency_cmap)


def plot_access_process_latencies(
    combined,
    values,
    warmup=None,
    rounds=None,
    ylabel=None,
    size_bytes=None,
    stride_bytes=None,
):
    repetitions = len(combined["r"].unique())
    print(repetitions)
    mean_cluster_latency = combined.groupby("hit_cluster")["latency"].mean()

    max_latency = mean_cluster_latency.get(1, 200.0) + 100.0
    min_latency = np.max([0, mean_cluster_latency.get(0, 0.0) - 100.0])

    num_overflow_indices = len(combined["overflow_index"].unique().tolist())
    y_axis_kind = "repetition" if num_overflow_indices == 1 else "overflow index"
    if ylabel is None:
        ylabel = y_axis_kind

    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(1.0 * plot.DINA4_WIDTH_INCHES, 0.5 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()

    latency_cmap = latency_colormap(
        combined, min_latency=min_latency, max_latency=max_latency
    )

    print(values.shape)
    c = plt.imshow(
        values,
        cmap=latency_cmap,
        vmin=min_latency,
        vmax=max_latency,
        # interpolation="nearest",
        interpolation="none",
        origin="upper",
        aspect="auto",
        # aspect="equal",
    )
    fig.colorbar(c, ax=ax)

    # rect = plt.Rectangle(
    #     (0,-10),
    #     values.shape[1], 10,
    #     facecolor='silver',
    #     clip_on=False,
    #     linewidth = 0,
    #
    # )
    # ax.add_patch(rect)
    # rx, ry = rect.get_xy()
    # cx = rx + rect.get_width()/2.0
    # cy = ry + rect.get_height()/2.0
    # ax.annotate("fill", (cx, cy), color='black', fontsize=0.5 * fontsize, ha='center', va='center',
    #             annotation_clip=False)

    xlabel = r"index into array $A$ of size $N$"
    if size_bytes is not None:
        xlabel += "=" + humanize.naturalsize(size_bytes, binary=True)

    if True:
        if None not in [warmup, rounds, size_bytes, stride_bytes]:
            xlabel = r"rounds through array $A$ of size $N=$" + humanize.naturalsize(
                size_bytes, binary=True
            )
            round_size = size_bytes / stride_bytes
            round_indices = np.arange(0, rounds + 1, step=1)
            xticks = round_indices * round_size
            xticklabels = ["R{}".format(r) for r in round_indices + warmup]
            ax.set_xticks(xticks, xticklabels)

    if True:
        if isinstance(repetitions, int):
            if y_axis_kind == "repetition":
                num_y_values = repetitions
                bin_size = 1
            elif y_axis_kind == "overflow index":
                num_y_values = len(values) / repetitions
                bin_size = repetitions
            else:
                raise ValueError("bad y axis type")

            print("num y values", num_y_values)
            step_size = int(np.amax([1, num_y_values / 10.0]))
            if step_size > 10:
                step_size = utils.round_up_to_multiple_of(step_size, 10.0)
            y_values = np.arange(num_y_values, step=step_size)
            yticks = y_values * bin_size
            yticklabels = y_values.astype(int)
            # print(y_values)
            # print(yticks)
            # print(yticklabels)
            ax.set_yticks(yticks, yticklabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


@main.command()
@click.option(
    "--mem", "mem", default="l1data", type=str, help="memory to microbenchmark"
)
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option(
    "--size",
    "known_cache_size_bytes",
    type=int,
    help="cache line size in bytes (stride)",
)
@click.option(
    "--line-size", "known_cache_line_bytes", type=int, help="cache size in bytes"
)
@click.option("--sets", "known_num_sets", type=int, help="number of cache sets")
@click.option("--stride", "stride_bytes", type=int, help="stride in bytes")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option(
    "--max-rounds", "max_rounds", type=int, default=1, help="maximum number of rounds"
)
@click.option(
    "--average", "average", type=bool, help="average of latencies over repetitions"
)
@click.option(
    "--cc", "--compute-capability", "compute_capability", type=int, help="compute capability"
)
@click.option(
    "--random", "random", type=bool, default=False, help="use random access pattern"
)
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
@click.option(
    "--debug", "debug", type=bool, is_flag=True, help="enable debug output"
)
def find_cache_set_mapping(
    mem,
    gpu,
    known_cache_size_bytes,
    known_cache_line_bytes,
    known_num_sets,
    stride_bytes,
    warmup,
    repetitions,
    max_rounds,
    average,
    cached,
    compute_capability,
    random,
    sim,
    force,
    debug,
):
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 5))
    warmup = warmup if warmup is not None else (1 if sim else 2)

    gpu = remote.find_gpu(gpu)

    if compute_capability is None:
        compute_capability = remote.get_compute_capability(gpu=gpu)

    sort = False
    unique = False

    if average is None:
        average = gpu == None
    
    # if compute_capability == 86:
    #     if average is None:
    #         average = False

    if gpu is not None and average:
        print("WARNING: averaging results for GPU {}, which might not have LRU access process".format(
            gpu.value))

    if known_cache_size_bytes is None:
        known_cache_size_bytes = get_known_cache_size_bytes( mem=mem, gpu=gpu)
    if known_cache_line_bytes is None:
        known_cache_line_bytes = get_known_cache_line_bytes( mem=mem, gpu=gpu)
    if known_num_sets is None:
        known_num_sets = get_known_cache_num_sets(mem=mem, gpu=gpu)

    derived_num_ways = known_cache_size_bytes // (
        known_cache_line_bytes * known_num_sets
    )

    print("average: {}".format(average))
    print(
        "known cache size: {} bytes ({})".format(
            known_cache_size_bytes,
            humanize.naturalsize(known_cache_size_bytes, binary=True),
        )
    )
    print(
        "known cache line size: {} bytes ({})".format(
            known_cache_line_bytes,
            humanize.naturalsize(known_cache_line_bytes, binary=True),
        )
    )
    print("num ways = {:<3}".format(derived_num_ways))


    # assert (
    #     known_cache_size_bytes
    #     == known_num_sets * derived_num_ways * known_cache_line_bytes
    # )

    if stride_bytes is None:
        stride_bytes = known_cache_line_bytes

    if gpu == "A4000":
        # stride_bytes = 32
        pass

    print("warmups = {:<3}".format(warmup))
    print("repetitions = {:<3}".format(repetitions))
    print("stride = {:<3} bytes".format(stride_bytes))

    # assert not (warmup < 1 and average), "cannot average without warmup"
    assert not (random and average), "cannot average random"

    match mem.lower():
        case "l1readonly":
            stride_bytes = known_cache_line_bytes
            pass

    cache_file = get_cache_file(
        prefix="cache_set_mapping",
        mem=mem,
        sim=sim,
        gpu=gpu,
        compute_capability=compute_capability,
        random=random,
    )

    print(cache_file)
    if cached and cache_file.is_file():
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = set_mapping(
            mem=mem,
            gpu=gpu,
            executable="random_set_mapping" if random else "set_mapping",
            size_bytes=known_cache_size_bytes,
            stride_bytes=stride_bytes,
            warmup=warmup,
            repetitions=repetitions,
            compute_capability=compute_capability,
            # random=random,
            max_rounds=max_rounds,
            sim=sim,
            force=force,
            debug=debug,
        )
        print(stderr)

        if average:
            # combined = combined.drop(columns=["r"])
            # we are averaging over k and r to get mean latency
            combined = (
                combined.groupby(["n", "overflow_index", "k", "index", "virt_addr"])
                .median()
                .reset_index()
            )
            combined["r"] = 0

        combined = compute_hits(combined, sim=sim, gpu=gpu)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)


    combined["rel_index"] = combined["index"] / 4
    combined["latency"] = combined["latency"].clip(lower=0.0)
    if repetitions < 10:
        for r, df in combined.groupby("r"):
            print("==== r={} ====".format(r))
            print(df.head(n=20))

    hit_cluster = 1 if mem == "l2" else 0

    # # manual averaging based on the pattern
    # def most_common_pattern(_df):
    #     print(_df)
    #     print(_df.shape)
    #     return _df
    #
    # def get_misses(_df):
    #     # print(_df)
    #     return _df.loc[_df["hit_cluster"] != 0, "index"]

    round_size = int(known_cache_size_bytes / stride_bytes)
    unique_indices = sorted(combined["index"].unique().tolist())
    unique_ks = sorted(combined["k"].unique().tolist())
    print(unique_indices[:10], unique_indices[-10:])
    print(unique_ks[:10], unique_ks[-10:])
    print(len(unique_indices))
    print(len(unique_ks))
    # assert len(unique_indices) == known_cache_size_bytes / stride_bytes
    assert len(unique_ks) == (known_cache_size_bytes * max_rounds) / stride_bytes

    assert len(combined["n"].unique().tolist()) == 1

    # if compute_capability == 86:
    #     if not random:
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             # assert len(_df["index"].unique().tolist()) == len(unique_indices)
    #             mask = combined["n"] == n
    #             mask &= combined["overflow_index"] == overflow_index
    #             mask &= combined["r"] == r
    #             stride = stride_bytes / 4
    #             wrapped = combined["index"] <= overflow_index - stride_bytes
    #             combined.loc[mask & wrapped, "latency"] = 300.0
    #
    #     if random:
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             rounds_1_and_2 = _df["k"] >= 1 * round_size
    #             rounds_1_and_2 &= _df["k"] < 3 * round_size
    #             # print(_df[rounds_1_and_2].head(n=150))
    #             # print(_df.loc[rounds_1_and_2, "hit_cluster"] > 0)
    #             num_misses = (_df.loc[rounds_1_and_2, "hit_cluster"] > 0).sum()
    #             if num_misses == 0:
    #                 print(color("n={} overflow index={} num misses={}".format(
    #                     n, overflow_index, num_misses), fg="yellow"))
    #             elif num_misses != 1:
    #                 raise ValueError("more than one miss in round 1 and 2")
    #         
    #
    #     if False:
    #         patterns = dict()
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             if (n, overflow_index) not in patterns:
    #                 patterns[(n, overflow_index)] = defaultdict(int)
    #
    #             misses = tuple(_df.loc[_df["hit_cluster"] != 0, "index"].tolist())
    #             # print(misses)
    #             patterns[(n, overflow_index)][misses] += 1
    #
    #         # pprint(patterns)
    #         # get most common patterns
    #         most_common_patterns = {
    #             k: sorted(v.items(), key=lambda x: x[1])[-1] for k, v in patterns.items()}
    #         # pprint(most_common_patterns)
    #
    #         new_combined = []
    #         for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
    #             misses = tuple(_df.loc[_df["hit_cluster"] != 0, "index"].tolist())
    #             if most_common_patterns[(n, overflow_index)][0] == misses:
    #                 # print("adding", (n, overflow_index, r))
    #                 mask = combined["n"] == n
    #                 mask &= combined["overflow_index"] == overflow_index
    #                 mask &= combined["r"] == r
    #                 assert len(combined[mask]) == len(unique_ks)
    #                 new_combined.append(combined[mask])
    #                 most_common_patterns[(n, overflow_index)] = (None, None)
    #
    #         combined = pd.concat(new_combined, ignore_index=True)
    #         assert (
    #             len(combined[["n", "overflow_index"]].drop_duplicates())
    #             == len(combined[["n", "overflow_index", "r"]].drop_duplicates()))
    #
    #         combined["r"] = 0
    #         repetitions = 1

    total_sets = OrderedDict()
    # combined = combined.sort_values(["n", "overflow_index", "k", "r"])
    combined = combined.sort_values(["n", "overflow_index", "r", "k"])
    # print("combined")
    # print(combined)
    print("combined", combined.shape)
    if average:
        repetitions = 1
        # max_rounds = 1

    print(
        "overflowed {} unique indices".format(len(combined["overflow_index"].unique()))
    )

    base_addr = int(combined["virt_addr"].min())
    combined["rel_virt_addr"] = combined["virt_addr"].astype(int) - base_addr

    # compute plot matrices
    max_cols = int(len(combined["k"].astype(int).unique())) # + 1
    max_indices = int(len(combined["index"].astype(int).unique()))
    # assert max_indices <= max_cols
    print("round size", round_size)
    # max_rows = len(combined["overflow_index"].astype(int).unique()) * repetitions
    repetitions = len(combined["r"].unique())
    print("repetitions", repetitions)
    max_rows = len(
        combined[["overflow_index", "r"]].astype(int).drop_duplicates()
    )

    latencies = np.zeros((max_rows, max_cols))
    hit_clusters = np.zeros((max_rows, max_cols))
    print("latencies", latencies.shape)
    

    for (overflow_addr_index, r), overflow_df in combined.groupby(
        ["overflow_index", "r"]
    ):
        assert (
            overflow_df.sort_values(["n", "k"])["index"]
            == overflow_df.sort_values(["n", "k", "index"])["index"]
        ).all()
        overflow_df = overflow_df.sort_values(["n", "k", "index"])
        # overflow_df = overflow_df.sort_values(["n", "index", "k"])
        stride = stride_bytes / 4
        overflow_index = overflow_addr_index / stride
        # print(overflow_addr_index, overflow_index, r)
        # print(max_cols)
        print(overflow_df.shape)
        assert max_cols == len(overflow_df)

        second_round_df = overflow_df.iloc[round_size:,:].copy()
        second_round_df = second_round_df[second_round_df["hit_cluster"] > hit_cluster]
        print(second_round_df.iloc[:5,:])

        row_idx = int(overflow_index * repetitions + r)
        if False:
            latencies[row_idx, :len(overflow_df)] = overflow_df["latency"].to_numpy()
            hit_clusters[row_idx, :len(overflow_df)] = overflow_df["hit_cluster"].to_numpy()
            continue

        # for _, (n, k, index, latency, hit_cluster) in second_round_df.iloc[:5,:][["n", "k", "index", "latency", "hit_cluster"]].iterrows():
        #     print(index, k)
        #     index = int(np.floor(index / stride_bytes))
        #     round = int(np.floor(k / round_size))
        #     col_idx = round * round_size + index
        #     print("=>", round, index)
        #
        # continue
        for _, (n, k, index, latency, hit_cluster) in overflow_df[
                ["n", "k", "index", "latency", "hit_cluster"]].iterrows():
            # print(index, k)
            index = int(np.floor(index / stride_bytes))
            round = int(np.floor(k / round_size))
            col_idx = round * round_size + index
            # print("=>", round, index)
            assert(col_idx < max_cols)
            # if col_idx < max_cols:
            if True:
                latencies[row_idx, col_idx] = latency
                hit_clusters[row_idx, col_idx] = hit_cluster

    print(latencies)

    all_misses = np.all(hit_clusters > hit_cluster, axis=0)
    (all_misses_max_idx,) = np.where(all_misses == True)
    all_misses_max_idx = (
        np.amax(all_misses_max_idx) if len(all_misses_max_idx) > 0 else -1
    )
    all_misses_max_addr = all_misses_max_idx * stride_bytes
    print(
        "all miss max idx={} max addr={}".format(
            all_misses_max_idx, all_misses_max_addr
        )
    )

    # return

    col_miss_count = hit_clusters.sum(axis=0)
    assert col_miss_count.shape[0] == max_cols
    # print(col_miss_count)

    # all_hits = np.all(hit_clusters == 0, axis=0)
    # (all_hits_indices,) = np.where(all_hits == True)
    # print(all_hits_indices)

    # shifted plot
    # if False:
    #     num_overflow_indices = len(combined["overflow_index"].unique())
    #     # num_overflow_indices = int(known_cache_size_bytes / stride_bytes)
    #     # print(num_overflow_indices)
    #     # assert(num_overflow_indices == len(combined["overflow_index"].unique()))
    #     # assert(num_overflow_indices == int(max_rows / repetitions))
    #     all_latencies = latencies.copy()
    #     all_latencies[:,round_size:] = 0.0
    #     shift = "left"
    #     for overflow_index in range(num_overflow_indices):
    #         row_start = overflow_index * repetitions
    #         row_end = (overflow_index+1)*repetitions
    #         if shift == "right":
    #             new_col_start = round_size + overflow_index
    #             new_col_end = max_cols
    #             col_start = round_size
    #             col_end = min(max_cols, col_start + (new_col_end - new_col_start))
    #         elif shift == "left":
    #             pass
    #             col_start = round_size + round_size
    #             col_end = max_cols
    #             new_col_start = round_size
    #             new_col_end = min(max_cols, new_col_start + (col_end - col_start))
    #         else:
    #             raise ValueError
    #         all_latencies[row_start:row_end,new_col_start:new_col_end] = latencies[row_start:row_end,col_start:col_end]
    #
    #     fig = plot_access_process_latencies(
    #         combined,
    #         all_latencies,
    #         warmup=warmup,
    #         rounds=max_rounds,
    #         ylabel="overflow index",
    #         size_bytes=known_cache_size_bytes,
    #         stride_bytes=stride_bytes,
    #     )
    #
    #     filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    #     filename = filename.with_stem(filename.name + "_shifted")
    #     filename.parent.mkdir(parents=True, exist_ok=True)
    #     print("saved plot to {}".format(filename))
    #     fig.savefig(filename)

    fig = plot_access_process_latencies(
        combined,
        latencies,
        warmup=warmup,
        rounds=max_rounds,
        # ylabel="overflow index",
        size_bytes=known_cache_size_bytes,
        stride_bytes=stride_bytes,
    )

    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    print("saved plot to {}".format(filename))
    fig.savefig(filename)

    return

    # per overflow plot
    if False:
        for (overflow_addr_index,), _ in combined.groupby(["overflow_index"]):
            stride = float(stride_bytes) / 4.0
            overflow_index = float(overflow_addr_index) / stride
            overflow_latencies = latencies[
                int(overflow_index * repetitions):int((overflow_index + 1) * repetitions), :]
            fig = plot_access_process_latencies(
                combined,
                overflow_latencies ,
                warmup=warmup,
                rounds=max_rounds,
                ylabel="repetition",
                size_bytes=known_cache_size_bytes,
                stride_bytes=stride_bytes,
            )

            filename = PLOT_DIR / "pchase_overflow" / cache_file.relative_to(CACHE_DIR)
            filename = filename.with_name("overflow_{}".format(int(overflow_index))).with_suffix(".pdf")
            filename.parent.mkdir(parents=True, exist_ok=True)
            print("saved plot to {}".format(filename))
            fig.savefig(filename)

    if False:
        # group by first miss in round 1 and 2
        print(combined.loc[
            (combined["k"] >= 1 * round_size) & (combined["k"] < 3 * round_size),:].head(n=100))

        # print(stride_bytes / 4)
        # total_sets = {k * 4: list() for k in range(1 * round_size, 3 * round_size, int(stride_bytes / 4))}
        total_sets = {int(k): list() for k in range(0, known_cache_size_bytes, stride_bytes)}
        # print(total_sets)
        # before = len(total_sets)
        if random:
            for (n, overflow_index, r), _df in combined.groupby(["n", "overflow_index", "r"]):
                rounds_3 = _df["k"] >= 3 * round_size

                rounds_1_and_2 = _df["k"] >= 1 * round_size
                rounds_1_and_2 &= _df["k"] < 3 * round_size
                first_miss = _df.loc[rounds_1_and_2 & (_df["hit_cluster"] > 0), "index"]
                assert len(first_miss) <= 1
                if len(first_miss) > 0:
                    first_miss = int(first_miss.iloc[0])
                    misses = tuple(_df.loc[rounds_3 & (_df["hit_cluster"] > 0), "index"].tolist())
                    total_sets[first_miss].append(misses)

        # pprint(total_sets)
        # assert before == len(total_sets)

        total_sets_df = pd.DataFrame.from_records(
            [(k, len(v)) for k, v in total_sets.items()], columns=["index", "count"])
        print(total_sets_df)

        fig = plt.figure(
            figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
            layout="constrained",
        )
        ax = plt.axes()
        ax.scatter(total_sets_df["index"], total_sets_df["count"], 10, marker="o")
        filename = PLOT_DIR / "set_freqs.pdf"
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename)

        total_sets_df["prob"] = total_sets_df["count"] / len(total_sets_df)
        print(total_sets_df)

    return

    #
    #
    #
    #
    #

    if not average:
        # remove first round of loading
        # combined = combined[combined["k"] > all_misses_max_idx]
        combined = combined[combined["k"] >= known_cache_size_bytes / stride_bytes]

    if average:
        for overflow_index, _ in combined.groupby("overflow_index"):
            mask = combined["overflow_index"] == overflow_index
            misses = mask & (combined["hit_cluster"] > 0)
            miss_rate = float(misses.sum()) / float(mask.sum()) * 100.0
            # print(combined.loc[misses,:].head(n=2*derived_num_ways))

            if misses.sum() != derived_num_ways:
                print(
                    color(
                        "overflow index={:>5} % 128 = {:<3} miss rate={:<4.2f} has {:<2} misses (expected {} ways)".format(
                            int(overflow_index),
                            int(overflow_index) % 128,
                            miss_rate,
                            misses.sum(),
                            derived_num_ways,
                        ),
                        fg="red",
                    )
                )

            # ROMAN: changed this to be unique
            key = tuple(combined.loc[misses, "virt_addr"].astype(int).unique())
            if key not in total_sets:
                total_sets[key] = 0
            total_sets[key] += 1
    else:
        for (overflow_index, r), _ in combined.groupby(["overflow_index", "r"]):
            mask = combined["overflow_index"] == overflow_index
            mask &= combined["r"] == r
            misses = mask & (combined["hit_cluster"] > 0)
            miss_rate = float(misses.sum()) / float(mask.sum()) * 100.0
            # if misses.sum() > 0:
            #     print(combined[mask])
            # print(combined.loc[misses,:].head(n=2*derived_num_ways))
            # rel_combined = combined[mask].copy()
            # rel_combined["virt_addr"] -= base_addr
            # print(rel_combined[:10])
            # print(rel_combined[-10:])

            if misses.sum() != derived_num_ways:
                print(
                    color(
                        "[r={:<3}] overflow index={:>5} % 128 = {:<3} miss rate={:<4.2f}% has {:<2} misses (expected {} ways)".format(
                            int(r),
                            int(overflow_index),
                            int(overflow_index) % 128,
                            miss_rate,
                            misses.sum(),
                            derived_num_ways,
                        ),
                        fg="red",
                    )
                )

            # key = tuple(combined.loc[misses, "virt_addr"].astype(int).to_numpy())
            missed_addresses = combined.loc[misses, "virt_addr"].astype(int).tolist()
            if unique:
                missed_addresses = list(np.unique(missed_addresses))
            if sort:
                missed_addresses = sorted(missed_addresses)
            key = tuple(missed_addresses)
            if key not in total_sets:
                total_sets[key] = 0
            total_sets[key] += 1

    

    if len(total_sets) <= 1:
        print(
            color("have {} set(s), try increasing N".format(len(total_sets)), fg="red")
        )
        return

    num_sets = len(total_sets)
    if num_sets > 1:
        print(num_sets)
        num_sets_log2 = int(np.log2(num_sets))
        print(
            color(
                "total sets={:<2} ({:<2} bits)".format(num_sets, num_sets_log2),
                fg="green" if num_sets == known_num_sets else "red",
            )
        )

        total_sets = sorted([(list(s), occurences) for s, occurences in total_sets.items()])

        expanded_sets = []
        for set_id, (set_addresses, occurences) in enumerate(total_sets):
            set_addresses = [addr - base_addr for addr in set_addresses]
            print(
                "===> cache set {:>3}: {:>4} addresses ( observed {:>3}x )\n{}".format(
                    set_id, len(set_addresses), occurences, set_addresses
                )
            )

            found = False
            for si in range(len(expanded_sets)):
                intersection_size = len(expanded_sets[si].intersection(set(set_addresses)))
                union_size = len(expanded_sets[si].union(set(set_addresses)))
                # intersects = intersection_size / union_size > 0.8
                # intersects = len(set(set_addresses)) - intersection_size <= 4
                intersects = (
                    union_size / intersection_size > 0.5 if intersection_size > 0 else False
                )
                if intersects:
                    expanded_sets[si] = expanded_sets[si].union(set(set_addresses))
                    found = True
                    break
            if not found:
                expanded_sets.append(set(set_addresses))

        print("expanded sets: {}".format(len(expanded_sets)))
        expanded_sets = sorted([sorted(list(s)) for s in expanded_sets])
        for si, s in enumerate(expanded_sets):
            print("===> expanded set {}:\n{}".format(si, s))

        largest_set = np.amax([len(s) for s, _ in total_sets])
        print("largest set={}".format(largest_set))

        # # print(combined["latency"].value_counts())
        # # print(combined["hit_cluster"].unique())
        # # min_latency = combined["latency"].min()
        # # max_latency = combined["latency"].max()
        #
        # mean_cluster_latency = combined.groupby("hit_cluster")["latency"].mean()
        # min_cluster_latency = combined.groupby("hit_cluster")["latency"].min()
        # max_cluster_latency = combined.groupby("hit_cluster")["latency"].max()
        # max_latency = mean_cluster_latency[1] + 100
        # min_latency = np.max([0, mean_cluster_latency[0] - 100])
        #
        # ylabel = r"overflow index"
        # xlabel = r"cache access process"
        # fontsize = plot.FONT_SIZE_PT
        # font_family = "Helvetica"
        #
        # plt.rcParams.update({"font.size": fontsize, "font.family": font_family})
        #
        # fig = plt.figure(
        #     figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        #     layout="constrained",
        # )
        # ax = plt.axes()
        #
        # white = (255,255,255)
        # orange = (255,140,0)
        # red = (255,0,0)
        #
        # def rgb_to_vec(color: typing.Tuple[int, int, int], alpha:float = 1.0):
        #     return np.array([color[0]/255.0, color[1]/255.0, color[2]/255.0, alpha])
        #
        # def gradient(start: typing.Tuple[int, int, int], end: typing.Tuple[int, int, int], n=256):
        #     vals = np.ones((n, 4))
        #     vals[:, 0] = np.linspace(start[0]/255.0, end[0]/255.0, n)
        #     vals[:, 1] = np.linspace(start[1]/255.0, end[1]/255.0, n)
        #     vals[:, 2] = np.linspace(start[2]/255.0, end[2]/255.0,  n)
        #     return ListedColormap(vals)
        #
        # white_to_orange = gradient(start=white, end=orange)
        # orange_to_red = gradient(start=orange, end=red)
        #
        # latency_range = max_latency - min_latency
        # mean_hit_cluster_latency = mean_cluster_latency[0]
        # mean_miss_cluster_latency = mean_cluster_latency[1]
        # assert min_latency <= mean_hit_cluster_latency <= mean_miss_cluster_latency <= max_latency
        #
        # tol = 0.2
        # start = min_latency
        # hit_end = min_cluster_latency[1] - tol * abs(min_cluster_latency[1] - max_cluster_latency[0])
        # # hit_end = mean_hit_cluster_latency + tol * (mean_miss_cluster_latency - mean_hit_cluster_latency)
        # # miss_end = mean_miss_cluster_latency + tol * (max_latency - mean_miss_cluster_latency)
        # miss_end = np.min([mean_cluster_latency[1] + 100, max_latency])
        # end = miss_end
        #
        # points = [start, hit_end, miss_end, end]
        # widths = [points[i+1] - points[i] for i in range(len(points) - 1)]
        #
        # assert np.sum(widths) == latency_range
        #
        # latency_cmap = np.vstack([
        #     np.repeat(rgb_to_vec(white).reshape(1, 4), repeats=int(np.round(widths[0])), axis=0),
        #     white_to_orange(np.linspace(0, 1, int(np.round(0.5 * widths[1])))),
        #     orange_to_red(np.linspace(0, 1, int(np.round(0.5 * widths[1])))),
        #     np.repeat(rgb_to_vec(red).reshape(1, 4), repeats=int(np.round(widths[2])), axis=0),
        # ])
        #
        # assert np.allclose(len(latency_cmap), int(np.round(latency_range)), atol=2)
        # latency_cmap = ListedColormap(latency_cmap)
        #
        # if False:
        #     # c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        #     c = ax.pcolormesh(latencies, cmap='RdBu')
        #     # ax.set_title('pcolormesh')
        #     # set the limits of the plot to the limits of the data
        #     # ax.axis([x.min(), x.max(), y.min(), y.max()])
        #     fig.colorbar(c, ax=ax)
        # else:
        #     c = plt.imshow(latencies, cmap=latency_cmap, vmin=min_latency, vmax=max_latency,
        #                    interpolation='nearest',
        #                    origin='upper',
        #                    aspect='auto',
        #                    # aspect="equal",
        #     )
        #     fig.colorbar(c, ax=ax)
        #
        # if False:
        #     # plot all hit locations
        #     print(all_hits_indices)
        #     print(len(all_hits_indices))
        #     for hit_index in all_hits_indices:
        #         ax.axvline(
        #             x=hit_index,
        #             color="black", # plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
        #             linestyle="-",
        #             linewidth=1,
        #             # label=r"L1 hit",
        #         )
        #
        # ax.set_ylabel(ylabel)
        # ax.set_xlabel(xlabel)
        # # xticks = np.arange(min_x, max_x, step=256 * KB)
        # # xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
        # # ax.set_xticks(xticks, xticklabels, rotation=45)
        # # ax.set_xlim(min_x, max_x)
        # # ax.set_ylim(0, 100.0)
        # # ax.legend()

        
        # data = np.array([
        #     np.pad(s, pad_width=(largest_set - len(s), 0), mode='constant', constant_values=0)
        #     for s, _ in total_sets
        # ])
        # print(data.shape)
        #
        # diffs = []
        # for s1, _ in total_sets:
        #     for s2, _ in total_sets:
        #         union = np.union1d(s1, s2)
        #         intersection = np.intersect1d(s1, s2)
        #         diffs.append(len(union) - len(intersection))
        #
        # diffs = np.array(diffs)
        # print("pairwise diff: min={} max={} mean={}".format(np.amin(diffs), np.amax(diffs), np.mean(diffs)))

    return

    def custom_dist(a, b):
        # assert list(a) == sorted(list(a))
        # assert list(b) == sorted(list(b))
        # a = np.unique(a)
        # b = np.unique(b)
        # print("a", a.shape)
        # print("b", b.shape)
        # assert a.shape == b.shape
        union = np.union1d(a, b)
        intersection = np.intersect1d(a, b)
        # print("union", union.shape)
        # print("intersection", intersection.shape)

        union = len(union[union != 0])
        intersection = len(intersection[intersection != 0])
        if union == 0:
            union_intersect = 0
        else:
            union_intersect = intersection / union
        # print("match = {:<4.2f}%".format(union_intersect * 100.0))
        diff = len(a[a != 0]) - intersection
        # print(diff)
        # return diff
        return union_intersect

    cluster_labels = DBSCAN(
        # eps=0.05, # 5 percent difference
        eps=0.001,  # 5 percent difference
        min_samples=1,
        metric=custom_dist,
        # metric_params={'w1':1,'w2':5,'w3':4},
    ).fit_predict(data)
    # ).fit_predict(data.reshape(-1, 1))
    print(cluster_labels)

    if gpu == "A4000":
        print("todo")
        return
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
        bit_pos_used = sorted(
            [
                int(str(bit).removeprefix("~").removeprefix("b"))
                for bit in unique_bits(f)
            ]
        )
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
                    "xor_bit": sym.logic.boolalg.Or(
                        *[
                            sym.logic.boolalg.Xor(term, bit)
                            if contains_var(f.args[i], var=bit)
                            else term
                            for i, term in enumerate(ff.args)
                        ]
                    ),
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
                            bit: index_bits[pos]
                            for bit, pos in reversed(list(zip(bits_used, bit_pos_used)))
                        }
                        ref_pred = int(bool(f.subs(vars)))
                        sub_pred = int(bool(sub_f.subs(vars)))
                        assert ref_pred == target_bit
                        print(
                            np.binary_repr(addr, width=64),
                            vars,
                            color(
                                sub_pred,
                                fg="red" if sub_pred != target_bit else "green",
                            ),
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
@click.option(
    "--mem", "mem", default="l1data", type=str, help="memory to microbenchmark"
)
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--start", "start_cache_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_cache_size_bytes", type=int, help="end cache size in bytes")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="warmup iterations")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def find_cache_replacement_policy(
    mem, gpu, start_cache_size_bytes, end_cache_size_bytes, repetitions, warmup, cached, sim, force
):
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
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 4))
    warmup = warmup if warmup is not None else (1 if sim else 4)

    gpu = remote.find_gpu(gpu)

    known_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    known_cache_line_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)

    sector_size_bytes = 32
    known_num_sets = get_known_cache_num_sets(mem=mem, gpu=gpu)
    stride_bytes = known_cache_line_bytes

    print(
        "known cache size: {} bytes ({})".format(
            known_cache_size_bytes,
            humanize.naturalsize(known_cache_size_bytes, binary=True),
        )
    )
    print(
        "known cache line size: {} bytes ({})".format(
            known_cache_line_bytes,
            humanize.naturalsize(known_cache_line_bytes, binary=True),
        )
    )
    print("known num sets: {}".format(known_num_sets))

    # 48 ways
    # terminology: num ways == cache lines per set == associativity
    # terminology: way size == num sets
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

    step_size_bytes = known_cache_line_bytes
    start_cache_size_bytes = known_cache_size_bytes + 1 * step_size_bytes
    end_cache_size_bytes = known_cache_size_bytes + known_num_sets * step_size_bytes

    # known_cache_size_bytes = 256 * KB
    # known_cache_size_bytes = 128 * KB
    # known_cache_size_bytes = 64 * KB
    # known_num_sets = 8

    assert step_size_bytes % stride_bytes == 0
    assert start_cache_size_bytes % stride_bytes == 0
    assert end_cache_size_bytes % stride_bytes == 0

    cache_file = get_cache_file(
        prefix="cache_replacement_policy", mem=mem, sim=sim, gpu=gpu
    )
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_cache_size_bytes,
            end_size_bytes=end_cache_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            repetitions=repetitions,
            warmup=warmup,
            sim=sim,
            force=force,
        )
        print(stderr)

        # cannot average like this for non-LRU cache processes
        # combined = combined.drop(columns=["r"])
        # combined = (
        #     combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        # )

        combined["set"] = (combined["n"] % known_cache_size_bytes) // step_size_bytes

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

    # remove incomplete rounds
    combined = combined[~combined["round"].isna()]

    combined = compute_cache_lines(
        combined,
        cache_size_bytes=known_cache_size_bytes,
        sector_size_bytes=sector_size_bytes,
        cache_line_bytes=known_cache_line_bytes,
    )

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


def agg_miss_rate(hit_clusters, hit_cluster):
    cluster_counts = hit_clusters.value_counts().reset_index()
    num_misses = cluster_counts.loc[cluster_counts["hit_cluster"] > hit_cluster, "count"].sum()
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
@click.option(
    "--mem", "mem", type=str, default="l1data", help="memory to microbenchmark"
)
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--start", "start_cache_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_cache_size_bytes", type=int, help="end cache size in bytes")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def find_cache_sets(mem, gpu, start_cache_size_bytes, end_cache_size_bytes, cached, sim, repetitions, warmup, force):
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

    gpu = remote.find_gpu(gpu)

    known_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    known_cache_line_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)
    predicted_num_sets = get_known_cache_num_sets(mem=mem, gpu=gpu)

    stride_bytes = 32
    step_size_bytes = 32

    if start_cache_size_bytes is None:
        start_cache_size_bytes = known_cache_size_bytes - 2 * known_cache_line_bytes
    if end_cache_size_bytes is None:
        end_cache_size_bytes = (
            known_cache_size_bytes + (2 + predicted_num_sets) * known_cache_line_bytes
        )

    if gpu == remote.DAS6_GPU.A4000:
        stride_bytes = known_cache_line_bytes
        step_size_bytes = known_cache_line_bytes
        predicted_num_sets = 32

    match mem.lower():
        case "l1readonly":
            # stride_bytes = 16
            pass

    print(
        "known: cache size = {} bytes ({})\t line size = {} bytes ({})".format(
            known_cache_size_bytes,
            humanize.naturalsize(known_cache_size_bytes, binary=True),
            known_cache_line_bytes,
            humanize.naturalsize(known_cache_line_bytes, binary=True),
        )
    )
    print("predicted num sets: {}".format(predicted_num_sets))
    print(
        "range: {:>10} to {:<10} step size={} steps={}".format(
            humanize.naturalsize(start_cache_size_bytes, binary=True),
            humanize.naturalsize(end_cache_size_bytes, binary=True),
            step_size_bytes,
            (end_cache_size_bytes - start_cache_size_bytes) / step_size_bytes
        )
    )
    
    assert step_size_bytes % stride_bytes == 0
    assert start_cache_size_bytes % stride_bytes == 0
    assert end_cache_size_bytes % stride_bytes == 0

    cache_file = get_cache_file(prefix="cache_sets", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = pchase(
            mem=mem,
            gpu=gpu,
            start_size_bytes=start_cache_size_bytes,
            end_size_bytes=end_cache_size_bytes,
            step_size_bytes=step_size_bytes,
            stride_bytes=stride_bytes,
            max_rounds=1,
            warmup=warmup,
            repetitions=repetitions,
            sim=sim,
            force=force,
            stream_output=sim == True,
        )
        print(stderr)

        # cannot average like this for non-LRU cache processes
        # combined = combined.drop(columns=["r"])
        # combined = (
        #     combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        # )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

    hit_cluster = 1 if mem == "l2" else 0

    for n, df in combined.groupby("n"):
        # reindex the numeric index
        df = df.reset_index()
        assert df.index.start == 0

        # count hits and misses
        num_hits = (df["hit_cluster"] <= hit_cluster).sum()
        num_misses = (df["hit_cluster"] > hit_cluster).sum()
        num_l1_misses = (df["hit_cluster"] == 1).sum()
        num_l2_misses = (df["hit_cluster"] == 2).sum()

        hit_rate = float(num_hits) / float(len(df)) * 100.0
        miss_rate = float(num_misses) / float(len(df)) * 100.0

        # extract miss pattern
        miss_pattern = df.index[df["hit_cluster"] > hit_cluster].tolist()
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

    derived_num_sets, _misses_per_set = compute_number_of_sets(combined, hit_cluster=hit_cluster)

    plot_df = combined.groupby("n").agg(
        {"hit_cluster": [partial(agg_miss_rate, hit_cluster=hit_cluster)]}
    ).reset_index()
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

    # min_x = utils.round_down_to_multiple_of(plot_df["n"].min(), known_cache_line_bytes)
    # max_x = utils.round_up_to_multiple_of(plot_df["n"].max(), known_cache_line_bytes)
    # print(min_x, max_x)

    num_ticks = 6
    min_x = plot_df["n"].min()
    max_x = plot_df["n"].max()
    tick_step_size_bytes = utils.round_up_to_next_power_of_two((max_x - min_x) / num_ticks)

    xticks = np.arange(
        utils.round_down_to_multiple_of(min_x, tick_step_size_bytes),
        max_x,
        step=tick_step_size_bytes,
    )
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    print(xticklabels)

    cache_line_boundaries = np.arange(min_x, max_x, step=known_cache_line_bytes)

    for i, cache_line_boundary in enumerate(cache_line_boundaries):
        ax.axvline(
            x=cache_line_boundary,
            color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
            linestyle="--",
            label="cache line boundary" if i == 0 else None,
        )

    marker_size = 12
    ax.scatter(
        plot_df["n"],
        plot_df["hit_cluster_agg_miss_rate"] * 100.0,
        marker_size,
        # linewidth=1.5,
        # linestyle='--',
        marker="x",
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label=get_label(sim=sim, gpu=gpu),
    )

    ax.grid(
        True,
        axis="y",
        linestyle="-",
        linewidth=1,
        color="black",
        alpha=0.1,
        zorder=1,
    )

    for set_idx in range(derived_num_sets):
        x = known_cache_size_bytes + (set_idx + 0.5) * known_cache_line_bytes
        set_mask = (
            (plot_df["n"] % known_cache_size_bytes) // known_cache_line_bytes
        ) == set_idx
        y = plot_df.loc[set_mask, "hit_cluster_agg_miss_rate"].mean() * 100

        label = r"$S_{{{}}}$".format(set_idx)
        plt.text(x, 0.9 * y, label, ha="center", va="top")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # xticks = np.arange(known_cache_size_bytes, max_x, step=256)
    # xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(min_x, max_x)

    ylim = plot_df["hit_cluster_agg_miss_rate"].max() * 2 * 100.0 + 10.0
    ax.set_ylim(0, np.clip(ylim, 10.0, 110.0))

    ax.legend(loc="upper left")
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--start", "start_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_size_bytes", type=int, help="end cache size in bytes")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--rounds", "rounds", type=int, default=1, help="number of rounds")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def find_cache_line_size(
    mem,
    gpu,
    start_size_bytes,
    end_size_bytes,
    cached,
    sim,
    repetitions,
    warmup,
    rounds,
    force,
):
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
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 10))
    warmup = warmup if warmup is not None else 1

    gpu = remote.find_gpu(gpu)

    known_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    predicted_cache_line_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)
    predicted_num_lines = get_known_cache_num_sets(mem=mem, gpu=gpu)

    stride_bytes = 32

    match mem.lower():
        case "l1readonly":
            # stride_bytes = 8
            pass

    step_size_bytes = 32


    # if mem == "l2":
    #     step_size_bytes = 32
    # elif mem == "l1data":
    #     step_size_bytes = 32
    # else:
    #     raise NotImplementedError("mem {} not yet supported".format(mem))

    print(
        "known cache size: {} bytes ({})".format(
            known_cache_size_bytes,
            humanize.naturalsize(known_cache_size_bytes, binary=True),
        )
    )
    print(
        "predicted cache line size: {} bytes ({})".format(
            predicted_cache_line_bytes,
            humanize.naturalsize(predicted_cache_line_bytes, binary=True),
        )
    )
    print("predicted num lines: {}".format(predicted_num_lines))

    start_size_bytes = known_cache_size_bytes - 2 * predicted_cache_line_bytes
    end_size_bytes = (
        known_cache_size_bytes + (2 + predicted_num_lines) * predicted_cache_line_bytes
    )

    match (gpu, mem.lower()):
        case (remote.DAS6_GPU.A4000, "l2"):
            step_size_bytes = 8
            stride_bytes = 8

            start_size_bytes = 3 * MB
            predicted_num_lines = 1
            end_size_bytes = (
                start_size_bytes + (predicted_num_lines) * predicted_cache_line_bytes
            )
            # predicted_num_lines = 12
            repetitions = 10

        case (remote.DAS6_GPU.A4000, "l1data"):
            

            # start_size_bytes = known_cache_size_bytes - 2 * predicted_cache_line_bytes
            start_size_bytes = 16 * KB
            predicted_num_lines = 20
            # start_size_bytes = 4 * KB
            # end_size_bytes = 16 * KB
            end_size_bytes = (
                start_size_bytes + (2 + predicted_num_lines) * predicted_cache_line_bytes
            )
            # predicted_num_lines = 12
            # repetitions = 100
        case (remote.DAS5_GPU.TITANXPASCAL, "l2"):
            step_size_bytes = 8
            stride_bytes = 8

            start_size_bytes = 2 * MB
            predicted_num_lines = 4 
            # start_size_bytes = 4 * KB
            # end_size_bytes = 16 * KB
            end_size_bytes = (
                start_size_bytes + (2 + predicted_num_lines) * predicted_cache_line_bytes
            )

            # start_size_bytes = 1 * MB
            # end_size_bytes = 3 * MB
            # step_size_bytes = 256 * KB
            # repetitions = 10

    # temp
    # start_size_bytes = known_cache_size_bytes - 1 * predicted_cache_line_bytes
    # end_size_bytes = start_size_bytes
    # end_size_bytes = (
    #     known_cache_size_bytes + 1 * predicted_cache_line_bytes
    # )
    print(
        "range: {:>10} to {:<10} step size={} steps={}".format(
            humanize.naturalsize(start_size_bytes, binary=True),
            humanize.naturalsize(end_size_bytes, binary=True),
            step_size_bytes,
            (end_size_bytes - start_size_bytes) / step_size_bytes
        )
    )

    assert step_size_bytes % stride_bytes == 0
    assert start_size_bytes % stride_bytes == 0
    assert step_size_bytes % stride_bytes == 0

    cache_file = get_cache_file(prefix="cache_line_size", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION, on_bad_lines="warn",
        )
    else:
        combined, stderr = pchase(
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
            stream_output=sim == True,
        )
        print(stderr)

        # cannot average like this for non-LRU cache processes
        # combined = combined.drop(columns=["r"])
        # combined = (
        #     combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        # )

        # print("NAN values")
        # print(combined.loc[combined.isna().any(axis=1),:])
        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

    # num_unique_indices = len(combined["index"].unique())

    # # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index
    # print("have {: >3} rounds (full round size is {: <5})".format(len(full_rounds), full_round_size))
    #
    # combined = combined[combined["round"].isin(full_rounds)]
    # # combined = combined[combined["round"].isin([0, 1])]

    hit_cluster = 1 if mem == "l2" else 0

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
        num_hits = (df["hit_cluster"] <= hit_cluster).sum()
        num_misses = (df["hit_cluster"] > hit_cluster).sum()
        num_l1_misses = (df["hit_cluster"] == 1).sum()
        num_l2_misses = (df["hit_cluster"] == 2).sum()

        hit_rate = float(num_hits) / float(len(df)) * 100.0
        miss_rate = float(num_misses) / float(len(df)) * 100.0

        human_size = humanize.naturalsize(n, binary=True)
        print(
            "size={: >10} lsbs={: <4} hits={} ({}) l1 misses={} l2 misses={} (miss rate={})".format(
                human_size,
                int(n % 128),
                color("{: <5}".format(num_hits), fg="green", bold=True),
                color("{: >3.2f}%".format(hit_rate), fg="green"),
                color("{: <5}".format(num_l1_misses), fg="red", bold=True),
                color("{: <5}".format(num_l2_misses), fg="red", bold=True),
                color("{: >3.2f}%".format(miss_rate), fg="red"),
            )
        )

    plot_df = combined.groupby("n").agg({
        "hit_cluster": partial(agg_miss_rate, hit_cluster=hit_cluster)}).reset_index()
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

    num_ticks = 6
    min_x = plot_df["n"].min()
    max_x = plot_df["n"].max()
    tick_step_size_bytes = utils.round_up_to_next_power_of_two((max_x - min_x) / num_ticks)

    xticks = np.arange(
        utils.round_down_to_multiple_of(min_x, tick_step_size_bytes),
        max_x,
        step=tick_step_size_bytes,
    )
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    print(xticklabels)

    cache_line_boundaries = np.arange(
        utils.round_down_to_multiple_of(plot_df["n"].min(), predicted_cache_line_bytes),
        utils.round_up_to_multiple_of(plot_df["n"].max(), predicted_cache_line_bytes),
        step=predicted_cache_line_bytes,
    )

    for i, cache_line_boundary in enumerate(cache_line_boundaries):
        ax.axvline(
            x=cache_line_boundary,
            # color=plot.plt_rgba(*plot.RGB_COLOR["purple1"], 0.5),
            color="black",
            alpha=0.2,
            # linestyle="--",
            linestyle="-",
            zorder=2,
            label="cache line boundary" if i == 0 else None,
        )

    ax.grid(
        True,
        axis="y",
        # linestyle="-",
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.1,
        zorder=1,
    )

    marker_size = 12
    ax.scatter(
        plot_df["n"],
        plot_df["miss_rate"] * 100.0,
        marker_size,
        # linewidth=1.5,
        # linestyle='--',
        marker="x",
        zorder=3,
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label=get_label(sim=sim, gpu=gpu),
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, np.clip(plot_df["miss_rate"].max() * 2 * 100.0, 10.0, 100.0) + 10.0)

    ax.legend(loc="upper left")
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--start", "start_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_size_bytes", type=int, help="end cache size in bytes")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--rounds", "rounds", type=int, default=1, help="number of rounds")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--warmup", "warmup", type=int, help="number of warmup iterations")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def latency_n_graph(
    mem,
    gpu,
    start_size_bytes,
    end_size_bytes,
    cached,
    rounds,
    repetitions,
    warmup,
    sim,
    force,
):
    """
    Compute latency-N graph.

    This is not by itself sufficient to deduce cache parameters but our simulator should match
    this behaviour.
    """
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 10))
    warmup = warmup if warmup is not None else (1 if sim else 2)
    known_cache_line_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)
    known_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    known_cache_sets = get_known_cache_num_sets(mem=mem, gpu=gpu)

    known_cache_size_bytes = 100 * KB
    known_cache_sets = 64

    gpu = remote.find_gpu(gpu)

    stride_bytes = 16
    step_size_bytes = 32

    if gpu == "A4000":
        stride_bytes = 128
        step_size_bytes = 128

    match mem.lower():
        case "l1readonly":
            pass

    start_size_bytes = start_size_bytes or (
        known_cache_size_bytes - 2 * known_cache_line_bytes
    )
    end_size_bytes = end_size_bytes or (
        known_cache_size_bytes + (known_cache_sets + 2) * known_cache_line_bytes
    )

    assert step_size_bytes % stride_bytes == 0
    assert start_size_bytes % stride_bytes == 0

    cache_file = get_cache_file(prefix="latency_n_graph", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = pchase(
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

        # cannot average like this for non-LRU cache processes
        # combined = combined.drop(columns=["r"])
        # combined = (
        #     combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        # )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

    # remove incomplete rounds
    # print(combined)
    # print(combined["round"].value_counts())
    # print(combined["round"].value_counts())
    if rounds > 1:
        combined = combined[~combined["round"].isna()]

    # print(combined)
    # first_n_round_mask = (combined["n"] == start_size_bytes) & (combined["round"] == 0)
    valid_index = combined.loc[combined["n"] == start_size_bytes, "index"].max()
    valid_index = np.max([0, valid_index - 32])
    print("start size", start_size_bytes)
    print("valid index", valid_index)
    # print(combined.loc[first_n_round_mask, "n"].unique())
    # assert len(combined.loc[first_n_round_mask, "n"].unique()) == 1
    # print(len(combined[first_n_round_mask]))
    # print(combined[first_n_round_mask][-10:])
    # print(combined[combined["n"] == start_size_bytes][-10:])
    combined = combined[combined["index"] >= valid_index]
    # max_index = known_cache_size_bytes / stride_bytes
    # print(max_index)
    # return
    for n, n_df in combined.groupby("n"):
        print(n)
        print(n_df)

    # return

    # remove incomplete rounds
    # round_sizes = combined["round"].value_counts()
    # full_round_size = round_sizes.max()
    # full_rounds = round_sizes[round_sizes == full_round_size].index

    # have different sizes per N
    # iter_size = int(combined["n"].value_counts().mean())
    # assert (combined["n"].value_counts() == iter_size).all()

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

    num_ticks = 6
    min_x = mean_latency["n"].min()
    max_x = mean_latency["n"].max()
    tick_step_size_bytes = utils.round_down_to_next_power_of_two((max_x - min_x) / num_ticks)

    xticks = np.arange(
        utils.round_up_to_multiple_of(min_x, tick_step_size_bytes),
        max_x,
        step=tick_step_size_bytes,
    )
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]
    print(xticklabels)

    ylabel = r"mean latency"
    xlabel = r"$N$ (bytes)"
    fontsize = plot.FONT_SIZE_PT
    font_family = "Helvetica"

    plt.rcParams.update({"font.size": fontsize, "font.family": font_family})

    fig = plt.figure(
        figsize=(0.5 * plot.DINA4_WIDTH_INCHES, 0.2 * plot.DINA4_HEIGHT_INCHES),
        layout="constrained",
    )
    ax = plt.axes()
    ax.plot(
        mean_latency["n"],
        mean_latency["latency"],
        linewidth=1.5,
        linestyle="--",
        marker="o",
        color=plot.plt_rgba(*plot.RGB_COLOR["green1"], 1.0),
        label=get_label(sim=sim, gpu=gpu),
    )
    # min_y = mean_latency["latency"].min()
    max_y = mean_latency["latency"].max()
    # ax.set_ylim(np.max([0, int(0.75 * min_y)]), int(1.25 * max_y))
    ax.set_ylim(0, int(1.25 * max_y))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks, xticklabels)
    ax.legend()

    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
@click.option(
    "--skip-l1",
    "skip_l1",
    type=bool,
    default=True,
    help="collect l2 latency by skipping L1",
)
def plot_latency_distribution(mem, gpu, cached, sim, repetitions, force, skip_l1):
    repetitions = max(1, repetitions or 1)

    gpu = remote.find_gpu(gpu)

    # plot latency distribution
    cache_file = get_cache_file(
        prefix="latency_distribution", mem=mem, sim=sim, gpu=gpu
    )
    if cached and cache_file.is_file():
        # open cached files
        latencies = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        latencies = pd.DataFrame(
            collect_full_latency_distribution(
                sim=sim, gpu=gpu, skip_l1=skip_l1, force=force
            ),
            columns=["latency"],
        )

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        latencies.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

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
        label=get_label(sim=sim, gpu=gpu),
    )
    for centroid, label in zip(latency_centroids, ["L1 Hit", "L2 Hit", "L2 Miss"]):
        centroid_bins = latency_hist_df["bin_start"] <= centroid + 2 * bin_size
        centroid_bins &= centroid - 2 * bin_size <= latency_hist_df["bin_end"]
        y = latency_hist_df.loc[centroid_bins, "count"].max()
        ax.annotate(
            "{}\n({:3.1f})".format(label, centroid),
            xy=(centroid, y),
            ha="center",
            va="bottom",
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, latency_hist_df["count"].max() * 1.5)
    ax.legend()
    filename = (PLOT_DIR / cache_file.relative_to(CACHE_DIR)).with_suffix(".pdf")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


def get_cache_file(prefix, mem, sim, gpu: typing.Optional[enum.Enum], compute_capability=None, **kwargs) -> Path:
    kind = "sim" if sim else "native"
    if gpu is None:
        cache_file_name = "{}-{}-{}".format(prefix, mem, kind)
    else:
        cache_file_name = "{}/{}-{}-{}".format(str(gpu.value).replace(" ", "-"), prefix, mem, kind)
    if isinstance(compute_capability, int):
        cache_file_name += "-cc{}".format(compute_capability)

    for k,v in kwargs.items():
        if isinstance(v, bool) and v == True:
            cache_file_name += f"-{k}"
        else:
            cache_file_name += f"-{k}{v}"
    return (CACHE_DIR / cache_file_name).with_suffix(".csv")


@main.command()
@click.option("--mem", "mem", type=str, default="l1data", help="mem to microbenchmark")
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--start", "start_size_bytes", type=int, help="start cache size in bytes")
@click.option("--end", "end_size_bytes", type=int, help="end cache size in bytes")
@click.option("--sim", "sim", type=bool, is_flag=True, help="simulate")
@click.option("--cached", "cached", type=bool, is_flag=True, help="use cached data")
@click.option("--warmup", "warmup", type=int, default=1, help="warmup iterations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--max-rounds", "max_rounds", type=int, help="maximum number of rounds")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def find_cache_size(
    mem,
    gpu,
    start_size_bytes,
    end_size_bytes,
    sim,
    cached,
    warmup,
    repetitions,
    max_rounds,
    force,
):
    """
    Step 1.

    Determine cache size C. We set s to 1. We then initialize
    N with a small value and increase it gradually until
    the first cache miss appears. C equals the maximum N
    where all memory accesses are cache hits.
    """
    repetitions = max(1, repetitions if repetitions is not None else (1 if sim else 5))
    max_rounds = max(1, max_rounds if max_rounds is not None else 1)
    gpu = remote.find_gpu(gpu)

    predicted_cache_size_bytes = get_known_cache_size_bytes(mem=mem, gpu=gpu)
    stride_bytes = get_known_cache_line_bytes(mem=mem, gpu=gpu)
    
    # match mem.lower():
    #     case "l1readonly":
    #         stride_bytes = 16

    match (gpu, mem.lower()):
        case (_, "l2"):
            # stride_bytes = 32
            # stride_bytes = 128
            step_size_bytes = 256 * KB
        case (_, "l1data"):
            # stride_bytes = 8
            # stride_bytes = 128
            step_size_bytes = 1 * KB
        case other:
            raise ValueError("unsupported config {}".format(other))

    start_size_bytes = int(start_size_bytes or step_size_bytes)
    end_size_bytes = int(end_size_bytes or (
        utils.round_up_to_multiple_of(
            2 * predicted_cache_size_bytes, multiple_of=step_size_bytes
        ))
    )

    match (gpu, mem.lower()):
        case (remote.DAS6_GPU.A4000, "l1data"):
            # stride_bytes = 128
            end_size_bytes = 128 * KB

    start_size_bytes = max(0, start_size_bytes)
    end_size_bytes = max(0, end_size_bytes)

    assert start_size_bytes % stride_bytes == 0
    assert step_size_bytes % stride_bytes == 0

    print(
        "predicted cache size: {} bytes ({})".format(
            predicted_cache_size_bytes,
            humanize.naturalsize(predicted_cache_size_bytes, binary=True),
        )
    )
    print(
        "range: {:>10} to {:<10} step size={} steps={}".format(
            humanize.naturalsize(start_size_bytes, binary=True),
            humanize.naturalsize(end_size_bytes, binary=True),
            step_size_bytes,
            (end_size_bytes - start_size_bytes) / step_size_bytes,
        )
    )

    cache_file = get_cache_file(prefix="cache_size", mem=mem, sim=sim, gpu=gpu)
    if cached and cache_file.is_file():
        # open cached files
        combined = pd.read_csv(
            cache_file, header=0, index_col=None, compression=CSV_COMPRESSION
        )
    else:
        combined, stderr = pchase(
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
            stream_output=sim == True,
        )
        print(stderr)

        # cannot average like this for non-LRU cache processes
        # combined = combined.drop(columns=["r"])
        # combined = (
        #     combined.groupby(["n", "k", "index", "virt_addr"]).median().reset_index()
        # )

        combined = compute_hits(combined, sim=sim, gpu=gpu)
        combined = compute_rounds(combined)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print("wrote cache file to ", cache_file)
        combined.to_csv(cache_file, index=False, compression=CSV_COMPRESSION)

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

    plot_df = combined.groupby("n").agg({"hit_cluster": partial(agg_miss_rate, hit_cluster=hit_cluster)}).reset_index()
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
        label=get_label(sim=sim, gpu=gpu),
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    num_ticks = 8
    tick_step_size_bytes = utils.round_up_to_next_power_of_two(plot_df["n"].max() / num_ticks)
    min_x = utils.round_down_to_multiple_of(plot_df["n"].min(), tick_step_size_bytes)
    max_x = utils.round_up_to_multiple_of(plot_df["n"].max(), tick_step_size_bytes)

    xticks = np.arange(
        np.max([min_x, tick_step_size_bytes]), max_x, step=tick_step_size_bytes
    )
    xticklabels = [humanize.naturalsize(n, binary=True) for n in xticks]

    ax.grid(
        True,
        axis="y",
        linestyle="-",
        linewidth=1,
        color="black",
        alpha=0.1,
        zorder=1,
    )

    ax.set_xticks(xticks, xticklabels, rotation=45)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0, 110.0)
    ax.legend(loc="upper left")

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
@click.option(
    "--gpu", "gpu", type=str, help="the remote gpu device to run the microbenchmark"
)
@click.option("--warmup", type=int, help="number of warmup interations")
@click.option("--repetitions", "repetitions", type=int, help="number of repetitions")
@click.option("--rounds", "rounds", type=int, help="number of rounds")
@click.option("--size", type=int, help="size in bytes")
@click.option("--stride", type=int, help="stride in bytes")
@click.option("--verbose", type=bool, is_flag=True, help="verbose output")
@click.option("--sim", type=bool, is_flag=True, help="use simulator")
@click.option(
    "--force", "force", type=bool, is_flag=True, help="force re-running experiments"
)
def run(mem, gpu, warmup, repetitions, rounds, size, stride, verbose, sim, force):
    gpu = remote.find_gpu(gpu)
    repetitions = repetitions or 1
    warmup = warmup or 1
    stride = stride or 32
    size = size or 24 * KB
    df, stderr = pchase(
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
        print(stderr)
    print(df)


if __name__ == "__main__":
    main()
