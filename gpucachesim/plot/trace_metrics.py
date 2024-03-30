import click
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import json
import yaml
from wasabi import color
from pprint import pprint
from pathlib import Path
from collections import defaultdict

import gpucachesim.plot as plot
import gpucachesim.utils as utils
import gpucachesim.benchmarks as benchmarks
import gpucachesim.stats
import gpucachesim.stats.native

from gpucachesim.benchmarks import (
    Target,
    Benchmarks,
    GPUConfig,
    BenchConfig,
    ProfileConfig,
    ProfileTargetConfig,
    SimulateConfig,
    SimulateTargetConfig,
    TraceConfig,
    TraceTargetConfig,
    REPO_ROOT_DIR,
    DEFAULT_BENCH_FILE,
)

np.set_printoptions(suppress=True, formatter={"float_kind": "{:f}".format})
pd.options.display.float_format = "{:.10f}".format


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


@main.command()
# @click.option("--path", help="Path to materialized benchmark config")
# @click.option("--bench", "bench_name", help="Benchmark name")
# @click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
def trace_formats():
    stats_file = REPO_ROOT_DIR / "results/trace-metrics.csv"
    df = pd.read_csv(stats_file, header=0)
    df["MB"] = df["num_bytes"].astype(float) / float(1024**2)
    df["KI"] = df["num_instructions"].astype(float) / 1000.0
    df["KI/sec"] = df["KI"].astype(float) / df["deserialization_time_sec"].astype(float)
    df["MB/sec"] = df["MB"].astype(float) / df["deserialization_time_sec"].astype(float)
    df["MB/KI"] = df["MB"].astype(float) / df["KI"].astype(float)

    print(
        df[
            [
                "format",
                "num_instructions",
                "deserialization_time_sec",
                "KI",
                "MB",
                "KI/sec",
                "MB/sec",
                "MB/KI",
            ]
        ]
    )
    print(df.groupby("format")[["KI/sec", "MB/sec", "MB/KI"]].mean())

    table = r" & Size & \multicolumn{2}{c}{Deserialization speed} \\"
    table += "\n"
    table += r" & MiB/KI & MiB/sec & KI/sec \\ \hline"
    table += "\n"
    for row_idx, (format, per_format) in enumerate(df.groupby("format")):
        # print(format)
        # print(per_format["MB/KI"].values)
        ki_per_sec = per_format["KI/sec"].mean()
        mb_per_sec = per_format["MB/sec"].mean()
        mb_per_ki = per_format["MB/KI"].mean()
        if row_idx % 2 == 0:
            table += r"\rowcolor{gray!10}"
        table += (
            str(format)
            + " & "
            + "{:>4.2f}".format(mb_per_ki)
            + " & "
            + "{:>5.2f}".format(mb_per_sec)
            + " & "
            + "{:>6.2f}".format(ki_per_sec)
            + r" \\"
        )
        table += "\n"

    print(table)


@main.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", "bench_name_arg", help="Benchmark name")
@click.option(
    "--config",
    "config_path",
    default=gpucachesim.stats.DEFAULT_CONFIG_FILE,
    help="Path to GPU config",
)
@click.option("-v", "--verbose", "verbose", type=bool, is_flag=True, help="enable verbose output")

# @click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
def trace_overhead(path, bench_name_arg, config_path, verbose):
    b = Benchmarks(path)
    # results_dir = Path(b.config["results_dir"])

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    targets = [
        (Target.Trace, Target.Profile),
        (Target.AccelsimTrace, Target.Profile),
        (Target.ExecDrivenSimulate, Target.Profile),
    ]

    for trace_target, profile_target in targets:
        if bench_name_arg is None:
            bench_names = b.benchmarks[trace_target.value].keys()
        elif isinstance(bench_name_arg, str):
            bench_names = [bench_name_arg]
        elif isinstance(bench_name_arg, list):
            bench_names = bench_name_arg
        else:
            raise ValueError

        benches = defaultdict(list)
        for bench_name in bench_names:
            trace_benches = b.benchmarks[trace_target.value][bench_name]
            profile_benches = b.benchmarks[profile_target.value][bench_name]
            benches[bench_name].extend(zip(trace_benches, profile_benches))

        all_stats = []
        for bench_name, bench_configs in benches.items():
            for trace_bench_config, profile_bench_config in bench_configs:
                name = trace_bench_config["name"]
                target = trace_bench_config["target"]
                input_idx = trace_bench_config["input_idx"]
                input_values = trace_bench_config["values"]

                if verbose:
                    print(" ===> {:>20} {:>15}@{:<4} {}".format(target, name, input_idx, input_values))

                profile_stats = gpucachesim.stats.native.NvprofStats(config, profile_bench_config)

                grouped = profile_stats.result_df.groupby(["kernel_launch_id"])
                mean_time_per_kernel_launch = grouped["exec_time_sec"].mean()
                # print(profile_stats.result_df["exec_time_sec"])
                # print(mean_time_per_kernel_launch)

                profile_exec_time_sec = mean_time_per_kernel_launch.sum()

                if trace_target == Target.ExecDrivenSimulate:
                    trace_bench_config: BenchConfig[SimulateTargetConfig] = trace_bench_config
                    trace_target_config: SimulateConfig = trace_bench_config["target_config"].value
                    stats_dir = Path(trace_target_config["stats_dir"])
                    # print(trace_dir)
                    # pprint(list(trace_dir.iterdir()))

                    with open(stats_dir / "trace_reconstruction_time.json", "r") as f:
                        # trace time is in millis
                        trace_exec_time_sec = float(json.load(f)) / 1_000.0

                else:
                    trace_bench_config: BenchConfig[TraceTargetConfig] = trace_bench_config
                    trace_target_config: TraceConfig = trace_bench_config["target_config"].value
                    trace_dir = Path(trace_target_config["traces_dir"])
                    # print(trace_dir)
                    # pprint(list(trace_dir.iterdir()))

                    with open(trace_dir / "trace_time.json", "r") as f:
                        # trace time is in millis
                        trace_exec_time_sec = float(json.load(f)) / 1_000.0

                slowdown = trace_exec_time_sec / profile_exec_time_sec
                if verbose:
                    print(
                        "overhead: {:>6.3f} (trace) vs {:>12.9f} (native) sec => slowdown factor {:>12}\t\t{:>20.3f}x".format(
                            trace_exec_time_sec,
                            profile_exec_time_sec,
                            plot.human_format_thousands(slowdown),
                            slowdown,
                        )
                    )

                values = pd.DataFrame.from_records([trace_bench_config["values"]])
                values.columns = ["input_" + c for c in values.columns]

                # this will be the new index
                values["target"] = trace_target.value
                values["benchmark"] = name
                values["input_id"] = input_idx

                exec_time_df = pd.DataFrame.from_dict(
                    dict(
                        profile_exec_time_sec=[profile_exec_time_sec],
                        trace_exec_time_sec=[trace_exec_time_sec],
                    )
                )

                values = exec_time_df.merge(values, how="cross")
                # print(values.T)
                all_stats.append(values)

                # profile_bench_config: BenchConfig[ProfileTargetConfig] = profile_bench_config
                # profile_target_config: ProfileConfig = profile_bench_config["target_config"].value
                # profile_dir = Path(profile_target_config["profile_dir"])
                # print(profile_dir)
                # pprint(list(profile_dir.iterdir()))

        all_stats = pd.concat(all_stats)
        all_stats["overhead"] = all_stats["trace_exec_time_sec"] / all_stats["profile_exec_time_sec"]
        all_stats["overhead_str"] = all_stats["overhead"].apply(lambda x: plot.human_format_thousands(x))
        if verbose:
            print(all_stats)

        slowdown = all_stats["overhead"]
        mean_slowdown = np.mean(slowdown)
        max_slowdown = np.amax(slowdown)
        min_slowdown = np.amin(slowdown)
        print(
            "MIN: {:>12} = {:>30.6f}x".format(
                plot.human_format_thousands(min_slowdown),
                min_slowdown,
            )
        )
        print(
            "MAX: {:>12} = {:>30.6f}x".format(
                plot.human_format_thousands(max_slowdown),
                max_slowdown,
            )
        )
        print(
            color(
                "\n\t => {:>20} MEAN SLOWDOWN FACTOR {:>12} = {:>30.6f}x".format(
                    trace_target.value,
                    plot.human_format_thousands(mean_slowdown),
                    mean_slowdown,
                ),
                fg="red",
            )
        )


if __name__ == "__main__":
    main()
