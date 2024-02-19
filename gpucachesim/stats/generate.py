import typing
import yaml
from os import PathLike
from pathlib import Path
from collections import defaultdict
from wasabi import color
import pandas as pd

import gpucachesim.stats.stats
import gpucachesim.stats.native
import gpucachesim.stats.accelsim
import gpucachesim.stats.playground

import gpucachesim.benchmarks as benchmarks
from gpucachesim.benchmarks import (
    Target,
    Benchmarks,
    GPUConfig,
    DEFAULT_BENCH_FILE,
)


def generate(
    path: PathLike,
    config_path: PathLike,
    bench_name: typing.Optional[str] = None,
    input_idx: typing.Optional[int] = None,
    quick=False,
    target: typing.Optional[str] = None,
    verbose=True,
    strict=True,
    nvprof=True,
    nsight=False,
    output_path: typing.Optional[PathLike] = None,
):
    b = Benchmarks(path)
    results_dir = Path(b.config["results_dir"])

    if target is not None:
        valid_targets = [t for t in Target if t.value.lower() == target.lower()]
    else:
        valid_targets = [
            Target.Profile,
            Target.Simulate,
            Target.ExecDrivenSimulate,
            Target.AccelsimSimulate,
            Target.PlaygroundSimulate,
        ]

    print("targets: {}".format([str(t.value) for t in valid_targets]))
    print("benchmarks: {}".format(bench_name))

    benches = defaultdict(list)
    for valid_target in valid_targets:
        if bench_name is None:
            valid_bench_names = b.benchmarks[valid_target.value].keys()
        elif isinstance(bench_name, str):
            valid_bench_names = [bench_name]
        elif isinstance(bench_name, list):
            valid_bench_names = bench_name
        else:
            raise ValueError

        for valid_bench_name in valid_bench_names:
            benches[valid_bench_name].extend(
                b.benchmarks[valid_target.value][valid_bench_name]
            )

    benches = dict(benches)

    print(
        "processing {} benchmark configurations ({} targets)".format(
            sum([len(b) for b in benches.values()]), len(valid_targets)
        )
    )

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    profilers = []
    if nvprof:
        profilers += ["nvprof"]
    if nsight:
        profilers += ["nsight"]

    for profiler in profilers:
        for valid_bench_name, bench_configs in benches.items():
            all_stats = []
            for bench_config in bench_configs:
                name = bench_config["name"]
                bench_target = bench_config["target"]
                input_idx = bench_config["input_idx"]
                input_values = bench_config["values"]
                target_name = f"[{bench_target}]"

                if quick:
                    if input_values.get("mode") not in ["serial", None]:
                        continue
                    # if input_values.get("memory_only") not in [False, None]:
                    #     continue
                    if input_values.get("cores_per_cluster") not in [
                        int(benchmarks.BASELINE["cores_per_cluster"]),
                        None,
                    ]:
                        continue
                    if input_values.get("num_clusters") not in [
                        int(benchmarks.BASELINE["num_clusters"]),
                        None,
                    ]:
                        continue

                current_bench_log_line = " ===> {:>20} {:>15}@{:<4} {}".format(
                    target_name, name, input_idx, input_values
                )

                try:
                    valid_bench_target = Target(bench_target)
                except Exception as e:
                    raise e

                try:
                    match (valid_bench_target, profiler):
                        case (Target.Profile, "nvprof"):
                            target_name += "[nvprof]"
                            bench_stats = gpucachesim.stats.native.NvprofStats(
                                config, bench_config
                            )
                        case (Target.Profile, "nsight"):
                            target_name += "[nsight]"
                            bench_stats = gpucachesim.stats.native.NsightStats(
                                config, bench_config
                            )
                        case (Target.Simulate, _):
                            bench_stats = gpucachesim.stats.stats.Stats(
                                config, bench_config
                            )
                        case (Target.ExecDrivenSimulate, _):
                            bench_stats = gpucachesim.stats.stats.ExecDrivenStats(
                                config, bench_config
                            )
                        case (Target.AccelsimSimulate, _):
                            bench_stats = gpucachesim.stats.accelsim.Stats(
                                config, bench_config
                            )
                        case (Target.PlaygroundSimulate, _):
                            bench_stats = gpucachesim.stats.playground.Stats(
                                config, bench_config
                            )
                        case other:
                            print(
                                color(
                                    f"WARNING: {name} has unknown target {other}",
                                    fg="red",
                                )
                            )
                            continue
                    print(current_bench_log_line)
                except Exception as e:
                    # allow babelstream for exec driven to be missing
                    if (valid_bench_target, name.lower()) == (
                        Target.ExecDrivenSimulate,
                        "babelstream",
                    ):
                        continue

                    print(color(current_bench_log_line, fg="red"))
                    if strict:
                        raise e
                    continue

                values = pd.DataFrame.from_records([bench_config["values"]])
                values.columns = ["input_" + c for c in values.columns]

                # this will be the new index
                values["target"] = bench_target
                values["benchmark"] = name
                values["input_id"] = input_idx

                values = bench_stats.result_df.merge(values, how="cross")
                assert "run" in values.columns
                if valid_bench_target != Target.Profile:
                    assert "is_release_build" in values.columns
                    if strict:
                        assert values["is_release_build"].all()

                if verbose:
                    print(values.T)
                all_stats.append(values)

            all_stats = pd.concat(all_stats)
            if verbose:
                print(all_stats)

            print(all_stats.value_counts(["target", "benchmark"]))
            stats_output_path = (
                results_dir / f"combined.stats.{profiler}.{valid_bench_name}.csv"
            )

            if output_path is not None:
                stats_output_path = Path(output_path)

            print(color(f"saving to {stats_output_path}", fg="cyan"))
            stats_output_path.parent.mkdir(parents=True, exist_ok=True)
            all_stats.to_csv(stats_output_path, index=False)
