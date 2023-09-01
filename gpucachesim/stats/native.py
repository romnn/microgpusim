from os import PathLike
from pathlib import Path
import typing
import json
import pandas as pd
from pprint import pprint

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common


class Stats(common.Stats):
    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["profile"]["profile_dir"])
        with open(self.path / "profile.commands.json", "rb") as f:
            self.commands = json.load(f)
        self.metrics = pd.read_json(self.path / "profile.metrics.json")
        self.use_duration = False
        self.config = config

    def duration_us(self):
        if "Duration" in self.metrics:
            # convert us to us (1e-6)
            # duration already us
            return self.metrics["Duration"].sum()
        elif "gpu__time_duration.sum_nsecond" in self.metrics:
            # convert ns to us
            return self.metrics["gpu__time_duration.sum_nsecond"].sum() * 1e-3
        else:
            raise ValueError("missing duration")

    def cycles(self) -> int:
        if self.use_duration:
            # clock speed is mhz, so *1e6
            # duration is us, so *1e-6
            # unit conversions cancel each other out
            duration = self.hw_duration_us()
            return duration * self.config["clock_speed"]
        else:
            # sm_efficiency: The percentage of time at least one warp
            # is active on a specific multiprocessor
            # mean_sm_efficiency = self.metrics["sm_efficiency"].mean() / 100.0
            # num_active_sm = self.data.config.spec["sm_count"] * mean_sm_efficiency
            # print("num active sms", num_active_sm)

            # nsight_col = "sm__cycles_elapsed.sum_cycle"
            nsight_col = "gpc__cycles_elapsed.avg_cycle"
            # nsight_col = "sm__cycles_active.avg_cycle"
            pprint(list(self.metrics.columns.tolist()))
            if "elapsed_cycles_sm" in self.metrics:
                sm_count = self.config["sm_count"]
                cycles = self.metrics["elapsed_cycles_sm"].sum()
                return cycles / sm_count
            elif nsight_col in self.metrics:
                return self.metrics[nsight_col].sum()
            else:
                raise ValueError("hw dataframe missing cycles")
                # hw_value *= mean_sm_efficiency
