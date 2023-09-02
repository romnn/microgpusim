from os import PathLike
from pathlib import Path
import typing
import json
import pandas as pd
from pprint import pprint

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common


class Stats(common.Stats):
    bench_config: BenchConfig
    config: GPUConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["profile"]["profile_dir"])
        with open(self.path / "profile.commands.json", "rb") as f:
            self.commands = json.load(f)
        self.df = pd.read_json(self.path / "profile.metrics.json")
        self.use_duration = False
        self.bench_config = bench_config
        self.config = config

    def duration_us(self):
        if "Duration" in self.df:
            # convert us to us (1e-6)
            # duration already us
            return self.df["Duration"].sum()
        elif "gpu__time_duration.sum_nsecond" in self.df:
            # convert ns to us
            return self.df["gpu__time_duration.sum_nsecond"].sum() * 1e-3
        else:
            raise ValueError("missing duration")

    def cycles(self) -> int:
        if self.use_duration:
            # clock speed is mhz, so *1e6
            # duration is us, so *1e-6
            # unit conversions cancel each other out
            duration = self.hw_duration_us()
            return duration * self.config.core_clock_speed
        else:
            # sm_efficiency: The percentage of time at least one warp
            # is active on a specific multiprocessor
            # mean_sm_efficiency = self.df["sm_efficiency"].mean() / 100.0
            # num_active_sm = self.data.config.spec["sm_count"] * mean_sm_efficiency
            # print("num active sms", num_active_sm)

            # nsight_col = "sm__cycles_elapsed.sum_cycle"
            nsight_col = "gpc__cycles_elapsed.avg_cycle"
            # nsight_col = "sm__cycles_active.avg_cycle"
            # pprint(list(self.df.columns.tolist()))
            if "elapsed_cycles_sm" in self.df:
                sm_count = self.config.num_total_cores
                # sm_count = self.config.num_clusters
                # print(self.df["elapsed_cycles_sm"]["value"])
                cycles = self.df["elapsed_cycles_sm"].sum()
                # this only holds until we have repetitions
                assert (cycles == self.df["elapsed_cycles_sm"]["value"]).all()
                return int(cycles / sm_count)
            elif nsight_col in self.df:
                return self.df[nsight_col].sum()
            else:
                raise ValueError("hw dataframe missing cycles")
                # hw_value *= mean_sm_efficiency

    def instructions(self):
        if "inst_issued" in self.df:
            # there is also inst_executed
            return self.df["inst_issued"].sum() * 20  # * self.config.num_total_cores
        elif "smsp__inst_executed.sum_inst" in self.df:
            # there is also sm__inst_executed.sum_inst
            # sm__sass_thread_inst_executed.sum_inst
            return self.df["smsp__inst_executed.sum_inst"].sum()
        else:
            raise ValueError("missing instructions")
