import pandas as pd
from pathlib import Path
from os import PathLike

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common
import gpucachesim.stats.accelsim as accelsim


class Stats(accelsim.Stats):
    def __init__(self, config: GPUConfig, global_bench_config: BenchConfig) -> None:
        self.bench_config = global_bench_config["playground_simulate"]
        self.path = Path(self.bench_config["stats_dir"])
        self.use_duration = False
        self.config = config
        self.repetitions = int(self.bench_config["repetitions"])
        self.load_converted_stats()
        self.load_raw_stats()
