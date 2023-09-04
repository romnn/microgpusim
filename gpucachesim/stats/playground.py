import pandas as pd
from pathlib import Path
from os import PathLike

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common
import gpucachesim.stats.accelsim as accelsim


class Stats(accelsim.Stats):
    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["playground_simulate"]["stats_dir"])
        self.use_duration = False
        self.config = config
        self.load_bench_config(bench_config)
