import pandas as pd
from pathlib import Path
from os import PathLike

from gpucachesim.benchmarks import (
    GPUConfig,
    BenchConfig,
    PlaygroundSimulateConfig,
    PlaygroundSimulateTargetConfig,
)

import gpucachesim.stats.common as common
import gpucachesim.stats.accelsim as accelsim


class Stats(accelsim.Stats):
    bench_config: BenchConfig[PlaygroundSimulateTargetConfig]
    target_config: PlaygroundSimulateConfig

    # def __init__(
    #     self,
    #     config: GPUConfig,
    #     bench_config: BenchConfig[PlaygroundSimulateTargetConfig],
    # ) -> None:
    #     self.bench_config = bench_config
    #     self.target_config = self.bench_config["target_config"].value
    #
    #     self.path = Path(self.target_config["stats_dir"])
    #     self.use_duration = False
    #     self.config = config
    #     self.repetitions = int(self.bench_config["common"]["repetitions"])
    #     self.load_converted_stats()
    #     self.load_raw_stats()
    #
    #     self.compute_result_df()
