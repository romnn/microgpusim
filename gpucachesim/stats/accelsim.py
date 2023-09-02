import pandas as pd
from pathlib import Path

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common


class Stats(common.Stats):
    bench_config: BenchConfig
    config: GPUConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["accelsim_simulate"]["stats_dir"])
        self.sim_df = pd.read_csv(
            self.path / "stats.sim.csv",
            header=0,
        )
        self.accesses_df = pd.read_csv(self.path / "stats.accesses.csv", header=None, names=["access", "count"])
        self.dram_df = pd.read_csv(
            self.path / "stats.dram.csv",
            header=0,
        )
        self.dram_banks_df = pd.read_csv(
            self.path / "stats.dram.banks.csv",
            header=0,
        )
        self.instructions_df = pd.read_csv(
            self.path / "stats.instructions.csv",
            header=None,
            names=["memory_space", "write", "count"],
        )
        self.l1i_stats = pd.read_csv(
            self.path / "stats.cache.l1i.csv",
            header=None,
            names=["cache_id", "access_type", "status", "count"],
        )
        self.l2d_stats = pd.read_csv(
            self.path / "stats.cache.l2d.csv",
            header=None,
            names=["cache_id", "access_type", "status", "count"],
        )

        self.use_duration = False
        self.bench_config = bench_config
        self.config = config

    def cycles(self) -> int:
        # return self.metrics["gpu_tot_sim_cycle"].sum()
        return self.sim_df["cycles"].sum()

    def instructions(self) -> int:
        return self.sim_df["instructions"].sum()
