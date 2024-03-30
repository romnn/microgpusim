import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence
import itertools

import gpucachesim.benchmarks as benchmarks
from gpucachesim.benchmarks import (
    GPUConfig,
    BenchConfig,
    AccelsimSimulateTargetConfig,
    AccelsimSimulateConfig,
)

# import gpucachesim.stats.common as common
import gpucachesim.stats.stats as stats

# from gpucachesim.stats.common import stat_cols, STAT_SUFFIXES


def raw_l2_cache_access_col(kind, status):
    return f"l2_cache_{kind.upper()}_{status.upper()}"


class Stats(stats.Stats):
    bench_config: BenchConfig[AccelsimSimulateTargetConfig]
    target_config: AccelsimSimulateConfig

    def __init__(
        self,
        config: GPUConfig,
        bench_config: BenchConfig[AccelsimSimulateTargetConfig],
        strict=True,
    ) -> None:
        self.bench_config = bench_config
        self.target_config = self.bench_config["target_config"].value
        self.path = Path(self.target_config["stats_dir"])
        self.use_duration = False
        self.config = config
        self.repetitions = self.bench_config["common"]["repetitions"]
        self.load_converted_stats()
        self.load_raw_stats()

        self.compute_result_df()
        # print(self.result_df[["kernel_name", "kernel_name_mangled", "kernel_function_signature"]].drop_duplicates())

        # add the input configs

        try:
            derived_kernel_cols = ["kernel_name", "kernel_launch_id", "run"]
            raw_kernel_cols = ["kernel", "kernel_id", "run"]

            def assertEqual(derived, original):
                if derived != original:
                    print("derived", derived)
                    print("original", original)
                assert derived == original

            if False:
                print(
                    self.result_df[
                        derived_kernel_cols
                        + [
                            "dram_reads",
                            "dram_writes",
                            "dram_accesses",
                            # "l2_write_hits",
                            # "l2_read_misses",
                        ]
                    ]
                )
                print(
                    self.raw_stats_df[
                        raw_kernel_cols
                        + [
                            # "l2_cache_global_read_total",
                            "total_dram_writes",
                            "total_dram_reads",
                        ]
                    ]
                )

            # sanity checks
            num_blocks = np.nan_to_num(self.result_df["num_blocks"].sum())
            # assert self.raw_stats_df["num_issued_blocks"].sum() == num_blocks
            assertEqual(
                derived=num_blocks,
                original=self.raw_stats_df["num_issued_blocks"].sum(),
            )

            cycles = np.nan_to_num(self.result_df["cycles"].sum())
            # assert self.raw_stats_df["gpu_tot_sim_cycle"].sum() == cycles
            assertEqual(derived=cycles, original=self.raw_stats_df["gpu_tot_sim_cycle"].sum())

            instructions = np.nan_to_num(self.result_df["instructions"].sum())
            # assert self.raw_stats_df["gpu_total_instructions"].sum() == instructions
            assertEqual(
                derived=instructions,
                original=self.raw_stats_df["gpu_total_instructions"].sum(),
            )

            dram_reads = np.nan_to_num(self.result_df["dram_reads"].sum())
            # assert self.raw_stats_df["total_dram_reads"].sum() == dram_reads
            assertEqual(
                derived=dram_reads,
                original=self.raw_stats_df["total_dram_reads"].sum(),
            )

            dram_writes = np.nan_to_num(self.result_df["dram_writes"].sum())
            # assert self.raw_stats_df["total_dram_writes"].sum() == dram_writes
            assertEqual(
                derived=dram_writes,
                original=self.raw_stats_df["total_dram_writes"].sum(),
            )

            dram_accesses = np.nan_to_num(self.result_df["dram_accesses"].sum())
            # assert self.raw_stats_df[["total_dram_writes", "total_dram_reads"]].sum().sum() == dram_accesses

            assertEqual(
                derived=dram_accesses,
                original=self.raw_stats_df[["total_dram_writes", "total_dram_reads"]].sum(axis=1).sum(),
            )

            # l2_cache_global_write_total == l2_cache_GLOBAL_ACC_W_TOTAL_ACCESS
            # l2_cache_global_read_total == l2_cache_GLOBAL_ACC_R_TOTAL_ACCESS
            # print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], ["HIT", "MISS", "SECTOR_MISS"]))
            # print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], stats.ACCESS_STATUSES))
            # print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], stats.ACCESS_STATUSES).sum().sum())
            # print(self.raw_stats_df["l2_cache_global_write_total"])

            # print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], ["HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS"]))
            # print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], ["HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS"]).T)
            l2_global_writes = np.nan_to_num(
                self._get_raw_l2_stats(["GLOBAL_ACC_W"], ["HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS"])
                .sum(axis=1)
                .sum()
            )
            assertEqual(
                derived=l2_global_writes,
                original=self.raw_stats_df["l2_cache_global_write_total"].sum(),
            )

            l2_global_reads = self._get_raw_l2_stats(["GLOBAL_ACC_R"], ["HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS"])
            # print(l2_global_reads)
            l2_global_reads = np.nan_to_num(l2_global_reads.sum(axis=1).sum())
            assertEqual(
                derived=l2_global_reads,
                original=self.raw_stats_df["l2_cache_global_read_total"].sum(),
            )

            l2_writes = np.nan_to_num(self.result_df["l2_writes"].sum())
            # assert self.raw_stats_df["l2_cache_global_write_total"].sum() == l2_writes
            assertEqual(
                derived=l2_writes,
                original=self.raw_stats_df["l2_cache_global_write_total"].sum(),
            )

            l2_reads = np.nan_to_num(self.result_df["l2_reads"].sum())
            # assert self.raw_stats_df["l2_cache_global_read_total"].sum() == l2_reads
            assertEqual(
                derived=l2_reads,
                original=self.raw_stats_df["l2_cache_global_read_total"].sum(),
            )

            l2_read_hits = np.nan_to_num(self.result_df["l2_read_hits"].sum())
            # assert self._get_raw_l2_read_stats(["HIT"]).sum().sum() == l2_read_hits
            assertEqual(
                derived=l2_read_hits,
                original=self._get_raw_l2_read_stats(["HIT", "HIT_RESERVED"]).sum(axis=1).sum(),
            )

            l2_write_hits = np.nan_to_num(self.result_df["l2_write_hits"].sum())
            # assert self._get_raw_l2_write_stats(["HIT"]).sum().sum() == l2_write_hits
            # print(self._get_raw_l2_write_stats(["HIT", "HIT_RESERVED"]))
            assertEqual(
                derived=l2_write_hits,
                original=self._get_raw_l2_write_stats(["HIT", "HIT_RESERVED"]).sum(axis=1).sum(),
            )

            l2_read_misses = np.nan_to_num(self.result_df["l2_read_misses"].sum())
            # assert self._get_raw_l2_read_stats(["MISS", "SECTOR_MISS"]).sum().sum() == l2_read_misses
            # print(self._get_raw_l2_read_stats(["MISS", "SECTOR_MISS"]).T)
            assertEqual(
                derived=l2_read_misses,
                original=self._get_raw_l2_read_stats(["MISS", "SECTOR_MISS"]).sum(axis=1).sum(),
            )

            l2_write_misses = np.nan_to_num(self.result_df["l2_write_misses"].sum())
            # assert self._get_raw_l2_write_stats(["MISS", "SECTOR_MISS"]).sum().sum() == l2_write_misses
            assertEqual(
                derived=l2_write_misses,
                original=self._get_raw_l2_write_stats(["MISS", "SECTOR_MISS"]).sum(axis=1).sum(),
            )

            l2_write_hit_rate = np.nan_to_num(self.result_df["l2_write_hit_rate"].sum())
            # assert (
            #     self.raw_stats_df["l2_cache_GLOBAL_ACC_W_HIT"] / self.raw_stats_df["l2_cache_global_write_total"]
            # ).sum() == l2_write_hit_rate
            assertEqual(
                derived=l2_write_hit_rate,
                original=(
                    self._get_raw_l2_write_stats(["HIT", "HIT_RESERVED"]).sum(axis=1)
                    / self.raw_stats_df["l2_cache_global_write_total"]
                ).sum(),
            )

            l2_read_hit_rate = np.nan_to_num(self.result_df["l2_read_hit_rate"].sum())
            # assert (
            #     self.raw_stats_df["l2_cache_GLOBAL_ACC_R_HIT"] / self.raw_stats_df["l2_cache_global_read_total"]
            # ).sum() == l2_read_hit_rate
            assertEqual(
                derived=l2_read_hit_rate,
                original=(
                    # self.raw_stats_df["l2_cache_GLOBAL_ACC_R_HIT"]
                    self._get_raw_l2_read_stats(["HIT", "HIT_RESERVED"]).sum(axis=1)
                    / self.raw_stats_df["l2_cache_global_read_total"]
                ).sum(),
            )

            # print(self.raw_stats_df[[c for c in self.raw_stats_df if "l1_data_cache" in c]])
            l1_reads = np.nan_to_num(self.result_df["l1_reads"].sum())
            # assert self.raw_stats_df["l1_data_cache_global_read_total"].sum() == l1_reads
            assertEqual(
                derived=l1_reads,
                original=self.raw_stats_df["l1_data_cache_global_read_total"].sum(),
            )
        except AssertionError as e:
            if strict:
                raise e
            print(f"WARNING: {e}")

    def load_raw_stats(self) -> None:
        raw_stats_dfs = []
        for r in range(self.repetitions):
            raw_stats_df = pd.read_csv(
                self.path / f"raw.stats.{r}.csv",
                header=None,
                names=["kernel", "kernel_id", "stat", "value"],
            )
            raw_stats_df = raw_stats_df.pivot(
                index=["kernel", "kernel_id"],
                columns=["stat"],
            )["value"]

            # print(raw_stats_df)
            # only keep the final kernel info
            # raw_stats_df = raw_stats_df.loc["final_kernel", 0].reset_index()
            raw_stats_df = raw_stats_df.reset_index()
            # print(raw_stats_df)
            # raw_stats_df.columns = ["stat", "value"]
            # raw_stats_df = raw_stats_df.set_index("stat").T

            raw_stats_df["run"] = r

            raw_stats_dfs.append(raw_stats_df)

        self.raw_stats_df = pd.concat(raw_stats_dfs)
        # self.raw_stats_df = common.compute_df_statistics(raw_stats_df, group_by=None)

    def _get_raw_l2_read_stats(self, status: Sequence[str]):
        return self._get_raw_l2_stats(benchmarks.READ_ACCESS_KINDS, status)

    def _get_raw_l2_write_stats(self, status: Sequence[str]):
        return self._get_raw_l2_stats(benchmarks.WRITE_ACCESS_KINDS, status)

    def _get_raw_l2_stats(self, kind: Sequence[str], status: Sequence[str]):

        cols = [raw_l2_cache_access_col(k, s) for (k, s) in itertools.product(kind, status)]
        return self.raw_stats_df[cols]

    def _compute_warp_instructions(self):
        num_warps = self.result_df["num_blocks"] * benchmarks.WARP_SIZE
        # print("this", self.raw_stats_df["warp_instruction_count"].values)
        # print(num_warps)
        self.result_df["warp_inst"] = self.raw_stats_df["warp_instruction_count"].values / num_warps
