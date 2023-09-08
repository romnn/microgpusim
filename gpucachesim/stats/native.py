from os import PathLike
from pathlib import Path
import typing
import json
import re
import pandas as pd
from pprint import pprint

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common


def normalize_device_name(name):
    # Strip off device numbers, e.g. (0), (1)
    # that some profiler versions add to the end of device name
    return re.sub(r" \(\d+\)$", "", name)


class Stats(common.Stats):
    bench_config: BenchConfig
    config: GPUConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["profile"]["profile_dir"])
        with open(self.path / "profile.commands.json", "rb") as f:
            commands = json.load(f)
            self.commands = pd.DataFrame.from_dict([{k: v["value"] for k, v in c.items()} for c in commands])
            self.commands_units = pd.DataFrame.from_dict([{k: v["unit"] for k, v in c.items()} for c in commands])

            # name refers to kernels now
            self.commands = self.commands.rename(columns={"Name": "Kernel"})
            # remove columns that are only relevant for memcopies
            # df = df.loc[:,df.notna().any(axis=0)]
            # self.commands = self.commands.drop(columns=["Size", "Throughput", "SrcMemType", "DstMemType"])
            # set the correct dtypes
            # self.commands = self.commands.astype(
            #     {
            #         "Start": "float64",
            #         "Duration": "float64",
            #         "Static SMem": "float64",
            #         "Dynamic SMem": "float64",
            #         "Device": "string",
            #         "Kernel": "string",
            #     }
            # )

            self.commands["Device"] = self.commands["Device"].apply(normalize_device_name)

        with open(self.path / "profile.metrics.json", "rb") as f:
            metrics = json.load(f)
            self.df = pd.DataFrame.from_dict([{k: v["value"] for k, v in m.items()} for m in metrics])
            self.units = pd.DataFrame.from_dict([{k: v["unit"] for k, v in m.items()} for m in metrics])

        self.use_duration = False
        self.bench_config = bench_config
        self.config = config

    @property
    def kernel_launches(self):
        return self.commands[~self.commands["Kernel"].str.contains(r"\[CUDA memcpy .*\]")]

    def duration_us(self) -> float:
        # pprint(self.kernel_launches)
        if "Duration" in self.commands:
            # convert us to us (1e-6)
            # duration already us
            return self.kernel_launches["Duration"].sum()
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
            duration = self.duration_us()
            return int(duration * float(self.config.core_clock_speed))
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
                # print(sm_count)
                cycles = self.df["elapsed_cycles_sm"].sum()
                # this only holds until we have repetitions
                # print(self.df["elapsed_cycles_sm"])
                # assert (cycles == self.df["elapsed_cycles_sm"]).all()
                # return int(cycles / sm_count)
                return int(cycles / sm_count)
            elif nsight_col in self.df:
                return self.df[nsight_col].sum()
            else:
                raise ValueError("hw dataframe missing cycles")
                # hw_value *= mean_sm_efficiency

    def num_blocks(self):
        return 0

    def instructions(self) -> int:
        total_instructions = (
            self.df[
                [
                    "inst_fp_16",
                    "inst_fp_32",
                    "inst_fp_64",
                    "inst_integer",
                    "inst_control",
                    "inst_compute_ld_st",
                    "inst_misc",
                ]
            ]
            .astype(int)
            .sum()
            .sum()
        )
        return total_instructions

    def executed_instructions(self):
        nvprof_key = "inst_issued"
        nvprof_key = "inst_executed"

        print("inst_issued", self.df["inst_issued"].sum() * self.config.num_total_cores)
        print(
            "inst_executed",
            self.df["inst_executed"].sum() * self.config.num_total_cores,
        )

        total_instructions = (
            self.df[
                [
                    "inst_fp_16",
                    "inst_fp_32",
                    "inst_fp_64",
                    "inst_integer",
                    "inst_control",
                    "inst_compute_ld_st",
                    "inst_misc",
                ]
            ]
            .astype(int)
            .sum()
            .sum()
        )
        print("total", total_instructions)

        sub_instructions = (
            self.df[
                [
                    "inst_fp_16",
                    "inst_fp_32",
                    "inst_fp_64",
                    "inst_integer",
                    # "inst_control",
                    "inst_compute_ld_st",
                    "inst_misc",
                ]
            ]
            .astype(int)
            .sum()
            .sum()
        )

        print(
            self.df[
                [
                    "inst_fp_16",
                    "inst_fp_32",
                    "inst_fp_64",
                    "inst_integer",
                    # "inst_control",
                    "inst_compute_ld_st",
                    "inst_misc",
                ]
            ]
            .astype(int)
            .sum()
        )
        print("sub", sub_instructions)

        # inst_fp_16	{'value': 0, 'unit': None}
        # inst_fp_32	{'value': 100, 'unit': None}
        # inst_fp_64	{'value': 0, 'unit': None}
        # inst_integer	{'value': 4896, 'unit': None}
        # inst_bit_convert	{'value': '0', 'unit': None}
        # inst_control	{'value': '1024', 'unit': None}
        # inst_compute_ld_st	{'value': '300', 'unit': None}
        # inst_misc	{'value': '4196', 'unit': None}

        if nvprof_key in self.df:
            return self.df[nvprof_key].sum() * self.config.num_total_cores
        elif "smsp__inst_executed.sum_inst" in self.df:
            # there is also sm__inst_executed.sum_inst
            # sm__sass_thread_inst_executed.sum_inst
            return self.df["smsp__inst_executed.sum_inst"].sum()
        else:
            raise ValueError("missing instructions")

    def exec_time_sec(self) -> float:
        return self.duration_us() * float(1e-6)

    def warp_instructions(self) -> float:
        nvprof_key = "inst_per_warp"
        if nvprof_key in self.df:
            return self.df[nvprof_key].mean()
        else:
            raise ValueError("nsight warp instructions")

    def dram_reads(self) -> int:
        nvprof_key = "dram_read_transactions"
        if nvprof_key in self.df:
            return int(self.df[nvprof_key].sum())
        else:
            return int(self.df["dram__sectors_read.sum_sector"].sum())

    def dram_writes(self) -> int:
        nvprof_key = "dram_write_transactions"
        if nvprof_key in self.df:
            return int(self.df[nvprof_key].sum())
        else:
            return int(self.df["dram__sectors_write.sum_sector"].sum())

    def dram_accesses(self) -> int:
        nvprof_key = "dram_read_transactions"
        if nvprof_key in self.df:
            return int(self.df[[nvprof_key, "dram_write_transactions"]].sum().sum())
        else:
            return int(self.df[["dram__sectors_read.sum_sector", "dram__sectors_write.sum_sector"]].sum().sum())

    def l2_reads(self) -> int:
        nvprof_key = "l2_tex_read_transactions"
        nvprof_key = "l2_read_transactions"
        if nvprof_key in self.df:
            return self.df[nvprof_key].sum()
        else:
            return self.df["lts__t_sectors_srcunit_tex_op_read.sum_sector"].sum()

    def l2_writes(self) -> int:
        nvprof_key = "l2_tex_write_transactions"
        nvprof_key = "l2_write_transactions"
        if nvprof_key in self.df:
            return self.df[nvprof_key].sum()
        else:
            return self.df["lts__t_sectors_srcunit_tex_op_write.sum_sector"].sum()

    def l2_accesses(self) -> int:
        # nvprof_read_key = "l2_tex_read_transactions"
        # nvprof_write_key = "l2_tex_write_transactions"

        nvprof_read_key = "l2_read_transactions"
        nvprof_write_key = "l2_write_transactions"
        if nvprof_read_key in self.df:
            return self.df[nvprof_read_key].sum() + self.df[nvprof_write_key].sum()
        else:
            return (
                self.df["lts__t_sectors_srcunit_tex_op_write.sum_sector"].sum()
                + self.df["lts__t_sectors_srcunit_tex_op_read.sum_sector"].sum()
            )

    def l2_read_hit_rate(self) -> float:
        nvprof_key = "l2_tex_read_hit_rate"
        # nvprof_key = "l2_read_hit_rate"
        return float(self.df[nvprof_key].mean()) / 100.0

    def l2_write_hit_rate(self) -> float:
        nvprof_key = "l2_tex_write_hit_rate"
        # nvprof_key = "l2_write_hit_rate"
        return float(self.df[nvprof_key].mean()) / 100.0

    def l2_read_miss_rate(self) -> float:
        return 1 - self.l2_read_hit_rate()

    def l2_write_miss_rate(self) -> float:
        return 1 - self.l2_write_hit_rate()

    def l2_read_hits(self) -> int:
        return int(float(self.l2_reads()) * self.l2_read_hit_rate())

    def l2_write_hits(self) -> int:
        return int(float(self.l2_writes()) * self.l2_write_hit_rate())

    def l2_read_misses(self) -> int:
        return int(float(self.l2_reads()) * self.l2_read_miss_rate())

    def l2_write_misses(self) -> int:
        return int(float(self.l2_writes()) * self.l2_write_miss_rate())

    def l2_hits(self) -> int:
        return self.l2_read_hits() + self.l2_write_hits()

    def l2_misses(self) -> int:
        return self.l2_read_misses() + self.l2_write_misses()
