from os import PathLike
from pathlib import Path
import typing
import json
import re
import pandas as pd
from pprint import pprint

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common

INDEX_COLS = ["Stream", "Context", "Device", "Kernel", "Correlation_ID"]

NUMERIC_METRIC_COLUMNS = [
    "elapsed_cycles_sm",
    "issue_slots",
    "unique_warps_launched",
    "pcie_total_data_transmitted",
    "pcie_total_data_received",
    "inst_issued",
    "inst_executed",
    "inst_fp_16",
    "inst_fp_32",
    "inst_fp_64",
    "inst_integer",
    "inst_control",
    "inst_compute_ld_st",
    "inst_misc",
    "ldst_issued",
    "ldst_executed",
    "atomic_transactions",
    "l2_atomic_transactions",
    "l2_tex_read_transactions",
    "l2_tex_write_transactions",
    "ecc_transactions",
    "dram_read_transactions",
    "dram_write_transactions",
    "shared_store_transactions",
    "shared_load_transactions",
    "local_load_transactions",
    "local_store_transactions",
    "gld_transactions",
    "gst_transactions",
    "sysmem_read_transactions",
    "sysmem_write_transactions",
    "l2_read_transactions",
    "l2_write_transactions",
    "tex_cache_transactions",
    "flop_count_hp",
    "flop_count_hp_add",
    "flop_count_hp_mul",
    "flop_count_hp_fma",
    "flop_count_dp",
    "flop_count_dp_add",
    "flop_count_dp_fma",
    "flop_count_dp_mul",
    "flop_count_sp",
    "flop_count_sp_add",
    "flop_count_sp_fma",
    "flop_count_sp_mul",
    "flop_count_sp_special",
    "inst_executed_global_loads",
    "inst_executed_local_loads",
    "inst_executed_shared_loads",
    "inst_executed_surface_loads",
    "inst_executed_global_stores",
    "inst_executed_local_stores",
    "inst_executed_shared_stores",
    "inst_executed_surface_stores",
    "inst_executed_global_atomics",
    "inst_executed_global_reductions",
    "inst_executed_surface_atomics",
    "l2_global_load_bytes",
    "l2_local_load_bytes",
    "l2_surface_load_bytes",
    "l2_local_global_store_bytes",
    "l2_global_reduction_bytes",
    "l2_global_atomic_store_bytes",
    "l2_surface_store_bytes",
    "l2_surface_reduction_bytes",
    "l2_surface_atomic_store_bytes",
    "sysmem_read_bytes",
    "sysmem_write_bytes",
    "dram_write_bytes",
    "dram_read_bytes",
    "global_load_requests",
    "local_load_requests",
    "surface_load_requests",
    "global_store_requests",
    "local_store_requests",
    "surface_store_requests",
    "global_atomic_requests",
    "global_reduction_requests",
    "surface_atomic_requests",
    "surface_reduction_requests",
    "texture_load_requests",
    "inst_per_warp",
    "ipc",
    "issued_ipc",
    "issue_slot_utilization",
    "eligible_warps_per_cycle",
    "inst_replay_overhead",
    "local_memory_overhead",
    "tex_cache_hit_rate",
    "l2_tex_read_hit_rate",
    "l2_tex_write_hit_rate",
    "l2_tex_hit_rate",
    "global_hit_rate",
    "local_hit_rate",
    "stall_inst_fetch",
    "stall_exec_dependency",
    "stall_memory_dependency",
    "stall_texture",
    "stall_sync",
    "stall_other",
    "stall_constant_memory_dependency",
    "stall_pipe_busy",
    "stall_memory_throttle",
    "stall_not_selected",
    "atomic_transactions_per_request",
    "shared_store_transactions_per_request",
    "shared_load_transactions_per_request",
    "local_load_transactions_per_request",
    "local_store_transactions_per_request",
    "gld_transactions_per_request",
    "gst_transactions_per_request",
    "sm_efficiency",
    "shared_efficiency",
    "flop_hp_efficiency",
    "flop_sp_efficiency",
    "flop_dp_efficiency",
    "gld_efficiency",
    "gst_efficiency",
    "branch_efficiency",
    "warp_execution_efficiency",
    "warp_nonpred_execution_efficiency",
    "dram_read_throughput",
    "dram_write_throughput",
    "l2_atomic_throughput",
    "ecc_throughput",
    "tex_cache_throughput",
    "l2_tex_read_throughput",
    "l2_tex_write_throughput",
    "l2_read_throughput",
    "l2_write_throughput",
    "sysmem_read_throughput",
    "sysmem_write_throughput",
    "local_load_throughput",
    "local_store_throughput",
    "shared_load_throughput",
    "shared_store_throughput",
    "gld_throughput",
    "gst_throughput",
    "gld_requested_throughput",
    "gst_requested_throughput",
]


def normalize_device_name(name):
    # Strip off device numbers, e.g. (0), (1)
    # that some profiler versions add to the end of device name
    return re.sub(r" \(\d+\)$", "", name)


def compute_df_statistics(df: pd.DataFrame, group_by: typing.List[str], agg=None):
    all_columns = set(df.columns.tolist())
    all_columns = all_columns.difference(group_by)
    all_columns = sorted(list(all_columns))
    # print(all_columns)

    if agg is None:
        agg = dict()

    grouped = df.groupby(group_by)

    df_mean = grouped.agg({**{c: "mean" for c in all_columns}, **agg})
    df_mean = df_mean.rename(columns={c: c + "_mean" for c in df_mean.columns})

    df_max = grouped.agg({**{c: "max" for c in all_columns}, **agg})
    df_max = df_max.rename(columns={c: c + "_max" for c in df_max.columns})

    df_min = grouped.agg({**{c: "min" for c in all_columns}, **agg})
    df_min = df_min.rename(columns={c: c + "_min" for c in df_min.columns})

    df_std = grouped.agg({**{c: "std" for c in all_columns}, **agg})
    df_std = df_std.rename(columns={c: c + "_std" for c in df_std.columns})

    return pd.concat([df_mean, df_max, df_min, df_std], axis=1)


class Stats(common.Stats):
    bench_config: BenchConfig
    config: GPUConfig
    df: pd.DataFrame
    commands_df: pd.DataFrame

    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["profile"]["profile_dir"])
        self.repetitions = bench_config["profile"]["repetitions"]
        print(self.repetitions)

        dfs = []
        command_dfs = []
        for r in range(self.repetitions):
            with open(self.path / f"profile.commands.{r}.json", "rb") as f:
                commands = json.load(f)
                commands_df = pd.DataFrame.from_records([{k: v["value"] for k, v in c.items()} for c in commands])
                _units = pd.DataFrame.from_records([{k: v["unit"] for k, v in c.items()} for c in commands])

                # name refers to kernels now
                commands_df = commands_df.rename(columns={"Name": "Kernel"})

                commands_df["Device"] = commands_df["Device"].apply(normalize_device_name)
                commands_df["run"] = r
                command_dfs.append(commands_df)

            with open(self.path / f"profile.metrics.{r}.json", "rb") as f:
                metrics = json.load(f)
                df = pd.DataFrame.from_records([{k: v["value"] for k, v in m.items()} for m in metrics])
                df["run"] = r

                _units = pd.DataFrame.from_records([{k: v["unit"] for k, v in m.items()} for m in metrics])
                dfs.append(df)

        df = pd.concat(dfs)
        # print(df.columns.tolist())
        df = df[NUMERIC_METRIC_COLUMNS + INDEX_COLS]
        df[NUMERIC_METRIC_COLUMNS] = df[NUMERIC_METRIC_COLUMNS].astype(float)
        self.df = compute_df_statistics(df, group_by=INDEX_COLS)

        # print(self.df.loc[:, self.df.columns.str.contains("elapsed_cycles_sm")].T)

        commands_df = pd.concat(command_dfs)
        # print(commands_df.select_dtypes(include=["object"]).columns)
        # print(commands_df)
        self.commands_df = compute_df_statistics(
            commands_df,
            group_by=INDEX_COLS,
            agg={"SrcMemType": "first", "DstMemType": "first"},
        )
        # print(self.commands_df)

        # self.df = self.df.groupby("run")
        # print(self.df)

        self.use_duration = False
        self.bench_config = bench_config
        self.config = config

    @property
    def kernel_launches(self):
        commands = self.commands_df.reset_index()
        return commands[~commands["Kernel"].str.contains(r"\[CUDA memcpy .*\]")]

    def duration_us(self) -> float:
        nvprof_key = "Duration_mean"
        nsight_key = "gpu__time_duration.sum_nsecond"

        if nvprof_key in self.commands_df:
            # convert us to us (1e-6)
            # duration already us
            return self.kernel_launches[nvprof_key].sum()
        elif nsight_key in self.df:
            # convert ns to us
            return self.df[nsight_key].sum() * 1e-3
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

            nvprof_key = "elapsed_cycles_sm_mean"
            # nsight_col = "sm__cycles_elapsed.sum_cycle"
            nsight_key = "gpc__cycles_elapsed.avg_cycle"
            # nsight_col = "sm__cycles_active.avg_cycle"
            # pprint(list(self.df.columns.tolist()))
            if nvprof_key in self.df:
                sm_count = self.config.num_total_cores
                # sm_count = self.config.num_clusters
                # print(sm_count)

                # print(self.df[["elapsed_cycles_sm", "run"]])
                # cycles_per_run = self.df.groupby("run")["elapsed_cycles_sm"].sum()
                cycles = self.df[nvprof_key].sum()
                # cycles = cycles_per_run.mean()
                # this only holds until we have repetitions
                # print(self.df["elapsed_cycles_sm"])
                # assert (cycles == self.df["elapsed_cycles_sm"]).all()
                # return int(cycles / sm_count)
                return int(cycles / sm_count)
            elif nsight_key in self.df:
                return self.df[nsight_key].sum()
            else:
                raise ValueError("hw dataframe missing cycles")
                # hw_value *= mean_sm_efficiency

    def num_blocks(self):
        return 0

    def instructions(self) -> int:
        inst_cols = [
            "inst_fp_16_mean",
            "inst_fp_32_mean",
            "inst_fp_64_mean",
            "inst_integer_mean",
            "inst_control_mean",
            "inst_compute_ld_st_mean",
            "inst_misc_mean",
        ]
        total_instructions = self.df[inst_cols].sum().sum()
        # print(total_instructions)
        # per_run_total_instructions = self.df[["run"] + inst_cols].astype(int)
        # print(per_run_total_instructions)
        # per_run_total_instructions = per_run_total_instructions.groupby("run")[inst_cols].sum()
        # per_run_total_instructions = per_run_total_instructions.sum(axis=1)
        return int(total_instructions)

    def executed_instructions(self):
        nvprof_key = "inst_issued_mean"
        nvprof_key = "inst_executed_mean"

        # there is also sm__inst_executed.sum_inst
        # sm__sass_thread_inst_executed.sum_inst
        nsight_key = "smsp__inst_executed.sum_inst"
        if nvprof_key in self.df:
            return self.df[nvprof_key].sum() * self.config.num_total_cores
        elif nsight_key in self.df:
            return self.df[nsight_key].sum()
        else:
            raise ValueError("missing instructions")

        # print("inst_issued", self.df["inst_issued"].sum() * self.config.num_total_cores)
        # print(
        #     "inst_executed",
        #     self.df["inst_executed"].sum() * self.config.num_total_cores,
        # )
        #
        # total_instructions = (
        #     self.df[
        #         [
        #             "inst_fp_16",
        #             "inst_fp_32",
        #             "inst_fp_64",
        #             "inst_integer",
        #             "inst_control",
        #             "inst_compute_ld_st",
        #             "inst_misc",
        #         ]
        #     ]
        #     .astype(int)
        #     .sum()
        #     .sum()
        # )
        # print("total", total_instructions)
        #
        # sub_instructions = (
        #     self.df[
        #         [
        #             "inst_fp_16",
        #             "inst_fp_32",
        #             "inst_fp_64",
        #             "inst_integer",
        #             # "inst_control",
        #             "inst_compute_ld_st",
        #             "inst_misc",
        #         ]
        #     ]
        #     .astype(int)
        #     .sum()
        #     .sum()
        # )
        #
        # print(
        #     self.df[
        #         [
        #             "inst_fp_16",
        #             "inst_fp_32",
        #             "inst_fp_64",
        #             "inst_integer",
        #             # "inst_control",
        #             "inst_compute_ld_st",
        #             "inst_misc",
        #         ]
        #     ]
        #     .astype(int)
        #     .sum()
        # )
        # print("sub", sub_instructions)

        # inst_fp_16	{'value': 0, 'unit': None}
        # inst_fp_32	{'value': 100, 'unit': None}
        # inst_fp_64	{'value': 0, 'unit': None}
        # inst_integer	{'value': 4896, 'unit': None}
        # inst_bit_convert	{'value': '0', 'unit': None}
        # inst_control	{'value': '1024', 'unit': None}
        # inst_compute_ld_st	{'value': '300', 'unit': None}
        # inst_misc	{'value': '4196', 'unit': None}

    def exec_time_sec(self) -> float:
        return self.duration_us() * float(1e-6)

    def warp_instructions(self) -> float:
        nvprof_key = "inst_per_warp_mean"
        if nvprof_key in self.df:
            return float(self.df[nvprof_key].mean())
        else:
            raise ValueError("nsight warp instructions")

    def dram_reads(self) -> float:
        nvprof_key = "dram_read_transactions_mean"
        if nvprof_key in self.df:
            return int(self.df[nvprof_key].sum())
        else:
            nsight_key = "dram__sectors_read.sum_sector"
            return int(self.df[nsight_key].sum())

    def dram_writes(self) -> float:
        nvprof_key = "dram_write_transactions_mean"
        if nvprof_key in self.df:
            return int(self.df[nvprof_key].sum())
        else:
            nsight_key = "dram__sectors_write.sum_sector"
            return int(self.df[nsight_key].sum())

    def dram_accesses(self) -> float:
        nvprof_keys = ["dram_read_transactions_mean", "dram_write_transactions_mean"]
        if set(nvprof_keys).issubset(self.df):
            return self.df[nvprof_keys].sum().sum()
        else:
            nsight_keys = [
                "dram__sectors_read.sum_sector",
                "dram__sectors_write.sum_sector",
            ]
            return self.df[nsight_keys].sum().sum()

    def l2_reads(self) -> int:
        nvprof_key = "l2_tex_read_transactions_mean"
        nvprof_key = "l2_read_transactions_mean"
        if nvprof_key in self.df:
            return self.df[nvprof_key].sum()
        else:
            return self.df["lts__t_sectors_srcunit_tex_op_read.sum_sector"].sum()

    def l2_writes(self) -> int:
        nvprof_key = "l2_tex_write_transactions_mean"
        nvprof_key = "l2_write_transactions_mean"
        if nvprof_key in self.df:
            return self.df[nvprof_key].sum()
        else:
            nsight_key = "lts__t_sectors_srcunit_tex_op_write.sum_sector"
            return self.df[nsight_key].sum()

    def l2_accesses(self) -> int:
        # nvprof_read_key = "l2_tex_read_transactions"
        # nvprof_write_key = "l2_tex_write_transactions"

        nvprof_keys = ["l2_read_transactions_mean", "l2_write_transactions_mean"]
        if set(nvprof_keys).issubset(self.df):
            # if nvprof_keys in self.df:
            return self.df[nvprof_keys].sum().sum()
        else:
            nsight_keys = [
                "lts__t_sectors_srcunit_tex_op_write.sum_sector",
                "lts__t_sectors_srcunit_tex_op_read.sum_sector",
            ]
            return self.df[nsight_keys].sum().sum()

    def l2_read_hit_rate(self) -> float:
        nvprof_key = "l2_tex_read_hit_rate_mean"
        # nvprof_key = "l2_read_hit_rate"
        return float(self.df[nvprof_key].mean()) / 100.0

    def l2_write_hit_rate(self) -> float:
        nvprof_key = "l2_tex_write_hit_rate_mean"
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
