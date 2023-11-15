import click
from pathlib import Path
import os
from enum import Enum
import typing
import yaml
from pprint import pprint

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from gpucachesim import ROOT_DIR
import gpucachesim.utils as utils


REPO_ROOT_DIR = ROOT_DIR.parent
DEFAULT_BENCH_FILE = REPO_ROOT_DIR / "test-apps/test-apps-materialized.yml"

WARP_SIZE = 32

READ_ACCESS_KINDS = [
    "GLOBAL_ACC_R",
    "LOCAL_ACC_R",
    "CONST_ACC_R",
    "TEXTURE_ACC_R",
    "INST_ACC_R",
    "L1_WR_ALLOC_R",
    "L2_WR_ALLOC_R",
]
WRITE_ACCESS_KINDS = ["GLOBAL_ACC_W", "LOCAL_ACC_W", "L1_WRBK_ACC", "L2_WRBK_ACC"]

ACCESS_KINDS = READ_ACCESS_KINDS + WRITE_ACCESS_KINDS

INDEX_COLS = [
    "kernel_name",
    "kernel_name_mangled",
    "kernel_launch_id",
    "run",
]

ACCESS_STATUSES = [
    "HIT",
    "HIT_RESERVED",
    "MISS",
    "RESERVATION_FAIL",
    "SECTOR_MISS",
    "MSHR_HIT",
]

SIMULATE_FUNCTIONAL_CONFIG_COLS = [
    "input_memory_only",
    "input_num_clusters",
    "input_cores_per_cluster",
]
SIMULATE_EXECUTION_CONFIG_COLS = [
    "input_mode",
    "input_threads",
    "input_run_ahead",
]
SIMULATE_INPUT_COLS = SIMULATE_EXECUTION_CONFIG_COLS + SIMULATE_FUNCTIONAL_CONFIG_COLS

BENCHMARK_INPUT_COLS = {
    "vectorAdd": ["input_dtype", "input_length"],
    "matrixmul": ["input_dtype", "input_rows"],
    "simple_matrixmul": ["input_dtype", "input_m", "input_n", "input_p"],
    "transpose": ["input_dim", "input_variant"],
    "babelstream": ["input_size"],
}

STAT_COLS = [
    "exec_time_sec",
    "cycles",
    "num_blocks",
    "instructions",
    "warp_inst",
    # dram stats
    "dram_reads",
    "dram_writes",
    # l2 stats
    "l2_accesses",
    "l2_reads",
    "l2_writes",
    "l2_read_hit_rate",
    "l2_write_hit_rate",
    "l2_read_miss_rate",
    "l2_write_miss_rate",
    "l2_hit_rate",
    "l2_miss_rate",
    "l2_read_hits",
    "l2_write_hits",
    "l2_read_misses",
    "l2_write_misses",
    "l2_hits",
    "l2_misses",
    # l1 rates
    "l1_hit_rate",
    "l1_miss_rate",
    # l1 accesses
    "l1_reads",
    "l1_writes",
    "l1_hits",
    "l1_misses",
    "l1_accesses",
]

# BENCH_TARGET_INDEX_COLS = ["target", "benchmark", "input_id"]
BENCH_TARGET_INDEX_COLS = ["target", "benchmark"]

NON_NUMERIC_COLS = {
    "target": "first",
    "benchmark": "first",
    "Host Name": "first",
    "Process Name": "first",
    "device": "first",
    "is_release_build": "first",
    "kernel_function_signature": "first",
    "kernel_name": "first",
    "kernel_name_mangled": "first",
    "input_id": "first",
    "input_memory_only": "first",
    "input_mode": "first",
    # makes no sense to aggregate
    "cores_per_cluster": "first",
    "num_clusters": "first",
    "total_cores": "first",
}


class GPUConfig:
    def __init__(self, config) -> None:
        self.config = config

    @property
    def _clock_domains(self) -> typing.Dict[str, float]:
        """<Core Clock>:<Interconnect Clock>:<L2 Clock>:<DRAM Clock>"""
        clock_domains = list(self.config["sim"]["gpgpu_clock_domains"].split(":"))
        return dict(
            core=clock_domains[0],
            interconnect=clock_domains[1],
            l2=clock_domains[2],
            dram=clock_domains[3],
        )

    @property
    def core_clock_speed(self) -> float:
        return self._clock_domains["core"]

    @property
    def num_clusters(self) -> int:
        return self.config["shader_core"]["gpgpu_n_clusters"]

    @property
    def cores_per_cluster(self) -> int:
        return self.config["shader_core"]["gpgpu_n_cores_per_cluster"]

    @property
    def num_total_cores(self) -> int:
        return self.num_clusters * self.cores_per_cluster


class Target(Enum):
    Profile = "Profile"
    Simulate = "Simulate"
    ExecDrivenSimulate = "ExecDrivenSimulate"
    AccelsimSimulate = "AccelsimSimulate"
    AccelsimTrace = "AccelsimTrace"
    PlaygroundSimulate = "PlaygroundSimulate"
    Trace = "Trace"


class Config(typing.TypedDict):
    results_dir: os.PathLike


class GenericBenchmarkConfig(typing.TypedDict):
    repetitions: int
    timeout: typing.Optional[str]
    concurrency: typing.Optional[int]
    enabled: bool
    results_dir: os.PathLike


class ProfileConfig(typing.TypedDict):
    profile_dir: os.PathLike


class ProfileTargetConfig(typing.NamedTuple):
    value: ProfileConfig


class SimulateConfig(typing.TypedDict):
    stats_dir: os.PathLike
    traces_dir: os.PathLike
    accelsim_traces_dir: os.PathLike
    parallel: typing.Optional[bool]


class SimulateTargetConfig(typing.NamedTuple):
    value: SimulateConfig


class TraceConfig(typing.TypedDict):
    traces_dir: os.PathLike
    save_json: bool
    full_trace: bool


class TraceTargetConfig(typing.NamedTuple):
    value: TraceConfig


class AccelsimSimulateConfig(typing.TypedDict):
    trace_config: os.PathLike
    inter_config: os.PathLike
    config_dir: os.PathLike
    config: os.PathLike
    traces_dir: os.PathLike
    stats_dir: os.PathLike


class AccelsimSimulateTargetConfig(typing.NamedTuple):
    value: AccelsimSimulateConfig


class AccelsimTraceConfig(typing.TypedDict):
    traces_dir: os.PathLike


class AccelsimTraceTargetConfig(typing.NamedTuple):
    value: AccelsimTraceConfig


class PlaygroundSimulateConfig(AccelsimSimulateConfig):
    pass


class PlaygroundSimulateTargetConfig(typing.NamedTuple):
    value: PlaygroundSimulateConfig


T = typing.TypeVar("T")


class BenchConfig(typing.TypedDict, typing.Generic[T]):
    name: str
    benchmark_idx: int
    uid: str

    path: os.PathLike
    executable: os.PathLike

    values: typing.Dict[str, typing.Any]
    args: typing.Sequence[str]
    input_idx: int

    common: GenericBenchmarkConfig

    target: str
    target_config: T


class BenchmarkLoader(SafeLoader):
    pass


def construct_profile_target_config(self, node):
    return ProfileTargetConfig(self.construct_mapping(node))


def construct_trace_target_config(self, node):
    return TraceTargetConfig(self.construct_mapping(node))


def construct_simulate_target_config(self, node):
    return SimulateTargetConfig(self.construct_mapping(node))


def construct_exec_driven_simulate_target_config(self, node):
    return SimulateTargetConfig(self.construct_mapping(node))


def construct_accelsim_simulate_target_config(self, node):
    return AccelsimSimulateTargetConfig(self.construct_mapping(node))


def construct_accelsim_trace_target_config(self, node):
    return AccelsimTraceTargetConfig(self.construct_mapping(node))


def construct_playground_simulate_target_config(self, node):
    return PlaygroundSimulateTargetConfig(self.construct_mapping(node))


BenchmarkLoader.add_constructor("!Profile", construct_profile_target_config)
BenchmarkLoader.add_constructor("!Trace", construct_trace_target_config)
BenchmarkLoader.add_constructor("!Simulate", construct_simulate_target_config)
BenchmarkLoader.add_constructor("!ExecDrivenSimulate", construct_exec_driven_simulate_target_config)
BenchmarkLoader.add_constructor("!AccelsimSimulate", construct_accelsim_simulate_target_config)
BenchmarkLoader.add_constructor("!AccelsimTrace", construct_accelsim_trace_target_config)
BenchmarkLoader.add_constructor("!PlaygroundSimulate", construct_playground_simulate_target_config)


class Benchmarks:
    config: Config

    def __init__(self, path: os.PathLike) -> None:
        """load the materialized benchmark config"""

        with open(path or DEFAULT_BENCH_FILE, "rb") as f:
            benchmarks = yaml.load(f, Loader=BenchmarkLoader)

        self.config = benchmarks["config"]
        self.benchmarks = benchmarks["benchmarks"]

    # def __getitem__(self, bench_name: str):
    #     return self.benchmarks[bench_name]

    # def get_bench_configs(self, bench_name: str):
    #     """Get the bench configs for all targets"""
    #     for target_benches in self.benchmarks.values():
    #         return target_benches[bench_name]
    #
    # def get_single_bench_config(self, bench_name: str, input_idx: int):
    #     """Get the bench configs for all targets"""
    #     for target_benches in self.benchmarks.values():
    #         return target_benches[bench_name][input_idx]
    #
    # def get_target_bench_config(self, target: Target, bench_name: str, input_idx: int):
    #     return self.benchmarks[target.value][bench_name][input_idx]


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


@main.command()
@click.option("--path", default=DEFAULT_BENCH_FILE, help="Path to materialized benchmark config")
@click.option("--baseline", type=bool, default=True, help="Baseline configurations")
def table(path, baseline):
    print("loading", path)
    b = Benchmarks(path)
    benches = b.benchmarks[Target.Simulate.value]

    table = r"Benchmark & Type & Input configurations \\ \hline"
    table += "\n"

    def get_name(bench_name):
        match bench_name.lower():
            case "simple_matrixmul":
                return "Matrixmul (naive)"
            case other:
                return other.capitalize()

    def get_kind(bench_name):
        match bench_name.lower():
            case "vectoradd":
                return "MB"
            case _:
                return "MB"

    row_idx = 0
    for bench_name, bench_configs in benches.items():
        # pprint(bench_configs)

        def is_baseline(config):
            # pprint(config)
            # print(config["name"])
            return not baseline or all(
                [
                    config["values"].get("memory_only") in [False, None],
                    config["values"].get("num_clusters") in [20, None],
                    config["values"].get("cores_per_cluster") in [1, None],
                    config["values"].get("mode") in ["serial", None],
                    # config["values"]["mode"].lower() == "serial",
                    # config["values"]["run_ahead"].lower() == "serial",
                ]
            )

        baseline_bench_configs = [config for config in bench_configs if is_baseline(config)]

        print(bench_name)
        num_rows = len(baseline_bench_configs)
        for input_idx, bench_config in enumerate(baseline_bench_configs):
            # sim_input_cols = SIMULATE_INPUT_COLS
            input_cols = BENCHMARK_INPUT_COLS[bench_name]
            if row_idx % 2 == 0:
                table += r"\rowcolor{gray!10}"
            if input_idx == 0:
                table += r"\multirow[t]{" + str(num_rows) + r"}{*}{"
                table += r"\shortstack[l]{" + get_name(bench_name) + r"}}"
                table += r" & \multirow[t]{" + str(num_rows) + r"}{*}{"
                table += r"\shortstack[l]{" + get_kind(bench_name) + "}}"
            else:
                table += " & "

            values = bench_config["values"]
            inputs = {k: values[k.removeprefix("input_")] for k in input_cols}
            # sim_inputs = {k: values[k.removeprefix("input_")] for k in sim_input_cols}
            table += r" & "
            for i, (k, v) in enumerate(inputs.items()):
                if i > 0:
                    table += ", "
                k = [kk.strip() for kk in k.removeprefix("input_").split("_")]
                table += "{}={}".format(" ".join(k), v)
            table += r" \\" + "\n"
            row_idx += 1

    print(table)
    utils.copy_to_clipboard(table)
    print("copied table to clipboard")


@main.command()
@click.option("--path", default=DEFAULT_BENCH_FILE, help="Path to materialized benchmark config")
def list(path):
    print("loading", path)
    b = Benchmarks(path)

    benchmark_names = list(b.benchmarks.keys())
    pprint(benchmark_names)


if __name__ == "__main__":
    main()
