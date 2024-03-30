import shutil
import click
from pathlib import Path
import os
from enum import Enum
import typing
import yaml
from pprint import pprint
from collections import defaultdict

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from gpucachesim import REPO_ROOT_DIR

import gpucachesim.utils as utils

BASELINE = dict(
    cores_per_cluster=1,
    num_clusters=28,
)

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
SIMULATE_INPUT_COLS = SIMULATE_FUNCTIONAL_CONFIG_COLS + SIMULATE_EXECUTION_CONFIG_COLS


BENCHMARK_INPUT_COL_LABELS = {
    "benchmark": "Benchmark",
    "input_dtype": "Data type",
    "input_length": "Length",
    "input_rows": "Number of rows",
    "input_m": "m",
    "input_n": "n",
    "input_p": "p",
    "input_dim": "Dimensions",
    "input_variant": "Variant",
    "input_size": "Size",
    # this is required for per kernel output
    "kernel_name": "Kernel",
    "kernel_launch_id": "Kernel ID",
}

BENCHMARK_INPUT_COLS = {
    "vectorAdd": ["input_dtype", "input_length"],
    "matrixmul": ["input_dtype", "input_rows"],
    "simple_matrixmul": ["input_dtype", "input_m", "input_n", "input_p"],
    "transpose": ["input_dim", "input_variant"],
    "babelstream": ["input_size"],
}
ALL_BENCHMARK_INPUT_COLS = set(utils.flatten(BENCHMARK_INPUT_COLS.values()))

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

BENCH_TARGET_INDEX_COLS = ["target", "benchmark"]

PREVIEW_COLS = list(BENCH_TARGET_INDEX_COLS + ["input_id"] + INDEX_COLS + SIMULATE_INPUT_COLS)

RATE_COLUMNS = [
    "l2_hit_rate",
    "l2_miss_rate",
    "l2_read_hit_rate",
    "l2_write_hit_rate",
    "l2_read_miss_rate",
    "l2_write_miss_rate",
    "l2_read_miss_rate",
    "l2_write_miss_rate",
    "l1_hit_rate",
    "l1_global_hit_rate",
    "l1_local_hit_rate",
    "mean_blocks_per_sm",
    "mean_blocks_per_sm_all_kernels",
]

NON_NUMERIC_COLS = {
    **{
        "target": "first",
        "benchmark": "first",
        "Host Name": "first",
        "Process Name": "first",
        "device": "first",
        "context_id": "first",
        "is_release_build": "first",
        "kernel_function_signature": "first",
        "kernel_name": "first",
        "kernel_name_mangled": "first",
        "input_id": "first",
        # "input_memory_only": "first",
        # "input_mode": "first",
        # makes no sense to aggregate
        "cores_per_cluster": "first",
        "num_clusters": "first",
        "total_cores": "first",
        # cannot measure trace time per kernel
        "trace_time_sec": "first",
    },
    # inputs should not be aggregated
    **{col: "first" for col in SIMULATE_INPUT_COLS},
    **{col: "first" for col in ALL_BENCHMARK_INPUT_COLS},
}


def _map_dtype(dtype: str) -> str:
    match dtype.lower():
        case "category" | "str":
            # return "string"
            return "object"
        case "bool":
            return "bool"
        case "float" | "int":
            return "float"
        case other:
            raise ValueError("unknown dtype {}".format(other))


SPECIAL_DTYPES = {
    # **{col: "float64" for col in stats_df.columns},
    # **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    "target": "category",
    "benchmark": "category",
    "Host Name": "str",
    "Process Name": "str",
    "device": "str",
    "context_id": "float",
    "is_release_build": "bool",
    "kernel_function_signature": "str",
    "kernel_name": "str",
    "kernel_name_mangled": "str",
    "input_id": "float",
    # "input_memory_only": "first",
    # "input_mode": "first",
    # makes no sense to aggregate
    "cores_per_cluster": "float",
    "num_clusters": "float",
    "total_cores": "float",
    "trace_time_sec": "float",
    # functional and exec simulation inputs
    "input_memory_only": "bool",
    "input_num_clusters": "float",
    "input_cores_per_cluster": "float",
    "input_mode": "category",
    "input_threads": "float",
    "input_run_ahead": "float",
    # benchmark inputs
    "input_m": "float",
    "input_length": "float",
    "input_variant": "category",
    "input_dtype": "float",
    "input_size": "float",
    "input_n": "float",
    "input_dim": "float",
    "input_rows": "float",
    "input_p": "float",
    # memory stats
    "access_kind": "category",
    "access_status": "category",
    "memory_space": "category",
}
SPECIAL_DTYPES = {col: _map_dtype(dtype) for col, dtype in SPECIAL_DTYPES.items()}
missing_dtypes = set(NON_NUMERIC_COLS.keys()) - set(SPECIAL_DTYPES.keys())
assert len(missing_dtypes) == 0, "missing dtypes for {}".format(missing_dtypes)


CATEGORICAL_COLS = set([col for col, dtype in SPECIAL_DTYPES.items() if dtype == "category"])


def benchmark_name_human_readable(name: str) -> str:
    match name.lower():
        case "vectoradd":
            return "VectorAdd"
        case "matrixmul":
            return "Matrixmul"
        case "simple_matrixmul":
            return "Naive Matrixmul"
        case "transpose":
            return "Transpose"
        case "babelstream":
            return "BabelStream"
        case other:
            return str(other)


# def default_dtypes() -> typing.Dict[str, str]:
#     special_dtypes = {
#         # **{col: "float64" for col in stats_df.columns},
#         # **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
#         "target": "str",
#         "benchmark": "str",
#         "Host Name": "str",
#         "Process Name": "str",
#         "device": "str",
#         "context_id": "float",
#         "is_release_build": "bool",
#         "kernel_function_signature": "str",
#         "kernel_name": "str",
#         "kernel_name_mangled": "str",
#         "input_id": "float",
#         # "input_memory_only": "first",
#         # "input_mode": "first",
#         # makes no sense to aggregate
#         "cores_per_cluster": "float",
#         "num_clusters": "float",
#         "total_cores": "float",
#         "input_memory_only": "bool",
#         "input_num_clusters": "float",
#         "input_cores_per_cluster": "float",
#         "input_mode": "str",
#         "input_threads": "float",
#         "input_run_ahead": "float",
#     }
#     missing_dtypes = set(benchmarks.NON_NUMERIC_COLS.keys()) - set(special_dtypes.keys())
#     assert len(missing_dtypes) == 0, "missing dtypes for {}".format(missing_dtypes)
#
#     dtypes = {
#         **{col: "float64" for col in stats_df.columns},
#         **special_dtypes,
#     }
#
#     return dtypes


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
    path: Path
    config: Config

    def __init__(self, path: typing.Optional[os.PathLike]) -> None:
        """load the materialized benchmark config"""
        if path is None:
            self.path = DEFAULT_BENCH_FILE
        else:
            self.path = Path(path)
        with open(self.path, "rb") as f:
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
def count_bench_configs(path):
    print("loading", path)
    b = Benchmarks(path)
    benches = b.benchmarks[Target.Simulate.value]

    total_bench_configs = sum([len(bench_configs) for bench_configs in benches.values()])
    print("total bench configs: {}".format(total_bench_configs))


@main.command()
@click.option("--path", default=DEFAULT_BENCH_FILE, help="Path to materialized benchmark config")
@click.option("--baseline", type=bool, default=True, help="Baseline configurations")
def table(path, baseline):
    print("loading", path)
    b = Benchmarks(path)
    benches = b.benchmarks[Target.Simulate.value]

    table = r"\rowcolor{gray!10} Benchmark & Type & \multicolumn{2}{c}{Input parameters} \\ \hline"
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

    # row_idx = 0
    for bench_name, bench_configs in benches.items():

        def is_baseline(config):
            return not baseline or all(
                [
                    config["values"].get("memory_only") in [False, None],
                    config["values"].get("num_clusters") in [int(BASELINE["num_clusters"]), None],
                    config["values"].get("cores_per_cluster") in [int(BASELINE["cores_per_cluster"]), None],
                    config["values"].get("mode") in ["serial", None],
                ]
            )

        baseline_bench_configs = [config for config in bench_configs if is_baseline(config)]

        print(bench_name)

        bench_input_values = defaultdict(set)

        for bench_config in baseline_bench_configs:
            input_cols = BENCHMARK_INPUT_COLS[bench_name]
            values = bench_config["values"]
            inputs = {k: values[k.removeprefix("input_")] for k in input_cols}
            for k, v in inputs.items():
                bench_input_values[k].add(v)

        for i, (k, input_values) in enumerate(bench_input_values.items()):
            if i == 0:
                table += r"\multirow[t]{" + str(len(bench_input_values)) + "}{*}{"
                table += r"\shortstack[l]{" + get_name(bench_name) + "}}"
                table += r" & \multirow[t]{" + str(len(bench_input_values)) + "}{*}{"
                table += r"\shortstack[l]{" + get_kind(bench_name) + "}}"
            else:
                table += r" & & "

            input_values = sorted([v for v in input_values])
            key = " ".join([kk.strip() for kk in k.removeprefix("input_").split("_")])
            table += r" & \textbf{" + str(key) + r"}"
            table += r" & " + (", ".join([str(v) for v in input_values]))

        table += r" \\ \hline" + "\n"

        # num_rows = len(baseline_bench_configs)
        # for input_idx, bench_config in enumerate(baseline_bench_configs):
        #     input_cols = BENCHMARK_INPUT_COLS[bench_name]
        #     if row_idx % 2 == 0:
        #         table += r"\rowcolor{gray!10}"
        #     if input_idx == 0:
        #         table += r"\multirow[t]{" + str(num_rows) + r"}{*}{"
        #         table += r"\shortstack[l]{" + get_name(bench_name) + r"}}"
        #         table += r" & \multirow[t]{" + str(num_rows) + r"}{*}{"
        #         table += r"\shortstack[l]{" + get_kind(bench_name) + "}}"
        #     else:
        #         table += " & "
        #
        #     values = bench_config["values"]
        #     inputs = {k: values[k.removeprefix("input_")] for k in input_cols}
        #     table += r" & "
        #     for i, (k, v) in enumerate(inputs.items()):
        #         if i > 0:
        #             table += ", "
        #         k = [kk.strip() for kk in k.removeprefix("input_").split("_")]
        #         table += "{}={}".format(" ".join(k), v)
        #     table += r" \\" + "\n"
        #     row_idx += 1

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


@main.command()
@click.option("--path", default=DEFAULT_BENCH_FILE, help="Path to materialized benchmark config")
def fix(path):
    print("loading", path)
    b = Benchmarks(path)

    result_dirs: typing.Set[Path] = set()
    for target, target_configs in b.benchmarks.items():
        for name, bench_configs in target_configs.items():
            print(target, name)
            for bench_config in bench_configs:
                result_dirs.add(Path(bench_config["results_dir"]))

    valid = set(
        [
            "profile",
            "simulate",
            "accelsim-simulate",
            "playground-simulate",
            "exec-driven-simulate",
            "trace",
            "accelsim-trace",
        ]
    )
    for result_dir in result_dirs:
        if not result_dir.is_dir():
            print("SKIP:", result_dir)
            continue
        print("fixing", result_dir)

        try:
            # if (result_dir / "Profile").is_dir() and (result_dir / "profile").is_dir():
            #     # Profile is newer
            #     shutil.rmtree(result_dir / "profile")
            if (result_dir / "profile/Profile").is_dir():
                os.rename(result_dir / "profile", result_dir / "profile-tmp")
                os.rename(result_dir / "profile/Profile", result_dir / "profile")
                shutil.rmtree(result_dir / "profile-tmp")
                os.rmdir(result_dir / "profile-tmp")
        except FileNotFoundError:
            pass

        try:
            if (result_dir / "profile-tmp").is_dir():
                os.rename(result_dir / "profile-tmp/Profile", result_dir / "profile")
                shutil.rmtree(result_dir / "profile-tmp")
                os.rmdir(result_dir / "profile-tmp")
        except FileNotFoundError:
            pass

        try:
            if (result_dir / "Profile").is_dir() and (result_dir / "profile").is_dir():
                # Profile is newer
                shutil.rmtree(result_dir / "profile")
            os.rename(result_dir / "Profile", result_dir / "profile")
            # shutil.move(result_dir / "Profile", result_dir / "profile")
        except FileNotFoundError:
            pass

        # except shutil.Error as e:
        #     if "already exists" in str(e):
        #         shutil.rmtree(result_dir / "profile")
        #         shutil.move(result_dir / "Profile", result_dir / "profile")

        try:
            os.rename(result_dir / "sim", result_dir / "simulate")
            # shutil.move(result_dir / "sim", result_dir / "simulate")
        except FileNotFoundError:
            pass

        try:
            if (result_dir / "Simulate").is_dir() and (result_dir / "simulate").is_dir():
                # Simulate is newer
                shutil.rmtree(result_dir / "simulate")
            os.rename(result_dir / "Simulate", result_dir / "simulate")
            # shutil.move(result_dir / "Simulate", result_dir / "simulate")
        except FileNotFoundError:
            pass

        try:
            # shutil.move(
            os.rename(
                result_dir / "sim/exec-driven",
                result_dir / "simulate/exec-driven-simulate",
            )
        except FileNotFoundError:
            pass

        try:
            # shutil.move(
            os.rename(
                result_dir / "simulate/exec-driven",
                result_dir / "simulate/exec-driven-simulate",
            )
        except FileNotFoundError:
            pass

        try:
            os.rename(result_dir / "playground-sim", result_dir / "playground-simulate")
        except FileNotFoundError:
            pass

        try:
            os.rename(result_dir / "accelsim-sim", result_dir / "accelsim-simulate")
        except FileNotFoundError:
            pass

        if (result_dir / "profile-tmp").is_dir():
            assert (result_dir / "profile").is_dir()
            os.rmdir(result_dir / "profile-tmp")

        remaining_dir_names = [dir.name for dir in result_dir.iterdir()]
        if not all([dir_name in valid for dir_name in remaining_dir_names]):
            raise ValueError(remaining_dir_names)


if __name__ == "__main__":
    main()
