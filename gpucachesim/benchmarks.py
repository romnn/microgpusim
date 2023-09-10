import click
from pathlib import Path
from os import PathLike
from enum import Enum
import typing
from typing import Dict, Any, Sequence, Optional, Generic, TypeVar
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from gpucachesim import ROOT_DIR

REPO_ROOT_DIR = ROOT_DIR.parent
DEFAULT_BENCH_FILE = REPO_ROOT_DIR / "test-apps/test-apps-materialized.yml"


class GPUConfig:
    def __init__(self, config) -> None:
        self.config = config

    @property
    def _clock_domains(self) -> Dict[str, float]:
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
    AccelsimSimulate = "AccelsimSimulate"
    AccelsimTrace = "AccelsimTrace"
    PlaygroundSimulate = "PlaygroundSimulate"
    Trace = "Trace"


class GenericBenchmarkConfig(typing.TypedDict):
    repetitions: int
    timeout: Optional[str]
    concurrency: Optional[int]
    enabled: bool
    results_dir: PathLike


class ProfileConfig(typing.TypedDict):
    profile_dir: PathLike


class ProfileTargetConfig(typing.NamedTuple):
    value: ProfileConfig


class SimulateConfig(typing.TypedDict):
    stats_dir: PathLike
    traces_dir: PathLike
    accelsim_traces_dir: PathLike
    parallel: Optional[bool]


class SimulateTargetConfig(typing.NamedTuple):
    value: SimulateConfig


class TraceConfig(typing.TypedDict):
    traces_dir: PathLike
    save_json: bool
    full_trace: bool


class TraceTargetConfig(typing.NamedTuple):
    value: TraceConfig


class AccelsimSimulateConfig(typing.TypedDict):
    trace_config: PathLike
    inter_config: PathLike
    config_dir: PathLike
    config: PathLike
    traces_dir: PathLike
    stats_dir: PathLike


class AccelsimSimulateTargetConfig(typing.NamedTuple):
    value: AccelsimSimulateConfig


class AccelsimTraceConfig(typing.TypedDict):
    traces_dir: PathLike


class AccelsimTraceTargetConfig(typing.NamedTuple):
    value: AccelsimTraceConfig


class PlaygroundSimulateConfig(AccelsimSimulateConfig):
    pass


class PlaygroundSimulateTargetConfig(typing.NamedTuple):
    value: PlaygroundSimulateConfig


T = TypeVar("T")


# class BenchConfig(typing.TypedDict):
class BenchConfig(typing.TypedDict, Generic[T]):
    name: str
    benchmark_idx: int
    uid: str

    path: PathLike
    executable: PathLike

    values: Dict[str, Any]
    args: Sequence[str]
    input_idx: int

    common: GenericBenchmarkConfig

    target: str
    target_config: T
    # target_config: (
    #     ProfileConfig
    #     | TraceConfig
    #     | SimulateConfig
    #     | AccelsimTraceConfig
    #     | AccelsimSimulateConfig
    #     | PlaygroundSimulateConfig
    # )


class BenchmarkLoader(SafeLoader):
    pass


def construct_profile_target_config(self, node):
    return ProfileTargetConfig(self.construct_mapping(node))


def construct_trace_target_config(self, node):
    return TraceTargetConfig(self.construct_mapping(node))


def construct_simulate_target_config(self, node):
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
BenchmarkLoader.add_constructor("!AccelsimSimulate", construct_accelsim_simulate_target_config)
BenchmarkLoader.add_constructor("!AccelsimTrace", construct_accelsim_trace_target_config)
BenchmarkLoader.add_constructor("!PlaygroundSimulate", construct_playground_simulate_target_config)


class Benchmarks:
    def __init__(self, path: PathLike) -> None:
        """load the materialized benchmark config"""

        with open(path or DEFAULT_BENCH_FILE, "rb") as f:
            benchmarks = yaml.load(f, Loader=BenchmarkLoader)

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


@click.command()
@click.option("--path", default=DEFAULT_BENCH_FILE, help="Path to materialized benchmark config")
def main(path):
    from pprint import pprint

    print(path)
    b = Benchmarks(path)

    benchmark_names = list(b.benchmarks.keys())
    pprint(benchmark_names)


if __name__ == "__main__":
    main()
