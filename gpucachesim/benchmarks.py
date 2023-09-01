import click
import yaml
from pathlib import Path
from os import PathLike
import typing

from gpucachesim import ROOT_DIR

REPO_ROOT_DIR = ROOT_DIR.parent
DEFAULT_BENCH_FILE = REPO_ROOT_DIR / "test-apps/test-apps-materialized.yml"


class SimConfig(typing.TypedDict):
    gpgpu_clock_domains: str

    # @property
    # def core_clock_speed(self) -> int:
    #     self.gpgpu_clock_domains()
    #
    # @property
    # def num_cores(self) -> int:
    #     kk


class GPUConfig(typing.TypedDict):
    sim: SimConfig


class ProfileConfig(typing.TypedDict):
    profile_dir: PathLike


class SimulateConfig(typing.TypedDict):
    profile_dir: PathLike


class BenchConfig(typing.TypedDict):
    profile: ProfileConfig
    simulate: SimulateConfig


class Benchmarks:
    def __init__(self, path: PathLike) -> None:
        """load the materialized benchmark config"""

        with open(path or DEFAULT_BENCH_FILE, "rb") as f:
            benchmarks = yaml.safe_load(f)

        self.benchmarks = benchmarks["benchmarks"]

    def __getitem__(self, bench_name: str):
        return self.benchmarks[bench_name]

    def get_bench_config(self, bench_name: str, input_idx: int):
        return self.benchmarks[bench_name][input_idx]


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
