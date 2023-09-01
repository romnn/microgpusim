import click
import yaml
from pathlib import Path
from os import PathLike
from typing import Optional

from gpucachesim import ROOT_DIR

REPO_ROOT_DIR = ROOT_DIR.parent
DEFAULT_BENCH_FILE = REPO_ROOT_DIR / "test-apps/test-apps-materialized.yml"


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
