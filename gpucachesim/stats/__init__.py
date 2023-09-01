import click

import gpucachesim.stats.stats as stats
import gpucachesim.stats.native as native
from gpucachesim.benchmarks import Benchmarks


@click.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--bench", help="Benchmark name")
@click.option("--input", default=0, help="Input index")
def main(path, bench, input):
    from pprint import pprint

    b = Benchmarks(path)
    if bench is None:
        raise NotImplemented
    print(bench, input)
    bench_config = b.get_bench_config(bench, input)
    # pprint(bench_config)

    our_stats = stats.Stats(bench_config["simulate"])
    native_stats = native.Stats(bench_config["simulate"])


if __name__ == "__main__":
    main()
